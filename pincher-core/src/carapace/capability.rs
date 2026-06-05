//! Capability gate for the Carapace Bridge.
//!
//! The [`CapabilityGate`] is the security enforcement layer between WASM guest
//! code and host functions.  Every host function invocation passes through the
//! gate, which checks the guest's [`CapabilityManifest`] for the appropriate
//! [`Permission`] before allowing the call to proceed.
//!
//! ## Hermit crab metaphor
//!
//! If the carapace is the hard outer shell protecting the crab, the capability
//! gate is the **hinge** of each leg — it only opens (allows a host call) when
//! the right permission token is present.  Without the token, the leg stays
//! locked and the call is denied.

use crate::capability::manifest::{CapabilityManifest, Permission};
use std::time::Duration;

// ── Error type ────────────────────────────────────────────────────────

/// Errors that can occur during capability gate checks.
#[derive(Debug, thiserror::Error)]
pub enum GateError {
    /// The guest does not possess the required permission for the requested
    /// host function.
    #[error("capability denied: guest lacks permission for {function:?} — {reason}")]
    PermissionDenied {
        /// The host function that was requested.
        function: HostFunction,
        /// Human-readable reason for denial.
        reason: String,
    },

    /// The guest's capability manifest has expired or is missing.
    #[error("capability manifest is invalid: {0}")]
    InvalidManifest(String),

    /// The requested operation targets a path or resource outside the
    /// guest's allowed scope.
    #[error("resource out of scope: {resource} is not covered by any permission")]
    OutOfScope {
        /// The resource (path, host, binary) that was out of scope.
        resource: String,
    },
}

/// Alias for results produced by the capability gate.
pub type GateResult<T> = Result<T, GateError>;

// ── HostFunction ──────────────────────────────────────────────────────

/// Enumeration of host functions that can be invoked by WASM guests.
///
/// Each variant carries the minimal context needed to perform a capability
/// check.  For example, `FileRead` carries the path the guest wants to read
/// so the gate can verify it against `Permission::FsRead`.
#[derive(Debug, Clone, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub enum HostFunction {
    /// Read a file from the host filesystem.
    FileRead {
        /// Path the guest wants to read.
        path: String,
    },
    /// Write a file on the host filesystem.
    FileWrite {
        /// Path the guest wants to write.
        path: String,
    },
    /// Make an outbound network request.
    NetRequest {
        /// Hostname or IP the guest wants to connect to.
        host: String,
        /// Port number.
        port: u16,
    },
    /// Execute a shell command on the host.
    ShellExec {
        /// Binary the guest wants to execute.
        binary: String,
    },
    /// Read an environment variable from the host.
    EnvRead {
        /// Name of the environment variable.
        name: String,
    },
}

impl HostFunction {
    /// Returns a short human-readable name for this host function (without
    /// the arguments), useful for logging and error messages.
    pub fn kind(&self) -> &'static str {
        match self {
            Self::FileRead { .. } => "file_read",
            Self::FileWrite { .. } => "file_write",
            Self::NetRequest { .. } => "net_request",
            Self::ShellExec { .. } => "shell_exec",
            Self::EnvRead { .. } => "env_read",
        }
    }

    /// Returns the set of [`Permission`] variants that would authorise this
    /// host function call, without checking path/host/binary specificity.
    ///
    /// This is used for fast pre-checks — if none of these permission
    /// *kinds* appear in the manifest, the call can be rejected immediately
    /// without examining arguments.
    pub fn required_permission_kind(&self) -> PermissionKind {
        match self {
            Self::FileRead { .. } => PermissionKind::FsRead,
            Self::FileWrite { .. } => PermissionKind::FsWrite,
            Self::NetRequest { .. } => PermissionKind::NetConnect,
            Self::ShellExec { .. } => PermissionKind::Execute,
            Self::EnvRead { .. } => PermissionKind::EnvRead,
        }
    }
}

/// Broad categories of permissions, used for fast pre-filtering.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PermissionKind {
    /// Filesystem read.
    FsRead,
    /// Filesystem write.
    FsWrite,
    /// Outbound network connection.
    NetConnect,
    /// Binary execution.
    Execute,
    /// Environment variable read.
    EnvRead,
}

// ── CapabilityGate ────────────────────────────────────────────────────

/// The capability gate enforces permission checks at the host-function
/// boundary.
///
/// It is constructed with a reference to the guest's [`CapabilityManifest`]
/// and exposes a single [`check`](CapabilityGate::check) method that
/// determines whether a given [`HostFunction`] invocation is allowed.
///
/// # Example
///
/// ```rust,ignore
/// use pincher_core::carapace::capability::{CapabilityGate, HostFunction};
/// use pincher_core::capability::manifest::CapabilityManifest;
///
/// let manifest = CapabilityManifest::empty(reflex_id);
/// let gate = CapabilityGate::new(&manifest);
///
/// let result = gate.check(&HostFunction::FileRead {
///     path: "/etc/passwd".into(),
/// });
/// assert!(result.is_err());
/// ```
#[derive(Debug)]
pub struct CapabilityGate<'a> {
    /// The manifest describing what the guest is allowed to do.
    manifest: &'a CapabilityManifest,
}

impl<'a> CapabilityGate<'a> {
    /// Create a new capability gate backed by the given manifest.
    pub fn new(manifest: &'a CapabilityManifest) -> Self {
        tracing::debug!(
            reflex_id = %manifest.reflex_id,
            permissions = manifest.permissions.len(),
            "capability gate created"
        );
        Self { manifest }
    }

    /// Check whether the guest is allowed to invoke the given host function.
    ///
    /// This performs a two-phase check:
    ///
    /// 1. **Kind check** — verify that the manifest contains at least one
    ///    permission of the right *kind* (e.g. `FsRead` for `FileRead`).
    ///    If no permission of the right kind exists, the call is denied
    ///    immediately.
    ///
    /// 2. **Scope check** — verify that the specific resource (path, host,
    ///    binary) is covered by at least one permission.  This uses the
    ///    `covers_*` methods on [`Permission`].
    pub fn check(&self, function: &HostFunction) -> GateResult<()> {
        tracing::debug!(function = ?function, "capability gate check");

        match function {
            HostFunction::FileRead { path } => {
                if self.manifest.has_fs_read(path) {
                    tracing::debug!(path, "file_read permitted");
                    Ok(())
                } else {
                    tracing::warn!(
                        path,
                        "file_read denied — no FsRead permission covers this path"
                    );
                    Err(GateError::PermissionDenied {
                        function: function.clone(),
                        reason: format!("no FsRead permission covers path {:?}", path),
                    })
                }
            }

            HostFunction::FileWrite { path } => {
                if self.manifest.has_fs_write(path) {
                    tracing::debug!(path, "file_write permitted");
                    Ok(())
                } else {
                    tracing::warn!(
                        path,
                        "file_write denied — no FsWrite permission covers this path"
                    );
                    Err(GateError::PermissionDenied {
                        function: function.clone(),
                        reason: format!("no FsWrite permission covers path {:?}", path),
                    })
                }
            }

            HostFunction::NetRequest { host, port } => {
                if self.manifest.allows_network() {
                    // Additional scope check: does any NetConnect permission
                    // cover this host?
                    let covered = self.manifest.permissions.iter().any(|p| {
                        if let Permission::NetConnect {
                            host: allowed_host,
                            port: allowed_port,
                        } = p
                        {
                            // Wildcard host "*" matches anything.
                            let host_ok = allowed_host == "*" || allowed_host == host;
                            let port_ok = *allowed_port == 0 || *allowed_port == *port;
                            host_ok && port_ok
                        } else {
                            false
                        }
                    });

                    if covered {
                        tracing::debug!(host, port, "net_request permitted");
                        Ok(())
                    } else {
                        tracing::warn!(
                            host,
                            port,
                            "net_request denied — no NetConnect permission covers this host:port"
                        );
                        Err(GateError::OutOfScope {
                            resource: format!("{}:{}", host, port),
                        })
                    }
                } else {
                    tracing::warn!(
                        host,
                        port,
                        "net_request denied — no network permission in manifest"
                    );
                    Err(GateError::PermissionDenied {
                        function: function.clone(),
                        reason: "no NetConnect permission in manifest".into(),
                    })
                }
            }

            HostFunction::ShellExec { binary } => {
                if self.manifest.has_execute(binary) {
                    tracing::debug!(binary, "shell_exec permitted");
                    Ok(())
                } else {
                    tracing::warn!(
                        binary,
                        "shell_exec denied — no Execute permission for this binary"
                    );
                    Err(GateError::PermissionDenied {
                        function: function.clone(),
                        reason: format!("no Execute permission covers binary {:?}", binary),
                    })
                }
            }

            HostFunction::EnvRead { name } => {
                // Environment variable reads are a special case.
                // For MVP, we allow reading any environment variable if the
                // manifest contains at least one `Permission::EnvRead`
                // (represented as an `FsRead` on a virtual "/env/*" path).
                // If no env-specific permission exists, we fall back to
                // checking for a general `FsRead { path: "/env/*" }` pattern.
                let env_allowed = self.manifest.permissions.iter().any(|p| {
                    match p {
                        Permission::FsRead { path } => {
                            // Convention: FsRead on "/env/*" grants env read.
                            path == "/env/*" || path == "/*" || path == "*"
                        }
                        _ => false,
                    }
                });

                if env_allowed {
                    tracing::debug!(name, "env_read permitted");
                    Ok(())
                } else {
                    tracing::warn!(name, "env_read denied — no env read permission in manifest");
                    Err(GateError::PermissionDenied {
                        function: function.clone(),
                        reason: format!(
                            "no FsRead permission on \"/env/*\" covers env var {:?}",
                            name
                        ),
                    })
                }
            }
        }
    }

    /// Returns a snapshot of all host-function kinds that the guest is
    /// allowed to invoke (based on the manifest's permission *kinds*, not
    /// specific scopes).  Useful for pre-filtering or UI display.
    pub fn allowed_function_kinds(&self) -> Vec<PermissionKind> {
        let mut kinds = Vec::new();
        for p in &self.manifest.permissions {
            match p {
                Permission::FsRead { .. } => {
                    if !kinds.contains(&PermissionKind::FsRead) {
                        kinds.push(PermissionKind::FsRead);
                    }
                }
                Permission::FsWrite { .. } => {
                    if !kinds.contains(&PermissionKind::FsWrite) {
                        kinds.push(PermissionKind::FsWrite);
                    }
                }
                Permission::NetConnect { .. } => {
                    if !kinds.contains(&PermissionKind::NetConnect) {
                        kinds.push(PermissionKind::NetConnect);
                    }
                }
                Permission::NetNone => {}
                Permission::Execute { .. } => {
                    if !kinds.contains(&PermissionKind::Execute) {
                        kinds.push(PermissionKind::Execute);
                    }
                }
            }
        }
        // EnvRead is derived from FsRead on "/env/*".
        if kinds.contains(&PermissionKind::FsRead) {
            kinds.push(PermissionKind::EnvRead);
        }
        kinds
    }
}

// ── SandboxPolicy ─────────────────────────────────────────────────────

/// A sandbox policy that combines the capability gate with resource limits.
///
/// This is the complete set of constraints applied to a WASM guest during
/// execution.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct SandboxPolicy {
    /// Maximum amount of WASM linear memory the guest may allocate, in bytes.
    pub memory_limit: u64,
    /// Maximum execution time before the guest is terminated.
    pub timeout: Duration,
    /// List of named capabilities the guest is allowed to use.  These names
    /// must correspond to entries in the guest's
    /// [`CapabilityManifest::required_capabilities`].
    pub allowed_capabilities: Vec<String>,
}

impl Default for SandboxPolicy {
    fn default() -> Self {
        Self {
            memory_limit: 64 * 1024 * 1024, // 64 MiB
            timeout: Duration::from_secs(30),
            allowed_capabilities: vec![],
        }
    }
}

impl SandboxPolicy {
    /// Create a policy with the given memory limit and timeout.
    pub fn new(memory_limit: u64, timeout: Duration) -> Self {
        Self {
            memory_limit,
            timeout,
            allowed_capabilities: vec![],
        }
    }

    /// Create a maximally restrictive policy — minimum memory, short timeout,
    /// no capabilities.
    pub fn restricted() -> Self {
        Self {
            memory_limit: 16 * 1024 * 1024, // 16 MiB
            timeout: Duration::from_secs(5),
            allowed_capabilities: vec![],
        }
    }

    /// Grant an additional named capability.
    pub fn allow_capability(&mut self, cap: impl Into<String>) {
        let cap = cap.into();
        if !self.allowed_capabilities.contains(&cap) {
            self.allowed_capabilities.push(cap);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::capability::manifest::Permission;
    use uuid::Uuid;

    fn manifest_with_perms(perms: Vec<Permission>) -> CapabilityManifest {
        CapabilityManifest {
            reflex_id: Uuid::new_v4(),
            permissions: perms,
            required_capabilities: vec![],
        }
    }

    #[test]
    fn file_read_allowed_when_permission_covers_path() {
        let manifest = manifest_with_perms(vec![Permission::FsRead {
            path: "/tmp/*".into(),
        }]);
        let gate = CapabilityGate::new(&manifest);
        assert!(gate
            .check(&HostFunction::FileRead {
                path: "/tmp/data.txt".into()
            })
            .is_ok());
    }

    #[test]
    fn file_read_denied_when_no_matching_permission() {
        let manifest = manifest_with_perms(vec![Permission::FsRead {
            path: "/tmp/*".into(),
        }]);
        let gate = CapabilityGate::new(&manifest);
        assert!(gate
            .check(&HostFunction::FileRead {
                path: "/etc/shadow".into()
            })
            .is_err());
    }

    #[test]
    fn file_write_denied_with_readonly_permission() {
        let manifest = manifest_with_perms(vec![Permission::FsRead {
            path: "/tmp/*".into(),
        }]);
        let gate = CapabilityGate::new(&manifest);
        assert!(gate
            .check(&HostFunction::FileWrite {
                path: "/tmp/out.txt".into()
            })
            .is_err());
    }

    #[test]
    fn net_request_allowed_with_matching_host() {
        let manifest = manifest_with_perms(vec![Permission::NetConnect {
            host: "api.example.com".into(),
            port: 443,
        }]);
        let gate = CapabilityGate::new(&manifest);
        assert!(gate
            .check(&HostFunction::NetRequest {
                host: "api.example.com".into(),
                port: 443
            })
            .is_ok());
    }

    #[test]
    fn net_request_denied_with_wrong_host() {
        let manifest = manifest_with_perms(vec![Permission::NetConnect {
            host: "api.example.com".into(),
            port: 443,
        }]);
        let gate = CapabilityGate::new(&manifest);
        assert!(gate
            .check(&HostFunction::NetRequest {
                host: "evil.com".into(),
                port: 443
            })
            .is_err());
    }

    #[test]
    fn shell_exec_allowed_with_matching_binary() {
        let manifest = manifest_with_perms(vec![Permission::Execute {
            binary: "ls".into(),
        }]);
        let gate = CapabilityGate::new(&manifest);
        assert!(gate
            .check(&HostFunction::ShellExec {
                binary: "ls".into()
            })
            .is_ok());
    }

    #[test]
    fn shell_exec_denied_with_wrong_binary() {
        let manifest = manifest_with_perms(vec![Permission::Execute {
            binary: "ls".into(),
        }]);
        let gate = CapabilityGate::new(&manifest);
        assert!(gate
            .check(&HostFunction::ShellExec {
                binary: "rm".into()
            })
            .is_err());
    }

    #[test]
    fn env_read_allowed_with_env_convention() {
        let manifest = manifest_with_perms(vec![Permission::FsRead {
            path: "/env/*".into(),
        }]);
        let gate = CapabilityGate::new(&manifest);
        assert!(gate
            .check(&HostFunction::EnvRead {
                name: "HOME".into()
            })
            .is_ok());
    }

    #[test]
    fn empty_manifest_denies_everything() {
        let manifest = CapabilityManifest::empty(Uuid::new_v4());
        let gate = CapabilityGate::new(&manifest);
        assert!(gate
            .check(&HostFunction::FileRead {
                path: "/tmp/test".into()
            })
            .is_err());
        assert!(gate
            .check(&HostFunction::FileWrite {
                path: "/tmp/test".into()
            })
            .is_err());
        assert!(gate
            .check(&HostFunction::NetRequest {
                host: "example.com".into(),
                port: 80
            })
            .is_err());
        assert!(gate
            .check(&HostFunction::ShellExec {
                binary: "ls".into()
            })
            .is_err());
        assert!(gate
            .check(&HostFunction::EnvRead {
                name: "PATH".into()
            })
            .is_err());
    }

    #[test]
    fn sandbox_policy_default_is_reasonable() {
        let policy = SandboxPolicy::default();
        assert_eq!(policy.memory_limit, 64 * 1024 * 1024);
        assert_eq!(policy.timeout, Duration::from_secs(30));
        assert!(policy.allowed_capabilities.is_empty());
    }

    #[test]
    fn sandbox_policy_restricted_is_tight() {
        let policy = SandboxPolicy::restricted();
        assert_eq!(policy.memory_limit, 16 * 1024 * 1024);
        assert_eq!(policy.timeout, Duration::from_secs(5));
    }
}
