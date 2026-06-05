//! Host functions exposed to WASM guests.
//!
//! This module defines the host-side functions that guest WASM modules can
//! invoke.  Each host function is **gated by a capability check** — if the
//! guest does not hold the appropriate [`Permission`], the call is denied
//! before any side effect occurs.
//!
//! ## Current status: MVP stubs
//!
//! The host functions in this module are **stubs** that return mock data.
//! They perform the capability gate check (real) and then return a
//! placeholder response.  The real implementations (interacting with the
//! filesystem, network, etc.) are post-MVP.
//!
//! ## Hermit crab metaphor
//!
//! The host functions are the **pincers** of the hermit crab — they reach
//! out into the outside world (filesystem, network, shell) on behalf of
//! the crab (guest), but only when the crab has the right permissions
//! (capabilities) to do so.

use super::capability::{CapabilityGate, GateError, HostFunction};

// ── Error type ────────────────────────────────────────────────────────

/// Errors that can occur when invoking host functions.
#[derive(Debug, thiserror::Error)]
pub enum HostError {
    /// The capability gate denied the host function call.
    #[error("host function denied: {0}")]
    CapabilityDenied(#[from] GateError),

    /// The host function encountered an error during execution.
    ///
    /// In the MVP, this is unlikely since functions are stubs, but it
    /// reserves the variant for real implementations.
    #[error("host function execution error: {function} — {reason}")]
    ExecutionError {
        /// Which host function failed.
        function: String,
        /// Why it failed.
        reason: String,
    },

    /// The guest attempted to invoke a host function that does not exist.
    #[error("unknown host function: {0}")]
    UnknownFunction(String),
}

/// Alias for results produced by host functions.
pub type HostResult<T> = Result<T, HostError>;

// ── Host function responses ───────────────────────────────────────────

/// Response from a `file_read` host function call.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct FileReadResponse {
    /// The path that was read.
    pub path: String,
    /// The file contents (mock in MVP).
    pub content: String,
    /// Size of the file in bytes.
    pub size: usize,
}

/// Response from a `file_write` host function call.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct FileWriteResponse {
    /// The path that was written.
    pub path: String,
    /// Number of bytes written.
    pub bytes_written: usize,
    /// Whether the write succeeded.
    pub success: bool,
}

/// Response from a `net_request` host function call.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct NetRequestResponse {
    /// The URL that was requested.
    pub url: String,
    /// HTTP status code (mock in MVP).
    pub status: u16,
    /// Response body (mock in MVP).
    pub body: String,
}

/// Response from a `shell_exec` host function call.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ShellExecResponse {
    /// The binary that was executed.
    pub binary: String,
    /// Exit code of the process (mock in MVP).
    pub exit_code: i32,
    /// Standard output (mock in MVP).
    pub stdout: String,
    /// Standard error (mock in MVP).
    pub stderr: String,
}

/// Response from an `env_read` host function call.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct EnvReadResponse {
    /// The variable name that was read.
    pub name: String,
    /// The variable value (mock in MVP).
    pub value: String,
    /// Whether the variable was found.
    pub found: bool,
}

// ── HostFunctionDispatcher ────────────────────────────────────────────

/// Dispatches host function calls from WASM guests after capability checks.
///
/// Each method corresponds to a host function that guests can invoke.  The
/// dispatcher first checks the capability gate, then (in the MVP) returns
/// mock data.  In the post-MVP, these methods will perform real I/O.
///
/// # Example
///
/// ```rust,ignore
/// use pincher_core::carapace::host::HostFunctionDispatcher;
/// use pincher_core::carapace::capability::{CapabilityGate, HostFunction};
/// use pincher_core::capability::manifest::CapabilityManifest;
///
/// let manifest = CapabilityManifest::empty(reflex_id);
/// let gate = CapabilityGate::new(&manifest);
/// let dispatcher = HostFunctionDispatcher::new(&gate);
///
/// // This will be denied because the manifest has no FsRead permission.
/// let result = dispatcher.file_read("/tmp/test.txt");
/// assert!(result.is_err());
/// ```
#[derive(Debug)]
pub struct HostFunctionDispatcher<'a> {
    /// The capability gate that governs all host function calls.
    gate: &'a CapabilityGate<'a>,
}

impl<'a> HostFunctionDispatcher<'a> {
    /// Create a new dispatcher backed by the given capability gate.
    pub fn new(gate: &'a CapabilityGate<'a>) -> Self {
        Self { gate }
    }

    /// **File read** — read the contents of a file on the host filesystem.
    ///
    /// Requires `Permission::FsRead` covering `path`.
    ///
    /// **MVP**: Returns mock data.
    pub fn file_read(&self, path: &str) -> HostResult<FileReadResponse> {
        let function = HostFunction::FileRead { path: path.into() };
        self.gate.check(&function)?;

        tracing::info!(path, "file_read — returning mock data (MVP)");

        // MVP stub: return mock content.
        Ok(FileReadResponse {
            path: path.into(),
            content: format!("[carapace mock] contents of {}", path),
            size: format!("[carapace mock] contents of {}", path).len(),
        })
    }

    /// **File write** — write data to a file on the host filesystem.
    ///
    /// Requires `Permission::FsWrite` covering `path`.
    ///
    /// **MVP**: Returns a mock success response.
    pub fn file_write(&self, path: &str, _content: &[u8]) -> HostResult<FileWriteResponse> {
        let function = HostFunction::FileWrite { path: path.into() };
        self.gate.check(&function)?;

        tracing::info!(path, "file_write — returning mock response (MVP)");

        // MVP stub: pretend we wrote the data.
        Ok(FileWriteResponse {
            path: path.into(),
            bytes_written: _content.len(),
            success: true,
        })
    }

    /// **Network request** — make an outbound HTTP/HTTPS request.
    ///
    /// Requires `Permission::NetConnect` covering `host:port`.
    ///
    /// **MVP**: Returns a mock HTTP 200 response.
    pub fn net_request(
        &self,
        host: &str,
        port: u16,
        _path: &str,
    ) -> HostResult<NetRequestResponse> {
        let function = HostFunction::NetRequest {
            host: host.into(),
            port,
        };
        self.gate.check(&function)?;

        tracing::info!(host, port, "net_request — returning mock response (MVP)");

        // MVP stub: return a mock HTTP 200.
        Ok(NetRequestResponse {
            url: format!("http://{}:{}{}", host, port, _path),
            status: 200,
            body: format!("[carapace mock] response from {}:{}", host, port),
        })
    }

    /// **Shell execute** — run a binary on the host.
    ///
    /// Requires `Permission::Execute` for the given `binary`.
    ///
    /// **MVP**: Returns a mock successful execution.
    pub fn shell_exec(&self, binary: &str, _args: &[&str]) -> HostResult<ShellExecResponse> {
        let function = HostFunction::ShellExec {
            binary: binary.into(),
        };
        self.gate.check(&function)?;

        tracing::info!(binary, "shell_exec — returning mock response (MVP)");

        // MVP stub: pretend the command succeeded.
        Ok(ShellExecResponse {
            binary: binary.into(),
            exit_code: 0,
            stdout: format!("[carapace mock] output of {}", binary),
            stderr: String::new(),
        })
    }

    /// **Environment read** — read an environment variable.
    ///
    /// Requires `Permission::FsRead` on the virtual path `/env/*`.
    ///
    /// **MVP**: Returns a mock value.
    pub fn env_read(&self, name: &str) -> HostResult<EnvReadResponse> {
        let function = HostFunction::EnvRead { name: name.into() };
        self.gate.check(&function)?;

        tracing::info!(name, "env_read — returning mock response (MVP)");

        // MVP stub: return a mock value.
        Ok(EnvReadResponse {
            name: name.into(),
            value: format!("[carapace mock] value of {}", name),
            found: true,
        })
    }
}

// ── Host function registry ────────────────────────────────────────────

/// Registry of all host functions that can be exposed to WASM guests.
///
/// This is used when constructing the WASM linker to register the host
/// functions that guests can import.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub enum HostFunctionKind {
    /// Read a file from the host filesystem.
    FileRead,
    /// Write a file on the host filesystem.
    FileWrite,
    /// Make an outbound network request.
    NetRequest,
    /// Execute a shell command.
    ShellExec,
    /// Read an environment variable.
    EnvRead,
}

impl HostFunctionKind {
    /// Returns all host function kinds.
    pub fn all() -> &'static [HostFunctionKind] {
        &[
            HostFunctionKind::FileRead,
            HostFunctionKind::FileWrite,
            HostFunctionKind::NetRequest,
            HostFunctionKind::ShellExec,
            HostFunctionKind::EnvRead,
        ]
    }

    /// Returns the WASM module name under which this function is exported.
    ///
    /// All carapace host functions live under the `"carapace"` namespace.
    pub fn wasm_module_name(&self) -> &'static str {
        "carapace"
    }

    /// Returns the WASM function name for this host function kind.
    pub fn wasm_function_name(&self) -> &'static str {
        match self {
            Self::FileRead => "file_read",
            Self::FileWrite => "file_write",
            Self::NetRequest => "net_request",
            Self::ShellExec => "shell_exec",
            Self::EnvRead => "env_read",
        }
    }

    /// Returns the fully-qualified WASM import name
    /// (`"carapace::file_read"`, etc.).
    pub fn fully_qualified_name(&self) -> String {
        format!("{}::{}", self.wasm_module_name(), self.wasm_function_name())
    }
}

impl std::fmt::Display for HostFunctionKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.wasm_function_name())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::capability::manifest::{CapabilityManifest, Permission};
    use uuid::Uuid;

    fn make_gate(perms: Vec<Permission>) -> CapabilityGate<'static> {
        let manifest = Box::leak(Box::new(CapabilityManifest {
            reflex_id: Uuid::new_v4(),
            permissions: perms,
            required_capabilities: vec![],
        }));
        CapabilityGate::new(manifest)
    }

    #[test]
    fn file_read_stub_returns_mock_data() {
        let gate = make_gate(vec![Permission::FsRead {
            path: "/tmp/*".into(),
        }]);
        let dispatcher = HostFunctionDispatcher::new(&gate);
        let result = dispatcher.file_read("/tmp/test.txt").unwrap();
        assert_eq!(result.path, "/tmp/test.txt");
        assert!(result.content.contains("[carapace mock]"));
    }

    #[test]
    fn file_read_denied_without_permission() {
        let gate = make_gate(vec![]);
        let dispatcher = HostFunctionDispatcher::new(&gate);
        assert!(dispatcher.file_read("/tmp/test.txt").is_err());
    }

    #[test]
    fn file_write_stub_returns_mock_success() {
        let gate = make_gate(vec![Permission::FsWrite {
            path: "/tmp/*".into(),
        }]);
        let dispatcher = HostFunctionDispatcher::new(&gate);
        let result = dispatcher.file_write("/tmp/out.txt", b"hello").unwrap();
        assert!(result.success);
        assert_eq!(result.bytes_written, 5);
    }

    #[test]
    fn net_request_stub_returns_200() {
        let gate = make_gate(vec![Permission::NetConnect {
            host: "api.example.com".into(),
            port: 443,
        }]);
        let dispatcher = HostFunctionDispatcher::new(&gate);
        let result = dispatcher
            .net_request("api.example.com", 443, "/v1/data")
            .unwrap();
        assert_eq!(result.status, 200);
    }

    #[test]
    fn shell_exec_stub_returns_success() {
        let gate = make_gate(vec![Permission::Execute {
            binary: "ls".into(),
        }]);
        let dispatcher = HostFunctionDispatcher::new(&gate);
        let result = dispatcher.shell_exec("ls", &["-la"]).unwrap();
        assert_eq!(result.exit_code, 0);
    }

    #[test]
    fn env_read_stub_returns_mock_value() {
        let gate = make_gate(vec![Permission::FsRead {
            path: "/env/*".into(),
        }]);
        let dispatcher = HostFunctionDispatcher::new(&gate);
        let result = dispatcher.env_read("HOME").unwrap();
        assert!(result.found);
        assert!(result.value.contains("[carapace mock]"));
    }

    #[test]
    fn host_function_kind_names() {
        assert_eq!(HostFunctionKind::FileRead.wasm_module_name(), "carapace");
        assert_eq!(HostFunctionKind::FileRead.wasm_function_name(), "file_read");
        assert_eq!(
            HostFunctionKind::FileRead.fully_qualified_name(),
            "carapace::file_read"
        );
    }

    #[test]
    fn all_host_function_kinds() {
        let all = HostFunctionKind::all();
        assert_eq!(all.len(), 5);
    }
}
