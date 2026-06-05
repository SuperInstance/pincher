//! Sandbox integration for PincherOS
//!
//! Provides capability-based sandboxing. When the `landlock` feature is
//! enabled, uses landlock for filesystem access restrictions. Otherwise,
//! uses a simplified approach with bubblewrap (bwrap) fallback.

use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use std::path::{Path, PathBuf};
use thiserror::Error;
use tracing::{debug, info, instrument, warn};

/// Sandbox errors.
#[derive(Debug, Error)]
pub enum SandboxError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),

    #[error("Sandbox execution failed: {0}")]
    Execution(String),

    #[error("Capability denied: {0}")]
    CapabilityDenied(String),

    #[error("bwrap not found: {0}")]
    BwrapNotFound(String),

    #[error("Landlock error: {0}")]
    Landlock(String),
}

/// Result type for sandbox operations.
pub type SandboxResult<T> = Result<T, SandboxError>;

/// Capabilities that can be granted to a sandboxed execution.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Capability {
    Network,
    FilesystemRead,
    FilesystemWrite,
    Subprocess,
    Gpu,
}

impl std::fmt::Display for Capability {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Capability::Network => write!(f, "network"),
            Capability::FilesystemRead => write!(f, "filesystem_read"),
            Capability::FilesystemWrite => write!(f, "filesystem_write"),
            Capability::Subprocess => write!(f, "subprocess"),
            Capability::Gpu => write!(f, "gpu"),
        }
    }
}

impl std::str::FromStr for Capability {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "network" => Ok(Capability::Network),
            "filesystem_read" => Ok(Capability::FilesystemRead),
            "filesystem_write" => Ok(Capability::FilesystemWrite),
            "subprocess" => Ok(Capability::Subprocess),
            "gpu" => Ok(Capability::Gpu),
            _ => Err(format!("Unknown capability: {}", s)),
        }
    }
}

/// A manifest of capabilities granted to a sandboxed execution.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CapabilityManifest {
    pub id: String,
    pub capabilities: HashSet<Capability>,
    pub read_paths: Vec<String>,
    pub write_paths: Vec<String>,
    pub allowed_hosts: Vec<String>,
    pub timeout_secs: u64,
    pub max_memory_mb: u64,
}

impl CapabilityManifest {
    pub fn new(id: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            capabilities: HashSet::new(),
            read_paths: Vec::new(),
            write_paths: Vec::new(),
            allowed_hosts: Vec::new(),
            timeout_secs: 30,
            max_memory_mb: 512,
        }
    }

    pub fn full(id: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            capabilities: HashSet::from([
                Capability::Network,
                Capability::FilesystemRead,
                Capability::FilesystemWrite,
                Capability::Subprocess,
                Capability::Gpu,
            ]),
            // SECURITY: Restrict read paths to specific safe directories, not "/"
            read_paths: vec!["/usr".to_string(), "/lib".to_string(), "/tmp".to_string()],
            write_paths: vec!["/tmp".to_string()],
            // SECURITY: No wildcard hosts — must be explicitly set
            allowed_hosts: Vec::new(),
            timeout_secs: 300,
            max_memory_mb: 2048,
        }
    }

    pub fn read_only(id: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            capabilities: HashSet::from([Capability::FilesystemRead, Capability::Subprocess]),
            read_paths: vec!["/".to_string()],
            write_paths: Vec::new(),
            allowed_hosts: Vec::new(),
            timeout_secs: 60,
            max_memory_mb: 256,
        }
    }

    pub fn with_capability(mut self, cap: Capability) -> Self {
        self.capabilities.insert(cap);
        self
    }

    pub fn with_read_path(mut self, path: impl Into<String>) -> Self {
        self.read_paths.push(path.into());
        self
    }

    pub fn with_write_path(mut self, path: impl Into<String>) -> Self {
        self.write_paths.push(path.into());
        self
    }

    pub fn has_capability(&self, cap: &Capability) -> bool {
        self.capabilities.contains(cap)
    }

    pub fn to_token(&self) -> SandboxResult<String> {
        Ok(serde_json::to_string(self)?)
    }

    pub fn from_token(token: &str) -> SandboxResult<Self> {
        Ok(serde_json::from_str(token)?)
    }

    pub fn sign(&self, secret_key: &[u8]) -> SandboxResult<SignedToken> {
        let manifest_json = serde_json::to_string(self)?;

        // SECURITY: Use blake3 keyed-hash (MAC) instead of plain hash.
        // This prevents token forging — only the holder of the secret key can sign.
        let key = blake3::derive_key("pincherOS/sandbox-token/v1", secret_key);
        let mut hasher = blake3::Hasher::new_keyed(&key);
        hasher.update(manifest_json.as_bytes());
        let mac = hasher.finalize();

        Ok(SignedToken {
            manifest: manifest_json,
            signature: format!("blake3-mac:{}", mac.to_hex()),
            algorithm: "blake3-keyed-mac".to_string(),
        })
    }

    pub fn verify(token: &SignedToken, secret_key: &[u8]) -> SandboxResult<bool> {
        let key = blake3::derive_key("pincherOS/sandbox-token/v1", secret_key);
        let mut hasher = blake3::Hasher::new_keyed(&key);
        hasher.update(token.manifest.as_bytes());
        let expected = format!("blake3-mac:{}", hasher.finalize().to_hex());
        Ok(token.signature == expected)
    }
}

/// A signed capability token.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SignedToken {
    pub manifest: String,
    pub signature: String,
    pub algorithm: String,
}

/// Sandbox configuration built from a capability manifest.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SandboxConfig {
    pub bwrap_args: Vec<String>,
    pub landlock_rules: Vec<LandlockRule>,
    pub use_bwrap: bool,
    pub use_landlock: bool,
    pub env_vars: Vec<(String, String)>,
    pub timeout_secs: u64,
}

/// A landlock filesystem access rule.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LandlockRule {
    pub path: String,
    pub access: String,
}

/// Build sandbox configuration from a capability manifest.
#[instrument(skip(manifest))]
pub fn build_sandbox(manifest: &CapabilityManifest) -> SandboxResult<SandboxConfig> {
    info!(manifest_id = %manifest.id, "Building sandbox configuration");

    let mut bwrap_args = vec![
        "bwrap".to_string(),
        "--ro-bind".to_string(),
        "/usr".to_string(),
        "/usr".to_string(),
        "--ro-bind".to_string(),
        "/lib".to_string(),
        "/lib".to_string(),
        "--ro-bind".to_string(),
        "/lib64".to_string(),
        "/lib64".to_string(),
        "--ro-bind".to_string(),
        "/bin".to_string(),
        "/bin".to_string(),
        "--ro-bind".to_string(),
        "/sbin".to_string(),
        "/sbin".to_string(),
        "--dev".to_string(),
        "/dev".to_string(),
        "--proc".to_string(),
        "/proc".to_string(),
        "--tmpfs".to_string(),
        "/tmp".to_string(),
    ];

    for path in &manifest.read_paths {
        if Path::new(path).exists() {
            bwrap_args.extend_from_slice(&["--ro-bind".to_string(), path.clone(), path.clone()]);
        }
    }

    for path in &manifest.write_paths {
        if Path::new(path).exists() {
            bwrap_args.extend_from_slice(&["--bind".to_string(), path.clone(), path.clone()]);
        } else {
            bwrap_args.extend_from_slice(&["--bind-try".to_string(), path.clone(), path.clone()]);
        }
    }

    if manifest.has_capability(&Capability::Network) {
        bwrap_args.push("--share-net".to_string());
    }

    bwrap_args.extend_from_slice(&["--unshare-all".to_string(), "--die-with-parent".to_string()]);

    let mut landlock_rules = Vec::new();

    for path in &manifest.read_paths {
        landlock_rules.push(LandlockRule {
            path: path.clone(),
            access: "read".to_string(),
        });
    }

    for path in &manifest.write_paths {
        landlock_rules.push(LandlockRule {
            path: path.clone(),
            access: "readwrite".to_string(),
        });
    }

    let use_bwrap = which_bwrap().is_some();
    if !use_bwrap {
        warn!("bwrap not found — sandbox will use landlock-only mode");
    }

    let config = SandboxConfig {
        bwrap_args,
        landlock_rules,
        use_bwrap,
        use_landlock: cfg!(feature = "landlock"),
        env_vars: vec![
            ("PATH".to_string(), "/usr/bin:/bin".to_string()),
            ("HOME".to_string(), "/tmp".to_string()),
        ],
        timeout_secs: manifest.timeout_secs,
    };

    info!(
        use_bwrap = config.use_bwrap,
        use_landlock = config.use_landlock,
        "Sandbox configuration built"
    );

    Ok(config)
}

/// Execute a command in a sandboxed environment.
#[instrument(skip(config))]
pub fn execute_sandboxed(
    cmd: &str,
    config: &SandboxConfig,
) -> SandboxResult<std::process::ExitStatus> {
    info!(cmd = cmd, "Executing sandboxed command");

    if config.use_bwrap {
        execute_with_bwrap(cmd, config)
    } else {
        execute_with_landlock(cmd, config)
    }
}

fn execute_with_bwrap(
    cmd: &str,
    config: &SandboxConfig,
) -> SandboxResult<std::process::ExitStatus> {
    let mut args = config.bwrap_args.clone();

    for (key, value) in &config.env_vars {
        args.extend_from_slice(&["--setenv".to_string(), key.clone(), value.clone()]);
    }

    args.push("--".to_string());
    args.push("sh".to_string());
    args.push("-c".to_string());
    args.push(cmd.to_string());

    debug!(args = ?args, "Executing bwrap command");

    let bwrap_path = which_bwrap()
        .ok_or_else(|| SandboxError::BwrapNotFound("bwrap binary not found in PATH".to_string()))?;

    let status = std::process::Command::new(bwrap_path)
        .args(&args[1..])
        .status()
        .map_err(|e| SandboxError::Execution(format!("Failed to execute bwrap: {}", e)))?;

    info!(exit_code = status.code(), "Sandboxed command completed");
    Ok(status)
}

fn execute_with_landlock(
    cmd: &str,
    _config: &SandboxConfig,
) -> SandboxResult<std::process::ExitStatus> {
    debug!(cmd = cmd, "Executing with landlock restrictions");

    #[cfg(feature = "landlock")]
    {
        let landlock_result = apply_landlock_rules(&_config.landlock_rules);
        match landlock_result {
            Ok(()) => {
                debug!("Landlock rules applied successfully");
            }
            Err(e) => {
                // SECURITY: Fail closed — if landlock fails, do NOT execute
                return Err(SandboxError::Landlock(format!(
                    "Failed to apply landlock rules — refusing to execute unsandboxed: {}",
                    e
                )));
            }
        }
    }

    #[cfg(not(feature = "landlock"))]
    {
        // SECURITY: Without landlock, we can't sandbox at all — refuse to execute
        return Err(SandboxError::Execution(
            "No sandboxing mechanism available (landlock feature not enabled). Refusing to execute.".into()
        ));
    }

    // Unreachable without a sandboxing mechanism; the error above is returned.
    // This block is kept as a template for future sandbox backends.
    #[allow(unreachable_code)]
    {
        let _ = cmd; // suppress unused variable warning
        unreachable!()
    }
}

#[cfg(feature = "landlock")]
fn apply_landlock_rules(rules: &[LandlockRule]) -> SandboxResult<()> {
    use landlock::{Access, AccessFs, Ruleset, RulesetAttr, RulesetCreated, RulesetStatus};

    let access_fs = AccessFs::from_all(Access::all());
    let mut ruleset = Ruleset::new()
        .handle_access(access_fs)
        .map_err(|e| SandboxError::Landlock(format!("Failed to create ruleset: {}", e)))?;

    for rule in rules {
        let path = PathBuf::from(&rule.path);
        if path.exists() {
            let access = match rule.access.as_str() {
                "read" => AccessFs::from_read(Access::all()),
                "write" => AccessFs::from_write(Access::all()),
                "readwrite" => AccessFs::from_all(Access::all()),
                "execute" => AccessFs::from_execute(Access::all()),
                _ => AccessFs::from_read(Access::all()),
            };

            ruleset = ruleset.add_rule(landlock::PathBeneath::new(&path, access));
        }
    }

    let status = ruleset
        .restrict_self()
        .map_err(|e| SandboxError::Landlock(format!("Failed to restrict self: {}", e)))?;

    match status {
        RulesetStatus::FullyEnforced => {
            info!("Landlock fully enforced");
        }
        RulesetStatus::PartiallyEnforced => {
            warn!("Landlock partially enforced");
        }
        RulesetStatus::NotEnforced => {
            warn!("Landlock not enforced");
        }
        _ => {}
    }

    Ok(())
}

fn which_bwrap() -> Option<PathBuf> {
    let paths = std::env::var("PATH").ok()?;
    for dir in paths.split(':') {
        let bwrap = PathBuf::from(dir).join("bwrap");
        if bwrap.exists() {
            return Some(bwrap);
        }
    }
    None
}
