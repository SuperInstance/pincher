//! Sandboxed command execution via **bubblewrap** (`bwrap`).
//!
//! If `bwrap` is available on `$PATH` the command is run inside a
//! lightweight namespace sandbox with restricted filesystem and network
//! access.  If `bwrap` is not found, the command is executed directly via
//! [`std::process::Command`] and a warning is logged.

use anyhow::{Context, Result};
use std::time::Instant;

/// The result of executing a sandboxed command.
#[derive(Debug, Clone, serde::Serialize)]
pub struct ExecutionResult {
    /// Process exit code (0 = success).
    pub exit_code: i32,
    /// Captured stdout (truncated to 4 KiB).
    pub stdout: String,
    /// Captured stderr.
    pub stderr: String,
    /// Wall-time in milliseconds.
    pub duration_ms: u64,
}

/// Configuration for the sandbox environment.
#[derive(Debug, Clone)]
pub struct SandboxConfig {
    /// Whether to allow network access inside the sandbox.
    pub allow_network: bool,
    /// Paths mounted read-only inside the sandbox.
    pub read_only_paths: Vec<String>,
    /// Paths mounted read-write inside the sandbox.
    pub read_write_paths: Vec<String>,
    /// Binaries that are permitted to execute.
    pub executable_whitelist: Vec<String>,
    /// Substrings that, if present in the command, cause an immediate veto.
    pub blocked_patterns: Vec<String>,
}

impl SandboxConfig {
    /// A sensible default restricted configuration:
    /// - no network
    /// - `/usr` and `/lib` read-only
    /// - `/tmp` read-write
    /// - only common utilities whitelisted
    /// - dangerous patterns blocked
    pub fn default_restricted() -> Self {
        Self {
            allow_network: false,
            read_only_paths: vec![
                "/usr".into(),
                "/lib".into(),
                "/lib64".into(),
                "/bin".into(),
                "/sbin".into(),
            ],
            read_write_paths: vec!["/tmp".into()],
            executable_whitelist: vec![
                "mkdir".into(),
                "cp".into(),
                "mv".into(),
                "ls".into(),
                "cat".into(),
                "echo".into(),
                "touch".into(),
                "rm".into(),
                "find".into(),
                "grep".into(),
                "head".into(),
                "tail".into(),
                "wc".into(),
                "sort".into(),
                "uniq".into(),
            ],
            blocked_patterns: vec![
                "rm -rf /".into(),
                "rm -rf /*".into(),
                "dd if=/dev/zero".into(),
                "dd if=/dev/random".into(),
                ":(){ :|:& };:".into(),
                "mkfs".into(),
                "> /dev/sda".into(),
                "chmod -R 777 /".into(),
                "chown -R".into(),
            ],
        }
    }

    /// A permissive configuration for trusted environments.
    pub fn permissive() -> Self {
        Self {
            allow_network: true,
            read_only_paths: vec![],
            read_write_paths: vec!["/".into()],
            executable_whitelist: vec![],
            blocked_patterns: vec![
                "rm -rf /".into(),
                "rm -rf /*".into(),
                ":(){ :|:& };:".into(),
            ],
        }
    }

    /// Returns `true` if the command contains any blocked pattern.
    pub fn is_blocked(&self, command: &str) -> bool {
        self.blocked_patterns
            .iter()
            .any(|p| command.contains(p.as_str()))
    }

    /// Returns `Ok(())` if the binary is in the whitelist (or the whitelist
    /// is empty = allow all).
    pub fn check_executable(&self, binary: &str) -> Result<()> {
        if self.executable_whitelist.is_empty() {
            return Ok(());
        }
        if self
            .executable_whitelist
            .iter()
            .any(|w| binary == w.as_str())
        {
            Ok(())
        } else {
            anyhow::bail!("binary '{}' is not in the executable whitelist", binary)
        }
    }
}

/// The sandbox executor.
pub struct Sandbox;

impl Sandbox {
    /// Execute `command` with `args` inside the sandbox described by `config`.
    ///
    /// If `bwrap` is on `$PATH`, the command is run inside a bubblewrap
    /// sandbox.  Otherwise, a warning is logged and the command is run
    /// directly (unsandboxed).
    pub fn execute(
        config: &SandboxConfig,
        command: &str,
        args: &[&str],
    ) -> Result<ExecutionResult> {
        // Check blocked patterns first — no execution at all.
        let full_cmd = format!("{} {}", command, args.join(" "));
        if config.is_blocked(&full_cmd) {
            anyhow::bail!("command matches a blocked pattern: {}", full_cmd);
        }

        // Check executable whitelist.
        config.check_executable(command)?;

        let start = Instant::now();

        let result = if which_bwrap() {
            Self::execute_bwrap(config, command, args)
        } else {
            // SECURITY: Fail closed — refuse to execute without sandbox.
            // Do NOT fall back to unsandboxed execution.
            tracing::error!("bwrap not found — refusing to execute without sandbox. Install bubblewrap (bwrap).");
            anyhow::bail!(
                "FATAL: bwrap not found — refusing to execute without sandbox. \
                 Install bubblewrap (bwrap) to enable sandboxed execution."
            )
        };

        let duration_ms = start.elapsed().as_millis() as u64;
        match result {
            Ok(mut r) => {
                r.duration_ms = duration_ms;
                Ok(r)
            }
            Err(e) => Err(e),
        }
    }

    /// Build and run a `bwrap` command line.
    fn execute_bwrap(
        config: &SandboxConfig,
        command: &str,
        args: &[&str],
    ) -> Result<ExecutionResult> {
        let mut bwrap_args: Vec<String> = Vec::new();

        // Mount /usr, /lib, etc. read-only.
        for path in &config.read_only_paths {
            bwrap_args.push("--ro-bind".into());
            bwrap_args.push(path.clone());
            bwrap_args.push(path.clone());
        }

        // Mount read-write paths.
        for path in &config.read_write_paths {
            bwrap_args.push("--bind".into());
            bwrap_args.push(path.clone());
            bwrap_args.push(path.clone());
        }

        // Bind /proc (needed for basic operation).
        bwrap_args.push("--proc".into());
        bwrap_args.push("/proc".into());

        // Bind /dev (minimal).
        bwrap_args.push("--dev".into());
        bwrap_args.push("/dev".into());

        // Network policy.
        if !config.allow_network {
            bwrap_args.push("--unshare-net".into());
        }

        // The actual command to run inside the sandbox.
        bwrap_args.push("--".into());
        bwrap_args.push(command.into());
        for arg in args {
            bwrap_args.push((*arg).into());
        }

        let output = std::process::Command::new("bwrap")
            .args(&bwrap_args)
            .output()
            .with_context(|| format!("failed to execute bwrap with command: {}", command))?;

        let stdout = String::from_utf8_lossy(&output.stdout).to_string();
        let stderr = String::from_utf8_lossy(&output.stderr).to_string();
        let exit_code = output.status.code().unwrap_or(-1);

        // Truncate stdout to 4 KiB for logging.
        let stdout_snippet = if stdout.len() > 4096 {
            stdout[..4096].to_string()
        } else {
            stdout
        };

        Ok(ExecutionResult {
            exit_code,
            stdout: stdout_snippet,
            stderr,
            duration_ms: 0, // filled in by caller
        })
    }

    /// Fallback: run the command directly without sandboxing.
    #[allow(dead_code)]
    fn execute_raw(command: &str, args: &[&str]) -> Result<ExecutionResult> {
        let output = std::process::Command::new(command)
            .args(args)
            .output()
            .with_context(|| format!("failed to execute command: {}", command))?;

        let stdout = String::from_utf8_lossy(&output.stdout).to_string();
        let stderr = String::from_utf8_lossy(&output.stderr).to_string();
        let exit_code = output.status.code().unwrap_or(-1);

        let stdout_snippet = if stdout.len() > 4096 {
            stdout[..4096].to_string()
        } else {
            stdout
        };

        Ok(ExecutionResult {
            exit_code,
            stdout: stdout_snippet,
            stderr,
            duration_ms: 0,
        })
    }
}

/// Returns `true` if `bwrap` is on `$PATH`.
fn which_bwrap() -> bool {
    std::process::Command::new("which")
        .arg("bwrap")
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
}
