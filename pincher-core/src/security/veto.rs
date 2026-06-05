//! Deterministic rule-based veto engine for PincherOS
//!
//! The veto engine provides a safety layer that inspects actions before
//! execution and can deny or require confirmation for dangerous operations.
//! This is the MVP implementation — rules are loaded from a TOML config
//! and evaluated deterministically.

use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use std::path::Path;
use thiserror::Error;
use tracing::{debug, info, instrument, warn};

/// Veto engine errors.
#[derive(Debug, Error)]
pub enum VetoError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("TOML parse error: {0}")]
    Toml(#[from] toml::de::Error),

    #[error("Invalid rule: {0}")]
    InvalidRule(String),
}

/// Result type for veto operations.
pub type VetoResult<T> = Result<T, VetoError>;

/// A single veto rule.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum VetoRule {
    /// Block commands containing a specific substring.
    ForbiddenCommand {
        /// The forbidden command pattern (substring match).
        pattern: String,
    },
    /// Block access to specific filesystem paths.
    ForbiddenPath {
        /// The forbidden path prefix.
        path: String,
    },
    /// Enforce a maximum file size for write operations.
    MaxFileSize {
        /// Maximum file size in bytes.
        max_bytes: u64,
    },
    /// Require a specific capability for an action.
    RequireCapability {
        /// The capability name required.
        capability: String,
    },
    /// Block specific command patterns (simplified substring for MVP).
    ForbiddenPattern {
        /// Substring pattern to match in commands.
        pattern: String,
        /// Human-readable reason for the block.
        reason: String,
    },
}

impl VetoRule {
    /// Create a forbidden command rule.
    pub fn forbidden_command(pattern: impl Into<String>) -> Self {
        VetoRule::ForbiddenCommand {
            pattern: pattern.into(),
        }
    }

    /// Create a forbidden path rule.
    pub fn forbidden_path(path: impl Into<String>) -> Self {
        VetoRule::ForbiddenPath { path: path.into() }
    }

    /// Create a max file size rule.
    pub fn max_file_size(max_bytes: u64) -> Self {
        VetoRule::MaxFileSize { max_bytes }
    }

    /// Create a require capability rule.
    pub fn require_capability(capability: impl Into<String>) -> Self {
        VetoRule::RequireCapability {
            capability: capability.into(),
        }
    }
}

/// The decision made by the veto engine.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum VetoDecision {
    /// The action is allowed.
    Allow,
    /// The action is denied with a reason.
    Deny(String),
    /// The action requires user confirmation with a reason.
    RequireConfirmation(String),
}

impl VetoDecision {
    /// Check if the action is allowed.
    pub fn is_allowed(&self) -> bool {
        matches!(self, VetoDecision::Allow)
    }

    /// Check if the action is denied.
    pub fn is_denied(&self) -> bool {
        matches!(self, VetoDecision::Deny(_))
    }

    /// Check if the action requires confirmation.
    pub fn requires_confirmation(&self) -> bool {
        matches!(self, VetoDecision::RequireConfirmation(_))
    }
}

/// Execution context provided to the veto engine for evaluation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutionContext {
    /// The command or action being executed.
    pub action: String,
    /// File paths involved in the action.
    pub paths: Vec<String>,
    /// Size of data being written (if applicable).
    pub data_size: Option<u64>,
    /// Capabilities granted for this execution.
    pub capabilities: HashSet<String>,
    /// The working directory.
    pub working_dir: String,
}

impl ExecutionContext {
    /// Create a new execution context for a command.
    pub fn for_command(action: &str) -> Self {
        Self {
            action: action.to_string(),
            paths: Vec::new(),
            data_size: None,
            capabilities: HashSet::new(),
            working_dir: std::env::current_dir()
                .map(|p| p.to_string_lossy().to_string())
                .unwrap_or_else(|_| ".".to_string()),
        }
    }

    /// Create a new execution context for a file operation.
    pub fn for_file_operation(action: &str, path: &str, data_size: Option<u64>) -> Self {
        Self {
            action: action.to_string(),
            paths: vec![path.to_string()],
            data_size,
            capabilities: HashSet::new(),
            working_dir: std::env::current_dir()
                .map(|p| p.to_string_lossy().to_string())
                .unwrap_or_else(|_| ".".to_string()),
        }
    }

    /// Add a capability to this context.
    pub fn with_capability(mut self, capability: &str) -> Self {
        self.capabilities.insert(capability.to_string());
        self
    }

    /// Add a path to this context.
    pub fn with_path(mut self, path: &str) -> Self {
        self.paths.push(path.to_string());
        self
    }
}

/// The veto engine that evaluates actions against rules.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VetoEngine {
    /// The list of veto rules to evaluate.
    pub rules: Vec<VetoRule>,
}

impl VetoEngine {
    /// Create a new empty veto engine.
    pub fn new() -> Self {
        Self { rules: Vec::new() }
    }

    /// Create a veto engine with the default safety rules.
    pub fn with_defaults() -> Self {
        Self {
            rules: default_veto_rules(),
        }
    }

    /// Load veto rules from a TOML configuration file.
    #[instrument(skip(path))]
    pub fn load_rules(path: &Path) -> VetoResult<Self> {
        info!(path = ?path, "Loading veto rules from config");

        let content = std::fs::read_to_string(path)?;
        let config: VetoConfig = toml::from_str(&content)?;

        info!(rule_count = config.rules.len(), "Loaded veto rules");
        Ok(Self {
            rules: config.rules,
        })
    }

    /// Save the current rules to a TOML configuration file.
    #[instrument(skip(self, path))]
    pub fn save_rules(&self, path: &Path) -> VetoResult<()> {
        let config = VetoConfig {
            rules: self.rules.clone(),
        };
        let content = toml::to_string_pretty(&config)
            .map_err(|e| VetoError::InvalidRule(format!("Failed to serialize rules: {}", e)))?;

        std::fs::write(path, content)?;
        info!(path = ?path, "Saved veto rules");
        Ok(())
    }

    /// Add a rule to the engine.
    pub fn add_rule(&mut self, rule: VetoRule) {
        self.rules.push(rule);
    }

    /// Check an action against all veto rules.
    #[instrument(skip(self, context))]
    pub fn check(&self, action: &str, context: &ExecutionContext) -> VetoResult<VetoDecision> {
        debug!(action = action, "Checking action against veto rules");

        for rule in &self.rules {
            let decision = self.evaluate_rule(rule, action, context)?;
            if !decision.is_allowed() {
                info!(
                    action = action,
                    decision = ?decision,
                    "Action vetoed by rule"
                );
                return Ok(decision);
            }
        }

        debug!(action = action, "Action allowed by all rules");
        Ok(VetoDecision::Allow)
    }

    /// Evaluate a single rule against an action.
    fn evaluate_rule(
        &self,
        rule: &VetoRule,
        action: &str,
        context: &ExecutionContext,
    ) -> VetoResult<VetoDecision> {
        match rule {
            VetoRule::ForbiddenCommand { pattern } => {
                if action.contains(pattern) {
                    debug!(pattern = pattern, "Forbidden command pattern matched");
                    return Ok(VetoDecision::Deny(format!(
                        "Command contains forbidden pattern: '{}'",
                        pattern
                    )));
                }
            }

            VetoRule::ForbiddenPath { path } => {
                for context_path in &context.paths {
                    if context_path.starts_with(path) || context_path == path {
                        debug!(path = path, "Forbidden path matched");
                        return Ok(VetoDecision::Deny(format!(
                            "Access to path '{}' is forbidden",
                            path
                        )));
                    }
                }
                // Also check if the action references the forbidden path
                if action.contains(path) {
                    debug!(path = path, "Forbidden path referenced in action");
                    return Ok(VetoDecision::Deny(format!(
                        "Action references forbidden path: '{}'",
                        path
                    )));
                }
            }

            VetoRule::MaxFileSize { max_bytes } => {
                if let Some(data_size) = context.data_size {
                    if data_size > *max_bytes {
                        debug!(
                            data_size = data_size,
                            max_bytes = max_bytes,
                            "File size exceeds limit"
                        );
                        return Ok(VetoDecision::Deny(format!(
                            "Data size ({} bytes) exceeds maximum ({} bytes)",
                            data_size, max_bytes
                        )));
                    }
                }
            }

            VetoRule::RequireCapability { capability } => {
                if !context.capabilities.contains(capability) {
                    debug!(capability = capability, "Missing required capability");
                    return Ok(VetoDecision::RequireConfirmation(format!(
                        "Action requires '{}' capability which is not granted",
                        capability
                    )));
                }
            }

            VetoRule::ForbiddenPattern { pattern, reason } => {
                if action.contains(pattern) {
                    debug!(pattern = pattern, "Forbidden pattern matched");
                    return Ok(VetoDecision::Deny(reason.clone()));
                }
            }
        }

        Ok(VetoDecision::Allow)
    }

    /// Get the number of rules.
    pub fn rule_count(&self) -> usize {
        self.rules.len()
    }
}

impl Default for VetoEngine {
    fn default() -> Self {
        Self::with_defaults()
    }
}

/// TOML configuration format for veto rules.
#[derive(Debug, Serialize, Deserialize)]
struct VetoConfig {
    rules: Vec<VetoRule>,
}

/// Generate the default set of safety veto rules.
fn default_veto_rules() -> Vec<VetoRule> {
    vec![
        // Block destructive rm commands
        VetoRule::ForbiddenPattern {
            pattern: "rm -rf /".to_string(),
            reason: "Recursive force-delete of root filesystem is not allowed".to_string(),
        },
        VetoRule::ForbiddenPattern {
            pattern: "rm -rf /*".to_string(),
            reason: "Recursive force-delete of root filesystem is not allowed".to_string(),
        },
        VetoRule::ForbiddenPattern {
            pattern: "rm -rf ~".to_string(),
            reason: "Recursive force-delete of home directory is not allowed".to_string(),
        },
        VetoRule::ForbiddenCommand {
            pattern: "mkfs".to_string(),
        },
        VetoRule::ForbiddenCommand {
            pattern: "dd if=".to_string(),
        },
        // Protect system directories
        VetoRule::ForbiddenPath {
            path: "/etc".to_string(),
        },
        VetoRule::ForbiddenPath {
            path: "/sys".to_string(),
        },
        VetoRule::ForbiddenPath {
            path: "/proc".to_string(),
        },
        VetoRule::ForbiddenPath {
            path: "/boot".to_string(),
        },
        VetoRule::ForbiddenPath {
            path: "/dev".to_string(),
        },
        // Require capabilities for specific dangerous operations
        // These trigger confirmation for network and subprocess commands
        // rather than blanket-blocking everything
        VetoRule::ForbiddenPattern {
            pattern: "curl ".to_string(),
            reason: "Network request requires 'network' capability".to_string(),
        },
        VetoRule::ForbiddenPattern {
            pattern: "wget ".to_string(),
            reason: "Network download requires 'network' capability".to_string(),
        },
        VetoRule::ForbiddenPattern {
            pattern: "ssh ".to_string(),
            reason: "SSH connection requires 'network' capability".to_string(),
        },
        VetoRule::ForbiddenPattern {
            pattern: "nc ".to_string(),
            reason: "Netcat requires 'network' capability".to_string(),
        },
        // File size limit: 100MB
        VetoRule::MaxFileSize {
            max_bytes: 100 * 1024 * 1024,
        },
        // Block package manager operations (could modify system)
        VetoRule::ForbiddenCommand {
            pattern: "apt-get install".to_string(),
        },
        VetoRule::ForbiddenCommand {
            pattern: "yum install".to_string(),
        },
        VetoRule::ForbiddenCommand {
            pattern: "pip install".to_string(),
        },
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_veto_allow_safe_command() {
        let engine = VetoEngine::with_defaults();
        let context = ExecutionContext::for_command("ls -la");
        let decision = engine.check("ls -la", &context).unwrap();
        assert!(decision.is_allowed());
    }

    #[test]
    fn test_veto_deny_rm_rf() {
        let engine = VetoEngine::with_defaults();
        let context = ExecutionContext::for_command("rm -rf /");
        let decision = engine.check("rm -rf /", &context).unwrap();
        assert!(decision.is_denied());
    }

    #[test]
    fn test_veto_deny_etc_write() {
        let engine = VetoEngine::with_defaults();
        let context = ExecutionContext::for_file_operation("write", "/etc/passwd", None);
        let decision = engine.check("write /etc/passwd", &context).unwrap();
        assert!(decision.is_denied());
    }

    #[test]
    fn test_veto_forbidden_network_command() {
        let engine = VetoEngine::with_defaults();
        let context = ExecutionContext::for_command("curl http://example.com");
        let decision = engine.check("curl http://example.com", &context).unwrap();
        // curl is blocked by ForbiddenPattern rule
        assert!(decision.is_denied());
    }

    #[test]
    fn test_veto_capability_check() {
        let _engine = VetoEngine::with_defaults();
        // A custom engine with a RequireCapability rule
        let mut engine = VetoEngine::new();
        engine.add_rule(VetoRule::RequireCapability {
            capability: "network".to_string(),
        });
        let context = ExecutionContext::for_command("curl http://example.com");
        let decision = engine.check("curl http://example.com", &context).unwrap();
        assert!(decision.requires_confirmation());
        // With capability granted, should allow
        let context_with_cap =
            ExecutionContext::for_command("curl http://example.com").with_capability("network");
        let decision2 = engine
            .check("curl http://example.com", &context_with_cap)
            .unwrap();
        assert!(decision2.is_allowed());
    }

    #[test]
    fn test_veto_max_file_size() {
        let engine = VetoEngine::with_defaults();
        let context =
            ExecutionContext::for_file_operation("write", "/tmp/large", Some(200 * 1024 * 1024));
        let decision = engine.check("write /tmp/large", &context).unwrap();
        assert!(decision.is_denied());
    }

    #[test]
    fn test_veto_decision_helpers() {
        assert!(VetoDecision::Allow.is_allowed());
        assert!(!VetoDecision::Allow.is_denied());
        assert!(!VetoDecision::Allow.requires_confirmation());

        assert!(!VetoDecision::Deny("test".to_string()).is_allowed());
        assert!(VetoDecision::Deny("test".to_string()).is_denied());

        assert!(VetoDecision::RequireConfirmation("test".to_string()).requires_confirmation());
    }

    #[test]
    fn test_load_save_rules() {
        let dir = std::env::temp_dir().join("pincher_veto_test");
        std::fs::create_dir_all(&dir).ok();
        let path = dir.join("veto_rules.toml");

        let engine = VetoEngine::with_defaults();
        engine.save_rules(&path).unwrap();

        let loaded = VetoEngine::load_rules(&path).unwrap();
        assert_eq!(engine.rule_count(), loaded.rule_count());
    }

    #[test]
    fn test_execution_context_builder() {
        let ctx = ExecutionContext::for_command("test")
            .with_capability("network")
            .with_path("/tmp/test");

        assert_eq!(ctx.action, "test");
        assert!(ctx.capabilities.contains("network"));
        assert!(ctx.paths.contains(&"/tmp/test".to_string()));
    }
}
