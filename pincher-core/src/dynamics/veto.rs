//! Deterministic veto engine — replaces MLP-based safety for MVP.

use crate::capability::manifest::CapabilityManifest;

/// The verdict returned by the veto engine.
#[derive(Debug, Clone)]
pub struct VetoVerdict {
    pub allowed: bool,
    pub reason: Option<String>,
    pub confidence: f64,
}

/// The deterministic veto engine.
pub struct VetoEngine {
    blocked_patterns: Vec<String>,
}

impl VetoEngine {
    /// Create a new veto engine with the default blocked patterns.
    pub fn new() -> Self {
        Self {
            blocked_patterns: vec![
                "rm -rf /".into(),
                "rm -rf /*".into(),
                "rm -r /".into(),
                "dd if=/dev/zero".into(),
                "dd if=/dev/random".into(),
                ":(){ :|:& };:".into(),
                "mkfs".into(),
                "> /dev/sda".into(),
                "chmod -R 777 /".into(),
                "chown -R".into(),
                "shutdown".into(),
                "reboot".into(),
                "halt".into(),
                "init 0".into(),
                "init 6".into(),
            ],
        }
    }

    /// Create a veto engine with custom blocked patterns.
    pub fn with_patterns(patterns: Vec<String>) -> Self {
        Self {
            blocked_patterns: patterns,
        }
    }

    /// Check a command against all veto rules.
    pub fn check(&self, command: &str, manifest: &CapabilityManifest) -> VetoVerdict {
        // 1. Check blocked patterns.
        for pattern in &self.blocked_patterns {
            if command.contains(pattern.as_str()) {
                return VetoVerdict {
                    allowed: false,
                    reason: Some(format!("command matches blocked pattern: {}", pattern)),
                    confidence: 1.0,
                };
            }
        }

        // 2. Check capability manifest.
        if let Some(verdict) = self.check_manifest(command, manifest) {
            return verdict;
        }

        // 3. If no manifest permissions at all, treat as novel.
        if manifest.permissions.is_empty() {
            return VetoVerdict {
                allowed: true,
                reason: Some("no capability manifest — treating as novel command".into()),
                confidence: 0.5,
            };
        }

        // 4. All checks passed.
        VetoVerdict {
            allowed: true,
            reason: None,
            confidence: 1.0,
        }
    }

    fn check_manifest(&self, command: &str, manifest: &CapabilityManifest) -> Option<VetoVerdict> {
        let parts: Vec<&str> = command.split_whitespace().collect();
        let binary = parts.first()?;

        if !manifest.has_execute(binary) && !manifest.permissions.is_empty() {
            let has_any_execute = manifest
                .permissions
                .iter()
                .any(|p| matches!(p, crate::capability::Permission::Execute { .. }));
            if has_any_execute {
                return Some(VetoVerdict {
                    allowed: false,
                    reason: Some(format!(
                        "binary '{}' is not in the execute permissions of manifest",
                        binary
                    )),
                    confidence: 1.0,
                });
            }
        }

        let uses_network = command.contains("curl")
            || command.contains("wget")
            || command.contains("nc ")
            || command.contains("ssh ");
        if uses_network && !manifest.allows_network() && !manifest.permissions.is_empty() {
            return Some(VetoVerdict {
                allowed: false,
                reason: Some(
                    "command appears to use network but manifest has no network permission".into(),
                ),
                confidence: 0.8,
            });
        }

        None
    }
}

impl Default for VetoEngine {
    fn default() -> Self {
        Self::new()
    }
}
