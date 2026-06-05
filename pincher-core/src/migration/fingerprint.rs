//! Hardware fingerprinting for PincherOS shell migration
//!
//! Collects hardware characteristics of the current shell (machine)
//! to determine compatibility when migrating a .nail between environments.
//! Uses BLAKE3 hashing for consistent fingerprint identification.

use serde::{Deserialize, Serialize};
use sysinfo::System;
use thiserror::Error;
use tracing::{info, instrument};

/// Fingerprint errors.
#[derive(Debug, Error)]
pub enum FingerprintError {
    #[error("System info error: {0}")]
    SystemInfo(String),

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
}

/// Result type for fingerprint operations.
pub type FingerprintResult<T> = Result<T, FingerprintError>;

/// A hardware fingerprint of the current shell (machine).
///
/// This captures the essential characteristics that affect reflex
/// portability: OS, CPU count, RAM, GPU, and network identity.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ShellFingerprint {
    /// Machine hostname.
    pub hostname: String,
    /// Operating system name.
    pub os: String,
    /// OS version.
    pub os_version: String,
    /// Number of CPU cores.
    pub cpu_count: usize,
    /// Total RAM in megabytes.
    pub ram_mb: u64,
    /// GPU information (if detectable).
    pub gpu: String,
    /// BLAKE3 hash of MAC addresses for network identity.
    pub mac_hash: String,
}

impl std::fmt::Display for ShellFingerprint {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{}@{} ({} {}, {} CPUs, {}MB RAM, GPU: {})",
            self.hostname, self.os, self.os, self.os_version, self.cpu_count, self.ram_mb, self.gpu
        )
    }
}

/// Collect the hardware fingerprint of the current machine.
#[instrument]
pub fn fingerprint() -> FingerprintResult<ShellFingerprint> {
    info!("Collecting hardware fingerprint");

    let mut sys = System::new_all();
    sys.refresh_all();

    let hostname = System::host_name().unwrap_or_else(|| "unknown".to_string());
    let os = System::name().unwrap_or_else(|| "unknown".to_string());
    let os_version = System::os_version().unwrap_or_else(|| "unknown".to_string());
    let cpu_count = sys.cpus().len();
    let ram_mb = sys.total_memory() / 1024 / 1024;

    // GPU detection — try to find GPU info
    let gpu = detect_gpu();

    // MAC address hash for network identity
    let mac_hash = compute_mac_hash();

    let fp = ShellFingerprint {
        hostname,
        os,
        os_version,
        cpu_count,
        ram_mb,
        gpu,
        mac_hash,
    };

    info!(fingerprint = %fp, "Hardware fingerprint collected");
    Ok(fp)
}

/// Compute a BLAKE3 hash of the fingerprint for identification.
pub fn fingerprint_hash(fp: &ShellFingerprint) -> String {
    let canonical = format!(
        "{}|{}|{}|{}|{}|{}|{}",
        fp.hostname, fp.os, fp.os_version, fp.cpu_count, fp.ram_mb, fp.gpu, fp.mac_hash
    );

    blake3::hash(canonical.as_bytes()).to_hex().to_string()
}

/// Compute a compatibility score between two shell fingerprints.
///
/// Returns a value between 0.0 (completely incompatible) and 1.0
/// (identical). The score considers:
/// - OS match (critical: 0 if different OS)
/// - CPU count similarity
/// - RAM similarity
/// - GPU presence
/// - Hostname match (minor factor)
pub fn compatibility_score(a: &ShellFingerprint, b: &ShellFingerprint) -> f32 {
    let mut score: f32 = 0.0;

    // OS match: critical — different OS families are problematic
    if a.os == b.os {
        score += 0.35;
        // OS version match: bonus
        if a.os_version == b.os_version {
            score += 0.10;
        } else {
            // Partial version match
            score += 0.05;
        }
    } else {
        // Completely different OS — major incompatibility
        // But still allow some score for cross-OS migration
        score += 0.05;
    }

    // CPU count similarity: more CPUs = more parallel capacity
    let cpu_ratio = if a.cpu_count > 0 && b.cpu_count > 0 {
        let min_cpu = a.cpu_count.min(b.cpu_count) as f32;
        let max_cpu = a.cpu_count.max(b.cpu_count) as f32;
        min_cpu / max_cpu
    } else {
        0.5 // Unknown
    };
    score += cpu_ratio * 0.15;

    // RAM similarity: affects context window and model loading
    let ram_ratio = if a.ram_mb > 0 && b.ram_mb > 0 {
        let min_ram = a.ram_mb.min(b.ram_mb) as f32;
        let max_ram = a.ram_mb.max(b.ram_mb) as f32;
        min_ram / max_ram
    } else {
        0.5
    };
    score += ram_ratio * 0.20;

    // GPU presence: both have GPU or both don't
    let gpu_match = (a.gpu != "none" && b.gpu != "none") || (a.gpu == "none" && b.gpu == "none");
    score += if gpu_match { 0.10 } else { 0.0 };

    // Hostname match: minor factor (same machine rebooted)
    if a.hostname == b.hostname {
        score += 0.05;
    }

    // MAC hash match: strong indicator of same physical machine
    if a.mac_hash == b.mac_hash && !a.mac_hash.is_empty() {
        score += 0.05;
    }

    score.min(1.0)
}

/// Detect GPU information.
fn detect_gpu() -> String {
    // Try lspci first (Linux)
    if let Ok(output) = std::process::Command::new("lspci").output() {
        let stdout = String::from_utf8_lossy(&output.stdout);
        for line in stdout.lines() {
            let line_lower = line.to_lowercase();
            if line_lower.contains("vga")
                || line_lower.contains("3d")
                || line_lower.contains("display")
            {
                // Extract GPU name
                if let Some(colon_pos) = line.find(':') {
                    let gpu_name = line[colon_pos + 1..].trim();
                    if !gpu_name.is_empty() {
                        return gpu_name.to_string();
                    }
                }
            }
        }
    }

    // Try nvidia-smi
    if let Ok(output) = std::process::Command::new("nvidia-smi")
        .args(["--query-gpu=name", "--format=csv,noheader"])
        .output()
    {
        if output.status.success() {
            let name = String::from_utf8_lossy(&output.stdout).trim().to_string();
            if !name.is_empty() {
                return name;
            }
        }
    }

    "none".to_string()
}

/// Compute a hash of MAC addresses for network identity.
fn compute_mac_hash() -> String {
    let mut mac_info = String::new();

    // Try reading from /sys/class/net (Linux)
    if let Ok(entries) = std::fs::read_dir("/sys/class/net") {
        for entry in entries.flatten() {
            let name = entry.file_name().to_string_lossy().to_string();
            let addr_path = format!("/sys/class/net/{}/address", name);
            if let Ok(addr) = std::fs::read_to_string(&addr_path) {
                let addr = addr.trim().to_string();
                // Skip loopback
                if addr != "00:00:00:00:00:00" {
                    mac_info.push_str(&addr);
                }
            }
        }
    }

    if mac_info.is_empty() {
        return String::new();
    }

    blake3::hash(mac_info.as_bytes()).to_hex()[..16].to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fingerprint() {
        let fp = fingerprint().unwrap();
        assert!(!fp.hostname.is_empty());
        assert!(!fp.os.is_empty());
        assert!(fp.cpu_count > 0);
        assert!(fp.ram_mb > 0);
    }

    #[test]
    fn test_fingerprint_hash_deterministic() {
        let fp = fingerprint().unwrap();
        let hash1 = fingerprint_hash(&fp);
        let hash2 = fingerprint_hash(&fp);
        assert_eq!(hash1, hash2);
        assert_eq!(hash1.len(), 64); // BLAKE3 hex length
    }

    #[test]
    fn test_fingerprint_hash_different() {
        let fp1 = ShellFingerprint {
            hostname: "host1".to_string(),
            os: "Linux".to_string(),
            os_version: "5.15".to_string(),
            cpu_count: 8,
            ram_mb: 16384,
            gpu: "nvidia".to_string(),
            mac_hash: "abc".to_string(),
        };

        let fp2 = ShellFingerprint {
            hostname: "host2".to_string(),
            os: "Linux".to_string(),
            os_version: "5.15".to_string(),
            cpu_count: 8,
            ram_mb: 16384,
            gpu: "nvidia".to_string(),
            mac_hash: "def".to_string(),
        };

        let hash1 = fingerprint_hash(&fp1);
        let hash2 = fingerprint_hash(&fp2);
        assert_ne!(hash1, hash2);
    }

    #[test]
    fn test_compatibility_score_identical() {
        let fp = fingerprint().unwrap();
        let score = compatibility_score(&fp, &fp);
        assert!((score - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_compatibility_score_different_os() {
        let fp1 = ShellFingerprint {
            hostname: "host1".to_string(),
            os: "Linux".to_string(),
            os_version: "5.15".to_string(),
            cpu_count: 8,
            ram_mb: 16384,
            gpu: "nvidia".to_string(),
            mac_hash: "abc".to_string(),
        };

        let fp2 = ShellFingerprint {
            hostname: "host2".to_string(),
            os: "Windows".to_string(),
            os_version: "11".to_string(),
            cpu_count: 4,
            ram_mb: 8192,
            gpu: "none".to_string(),
            mac_hash: "def".to_string(),
        };

        let score = compatibility_score(&fp1, &fp2);
        // Different OS, less RAM, and no GPU should have low score
        assert!(score < 0.5);
    }

    #[test]
    fn test_compatibility_score_similar() {
        let fp1 = ShellFingerprint {
            hostname: "host1".to_string(),
            os: "Linux".to_string(),
            os_version: "5.15".to_string(),
            cpu_count: 8,
            ram_mb: 16384,
            gpu: "nvidia".to_string(),
            mac_hash: "abc".to_string(),
        };

        let fp2 = ShellFingerprint {
            hostname: "host2".to_string(),
            os: "Linux".to_string(),
            os_version: "5.19".to_string(),
            cpu_count: 6,
            ram_mb: 8192,
            gpu: "nvidia".to_string(),
            mac_hash: "def".to_string(),
        };

        let score = compatibility_score(&fp1, &fp2);
        // Same OS, similar hardware should have high score
        assert!(score > 0.6);
    }

    #[test]
    fn test_shell_fingerprint_display() {
        let fp = ShellFingerprint {
            hostname: "testhost".to_string(),
            os: "Linux".to_string(),
            os_version: "5.15".to_string(),
            cpu_count: 4,
            ram_mb: 8192,
            gpu: "NVIDIA RTX 3080".to_string(),
            mac_hash: "abc123".to_string(),
        };

        let display = format!("{}", fp);
        assert!(display.contains("testhost"));
        assert!(display.contains("Linux"));
        assert!(display.contains("4 CPUs"));
        assert!(display.contains("8192MB RAM"));
    }
}
