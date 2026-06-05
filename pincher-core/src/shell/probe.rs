//! Shell hardware probing — detects device capabilities and classifies into tiers.

use anyhow::Context;
use blake3::Hasher;
use serde::{Deserialize, Serialize};
use sysinfo::System;
use uuid::Uuid;

/// Classification of the device based on available RAM.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DeviceTier {
    Micro,
    Small,
    Medium,
    Large,
}

/// A snapshot of the host hardware and OS.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ShellProfile {
    pub hostname: String,
    pub cpu_model: String,
    pub cpu_cores: usize,
    pub ram_mb: u64,
    pub gpu_present: bool,
    pub gpu_name: Option<String>,
    pub kernel_version: String,
    pub os_name: String,
    pub arch: String,
    pub fingerprint: String,
}

impl ShellProfile {
    /// Probes the current machine and builds a ShellProfile.
    pub fn probe() -> anyhow::Result<Self> {
        let mut sys = System::new_all();
        sys.refresh_all();

        let hostname = System::host_name().unwrap_or_else(|| "unknown".to_string());

        let cpu_model = sys
            .cpus()
            .first()
            .map(|c| c.brand().to_string())
            .unwrap_or_else(|| "unknown".to_string());

        let cpu_cores = sys.cpus().len();
        let ram_mb = sys.total_memory() / 1024 / 1024;

        let (gpu_present, gpu_name) = detect_gpu();

        let kernel_version = System::kernel_version().unwrap_or_else(|| "unknown".to_string());
        let os_name = System::name().unwrap_or_else(|| "unknown".to_string());
        let arch = std::env::consts::ARCH.to_string();

        let mut profile = Self {
            hostname,
            cpu_model,
            cpu_cores,
            ram_mb,
            gpu_present,
            gpu_name,
            kernel_version,
            os_name,
            arch,
            fingerprint: String::new(),
        };

        profile.fingerprint = profile.compute_fingerprint();
        Ok(profile)
    }

    /// Creates a profile with default/unknown values (for testing).
    pub fn default_for_test() -> Self {
        let mut profile = Self {
            hostname: "test-host".to_string(),
            cpu_model: "test-cpu".to_string(),
            cpu_cores: 4,
            ram_mb: 8192,
            gpu_present: false,
            gpu_name: None,
            kernel_version: "0.0.0-test".to_string(),
            os_name: "TestOS".to_string(),
            arch: "x86_64".to_string(),
            fingerprint: String::new(),
        };
        profile.fingerprint = profile.compute_fingerprint();
        profile
    }

    /// Returns the DeviceTier based on total RAM.
    pub fn device_tier(&self) -> DeviceTier {
        if self.ram_mb < 2048 {
            DeviceTier::Micro
        } else if self.ram_mb < 4096 {
            DeviceTier::Small
        } else if self.ram_mb < 16384 {
            DeviceTier::Medium
        } else {
            DeviceTier::Large
        }
    }

    /// Checks whether this shell supports a named capability.
    pub fn supports_capability(&self, cap: &str) -> bool {
        match cap {
            "cuda" | "gpu" => self.gpu_present,
            "large_ram" => self.device_tier() == DeviceTier::Large,
            "medium_ram" => matches!(self.device_tier(), DeviceTier::Medium | DeviceTier::Large),
            _ => false,
        }
    }

    fn compute_fingerprint(&self) -> String {
        let mut hasher = Hasher::new();
        hasher.update(self.hostname.as_bytes());
        hasher.update(self.cpu_model.as_bytes());
        hasher.update(&self.cpu_cores.to_le_bytes());
        hasher.update(&self.ram_mb.to_le_bytes());
        hasher.update(&(self.gpu_present as u8).to_le_bytes());
        if let Some(ref gpu) = self.gpu_name {
            hasher.update(gpu.as_bytes());
        }
        hasher.update(self.kernel_version.as_bytes());
        hasher.update(self.os_name.as_bytes());
        hasher.update(self.arch.as_bytes());
        hasher.finalize().to_hex().to_string()
    }

    /// Persists this profile to the database.
    pub fn save_to_db(&self, conn: &rusqlite::Connection) -> anyhow::Result<Uuid> {
        let id = Uuid::new_v4();
        let now = chrono::Utc::now().to_rfc3339();
        conn.execute(
            "INSERT OR REPLACE INTO shells (fingerprint, hostname, os, cpu_count, ram_mb, gpu, last_seen) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)",
            rusqlite::params![
                self.fingerprint,
                self.hostname,
                self.os_name,
                self.cpu_cores as i64,
                self.ram_mb as i64,
                self.gpu_name.as_deref().unwrap_or("none"),
                now,
            ],
        )
        .context("failed to insert shell profile into database")?;
        Ok(id)
    }
}

fn detect_gpu() -> (bool, Option<String>) {
    if let Ok(output) = std::process::Command::new("nvidia-smi")
        .arg("--query-gpu=name")
        .arg("--format=csv,noheader")
        .output()
    {
        if output.status.success() {
            let name = String::from_utf8_lossy(&output.stdout).trim().to_string();
            if !name.is_empty() {
                return (true, Some(name));
            }
        }
    }

    #[cfg(target_os = "linux")]
    {
        if std::path::Path::new("/proc/driver/nvidia").exists() {
            return (true, Some("NVIDIA (detected via /proc)".to_string()));
        }
    }

    (false, None)
}
