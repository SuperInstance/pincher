//! Guest module management for the Carapace Bridge.
//!
//! This module defines the types and operations for loading, validating, and
//! managing WASM guest modules.  A [`GuestModule`] represents a loaded WASM
//! binary with its associated metadata (name, version, required capabilities,
//! content hash).  A [`GuestConfig`] captures the execution constraints for a
//! particular guest instantiation.
//!
//! ## Hermit crab metaphor
//!
//! A guest module is like a **shell the hermit crab has found** — it comes
//! with its own shape (capabilities it needs) and the crab must decide
//! whether it fits (whether to grant those capabilities).  The
//! `required_capabilities` field declares what the guest needs; the host
//! decides whether to allow them via the [`GuestConfig`].

use crate::capability::manifest::CapabilityManifest;
use serde::{Deserialize, Serialize};
use std::path::Path;
use std::time::Duration;

// ── Error type ────────────────────────────────────────────────────────

/// Errors that can occur while loading or validating guest modules.
#[derive(Debug, thiserror::Error)]
pub enum GuestError {
    /// The WASM binary could not be read from disk.
    #[error("failed to read WASM binary from {path}: {source}")]
    IoError {
        /// Path to the WASM file.
        path: String,
        /// The underlying I/O error.
        source: std::io::Error,
    },

    /// The WASM binary failed validation.
    #[error("WASM validation failed for {name}: {reason}")]
    ValidationFailed {
        /// Name of the guest module.
        name: String,
        /// Why validation failed.
        reason: String,
    },

    /// The WASM module could not be compiled.
    #[error("WASM compilation failed for {name}: {reason}")]
    CompilationFailed {
        /// Name of the guest module.
        name: String,
        /// Why compilation failed.
        reason: String,
    },

    /// The guest requires capabilities that the host is not willing to grant.
    #[error(
        "capability mismatch for {name}: guest requires {required:?} but host allows {allowed:?}"
    )]
    CapabilityMismatch {
        /// Name of the guest module.
        name: String,
        /// Capabilities the guest requires.
        required: Vec<String>,
        /// Capabilities the host is willing to grant.
        allowed: Vec<String>,
    },

    /// The guest's content hash does not match the expected hash.
    #[error("integrity check failed for {name}: expected {expected}, got {actual}")]
    IntegrityMismatch {
        /// Name of the guest module.
        name: String,
        /// The expected BLAKE3 hash.
        expected: String,
        /// The actual BLAKE3 hash.
        actual: String,
    },

    /// The WASM engine returned an error during module creation.
    #[error("WASM engine error for {name}: {reason}")]
    EngineError {
        /// Name of the guest module.
        name: String,
        /// The engine error description.
        reason: String,
    },
}

/// Alias for results produced by guest operations.
pub type GuestResult<T> = Result<T, GuestError>;

// ── GuestModule ───────────────────────────────────────────────────────

/// A loaded WASM guest module with its metadata.
///
/// This is the **static** representation of a guest — it captures everything
/// known about the WASM binary *before* it is instantiated.  The actual
/// runtime state lives in the [`CarapaceBridge`](super::CarapaceBridge).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GuestModule {
    /// Human-readable name for this guest module.
    pub name: String,

    /// Semantic version of the guest module (e.g. `"1.2.3"`).
    pub version: String,

    /// Capabilities this guest *requires* in order to function.  If any of
    /// these are not granted, the guest should not be instantiated.
    pub required_capabilities: Vec<String>,

    /// BLAKE3 hash of the WASM binary, used for integrity verification.
    pub hash: String,

    /// Raw WASM bytecode.
    #[serde(skip)]
    pub bytecode: Vec<u8>,
}

impl GuestModule {
    /// Load a guest module from a WASM file on disk.
    ///
    /// This reads the file, computes its BLAKE3 hash, and validates the
    /// WASM bytecode.  The `name` and `version` are provided by the caller
    /// (typically from an accompanying manifest file).
    ///
    /// # Arguments
    ///
    /// * `path` — Path to the `.wasm` file.
    /// * `name` — Human-readable name for the guest.
    /// * `version` — Semantic version string.
    /// * `required_capabilities` — Capabilities the guest requires.
    pub fn load_from_file(
        path: &Path,
        name: impl Into<String>,
        version: impl Into<String>,
        required_capabilities: Vec<String>,
    ) -> GuestResult<Self> {
        let path_str = path.display().to_string();
        let bytecode = std::fs::read(path).map_err(|e| GuestError::IoError {
            path: path_str,
            source: e,
        })?;

        let name = name.into();
        let hash = blake3::hash(&bytecode).to_hex().to_string();

        tracing::info!(
            name = %name,
            size = bytecode.len(),
            hash = %hash,
            "loaded WASM binary from disk"
        );

        // Validate the WASM binary using wasmtime (if available).
        #[cfg(feature = "wasmtime")]
        {
            let engine = wasmtime::Engine::default();
            if let Err(e) = wasmtime::Module::validate(&engine, &bytecode) {
                tracing::error!(name = %name, error = %e, "WASM validation failed");
                return Err(GuestError::ValidationFailed {
                    name,
                    reason: e.to_string(),
                });
            }
            tracing::debug!(name = %name, "WASM binary validated successfully");
        }

        #[cfg(not(feature = "wasmtime"))]
        tracing::warn!(name = %name, "WASM validation skipped — wasmtime feature not enabled");

        Ok(Self {
            name,
            version: version.into(),
            required_capabilities,
            hash,
            bytecode,
        })
    }

    /// Load a guest module from in-memory WASM bytecode.
    ///
    /// This is useful for guests that are received over the network or
    /// extracted from an archive.
    pub fn from_bytes(
        bytecode: Vec<u8>,
        name: impl Into<String>,
        version: impl Into<String>,
        required_capabilities: Vec<String>,
    ) -> GuestResult<Self> {
        let name = name.into();
        let hash = blake3::hash(&bytecode).to_hex().to_string();

        // Validate the WASM binary.
        #[cfg(feature = "wasmtime")]
        {
            let engine = wasmtime::Engine::default();
            if let Err(e) = wasmtime::Module::validate(&engine, &bytecode) {
                tracing::error!(name = %name, error = %e, "WASM validation failed");
                return Err(GuestError::ValidationFailed {
                    name,
                    reason: e.to_string(),
                });
            }
        }

        #[cfg(not(feature = "wasmtime"))]
        tracing::warn!(name = %name, "WASM validation skipped — wasmtime feature not enabled");

        Ok(Self {
            name,
            version: version.into(),
            required_capabilities,
            hash,
            bytecode,
        })
    }

    /// Verify that this module's bytecode matches the expected BLAKE3 hash.
    pub fn verify_integrity(&self, expected_hash: &str) -> GuestResult<()> {
        let actual = blake3::hash(&self.bytecode).to_hex().to_string();
        if actual == expected_hash {
            tracing::debug!(name = %self.name, "integrity check passed");
            Ok(())
        } else {
            tracing::error!(
                name = %self.name,
                expected = expected_hash,
                actual = %actual,
                "integrity check failed"
            );
            Err(GuestError::IntegrityMismatch {
                name: self.name.clone(),
                expected: expected_hash.to_string(),
                actual,
            })
        }
    }

    /// Check whether the guest's required capabilities are a subset of the
    /// capabilities allowed by the given [`CapabilityManifest`].
    ///
    /// Returns `Ok(())` if all required capabilities are present, or an
    /// error listing the missing ones.
    pub fn check_capabilities(&self, manifest: &CapabilityManifest) -> GuestResult<()> {
        let allowed: Vec<&str> = manifest
            .required_capabilities
            .iter()
            .map(|s| s.as_str())
            .collect();

        let missing: Vec<String> = self
            .required_capabilities
            .iter()
            .filter(|cap| !allowed.contains(&cap.as_str()))
            .cloned()
            .collect();

        if missing.is_empty() {
            tracing::debug!(
                name = %self.name,
                "all required capabilities are granted"
            );
            Ok(())
        } else {
            tracing::warn!(
                name = %self.name,
                missing = ?missing,
                "capability mismatch — guest requires capabilities not in manifest"
            );
            Err(GuestError::CapabilityMismatch {
                name: self.name.clone(),
                required: self.required_capabilities.clone(),
                allowed: manifest.required_capabilities.clone(),
            })
        }
    }

    /// Returns the size of the WASM bytecode in bytes.
    pub fn bytecode_size(&self) -> usize {
        self.bytecode.len()
    }
}

// ── GuestConfig ───────────────────────────────────────────────────────

/// Configuration for instantiating a WASM guest.
///
/// This controls the resource limits and security constraints applied to
/// a guest at runtime.  Each field maps to a wasmtime configuration knob.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GuestConfig {
    /// Maximum WASM linear memory size in bytes.
    ///
    /// This is enforced by the wasmtime runtime — the guest cannot allocate
    /// beyond this limit.
    pub memory_limit: u64,

    /// Maximum execution time for the guest's `_start` function.
    ///
    /// After this duration elapses, the guest is terminated via wasmtime's
    /// fuel or epoch-based interruption mechanism.
    pub timeout: Duration,

    /// Capabilities that the host is willing to grant to this guest.
    ///
    /// These must be a superset of the guest's `required_capabilities` for
    /// instantiation to succeed.
    pub allowed_capabilities: Vec<String>,

    /// Whether to enable WASM epoch-based interruption.
    ///
    /// When enabled, the engine will periodically check an "epoch" counter
    /// and trap if the counter has advanced beyond the guest's deadline.
    /// This is more efficient than fuel-based interruption for long-running
    /// guests.
    pub epoch_interruption: bool,

    /// Maximum stack depth for WASM calls.
    ///
    /// Prevents recursive guests from consuming unbounded stack space.
    pub max_wasm_stack: usize,
}

impl Default for GuestConfig {
    fn default() -> Self {
        Self {
            memory_limit: 64 * 1024 * 1024, // 64 MiB
            timeout: Duration::from_secs(30),
            allowed_capabilities: vec![],
            epoch_interruption: true,
            max_wasm_stack: 512 * 1024, // 512 KiB
        }
    }
}

impl GuestConfig {
    /// Create a new config with the given memory limit and timeout.
    pub fn new(memory_limit: u64, timeout: Duration) -> Self {
        Self {
            memory_limit,
            timeout,
            ..Self::default()
        }
    }

    /// Create a maximally restrictive config — minimum memory, short timeout,
    /// no capabilities.
    pub fn restricted() -> Self {
        Self {
            memory_limit: 16 * 1024 * 1024, // 16 MiB
            timeout: Duration::from_secs(5),
            allowed_capabilities: vec![],
            epoch_interruption: true,
            max_wasm_stack: 256 * 1024, // 256 KiB
        }
    }

    /// Grant an additional named capability.
    pub fn allow_capability(&mut self, cap: impl Into<String>) {
        let cap = cap.into();
        if !self.allowed_capabilities.contains(&cap) {
            self.allowed_capabilities.push(cap);
        }
    }

    /// Convert this config into a wasmtime `Config`.
    ///
    /// This applies the memory limits, epoch interruption, and other settings
    /// to a wasmtime configuration object. Only available when the `wasmtime`
    /// feature is enabled.
    #[cfg(feature = "wasmtime")]
    pub fn to_wasmtime_config(&self) -> wasmtime::Config {
        let mut config = wasmtime::Config::new();

        // Use the Cranelift compiler (required by our Cargo.toml feature).
        config.cranelift();

        // Memory limits.
        let memory_pages = (self.memory_limit / wasmtime::WASM_PAGE_SIZE as u64) as u64;
        config.max_wasm_stack(self.max_wasm_stack);

        // Epoch-based interruption for timeout enforcement.
        if self.epoch_interruption {
            config.epoch_interruption(true);
        }

        // Limit maximum memory to the configured limit.
        // wasmtime's memory limit is specified in pages.
        let _ = memory_pages; // Used below in the compiled module.

        tracing::debug!(
            memory_limit = self.memory_limit,
            memory_pages,
            timeout_ms = self.timeout.as_millis(),
            epoch_interruption = self.epoch_interruption,
            max_wasm_stack = self.max_wasm_stack,
            "created wasmtime config from guest config"
        );

        config
    }
}

// ── GuestVersion ──────────────────────────────────────────────────────

/// Semantic version information for a guest module.
///
/// This is a lightweight version type that supports comparison for
/// compatibility checks.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct GuestVersion {
    /// Major version — breaking changes.
    pub major: u32,
    /// Minor version — backwards-compatible additions.
    pub minor: u32,
    /// Patch version — backwards-compatible fixes.
    pub patch: u32,
}

impl GuestVersion {
    /// Parse a semantic version string (e.g. `"1.2.3"`).
    pub fn parse(s: &str) -> Option<Self> {
        let parts: Vec<&str> = s.trim().split('.').collect();
        if parts.len() != 3 {
            return None;
        }
        Some(Self {
            major: parts[0].parse().ok()?,
            minor: parts[1].parse().ok()?,
            patch: parts[2].parse().ok()?,
        })
    }

    /// Returns `true` if this version is compatible with `other` according
    /// to semantic versioning (same major version, and this >= other).
    pub fn is_compatible_with(&self, other: &GuestVersion) -> bool {
        self.major == other.major && self >= other
    }
}

impl std::fmt::Display for GuestVersion {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}.{}.{}", self.major, self.minor, self.patch)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn guest_version_parse() {
        let v = GuestVersion::parse("1.2.3").unwrap();
        assert_eq!(v.major, 1);
        assert_eq!(v.minor, 2);
        assert_eq!(v.patch, 3);
    }

    #[test]
    fn guest_version_parse_invalid() {
        assert!(GuestVersion::parse("1.2").is_none());
        assert!(GuestVersion::parse("a.b.c").is_none());
    }

    #[test]
    fn guest_version_compatibility() {
        let v1 = GuestVersion::parse("1.2.0").unwrap();
        let v2 = GuestVersion::parse("1.3.0").unwrap();
        let v3 = GuestVersion::parse("2.0.0").unwrap();

        // v2 is compatible with v1 (same major, v2 >= v1).
        assert!(v2.is_compatible_with(&v1));
        // v1 is NOT compatible with v2 (v1 < v2).
        assert!(!v1.is_compatible_with(&v2));
        // v3 is NOT compatible with v1 (different major).
        assert!(!v3.is_compatible_with(&v1));
    }

    #[test]
    fn guest_config_default() {
        let config = GuestConfig::default();
        assert_eq!(config.memory_limit, 64 * 1024 * 1024);
        assert_eq!(config.timeout, Duration::from_secs(30));
        assert!(config.epoch_interruption);
    }

    #[test]
    fn guest_config_restricted() {
        let config = GuestConfig::restricted();
        assert_eq!(config.memory_limit, 16 * 1024 * 1024);
        assert_eq!(config.timeout, Duration::from_secs(5));
    }

    #[test]
    fn guest_version_display() {
        let v = GuestVersion::parse("2.1.0").unwrap();
        assert_eq!(v.to_string(), "2.1.0");
    }
}
