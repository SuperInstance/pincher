//! Migration module — .nail shell portability for PincherOS

pub mod fingerprint;
pub mod pack;

// Re-export key types
pub use fingerprint::{
    compatibility_score, fingerprint, fingerprint_hash, FingerprintError, FingerprintResult,
    ShellFingerprint,
};
pub use pack::{
    pack_nail, read_identity, read_manifest, unpack_nail, verify_nail, AgentConfig, AgentIdentity,
    AgentPreferences, NailChecksums, NailManifest, PackError, PackResult,
};
