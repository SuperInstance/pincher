//! Security module for PincherOS
//!
//! Provides the veto engine (deterministic safety rules) and sandbox
//! integration (capability-based isolation) for safe reflex execution.

pub mod sandbox;
pub mod veto;

// Re-export key types from veto
pub use veto::{ExecutionContext, VetoDecision, VetoEngine, VetoError, VetoResult, VetoRule};

// Re-export key types from sandbox
pub use sandbox::{
    Capability, LandlockRule, SandboxConfig, SandboxError, SandboxResult, SignedToken,
};
