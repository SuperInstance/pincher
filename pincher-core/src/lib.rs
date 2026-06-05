//! # PincherOS Core Library
//!
//! `pincher-core` is the core library for PincherOS — a "post-model operating
//! system" using the hermit crab metaphor.

pub mod capability;
pub mod carapace;
pub mod db;
pub mod dynamics;
pub mod embed;
pub mod immunology;
pub mod intent;
pub mod migration;
pub mod reflex;
pub mod resource;
pub mod rpc;
pub mod sandbox;
pub mod security;
pub mod shell;

// ── Crate-level re-exports ──────────────────────────────────────────

pub use embed::{cosine_similarity, download_model, EmbedError, EmbedResult, EMBEDDING_DIM};

pub use reflex::{
    EngineError, EngineResult, EngineStatus, Execution, MatchError, MatchThresholds, MatchType,
    Reflex, ReflexEngine,
};

pub use db::{
    schema::{
        bytes_to_embed, embed_to_bytes, ActionLogRow, ReflexRow, SessionRow, ShellRow,
        EMBEDDING_DIM as DB_EMBEDDING_DIM,
    },
    Database, DbError, DbResult,
};

pub use resource::{
    PidController, ResourceBudget, ResourceController as ResourceCtrl, ResourceError,
    ResourceMetrics, ResourceResult, ResourceState, ResourceThresholds,
};

pub use security::{
    veto::{
        ExecutionContext, VetoDecision, VetoEngine as SecVetoEngine, VetoError, VetoResult,
        VetoRule,
    },
    Capability as SecCapability, LandlockRule, SandboxConfig, SandboxError as SecSandboxError,
    SandboxResult as SecSandboxResult, SignedToken,
};

pub use capability::{
    manifest::{CapabilityManifest, Permission},
    token::CapabilityToken,
};

pub use migration::{
    compatibility_score, fingerprint, fingerprint_hash, pack_nail, read_identity, read_manifest,
    unpack_nail, verify_nail, AgentConfig, AgentIdentity, AgentPreferences, FingerprintError,
    FingerprintResult, NailChecksums, NailManifest, PackError, PackResult, ShellFingerprint,
};

pub use rpc::{
    start_rpc_server, EngineCommand, JsonRpcRequest, JsonRpcResponse, RpcError, RpcErrorValue,
    RpcRequest, RpcResponse,
};
