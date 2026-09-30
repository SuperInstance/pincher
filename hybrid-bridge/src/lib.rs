#![deny(unsafe_code)]

//! # Hybrid Bridge
//!
//! The communication backbone of the **Hybrid Manifold** — connecting the
//! Matrix Engine, Room Agents, and Veto Engine through high-performance
//! async channels on ARM64.
//!
//! ## Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────┐
//! │                     HybridEngine                                │
//! │  ┌─────────┐    ┌──────────┐    ┌──────────┐    ┌───────────┐  │
//! │  │ MATRIX   │ ──►│  ROOMS   │ ──►│  VETO    │ ──►│ EXECUTION │  │
//! │  │  Phase   │    │  Phase   │    │  Phase   │    │  Phase    │  │
//! │  └────┬─────┘    └────┬─────┘    └────┬─────┘    └─────┬─────┘  │
//! │       │               │               │               │        │
//! │       └── broadcast ──┘    mpsc ──────┘   broadcast ──┘        │
//! │                              ┌─┐                               │
//! │                              │ │  ← HybridBridge channels      │
//! │                              └─┘                               │
//! └─────────────────────────────────────────────────────────────────┘
//! ```
//!
//! ## Performance Targets (ARM64)
//!
//! | Operation | Target |
//! |-----------|--------|
//! | Matrix fast cycle | < 3ms |
//! | Room analysis (per agent) | < 100ms |
//! | Veto resolution (5000 rooms) | < 10ms |
//! | End-to-end hybrid cycle | < 1s |
//!
//! ## Usage
//!
//! ```rust,no_run
//! use hybrid_bridge::prelude::*;
//!
//! # async fn example() {
//! // Create the communication backbone
//! let bridge = std::sync::Arc::new(HybridBridge::new());
//!
//! // Subscribe to matrix snapshots
//! let mut snapshot_rx = bridge.subscribe_matrix();
//!
//! // Submit a proposal
//! bridge.submit_proposal(RoomProposal {
//!     ticker: "AAPL".into(),
//!     gate: TernaryGate::Bullish,
//!     conviction: 0.85,
//!     confidence: 0.72,
//!     narrative_sig: "abc123".into(),
//!     matrix_agreement: 0.91,
//!     veto_override: false,
//!     timestamp: 1000,
//! }).await.unwrap();
//! # }
//! ```

// ── Module declarations ──────────────────────────────────────────────

/// Core `HybridBridge` — async broadcast/mpsc backbone connecting all layers.
pub mod bridge;

/// Chaos testing utilities — NaN/Inf injection, detection, and recovery.
pub mod chaos;

/// CLI front-end for inspecting and controlling the Hybrid Manifold.
pub mod cli;

/// Market data feed adapters (CSV, WebSocket, etc.) for matrix ingestion.
pub mod datafeed;

/// Engine traits (`MatrixEngine`, `RoomAgent`, `VetoEngine`, `HybridEngine`)
/// and the concrete `HybridEngineImpl` orchestrator.
pub mod engine;

/// Error types for hybrid bridge operations.
pub mod error;

/// In-memory `Array3<f32>`-backed mock of the `MatrixEngine` trait.
pub mod mock_matrix;

/// Deterministic mock `RoomAgent` for integration testing.
pub mod mock_room;

/// Configurable mock `VetoEngine` with SAEP constraint support.
pub mod mock_veto;

/// Bridging layer between Hybrid Manifold and ternary-types ecosystem.
pub mod ternary_bridge;

/// Shared type definitions — topology, proposals, portfolios, tensor utilities.
pub mod types;

// ── Prelude ──────────────────────────────────────────────────────────

/// Convenience re-exports for downstream consumers.
pub mod prelude {
    pub use crate::bridge::{BridgeMetricSnapshot, BridgeMetrics, HybridBridge};
    pub use crate::datafeed::{CsvFileFeed, MarketDataFeed, StockTick};
    pub use crate::engine::{
        DefaultVetoEngine, HybridConfig, HybridEngine, HybridEngineImpl, MatrixEngine, RoomAgent,
        VetoEngine,
    };
    pub use crate::error::{HybridError, HybridResult};
    pub use crate::mock_room::MockRoomAgent;
    pub use crate::mock_veto::MockVetoEngine;
    pub use crate::types::{
        detect_non_finite, mask_non_finite, CheckFn, FeatureSuggestion, FinalPosition,
        GovernanceLayer, HybridMessage, MatrixMetadata, MatrixSnapshot, PartialSnapshot,
        PortfolioVector, RoomProposal, SaepAction, SaepConstraint, TernaryGate,
        TopologicalSignature, Violation,
    };
}

// ── Crate-level re-exports (for ergonomic `use hybrid_bridge::*`) ────

pub use bridge::{BridgeMetricSnapshot, BridgeMetrics, HybridBridge};
pub use datafeed::{CsvFileFeed, MarketDataFeed, StockTick};
pub use engine::{
    DefaultVetoEngine, HybridConfig, HybridEngine, HybridEngineImpl,
    MatrixEngine as MatrixEngineTrait, RoomAgent as RoomAgentTrait, VetoEngine as VetoEngineTrait,
};
pub use error::{HybridError, HybridResult};
pub use mock_room::MockRoomAgent;
pub use mock_veto::MockVetoEngine;
pub use types::{
    detect_non_finite, mask_non_finite, CheckFn, FeatureSuggestion, FinalPosition, GovernanceLayer,
    HybridMessage, MatrixSnapshot, PortfolioVector, RoomProposal, SaepAction, SaepConstraint,
    TernaryGate, TopologicalSignature, Violation,
};

// ── Tests ────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Smoke test: all types can be constructed and sent across channels.
    #[tokio::test]
    async fn test_smoke_bridge_send_receive() {
        let bridge = HybridBridge::default();
        let mut system_rx = bridge.subscribe_system_events();

        // Send a system event
        bridge.emit_system_event("test".into(), "smoke test".into());

        // Receive via system channel
        match system_rx.recv().await.unwrap() {
            HybridMessage::SystemEvent { kind, payload } => {
                assert_eq!(kind, "test");
                assert_eq!(payload, "smoke test");
            }
            _ => panic!("Expected SystemEvent"),
        }
    }

    /// Test that the prelude exports everything needed (compile-time check).
    #[test]
    fn test_prelude_exports() {
        fn _check(_: prelude::HybridBridge) {}
        fn _check2(_: prelude::TernaryGate) {}
        fn _check3(_: prelude::HybridError) {}
        fn _check4(_: prelude::MatrixSnapshot) {}
        fn _check5(_: prelude::PortfolioVector) {}
        fn _check6(_: prelude::RoomProposal) {}
        fn _check7(_: prelude::FeatureSuggestion) {}
        fn _check8(_: prelude::FinalPosition) {}
        fn _check9(_: prelude::TopologicalSignature) {}
        fn _check10(_: prelude::GovernanceLayer) {}
        fn _check11(_: prelude::SaepAction) {}
        fn _check12(_: prelude::Violation) {}
        fn _check13(_: prelude::HybridConfig) {}
        fn _check14(_: prelude::MockRoomAgent) {}
        fn _check15(_: prelude::MockVetoEngine) {}
    }

    /// Test the default veto engine is constructable from prelude.
    #[test]
    fn test_default_veto_from_prelude() {
        let _veto = prelude::DefaultVetoEngine::new();
    }

    /// Test that HybridEngineImpl type is recognized (compile check).
    #[test]
    fn test_engine_impl_exists() {
        fn _check_engine<M: engine::MatrixEngine + 'static, V: engine::VetoEngine + 'static>(
            _engine: HybridEngineImpl<M, V>,
        ) {
        }
    }
}
