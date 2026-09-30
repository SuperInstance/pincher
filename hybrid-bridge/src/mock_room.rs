//! Mock Room Agent for integration tests.
//!
//! Provides an in-memory implementation of the `RoomAgent` trait that
//! produces proposals based on matrix snapshot data.

use crate::engine::RoomAgent;
use crate::types::{
    FeatureSuggestion, MatrixSnapshot, RoomProposal, TernaryGate, TopologicalSignature,
};
use async_trait::async_trait;
use ndarray::Array1;
use ndarray::Array2;
use tracing::debug;

/// A mock room agent that generates deterministic proposals.
pub struct MockRoomAgent {
    ticker: String,
    narrative: String,
}

impl MockRoomAgent {
    /// Create a new mock room agent for the given ticker.
    pub fn new(ticker: &str) -> Self {
        Self {
            ticker: ticker.to_string(),
            narrative: String::new(),
        }
    }

    /// Set a narrative for the room agent.
    pub fn with_narrative(mut self, narrative: &str) -> Self {
        self.narrative = narrative.to_string();
        self
    }
}

#[async_trait]
impl RoomAgent for MockRoomAgent {
    async fn analyze(
        &self,
        snapshot: &MatrixSnapshot,
        _feature_slice: Option<Array2<f32>>,
        _cross_section: Option<Array1<f32>>,
    ) -> RoomProposal {
        // Find the topology for this ticker to guide proposal generation
        let topology = snapshot.topologies.iter().find(|t| t.ticker == self.ticker);

        // Determine gate based on ticker hash for reproducibility
        let hash: u64 = self.ticker.bytes().map(|b| b as u64).sum();
        let gate_idx = hash % 3;
        let gate = match gate_idx {
            0 => TernaryGate::Bullish,
            1 => TernaryGate::Bearish,
            _ => TernaryGate::Neutral,
        };

        // Use topology-derived confidence if available
        let confidence = topology.map(|t| t.confidence).unwrap_or(0.7);

        // Simulate conviction based on the confidence and some noise based on ticker
        let conviction = (0.5 + (hash % 100) as f64 / 200.0).clamp(0.0, 1.0);

        let matrix_agreement = topology
            .map(|t| (t.confidence * 0.8 + 0.2).clamp(0.0, 1.0))
            .unwrap_or(0.7);

        // Generate a simple narrative signature from the narrative text
        let narrative_sig = if self.narrative.is_empty() {
            format!("mock:{}:{}", self.ticker, snapshot.tick)
        } else {
            format!("{}:{}", self.narrative, snapshot.tick)
        };

        debug!(
            "MockRoomAgent({}) analyze → {:?} conviction={:.2}",
            self.ticker, gate, conviction
        );

        RoomProposal {
            ticker: self.ticker.clone(),
            gate,
            conviction,
            confidence,
            narrative_sig,
            matrix_agreement,
            veto_override: false,
            timestamp: snapshot.tick,
        }
    }

    async fn on_symmetry_alert(
        &self,
        peer_ticker: &str,
        symmetry_score: f64,
        _topology: &TopologicalSignature,
    ) {
        debug!(
            "MockRoomAgent({}) received symmetry alert from {} (score={:.2})",
            self.ticker, peer_ticker, symmetry_score
        );
    }

    async fn suggest_feature(&self, _suggestion: FeatureSuggestion) {}

    async fn set_regime(&mut self, _label: String) {}

    async fn update_narrative(&mut self, narrative: String) {
        self.narrative = narrative;
    }
}
