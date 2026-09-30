//! Type definitions for the Hybrid Manifold communication layer.
//!
//! Defines the shared data structures flowing between the Matrix Engine,
//! Room Agents, and Veto Engine through the HybridBridge.

use ndarray::Array2;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;

// ─────────────────────────────────────────────────────────────────────
// 1. Topology & Matrix Types
// ─────────────────────────────────────────────────────────────────────

/// Topological snapshot of a single ticker's position in the manifold.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct TopologicalSignature {
    pub ticker: String,
    pub betti_numbers: Vec<usize>, // [β₀, β₁, β₂]
    pub persistence_landscape: Vec<f64>,
    pub wasserstein_distance_centroid: f64,
    pub regime_label: String, // e.g., "rotation", "fragmentation", "stable"
    pub confidence: f64,      // [0.0, 1.0]
}

/// Lightweight metadata returned after a fast-path matrix cycle.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MatrixMetadata {
    pub tick: u64,
    pub n_stocks: usize,
    pub n_features: usize,
    pub n_history: usize,
    pub mean_correlation: f64,
    pub timestamp_ms: i64,
}

/// Partial snapshot returned after a medium-path cycle.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PartialSnapshot {
    pub tick: u64,
    pub n_stocks: usize,
    pub correlation_matrix_cond: f64,
    pub top_eigenvalues: Vec<f64>,
    pub regime: String,
    pub timestamp_ms: i64,
}

/// Result of a full matrix cycle — the richest output.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MatrixSnapshot {
    /// Timestamp (Unix millis)
    pub tick: u64,
    /// Number of active stocks
    pub n_stocks: usize,
    /// Top N eigenvalues (explained variance)
    pub eigenvalues: Vec<f64>,
    /// Top N eigenvectors (factor loadings), truncated to top_k
    pub eigenvectors: Array2<f64>,
    /// Per-stock topological signatures (for all active stocks)
    pub topologies: Vec<TopologicalSignature>,
    /// Matrix-level Betti counts (aggregated)
    pub universe_betti: [usize; 3],
    /// Global regime flag
    pub regime: String,
    /// Correlation matrix condition number (stability indicator)
    pub condition_number: f64,
}

// ─────────────────────────────────────────────────────────────────────
// 2. Ternary Gate & Room Proposal Types
// ─────────────────────────────────────────────────────────────────────

/// The ternary gate: direction decision.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum TernaryGate {
    Bullish, // +1
    Neutral, // 0 (leminal zone)
    Bearish, // -1
}

impl TernaryGate {
    /// Convert to a numeric weight {-1, 0, +1}.
    pub fn to_i8(&self) -> i8 {
        match self {
            TernaryGate::Bullish => 1,
            TernaryGate::Neutral => 0,
            TernaryGate::Bearish => -1,
        }
    }
}

/// A room's proposal to the Veto Engine.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RoomProposal {
    pub ticker: String,
    pub gate: TernaryGate,
    pub conviction: f64,       // Continuous weight [0.0, 1.0]
    pub confidence: f64,       // Reflex confidence [0.0, 1.0]
    pub narrative_sig: String, // Hash of the narrative (for audit)
    pub matrix_agreement: f64, // How much does this agree with matrix consensus? [0,1]
    pub veto_override: bool,   // Agent flags this as "skeptical"
    pub timestamp: u64,
}

/// Feature suggestion — rooms can propose new matrix features.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeatureSuggestion {
    pub ticker: String,
    pub feature_name: String,  // e.g., "lithium_correlation"
    pub source: String,        // e.g., "earnings_call_analysis"
    pub urgency: f64,          // [0, 1]
    pub sample_data: Vec<f64>, // 20 data points for initialization
}

// ─────────────────────────────────────────────────────────────────────
// 3. Veto Engine & Portfolio Types
// ─────────────────────────────────────────────────────────────────────

/// Governance layer for SAEP constraint resolution.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum GovernanceLayer {
    Room,
    Sector,
    Portfolio,
    Market,
}

/// What a SAEP constraint does when violated.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum SaepAction {
    /// Log a warning, allow action
    Warn,
    /// Cap the position at the constraint limit
    Limit,
    /// Reject the proposal entirely
    Veto,
    /// Freeze all actions across the affected layer
    Freeze,
}

/// A single stock's final position (after veto).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FinalPosition {
    pub ticker: String,
    pub weight: f64,               // Final allocation [-1.0, 1.0]
    pub raw_gate: TernaryGate,     // What the room wanted
    pub veto_applied: Vec<String>, // Which SAEP constraints fired
    pub veto_severity: f64,        // [0 = no veto, 1 = full veto]
}

/// The final portfolio output.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PortfolioVector {
    pub positions: Vec<FinalPosition>,
    pub gross_exposure: f64,
    pub net_exposure: f64,
    pub sector_concentrations: HashMap<String, f64>,
    pub portfolio_var: f64,
    pub timestamp: u64,
}

/// Type alias for the SAEP constraint check function.
pub type CheckFn =
    Arc<dyn Fn(&RoomProposal, &HashMap<String, f64>) -> Result<(), Violation> + Send + Sync>;

/// A SAEP constraint pattern.
pub struct SaepConstraint {
    pub id: String,
    pub layer: GovernanceLayer,
    pub check_fn: CheckFn,
    pub action: SaepAction,
    pub escalate_to: Option<GovernanceLayer>,
}

impl std::fmt::Debug for SaepConstraint {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SaepConstraint")
            .field("id", &self.id)
            .field("layer", &self.layer)
            .field("action", &self.action)
            .field("escalate_to", &self.escalate_to)
            .finish()
    }
}

impl Clone for SaepConstraint {
    fn clone(&self) -> Self {
        SaepConstraint {
            id: self.id.clone(),
            layer: self.layer.clone(),
            check_fn: self.check_fn.clone(),
            action: self.action.clone(),
            escalate_to: self.escalate_to.clone(),
        }
    }
}

/// Result of checking a SAEP constraint.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Violation {
    pub constraint_id: String,
    pub message: String,
    pub severity: f64,
}

// ─────────────────────────────────────────────────────────────────────
// 4. Hybrid Message Enum
// ─────────────────────────────────────────────────────────────────────

/// Messages sent between the Matrix Engine and Room Agents via the bridge.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum HybridMessage {
    /// From Matrix to all Rooms: new snapshot available
    SnapshotBroadcast(MatrixSnapshot),
    /// From Room to Veto: new proposal
    ProposalSubmission(RoomProposal),
    /// From Room to Matrix: feature suggestion
    FeatureSuggestion(FeatureSuggestion),
    /// From Veto to all: final portfolio vector
    PortfolioVectorBroadcast(PortfolioVector),
    /// From Matrix to specific Room: slice update
    SliceUpdate { ticker: String, data: Array2<f32> },
    /// Symmetry alert broadcast
    SymmetryAlert { tickers: Vec<String>, score: f64 },
    /// System event (freeze, error, etc.)
    SystemEvent { kind: String, payload: String },
}

impl HybridMessage {
    /// Returns a human-readable label for the message variant.
    pub fn variant_name(&self) -> &'static str {
        match self {
            HybridMessage::SnapshotBroadcast(_) => "SnapshotBroadcast",
            HybridMessage::ProposalSubmission(_) => "ProposalSubmission",
            HybridMessage::FeatureSuggestion(_) => "FeatureSuggestion",
            HybridMessage::PortfolioVectorBroadcast(_) => "PortfolioVectorBroadcast",
            HybridMessage::SliceUpdate { .. } => "SliceUpdate",
            HybridMessage::SymmetryAlert { .. } => "SymmetryAlert",
            HybridMessage::SystemEvent { .. } => "SystemEvent",
        }
    }
}

// ─────────────────────────────────────────────────────────────────────
// 5. Tensor Utility Functions
// ─────────────────────────────────────────────────────────────────────

/// Check a feature tensor for NaN/Inf values, returning the problematic
/// coordinates. If `n_flagged > 0`, the matrix should enter safe mode.
pub fn detect_non_finite(tensor: &ndarray::Array3<f32>) -> Vec<(usize, usize, usize)> {
    let mut flagged = Vec::new();
    for ((i, j, k), &val) in tensor.indexed_iter() {
        if !val.is_finite() {
            flagged.push((i, j, k));
        }
    }
    flagged
}

/// Mask non-finite values in the tensor — replace NaN/Inf with 0.0.
/// Returns the count of masked cells for diagnostics.
pub fn mask_non_finite(tensor: &mut ndarray::Array3<f32>) -> usize {
    let mut count = 0;
    for val in tensor.iter_mut() {
        if !val.is_finite() {
            *val = 0.0;
            count += 1;
        }
    }
    count
}
