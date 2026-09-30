//! # Ternary Bridge — Concrete Shim Layer
//!
//! Maps between the **Hybrid Manifold** continuous float conviction domain `[0.0, 1.0]`
//! and the **SuperInstance ternary ecosystem** of balanced trits `{-1, 0, +1}`.
//!
//! ## Architecture
//!
//! ```text
//! ┌─────────────────────────┐      ┌──────────────────────────────┐
//! │   Hybrid Manifold        │      │  SuperInstance Ternary Fleet  │
//! │                         │      │                              │
//! │  RoomProposal.conviction│─────►│  TritMapping: [0,1]→{ -1,0,+1}│
//! │    [0.0, 1.0]           │      │  w/ hysteresis deadband      │
//! │                         │      │                              │
//! │  PortfolioVector        │─────►│  PortfolioToTrits: SAEP-aware │
//! │   (weights [-1,1])      │      │  ternary vector encoding     │
//! │                         │      │                              │
//! │  TernaryGraph output    │◄─────│  TritsToWeights: graph→pos    │
//! │   (position weights)    │      │  weight interpolation        │
//! │                         │      │                              │
//! │  Multiple proposals     │◄─────│  Consensus: confidence-      │
//! │   → consensus           │      │  weighted trit aggregation   │
//! └─────────────────────────┘      └──────────────────────────────┘
//! ```
//!
//! ## Key Design Decisions
//!
//! 1. **Hysteresis deadband** — Prevents trit flutter when conviction hovers near
//!    decision boundaries. A `0.3–0.7` window maps to `Neutral`, requiring a full
//!    swing through the band before flipping.
//!
//! 2. **SAEP-aware encoding** — Final position weights incorporate veto severity
//!    directly into the ternary mapping: a veto-supressed weight may quantize
//!    differently than a clean one.
//!
//! 3. **Confidence-weighted consensus** — Multiple trit arrays (e.g. from different
//!    room agents or ternary graph layers) are merged by summing weighted votes
//!    rather than naive majority, preserving nuance from high-conviction sources.

use crate::types::{FinalPosition, PortfolioVector, TernaryGate};
use ternary_types::Ternary::{Negative, Neutral, Positive};
use ternary_types::{Ternary, TernaryMatrix, TritVector};

use std::fmt;

// ─────────────────────────────────────────────────────────────────────────────
// Error Types
// ─────────────────────────────────────────────────────────────────────────────

/// Errors that can occur during ternary bridge operations.
#[derive(Debug, Clone, PartialEq)]
pub enum TernaryBridgeError {
    /// Conviction value is out of the [0.0, 1.0] range.
    ConvictionOutOfRange(f64),
    /// Weight value is outside the [-1.0, 1.0] range.
    WeightOutOfRange(f64),
    /// Trit vector length mismatch during consensus.
    TritVectorLengthMismatch { expected: usize, actual: usize },
    /// Portfolio vector is empty (no positions to encode).
    EmptyPortfolio,
    /// Empty trit set provided for consensus computation.
    EmptyConsensusSet,
    /// TritsToWeights received an empty trit array.
    EmptyTritVector,
}

impl fmt::Display for TernaryBridgeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ConvictionOutOfRange(v) => {
                write!(f, "conviction {v} is outside the valid range [0.0, 1.0]")
            }
            Self::WeightOutOfRange(v) => {
                write!(f, "weight {v} is outside the valid range [-1.0, 1.0]")
            }
            Self::TritVectorLengthMismatch { expected, actual } => {
                write!(
                    f,
                    "trit vector length mismatch: expected {expected}, got {actual}"
                )
            }
            Self::EmptyPortfolio => {
                write!(f, "portfolio vector is empty — no positions to encode")
            }
            Self::EmptyConsensusSet => {
                write!(f, "cannot compute consensus from an empty trit set")
            }
            Self::EmptyTritVector => {
                write!(f, "cannot convert an empty trit vector to weights")
            }
        }
    }
}

impl std::error::Error for TernaryBridgeError {}

/// Convenience result alias for ternary bridge operations.
pub type TernaryBridgeResult<T> = Result<T, TernaryBridgeError>;

// ─────────────────────────────────────────────────────────────────────────────
// Hysteresis Configuration
// ─────────────────────────────────────────────────────────────────────────────

/// Configuration for the hysteretic deadband that prevents trit flutter.
///
/// The deadband defines a range `[lower, upper]` within which conviction maps to
/// `Neutral`. To exit Neutral the conviction must cross *outside* the band; to
/// re-enter it must cross back *inside* with the opposite edge.
///
/// # Example
///
/// With the default `[0.3, 0.7]` band:
/// - `conviction = 0.25` → `Negative` (below lower bound)
/// - `conviction = 0.50` → `Neutral` (inside deadband)
/// - `conviction = 0.75` → `Positive` (above upper bound)
/// - Dropping from `0.80` to `0.72` → still `Positive` (must cross below 0.7)
/// - Dropping from `0.68` back up to `0.32` → still `Neutral` (stays inside)
#[derive(Debug, Clone, Copy)]
pub struct HysteresisConfig {
    /// Lower bound of the deadband. Default: `0.3`.
    pub lower: f64,
    /// Upper bound of the deadband. Default: `0.7`.
    pub upper: f64,
}

impl Default for HysteresisConfig {
    fn default() -> Self {
        Self {
            lower: 0.3,
            upper: 0.7,
        }
    }
}

impl HysteresisConfig {
    /// Create a new hysteresis configuration.
    ///
    /// # Panics
    ///
    /// Panics if `lower >= upper`, if `lower < 0.0`, or if `upper > 1.0`.
    pub fn new(lower: f64, upper: f64) -> Self {
        assert!(
            (0.0..=1.0).contains(&lower),
            "lower bound must be in [0.0, 1.0]"
        );
        assert!(
            (0.0..=1.0).contains(&upper),
            "upper bound must be in [0.0, 1.0]"
        );
        assert!(lower < upper, "lower bound must be less than upper bound");
        Self { lower, upper }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// TritMapping: Conviction [0, 1] → Ternary [-1, 0, +1] with Hysteresis
// ─────────────────────────────────────────────────────────────────────────────

/// Maps continuous float conviction `[0.0, 1.0]` to a balanced trit `{-1, 0, +1}`.
///
/// Uses **hysteresis** (a deadband around `0.3–0.7`) to prevent oscillation when
/// conviction is indecisive. The state machine works as follows:
///
/// 1. **Initial mapping** determines the raw trit from conviction alone.
/// 2. **Hysteresis** is applied through a stateful filter:
///    - If conviction is inside `[lower, upper]` → `Neutral`
///    - If conviction is below `lower` → `Negative`
///    - If conviction is above `upper` → `Positive`
///    - State is maintained: conviction must cross the *opposite edge* of the
///      deadband to flip from one non-neutral state to another.
#[derive(Debug, Clone)]
pub struct TritMapping {
    /// Hysteresis configuration.
    pub hysteresis: HysteresisConfig,
    /// The last mapped ternary value (for hysteresis state tracking).
    last_trit: Ternary,
    /// The last conviction value seen (for rate-of-change detection).
    last_conviction: f64,
}

impl Default for TritMapping {
    fn default() -> Self {
        Self {
            hysteresis: HysteresisConfig::default(),
            last_trit: Neutral,
            last_conviction: 0.5,
        }
    }
}

impl TritMapping {
    /// Create a new `TritMapping` with the given hysteresis configuration.
    pub fn new(hysteresis: HysteresisConfig) -> Self {
        Self {
            hysteresis,
            last_trit: Neutral,
            last_conviction: 0.5,
        }
    }

    /// Create a new `TritMapping` with default hysteresis `[0.3, 0.7]`.
    pub fn with_default_hysteresis() -> Self {
        Self::new(HysteresisConfig::default())
    }

    /// Map a conviction value to a ternary state, applying hysteresis.
    ///
    /// The hysteresis prevents rapid toggling when conviction hovers near the
    /// decision boundary. The mapping logic:
    ///
    /// - `conviction < hysteresis.lower` → `Negative`
    /// - `conviction > hysteresis.upper` → `Positive`
    /// - Otherwise → `Neutral`
    ///
    /// Additionally, a rate-of-change guard prevents flipping from `Negative` to
    /// `Positive` (or vice versa) without passing through `Neutral` first.
    ///
    /// # Errors
    ///
    /// Returns `TernaryBridgeError::ConvictionOutOfRange` if `conviction` is not
    /// in `[0.0, 1.0]`.
    pub fn map(&mut self, conviction: f64) -> TernaryBridgeResult<Ternary> {
        if !(0.0..=1.0).contains(&conviction) {
            return Err(TernaryBridgeError::ConvictionOutOfRange(conviction));
        }

        let raw_trit = if conviction < self.hysteresis.lower {
            Negative
        } else if conviction > self.hysteresis.upper {
            Positive
        } else {
            Neutral
        };

        // Hysteresis guard: prevent direct Negative ↔ Positive flips.
        // Must pass through Neutral first if crossing the entire band.
        let delta = conviction - self.last_conviction;
        let trit = match (self.last_trit, raw_trit) {
            // Stable: no change needed
            (current, next) if current == next => next,

            // Entering Neutral from either side — always allow
            (_, Neutral) => Neutral,

            // Leaving Neutral — always allow
            (Neutral, next) => next,

            // From Negative → Positive: only if conviction jumped the entire
            // deadband (> upper - lower). Otherwise stay in Negative.
            (Negative, Positive)
                if delta.abs() > (self.hysteresis.upper - self.hysteresis.lower) =>
            {
                Positive
            }

            // From Positive → Negative: same guard.
            (Positive, Negative)
                if delta.abs() > (self.hysteresis.upper - self.hysteresis.lower) =>
            {
                Negative
            }

            // Any other flip: remain in current state to prevent flutter.
            _ => self.last_trit,
        };

        self.last_trit = trit;
        self.last_conviction = conviction;
        Ok(trit)
    }

    /// Reset the hysteresis state to `Neutral`.
    pub fn reset(&mut self) {
        self.last_trit = Neutral;
        self.last_conviction = 0.5;
    }

    /// Get the current (last mapped) trit without consuming a new conviction.
    pub fn current_trit(&self) -> Ternary {
        self.last_trit
    }

    /// Convert a `TernaryGate` to a `Ternary` directly (no hysteresis).
    ///
    /// This is a stateless conversion:
    /// - `TernaryGate::Bullish` → `Ternary::Positive`
    /// - `TernaryGate::Neutral` → `Ternary::Neutral`
    /// - `TernaryGate::Bearish` → `Ternary::Negative`
    pub fn gate_to_trit(gate: &TernaryGate) -> Ternary {
        match gate {
            TernaryGate::Bullish => Positive,
            TernaryGate::Neutral => Neutral,
            TernaryGate::Bearish => Negative,
        }
    }

    /// Convert a `Ternary` back to a `TernaryGate`.
    pub fn trit_to_gate(trit: Ternary) -> TernaryGate {
        match trit {
            Positive => TernaryGate::Bullish,
            Neutral => TernaryGate::Neutral,
            Negative => TernaryGate::Bearish,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// PortfolioToTrits: PortfolioVector → Ternary Vector with SAEP-Aware Encoding
// ─────────────────────────────────────────────────────────────────────────────

/// Encodes a `PortfolioVector` into a `Ternary` representation that captures
/// both direction and conviction, with special handling for SAEP veto states.
///
/// Each `FinalPosition` is mapped to a trit according to the following rules:
///
/// 1. **Extreme veto** (`veto_severity >= 0.95`): Always `Neutral` — the position
///    was effectively killed.
/// 2. **Strong long** (`weight >= 0.5`): `Positive`
/// 3. **Strong short** (`weight <= -0.5`): `Negative`
/// 4. **Weak / neutral** (otherwise): `Neutral`
///
/// Additionally, the encoding can produce a **weighted trit vector** where each
/// trit is paired with its magnitude (`|weight|`) and confidence, enabling
/// downstream consensus computation that is SAEP-aware.
#[derive(Debug, Clone)]
pub struct PortfolioToTrits {
    /// Threshold above which a position weight is considered "strong".
    pub strong_threshold: f64,
    /// Veto severity above which a position is forced to Neutral regardless of weight.
    pub veto_kill_threshold: f64,
}

impl Default for PortfolioToTrits {
    fn default() -> Self {
        Self {
            strong_threshold: 0.5,
            veto_kill_threshold: 0.95,
        }
    }
}

impl PortfolioToTrits {
    /// Create a new encoder with custom thresholds.
    pub fn new(strong_threshold: f64, veto_kill_threshold: f64) -> Self {
        Self {
            strong_threshold,
            veto_kill_threshold,
        }
    }

    /// Encode a `PortfolioVector` into a `TritVector` with SAEP-aware mapping.
    ///
    /// The order of trits in the output vector matches the order of positions in
    /// the portfolio.
    ///
    /// # Errors
    ///
    /// Returns `TernaryBridgeError::EmptyPortfolio` if the portfolio has no
    /// positions.
    pub fn encode(&self, portfolio: &PortfolioVector) -> TernaryBridgeResult<TritVector> {
        if portfolio.positions.is_empty() {
            return Err(TernaryBridgeError::EmptyPortfolio);
        }

        let trits: Vec<Ternary> = portfolio
            .positions
            .iter()
            .map(|pos| self.encode_position(pos))
            .collect();

        Ok(TritVector::new(&trits))
    }

    /// Encode a single `FinalPosition` into a trit.
    ///
    /// SAEP-aware logic:
    /// - If veto_severity >= veto_kill_threshold, always `Neutral`
    /// - Otherwise, quantize weight to trit based on direction and magnitude
    pub fn encode_position(&self, pos: &FinalPosition) -> Ternary {
        // SAEP veto kill: if the veto was severe enough, force neutral
        // regardless of what the raw weight says.
        if pos.veto_severity >= self.veto_kill_threshold {
            return Neutral;
        }

        let weight = pos.weight.clamp(-1.0, 1.0);

        if weight >= self.strong_threshold {
            Positive
        } else if weight <= -self.strong_threshold {
            Negative
        } else {
            // Check raw_gate for weak-directional positions
            match pos.raw_gate {
                TernaryGate::Bullish => Positive,
                TernaryGate::Bearish => Negative,
                TernaryGate::Neutral => Neutral,
            }
        }
    }

    /// Encode a portfolio into a weighted trit representation.
    ///
    /// Returns a vector of `(Ternary, f64, f64)` tuples: `(trit, magnitude, confidence)`.
    /// The magnitude is `|pos.weight|`, and the confidence factor incorporates
    /// the veto severity (`1.0 - veto_severity`).
    ///
    /// This is useful for downstream consensus that needs to weigh sources.
    ///
    /// # Errors
    ///
    /// Returns `TernaryBridgeError::EmptyPortfolio` if the portfolio has no
    /// positions.
    pub fn encode_weighted(
        &self,
        portfolio: &PortfolioVector,
    ) -> TernaryBridgeResult<Vec<(Ternary, f64, f64)>> {
        if portfolio.positions.is_empty() {
            return Err(TernaryBridgeError::EmptyPortfolio);
        }

        Ok(portfolio
            .positions
            .iter()
            .map(|pos| {
                let trit = self.encode_position(pos);
                let magnitude = pos.weight.abs();
                let confidence = 1.0 - pos.veto_severity;
                (trit, magnitude, confidence)
            })
            .collect())
    }

    /// Encode each position's ticker-trit pair for diagnostic output.
    ///
    /// Returns `Vec<(ticker: &str, trit: Ternary, weight: f64, severity: f64)>`.
    /// Useful for logging, debugging, or rendering a trit-based position dashboard.
    pub fn encode_with_labels<'a>(
        &self,
        portfolio: &'a PortfolioVector,
    ) -> TernaryBridgeResult<Vec<(&'a str, Ternary, f64, f64)>> {
        if portfolio.positions.is_empty() {
            return Err(TernaryBridgeError::EmptyPortfolio);
        }

        Ok(portfolio
            .positions
            .iter()
            .map(|pos| {
                let trit = self.encode_position(pos);
                (pos.ticker.as_str(), trit, pos.weight, pos.veto_severity)
            })
            .collect())
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// TritsToWeights: Ternary Graph Output → Position Weights
// ─────────────────────────────────────────────────────────────────────────────

/// Converts `TritVector` or `TernaryMatrix` outputs from the ternary graph
/// ecosystem back into continuous float weights `[-1.0, 1.0]`.
///
/// Two interpolation modes are supported:
///
/// - **Linear** (`Linear`): maps `{-1 → -1.0, 0 → 0.0, +1 → +1.0}` directly.
///   This is a direct pass-through suitable for graphs that output in trit space
///   and where you want the raw decision.
///
/// - **Confidence** (`Confidence { high, low }`): maps `{-1 → -low, 0 → 0.0,
///   +1 → +high}` where `high` and `low` represent confidence levels. Positive
///   trits get positive weight at the `high` level, negative trits get negative
///   weight at the `low` level. This is useful when you want to preserve graph
///   direction but scale it by external conviction.
#[derive(Debug, Default, Clone, Copy, PartialEq)]
pub enum WeightInterpolation {
    /// Direct linear mapping: `-1→-1.0, 0→0.0, +1→+1.0`
    #[default]
    Linear,
    /// Confidence-scaled mapping with explicit high/low bounds.
    Confidence {
        /// Weight assigned to `Positive` trits (e.g. 0.75).
        high: f64,
        /// Weight assigned to `Negative` trits (e.g. -0.60).
        low: f64,
    },
}

/// Converts ternary graph outputs (trit arrays and matrices) back into
/// continuous float position weights.
#[derive(Debug, Clone)]
pub struct TritsToWeights {
    /// Interpolation mode for trit-to-weight conversion.
    pub interpolation: WeightInterpolation,
}

impl Default for TritsToWeights {
    fn default() -> Self {
        Self {
            interpolation: WeightInterpolation::Linear,
        }
    }
}

impl TritsToWeights {
    /// Create a new converter with the given interpolation mode.
    pub fn new(interpolation: WeightInterpolation) -> Self {
        Self { interpolation }
    }

    /// Convert a `TritVector` into a `Vec<f64>` of weights.
    ///
    /// Each trit is mapped independently.
    ///
    /// # Errors
    ///
    /// Returns `TernaryBridgeError::EmptyTritVector` if the vector is empty.
    pub fn trits_to_weights(&self, trits: &TritVector) -> TernaryBridgeResult<Vec<f64>> {
        if trits.is_empty() {
            return Err(TernaryBridgeError::EmptyTritVector);
        }

        let weights: Vec<f64> = trits
            .as_slice()
            .iter()
            .map(|&t| self.trit_to_weight(t))
            .collect();

        Ok(weights)
    }

    /// Convert a single `Ternary` to a weight based on the interpolation mode.
    pub fn trit_to_weight(&self, trit: Ternary) -> f64 {
        match self.interpolation {
            WeightInterpolation::Linear => trit.to_f64(),
            WeightInterpolation::Confidence { high, low } => match trit {
                Positive => high.clamp(0.0, 1.0),
                Neutral => 0.0,
                Negative => low.clamp(-1.0, 0.0),
            },
        }
    }

    /// Convert a `TernaryMatrix` (e.g. output from a ternary graph layer) into
    /// a flat `Vec<f64>` of weights (row-major order).
    ///
    /// # Errors
    ///
    /// Returns `TernaryBridgeError::EmptyTritVector` if the matrix has zero rows
    /// or columns.
    pub fn matrix_to_weights(&self, matrix: &TernaryMatrix) -> TernaryBridgeResult<Vec<f64>> {
        if matrix.rows() == 0 || matrix.cols() == 0 {
            return Err(TernaryBridgeError::EmptyTritVector);
        }

        let weights: Vec<f64> = matrix
            .as_slice()
            .iter()
            .map(|&t| self.trit_to_weight(t))
            .collect();

        Ok(weights)
    }

    /// Convert a `TernaryMatrix` into a vector of per-row weight aggregates.
    ///
    /// Each row is summed (mod 3 via balanced addition) and then mapped to a
    /// weight. This is useful when the matrix represents per-column features
    /// and each row corresponds to a position/ticker.
    ///
    /// # Errors
    ///
    /// Returns `TernaryBridgeError::EmptyTritVector` if the matrix has zero rows
    /// or columns.
    pub fn matrix_row_weights(&self, matrix: &TernaryMatrix) -> TernaryBridgeResult<Vec<f64>> {
        if matrix.rows() == 0 || matrix.cols() == 0 {
            return Err(TernaryBridgeError::EmptyTritVector);
        }

        let weights: Vec<f64> = (0..matrix.rows())
            .map(|r| {
                let row = matrix
                    .row(r)
                    .expect("row index should be valid; checked bounds above");
                // Sum the row trits mod 3 into a single trit
                let mut sum = Neutral;
                for &t in row.as_slice() {
                    sum = sum + t;
                }
                self.trit_to_weight(sum)
            })
            .collect();

        Ok(weights)
    }

    /// Create `FinalPosition` entries from a trit vector and corresponding tickers.
    ///
    /// This is the primary conversion path from ternary graph output → portfolio
    /// positions. Each ticker gets a `FinalPosition` with no veto applied.
    ///
    /// # Errors
    ///
    /// Returns `TernaryBridgeError::TritVectorLengthMismatch` if the number of
    /// trits does not match the number of tickers.
    pub fn to_final_positions(
        &self,
        trits: &TritVector,
        tickers: &[String],
    ) -> TernaryBridgeResult<Vec<FinalPosition>> {
        if trits.len() != tickers.len() {
            return Err(TernaryBridgeError::TritVectorLengthMismatch {
                expected: trits.len(),
                actual: tickers.len(),
            });
        }

        let positions: Vec<FinalPosition> = tickers
            .iter()
            .zip(trits.as_slice().iter())
            .map(|(ticker, &trit)| {
                let weight = self.trit_to_weight(trit);
                let gate = TritMapping::trit_to_gate(trit);
                FinalPosition {
                    ticker: ticker.clone(),
                    weight,
                    raw_gate: gate,
                    veto_applied: vec![],
                    veto_severity: 0.0,
                }
            })
            .collect();

        Ok(positions)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Consensus: Confidence-Weighted Aggregation from Trit Arrays
// ─────────────────────────────────────────────────────────────────────────────

/// A single trit vote with associated confidence.
#[derive(Debug, Clone, Copy)]
pub struct WeightedTritVote {
    /// The trit value.
    pub trit: Ternary,
    /// Confidence weight in `[0.0, 1.0]`.
    pub confidence: f64,
    /// Vote weight (magnitude) in `[0.0, 1.0]`.
    pub weight: f64,
}

impl WeightedTritVote {
    /// Create a new weighted trit vote.
    ///
    /// # Panics
    ///
    /// Panics if `confidence` or `weight` are not in `[0.0, 1.0]`.
    pub fn new(trit: Ternary, confidence: f64, weight: f64) -> Self {
        assert!(
            (0.0..=1.0).contains(&confidence),
            "confidence must be in [0.0, 1.0]"
        );
        assert!(
            (0.0..=1.0).contains(&weight),
            "weight must be in [0.0, 1.0]"
        );
        Self {
            trit,
            confidence,
            weight,
        }
    }
}

/// Configuration for consensus computation.
#[derive(Debug, Clone)]
pub struct ConsensusConfig {
    /// Minimum confidence threshold — votes below this are ignored.
    pub min_confidence: f64,
    /// Weight given to majority agreement (boost factor). Default: 1.5.
    pub majority_boost: f64,
    /// If the weighted consensus score magnitude is below this, output Neutral.
    /// Default: 0.1.
    pub consensus_deadband: f64,
}

impl Default for ConsensusConfig {
    fn default() -> Self {
        Self {
            min_confidence: 0.1,
            majority_boost: 1.5,
            consensus_deadband: 0.1,
        }
    }
}

/// Computes a confidence-weighted consensus from multiple trit arrays.
///
/// This is used when multiple sources (room agents, ternary graph layers, etc.)
/// produce trit arrays for the same set of positions and we need a single
/// combined result.
pub struct ConsensusComputer {
    config: ConsensusConfig,
}

impl Default for ConsensusComputer {
    fn default() -> Self {
        Self::new(ConsensusConfig::default())
    }
}

impl ConsensusComputer {
    /// Create a new consensus computer with the given configuration.
    pub fn new(config: ConsensusConfig) -> Self {
        Self { config }
    }

    /// Compute a consensus `TritVector` from multiple weighted votes.
    ///
    /// Each vote source provides a `TritVector` and associated confidence.
    /// The algorithm:
    ///
    /// 1. Discard votes below `min_confidence`.
    /// 2. For each trit position, sum weighted votes:
    ///    - `Positive` votes contribute `+confidence * weight`
    ///    - `Negative` votes contribute `-confidence * weight`
    ///    - `Neutral` votes contribute `0.0`
    /// 3. If the sum exceeds `consensus_deadband`, the trit is `Positive`.
    /// 4. If the sum is below `-consensus_deadband`, the trit is `Negative`.
    /// 5. Otherwise, the trit is `Neutral`.
    ///
    /// # Errors
    ///
    /// Returns `TernaryBridgeError::EmptyConsensusSet` if `votes` is empty.
    /// Returns `TernaryBridgeError::TritVectorLengthMismatch` if the trit vectors
    /// have different lengths.
    pub fn compute_consensus(&self, votes: &[WeightedTritVote]) -> TernaryBridgeResult<TritVector> {
        if votes.is_empty() {
            return Err(TernaryBridgeError::EmptyConsensusSet);
        }

        // Group votes by the implicit index — each call handles one position.
        // For a full multi-position consensus, use `compute_consensus_multi`.
        let mut score = 0.0_f64;
        let mut total_confidence = 0.0_f64;

        for vote in votes {
            if vote.confidence < self.config.min_confidence {
                continue;
            }

            let contribution = vote.confidence * vote.weight;
            total_confidence += contribution;

            match vote.trit {
                Positive => score += contribution,
                Negative => score -= contribution,
                Neutral => {} // No contribution
            }
        }

        // Apply majority boost: if there's agreement, amplify the signal
        if total_confidence > 0.0 {
            let agreement_ratio = score.abs() / total_confidence;
            if agreement_ratio > 0.5 {
                score *= self.config.majority_boost;
            }
        }

        // Quantize to trit with deadband
        let trit = if score > self.config.consensus_deadband {
            Positive
        } else if score < -self.config.consensus_deadband {
            Negative
        } else {
            Neutral
        };

        Ok(TritVector::new(&[trit]))
    }

    /// Compute a consensus from multiple sources, each providing a `TritVector`
    /// and a confidence score.
    ///
    /// Each source's vector has one trit per position. The consensus is computed
    /// per-position by collecting all source votes for that position and running
    /// the single-position consensus algorithm.
    ///
    /// # Errors
    ///
    /// Returns `TernaryBridgeError::EmptyConsensusSet` if `sources` is empty.
    /// Returns `TernaryBridgeError::TritVectorLengthMismatch` if the trit vectors
    /// have different lengths.
    pub fn compute_consensus_multi(
        &self,
        sources: &[(&TritVector, f64)],
    ) -> TernaryBridgeResult<TritVector> {
        if sources.is_empty() {
            return Err(TernaryBridgeError::EmptyConsensusSet);
        }

        let n_positions = sources[0].0.len();

        // Validate all vectors have the same length
        for (vec, _conf) in sources {
            if vec.len() != n_positions {
                return Err(TernaryBridgeError::TritVectorLengthMismatch {
                    expected: n_positions,
                    actual: vec.len(),
                });
            }
        }

        // For each position, collect votes from all sources
        let mut result_trits = Vec::with_capacity(n_positions);

        for pos_idx in 0..n_positions {
            let mut score = 0.0_f64;
            let mut total_confidence = 0.0_f64;

            for (vec, confidence) in sources {
                if *confidence < self.config.min_confidence {
                    continue;
                }

                let trit = vec.get(pos_idx).unwrap_or(Neutral);
                let contribution = confidence * *confidence; // square it for confidence dampening
                total_confidence += contribution;

                match trit {
                    Positive => score += contribution,
                    Negative => score -= contribution,
                    Neutral => {}
                }
            }

            // Apply majority boost
            if total_confidence > 0.0 {
                let agreement_ratio = score.abs() / total_confidence;
                if agreement_ratio > 0.5 {
                    score *= self.config.majority_boost;
                }
            }

            let trit = if score > self.config.consensus_deadband {
                Positive
            } else if score < -self.config.consensus_deadband {
                Negative
            } else {
                Neutral
            };

            result_trits.push(trit);
        }

        Ok(TritVector::new(&result_trits))
    }

    /// Compute a consensus from the raw `RoomProposal` slice, using each
    /// proposal's gate and conviction as the source of trit and confidence.
    ///
    /// This is the primary integration point with the Hybrid Manifold: room
    /// agents submit proposals, and this function produces a consensus trit
    /// vector from them.
    ///
    /// # Errors
    ///
    /// Returns `TernaryBridgeError::EmptyConsensusSet` if `proposals` is empty.
    pub fn compute_consensus_from_proposals(
        &self,
        proposals: &[crate::types::RoomProposal],
    ) -> TernaryBridgeResult<TritVector> {
        if proposals.is_empty() {
            return Err(TernaryBridgeError::EmptyConsensusSet);
        }

        let votes: Vec<WeightedTritVote> = proposals
            .iter()
            .map(|p| {
                let trit = TritMapping::gate_to_trit(&p.gate);
                // Use matrix_agreement as a confidence multiplier:
                // a proposal that agrees with the matrix consensus has higher weight.
                let adjusted_confidence = p.confidence * p.matrix_agreement;
                WeightedTritVote::new(trit, adjusted_confidence, p.conviction)
            })
            .collect();

        self.compute_consensus(&votes)
    }

    /// Compute the consensus score as a continuous float (not quantized to trit).
    ///
    /// Returns a value in `[-1.0, 1.0]` representing the raw weighted agreement.
    /// Useful for debugging or for downstream systems that want the continuous
    /// score before trit quantization.
    pub fn consensus_score(&self, votes: &[WeightedTritVote]) -> TernaryBridgeResult<f64> {
        if votes.is_empty() {
            return Err(TernaryBridgeError::EmptyConsensusSet);
        }

        let mut score = 0.0_f64;
        let mut total_confidence = 0.0_f64;

        for vote in votes {
            if vote.confidence < self.config.min_confidence {
                continue;
            }

            let contribution = vote.confidence * vote.weight;
            total_confidence += contribution;

            match vote.trit {
                Positive => score += contribution,
                Negative => score -= contribution,
                Neutral => {}
            }
        }

        // Normalize to [-1.0, 1.0]
        if total_confidence > 0.0 {
            Ok((score / total_confidence).clamp(-1.0, 1.0))
        } else {
            Ok(0.0)
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Convenience Functions
// ─────────────────────────────────────────────────────────────────────────────

/// Convert a `TernaryGate` to the `Ternary` type: `Bullish→Positive, Neutral→Neutral, Bearish→Negative`.
pub fn gate_to_ternary(gate: &TernaryGate) -> Ternary {
    TritMapping::gate_to_trit(gate)
}

/// Convert a `Ternary` back to a `TernaryGate`.
pub fn ternary_to_gate(trit: Ternary) -> TernaryGate {
    TritMapping::trit_to_gate(trit)
}

/// Convert a `TritVector` to a `Vec<TernaryGate>`.
pub fn trit_vector_to_gates(trits: &TritVector) -> Vec<TernaryGate> {
    trits
        .as_slice()
        .iter()
        .map(|&t| ternary_to_gate(t))
        .collect()
}

/// Convert a slice of `TernaryGate` to a `TritVector`.
pub fn gates_to_trit_vector(gates: &[TernaryGate]) -> TritVector {
    let trits: Vec<Ternary> = gates.iter().map(gate_to_ternary).collect();
    TritVector::new(&trits)
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::FinalPosition;
    use std::collections::HashMap;

    // ── TritMapping Tests ────────────────────────────────────────────────

    #[test]
    fn test_trit_mapping_negative() {
        let mut mapper = TritMapping::default();
        assert_eq!(mapper.map(0.0).unwrap(), Negative);
        assert_eq!(mapper.map(0.2).unwrap(), Negative);
        assert_eq!(mapper.map(0.29).unwrap(), Negative);
    }

    #[test]
    fn test_trit_mapping_neutral() {
        let mut mapper = TritMapping::default();
        assert_eq!(mapper.map(0.3).unwrap(), Neutral);
        assert_eq!(mapper.map(0.5).unwrap(), Neutral);
        assert_eq!(mapper.map(0.7).unwrap(), Neutral);
    }

    #[test]
    fn test_trit_mapping_positive() {
        let mut mapper = TritMapping::default();
        assert_eq!(mapper.map(0.71).unwrap(), Positive);
        assert_eq!(mapper.map(0.9).unwrap(), Positive);
        assert_eq!(mapper.map(1.0).unwrap(), Positive);
    }

    #[test]
    fn test_trit_mapping_hysteresis_flip_guard() {
        let mut mapper = TritMapping::default();
        // Start negative
        assert_eq!(mapper.map(0.2).unwrap(), Negative);
        // Jump to positive — this crosses the full band (0.3-0.7 = 0.4 delta),
        // and 0.2→0.8 has delta=0.6 > 0.4, so the guard allows it.
        assert_eq!(mapper.map(0.8).unwrap(), Positive);
    }

    #[test]
    fn test_trit_mapping_hysteresis_small_delta_blocks() {
        let mut mapper = TritMapping::default();
        // Go to negative
        assert_eq!(mapper.map(0.2).unwrap(), Negative);
        // Small positive move (0.2→0.5, delta=0.3) — not enough to cross the
        // full band (0.4), so we stay Negative.
        assert_eq!(mapper.map(0.5).unwrap(), Neutral); // Enters deadband → Neutral
    }

    #[test]
    fn test_trit_mapping_out_of_range() {
        let mut mapper = TritMapping::default();
        assert!(mapper.map(-0.1).is_err());
        assert!(mapper.map(1.1).is_err());
    }

    #[test]
    fn test_trit_mapping_reset() {
        let mut mapper = TritMapping::default();
        mapper.map(0.9).unwrap();
        assert_eq!(mapper.current_trit(), Positive);
        mapper.reset();
        assert_eq!(mapper.current_trit(), Neutral);
    }

    #[test]
    fn test_gate_to_trit_conversion() {
        assert_eq!(TritMapping::gate_to_trit(&TernaryGate::Bullish), Positive);
        assert_eq!(TritMapping::gate_to_trit(&TernaryGate::Neutral), Neutral);
        assert_eq!(TritMapping::gate_to_trit(&TernaryGate::Bearish), Negative);
    }

    #[test]
    fn test_trit_to_gate_conversion() {
        assert_eq!(TritMapping::trit_to_gate(Positive), TernaryGate::Bullish);
        assert_eq!(TritMapping::trit_to_gate(Neutral), TernaryGate::Neutral);
        assert_eq!(TritMapping::trit_to_gate(Negative), TernaryGate::Bearish);
    }

    #[test]
    fn test_custom_hysteresis() {
        let config = HysteresisConfig::new(0.2, 0.8);
        let mut mapper = TritMapping::new(config);
        assert_eq!(mapper.map(0.25).unwrap(), Neutral); // inside [0.2, 0.8]
        assert_eq!(mapper.map(0.1).unwrap(), Negative); // below lower
        assert_eq!(mapper.map(0.9).unwrap(), Positive); // above upper
    }

    #[test]
    #[should_panic(expected = "lower bound must be less than upper bound")]
    fn test_invalid_hysteresis_equal() {
        HysteresisConfig::new(0.5, 0.5);
    }

    #[test]
    #[should_panic(expected = "lower bound must be in [0.0, 1.0]")]
    fn test_invalid_hysteresis_negative_lower() {
        HysteresisConfig::new(-0.1, 0.5);
    }

    // ── PortfolioToTrits Tests ──────────────────────────────────────────

    fn sample_portfolio() -> PortfolioVector {
        PortfolioVector {
            positions: vec![
                FinalPosition {
                    ticker: "AAPL".into(),
                    weight: 0.85,
                    raw_gate: TernaryGate::Bullish,
                    veto_applied: vec![],
                    veto_severity: 0.0,
                },
                FinalPosition {
                    ticker: "TSLA".into(),
                    weight: -0.72,
                    raw_gate: TernaryGate::Bearish,
                    veto_applied: vec![],
                    veto_severity: 0.0,
                },
                FinalPosition {
                    ticker: "MSFT".into(),
                    weight: 0.1,
                    raw_gate: TernaryGate::Bullish,
                    veto_applied: vec![],
                    veto_severity: 0.0,
                },
            ],
            gross_exposure: 1.67,
            net_exposure: 0.23,
            sector_concentrations: HashMap::new(),
            portfolio_var: 0.05,
            timestamp: 1000,
        }
    }

    #[test]
    fn test_portfolio_encode_basic() {
        let encoder = PortfolioToTrits::default();
        let portfolio = sample_portfolio();
        let trits = encoder.encode(&portfolio).unwrap();

        assert_eq!(trits.len(), 3);
        assert_eq!(trits.get(0), Some(Positive)); // AAPL: 0.85 ≥ 0.5
        assert_eq!(trits.get(1), Some(Negative)); // TSLA: -0.72 ≤ -0.5
        assert_eq!(trits.get(2), Some(Positive)); // MSFT: 0.1 < 0.5, but raw_gate=Bullish
    }

    #[test]
    fn test_portfolio_encode_veto_kill() {
        let encoder = PortfolioToTrits::default();
        let mut portfolio = sample_portfolio();
        // Apply veto to AAPL
        portfolio.positions[0].veto_severity = 0.99;

        let trits = encoder.encode(&portfolio).unwrap();
        assert_eq!(trits.get(0), Some(Neutral)); // AAPL killed by veto
        assert_eq!(trits.get(1), Some(Negative)); // TSLA unchanged
    }

    #[test]
    fn test_portfolio_encode_empty() {
        let encoder = PortfolioToTrits::default();
        let portfolio = PortfolioVector {
            positions: vec![],
            gross_exposure: 0.0,
            net_exposure: 0.0,
            sector_concentrations: HashMap::new(),
            portfolio_var: 0.0,
            timestamp: 0,
        };

        assert_eq!(
            encoder.encode(&portfolio).unwrap_err(),
            TernaryBridgeError::EmptyPortfolio
        );
    }

    #[test]
    fn test_portfolio_encode_weighted() {
        let encoder = PortfolioToTrits::default();
        let portfolio = sample_portfolio();
        let weighted = encoder.encode_weighted(&portfolio).unwrap();

        assert_eq!(weighted.len(), 3);
        // AAPL: (Positive, 0.85, 1.0)
        assert_eq!(weighted[0], (Positive, 0.85, 1.0));
        // TSLA: (Negative, 0.72, 1.0)
        assert_eq!(weighted[1], (Negative, 0.72, 1.0));
    }

    #[test]
    fn test_portfolio_encode_with_labels() {
        let encoder = PortfolioToTrits::default();
        let portfolio = sample_portfolio();
        let labeled = encoder.encode_with_labels(&portfolio).unwrap();

        assert_eq!(labeled.len(), 3);
        assert_eq!(labeled[0].0, "AAPL");
        assert_eq!(labeled[0].1, Positive);
        assert_eq!(labeled[1].0, "TSLA");
        assert_eq!(labeled[1].1, Negative);
    }

    #[test]
    fn test_custom_thresholds() {
        let encoder = PortfolioToTrits::new(0.3, 0.98);
        let portfolio = sample_portfolio();
        let trits = encoder.encode(&portfolio).unwrap();

        // With strong_threshold = 0.3:
        // AAPL: 0.85 ≥ 0.3 → Positive
        // TSLA: -0.72 ≤ -0.3 → Negative
        // MSFT: 0.1 < 0.3, but raw_gate=Bullish → Positive
        assert_eq!(trits.get(0), Some(Positive));
        assert_eq!(trits.get(1), Some(Negative));
        assert_eq!(trits.get(2), Some(Positive));
    }

    // ── TritsToWeights Tests ────────────────────────────────────────────

    #[test]
    fn test_linear_interpolation() {
        let converter = TritsToWeights::default();
        assert!((converter.trit_to_weight(Positive) - 1.0).abs() < 1e-10);
        assert!((converter.trit_to_weight(Neutral) - 0.0).abs() < 1e-10);
        assert!((converter.trit_to_weight(Negative) - (-1.0)).abs() < 1e-10);
    }

    #[test]
    fn test_confidence_interpolation() {
        let converter = TritsToWeights::new(WeightInterpolation::Confidence {
            high: 0.75,
            low: -0.6,
        });
        assert!((converter.trit_to_weight(Positive) - 0.75).abs() < 1e-10);
        assert!((converter.trit_to_weight(Neutral) - 0.0).abs() < 1e-10);
        assert!((converter.trit_to_weight(Negative) - (-0.6)).abs() < 1e-10);
    }

    #[test]
    fn test_trits_to_weights_vector() {
        let converter = TritsToWeights::default();
        let trits = TritVector::new(&[Positive, Neutral, Negative, Positive]);
        let weights = converter.trits_to_weights(&trits).unwrap();

        assert_eq!(weights.len(), 4);
        assert!((weights[0] - 1.0).abs() < 1e-10);
        assert!((weights[1] - 0.0).abs() < 1e-10);
        assert!((weights[2] - (-1.0)).abs() < 1e-10);
        assert!((weights[3] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_trits_to_weights_empty() {
        let converter = TritsToWeights::default();
        let trits = TritVector::zeros(0);
        assert_eq!(
            converter.trits_to_weights(&trits).unwrap_err(),
            TernaryBridgeError::EmptyTritVector
        );
    }

    #[test]
    fn test_matrix_to_weights() {
        let converter = TritsToWeights::default();
        let mut matrix = TernaryMatrix::new(2, 3);
        matrix.set(0, 0, Positive);
        matrix.set(0, 1, Neutral);
        matrix.set(0, 2, Negative);
        matrix.set(1, 0, Negative);
        matrix.set(1, 1, Positive);
        matrix.set(1, 2, Neutral);

        let weights = converter.matrix_to_weights(&matrix).unwrap();
        assert_eq!(weights.len(), 6);
        assert!((weights[0] - 1.0).abs() < 1e-10);
        assert!((weights[1] - 0.0).abs() < 1e-10);
        assert!((weights[2] - (-1.0)).abs() < 1e-10);
        assert!((weights[3] - (-1.0)).abs() < 1e-10);
    }

    #[test]
    fn test_matrix_row_weights() {
        let converter = TritsToWeights::default();
        let mut matrix = TernaryMatrix::new(2, 2);
        matrix.set(0, 0, Positive);
        matrix.set(0, 1, Positive); // row sum: +1 + +1 = -1 (mod 3) → Negative → -1.0
        matrix.set(1, 0, Positive);
        matrix.set(1, 1, Negative); // row sum: +1 + -1 = 0 (mod 3) → Neutral → 0.0

        let weights = converter.matrix_row_weights(&matrix).unwrap();
        assert_eq!(weights.len(), 2);
        assert!((weights[0] - (-1.0)).abs() < 1e-10); // mod 3: 1+1=2 ≡ -1
        assert!((weights[1] - 0.0).abs() < 1e-10); // 1 + (-1) = 0
    }

    #[test]
    fn test_to_final_positions() {
        let converter = TritsToWeights::default();
        let trits = TritVector::new(&[Positive, Neutral, Negative]);
        let tickers = vec!["AAPL".into(), "MSFT".into(), "TSLA".into()];

        let positions = converter.to_final_positions(&trits, &tickers).unwrap();
        assert_eq!(positions.len(), 3);
        assert_eq!(positions[0].ticker, "AAPL");
        assert!((positions[0].weight - 1.0).abs() < 1e-10);
        assert_eq!(positions[0].raw_gate, TernaryGate::Bullish);
        assert_eq!(positions[1].ticker, "MSFT");
        assert!((positions[1].weight - 0.0).abs() < 1e-10);
        assert_eq!(positions[1].raw_gate, TernaryGate::Neutral);
        assert_eq!(positions[2].ticker, "TSLA");
        assert!((positions[2].weight - (-1.0)).abs() < 1e-10);
        assert_eq!(positions[2].raw_gate, TernaryGate::Bearish);
    }

    #[test]
    fn test_to_final_positions_mismatch() {
        let converter = TritsToWeights::default();
        let trits = TritVector::new(&[Positive, Neutral]);
        let tickers = vec!["AAPL".into()];

        assert_eq!(
            converter.to_final_positions(&trits, &tickers).unwrap_err(),
            TernaryBridgeError::TritVectorLengthMismatch {
                expected: 2,
                actual: 1,
            }
        );
    }

    // ── Consensus Tests ─────────────────────────────────────────────────

    #[test]
    fn test_consensus_unanimous_positive() {
        let computer = ConsensusComputer::default();
        let votes = vec![
            WeightedTritVote::new(Positive, 0.9, 0.8),
            WeightedTritVote::new(Positive, 0.85, 0.7),
            WeightedTritVote::new(Positive, 0.95, 0.9),
        ];

        let result = computer.compute_consensus(&votes).unwrap();
        assert_eq!(result.get(0), Some(Positive));
    }

    #[test]
    fn test_consensus_unanimous_negative() {
        let computer = ConsensusComputer::default();
        let votes = vec![
            WeightedTritVote::new(Negative, 0.9, 0.8),
            WeightedTritVote::new(Negative, 0.85, 0.7),
        ];

        let result = computer.compute_consensus(&votes).unwrap();
        assert_eq!(result.get(0), Some(Negative));
    }

    #[test]
    fn test_consensus_split_returns_neutral() {
        let computer = ConsensusComputer::default();
        let votes = vec![
            WeightedTritVote::new(Positive, 0.9, 0.3),
            WeightedTritVote::new(Negative, 0.9, 0.3),
        ];

        let result = computer.compute_consensus(&votes).unwrap();
        // 0.9*0.3 - 0.9*0.3 = 0.0, which is below deadband
        assert_eq!(result.get(0), Some(Neutral));
    }

    #[test]
    fn test_consensus_empty_set() {
        let computer = ConsensusComputer::default();
        let votes = vec![];
        assert_eq!(
            computer.compute_consensus(&votes).unwrap_err(),
            TernaryBridgeError::EmptyConsensusSet
        );
    }

    #[test]
    fn test_consensus_low_confidence_filtered() {
        let computer = ConsensusComputer::default();
        let votes = vec![
            WeightedTritVote::new(Positive, 0.05, 0.9), // below min_confidence (0.1)
            WeightedTritVote::new(Positive, 0.95, 0.8),
        ];

        let result = computer.compute_consensus(&votes).unwrap();
        assert_eq!(result.get(0), Some(Positive));
    }

    #[test]
    fn test_consensus_majority_boost() {
        let computer = ConsensusComputer::default();
        let votes = vec![
            WeightedTritVote::new(Positive, 0.6, 0.5),
            WeightedTritVote::new(Positive, 0.6, 0.5),
            WeightedTritVote::new(Negative, 0.6, 0.5),
        ];

        let result = computer.compute_consensus(&votes).unwrap();
        // score = 0.6*0.5 + 0.6*0.5 - 0.6*0.5 = 0.3
        // total_confidence = 0.9
        // agreement_ratio = 0.3/0.9 = 0.333... < 0.5, no boost
        // 0.3 > deadband 0.1 → Positive
        assert_eq!(result.get(0), Some(Positive));
    }

    #[test]
    fn test_consensus_score_continuous() {
        let computer = ConsensusComputer::default();
        let votes = vec![
            WeightedTritVote::new(Positive, 0.9, 1.0),
            WeightedTritVote::new(Negative, 0.3, 0.5), // low confidence
        ];

        let score = computer.consensus_score(&votes).unwrap();
        // score = (0.9*1.0 - 0.3*0.5) / (0.9*1.0 + 0.3*0.5)
        // = (0.9 - 0.15) / (0.9 + 0.15) = 0.75 / 1.05 ≈ 0.714
        assert!((score - 0.714285).abs() < 1e-4);
    }

    #[test]
    fn test_consensus_multi_vector() {
        let computer = ConsensusComputer::default();

        let source1 = TritVector::new(&[Positive, Negative, Neutral]);
        let source2 = TritVector::new(&[Positive, Neutral, Positive]);
        let source3 = TritVector::new(&[Neutral, Negative, Neutral]);

        let sources = vec![(&source1, 0.9), (&source2, 0.8), (&source3, 0.6)];

        let result = computer.compute_consensus_multi(&sources).unwrap();
        assert_eq!(result.len(), 3);

        // Position 0: (+1, +1, 0) → mostly positive → Positive
        // Position 1: (-1, 0, -1) → mostly negative → Negative
        // Position 2: (0, +1, 0) → weak positive → could be Positive or Neutral
        assert_eq!(result.get(0), Some(Positive));
        assert_eq!(result.get(1), Some(Negative));
    }

    #[test]
    fn test_consensus_from_proposals() {
        let computer = ConsensusComputer::default();

        let proposals = vec![
            crate::types::RoomProposal {
                ticker: "AAPL".into(),
                gate: TernaryGate::Bullish,
                conviction: 0.85,
                confidence: 0.9,
                narrative_sig: "a".into(),
                matrix_agreement: 0.95,
                veto_override: false,
                timestamp: 1,
            },
            crate::types::RoomProposal {
                ticker: "AAPL".into(),
                gate: TernaryGate::Bullish,
                conviction: 0.7,
                confidence: 0.8,
                narrative_sig: "b".into(),
                matrix_agreement: 0.85,
                veto_override: false,
                timestamp: 1,
            },
            crate::types::RoomProposal {
                ticker: "AAPL".into(),
                gate: TernaryGate::Neutral,
                conviction: 0.5,
                confidence: 0.6,
                narrative_sig: "c".into(),
                matrix_agreement: 0.7,
                veto_override: false,
                timestamp: 1,
            },
        ];

        let result = computer
            .compute_consensus_from_proposals(&proposals)
            .unwrap();
        assert_eq!(result.get(0), Some(Positive));
    }

    // ── Convenience Function Tests ──────────────────────────────────────

    #[test]
    fn test_gate_to_ternary_fn() {
        assert_eq!(gate_to_ternary(&TernaryGate::Bullish), Positive);
        assert_eq!(gate_to_ternary(&TernaryGate::Neutral), Neutral);
        assert_eq!(gate_to_ternary(&TernaryGate::Bearish), Negative);
    }

    #[test]
    fn test_ternary_to_gate_fn() {
        assert_eq!(ternary_to_gate(Positive), TernaryGate::Bullish);
        assert_eq!(ternary_to_gate(Neutral), TernaryGate::Neutral);
        assert_eq!(ternary_to_gate(Negative), TernaryGate::Bearish);
    }

    #[test]
    fn test_trit_vector_to_gates() {
        let trits = TritVector::new(&[Positive, Neutral, Negative]);
        let gates = trit_vector_to_gates(&trits);
        assert_eq!(
            gates,
            vec![
                TernaryGate::Bullish,
                TernaryGate::Neutral,
                TernaryGate::Bearish
            ]
        );
    }

    #[test]
    fn test_gates_to_trit_vector() {
        let gates = vec![
            TernaryGate::Bullish,
            TernaryGate::Neutral,
            TernaryGate::Bearish,
        ];
        let trits = gates_to_trit_vector(&gates);
        assert_eq!(trits.as_slice(), &[Positive, Neutral, Negative]);
    }

    // ── Error Display Tests ─────────────────────────────────────────────

    #[test]
    fn test_error_display() {
        let err = TernaryBridgeError::ConvictionOutOfRange(1.5);
        assert_eq!(
            err.to_string(),
            "conviction 1.5 is outside the valid range [0.0, 1.0]"
        );

        let err = TernaryBridgeError::WeightOutOfRange(-2.0);
        assert_eq!(
            err.to_string(),
            "weight -2 is outside the valid range [-1.0, 1.0]"
        );

        let err = TernaryBridgeError::TritVectorLengthMismatch {
            expected: 5,
            actual: 3,
        };
        assert_eq!(
            err.to_string(),
            "trit vector length mismatch: expected 5, got 3"
        );

        let err = TernaryBridgeError::EmptyPortfolio;
        assert_eq!(
            err.to_string(),
            "portfolio vector is empty — no positions to encode"
        );
    }
}
