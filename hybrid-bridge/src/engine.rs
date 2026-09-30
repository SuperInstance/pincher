//! Hybrid Engine — orchestrates the full Matrix → Rooms → Veto → Execution cycle.
//!
//! This module defines the `HybridEngine` trait and its concrete implementation
//! `HybridEngineImpl`, which coordinates the three layers:
//!
//! 1. **Matrix Phase** — Run fast/medium/full cycle on the feature tensor
//! 2. **Rooms Phase** — Fan out the snapshot to all room agents, collect proposals
//! 3. **Veto Phase** — Resolve proposals into a final portfolio vector
//! 4. **Execution Phase** — Broadcast the portfolio and emit system events

use crate::bridge::HybridBridge;
use crate::types::{
    FeatureSuggestion, FinalPosition, MatrixMetadata, MatrixSnapshot, PartialSnapshot,
    PortfolioVector, RoomProposal, SaepAction, SaepConstraint, TernaryGate, TopologicalSignature,
};
use async_trait::async_trait;
use ndarray::Array1;
use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use tokio::sync::{RwLock, Semaphore};
use tracing::{debug, info, instrument, trace, warn};

// ─────────────────────────────────────────────────────────────────────
// Phase Timing Configuration (ARM64-optimized)
// ─────────────────────────────────────────────────────────────────────

/// Cycle counts for each matrix path.
pub const FAST_CYCLE_INTERVAL: u64 = 1; // every tick
pub const MEDIUM_CYCLE_INTERVAL: u64 = 5; // every 5 ticks
pub const FULL_CYCLE_INTERVAL: u64 = 20; // every 20 ticks

/// Maximum number of room agents that can analyze concurrently.
pub const MAX_CONCURRENT_ROOMS: usize = 128;

// ─────────────────────────────────────────────────────────────────────
// Trait Definitions (from HYBRID-API-SPEC sections 1-3)
// ─────────────────────────────────────────────────────────────────────

/// Matrix Engine Interface
///
/// Owns the feature tensor `X[n, m, h]` and exposes computation results to Room Agents.
#[async_trait]
pub trait MatrixEngine: Send + Sync {
    /// Ingest a new data point for ticker `t` at time `tick`.
    /// Updates the feature tensor X[t, :, :] with new row.
    async fn ingest(&self, ticker: &str, features: &[f64], tick: u64);

    /// Run the fast-path update (tensor ingest + simple statistics).
    /// Guaranteed < 5ms on ARM64 at n=5000.
    async fn fast_cycle(&self, tick: u64) -> MatrixMetadata;

    /// Run the medium-path update (correlation matrix + streaming PCA).
    /// ~200-500ms, run every 5 ticks.
    async fn medium_cycle(&self, tick: u64) -> PartialSnapshot;

    /// Run the full slow-path update (eigendecomposition + TDA).
    /// ~2-5 seconds, run every 20-60 ticks or on regime change.
    async fn full_cycle(&self, tick: u64) -> MatrixSnapshot;

    /// Provide a slice of the feature tensor for a single ticker.
    async fn get_slice(&self, ticker: &str) -> Option<ndarray::Array2<f32>>;

    /// Provide the cross-sectional view: all stocks, single feature, single time.
    async fn get_cross_section(&self, feature: &str, time_idx: usize) -> Option<Array1<f32>>;

    /// Register a new ticker. Appends a row to the tensor (O(1), ~10ms).
    async fn add_ticker(&self, ticker: &str, initial_features: Option<&[f64]>);

    /// Remove a ticker. Does not delete — marks inactive, reuses row.
    async fn remove_ticker(&self, ticker: &str);
}

/// Room Agent Interface
///
/// Consumes Matrix Slices, adds interpretation, and emits proposals.
#[async_trait]
pub trait RoomAgent: Send + Sync {
    /// The main analysis tick. The room reads its slice from the matrix
    /// and produces a proposal.
    async fn analyze(
        &self,
        matrix_snapshot: &MatrixSnapshot,
        feature_slice: Option<ndarray::Array2<f32>>,
        cross_section: Option<Array1<f32>>,
    ) -> RoomProposal;

    /// Receive a symmetry alert from the matrix.
    async fn on_symmetry_alert(
        &self,
        peer_ticker: &str,
        symmetry_score: f64,
        topology: &TopologicalSignature,
    );

    /// Suggest a new feature to the matrix engine.
    async fn suggest_feature(&self, suggestion: FeatureSuggestion);

    /// Set a stock-specific regime label.
    async fn set_regime(&mut self, label: String);

    /// Update the room's narrative (qualitative context).
    async fn update_narrative(&mut self, narrative: String);
}

/// Veto Engine Interface
///
/// Aggregates room proposals into a final position vector subject to SAEP constraints.
#[async_trait]
pub trait VetoEngine: Send + Sync {
    /// Register a SAEP constraint.
    async fn register_constraint(&mut self, constraint: SaepConstraint);

    /// Process all room proposals and produce the final portfolio.
    async fn resolve(
        &self,
        proposals: &[RoomProposal],
        current_portfolio: Option<&PortfolioVector>,
    ) -> PortfolioVector;

    /// Get current portfolio state.
    async fn get_portfolio(&self) -> PortfolioVector;

    /// Emergency freeze — halts all actions.
    async fn freeze(&mut self, reason: &str);

    /// Unfreeze.
    async fn unfreeze(&mut self, reason: &str);
}

// ─────────────────────────────────────────────────────────────────────
// HybridEngine Trait
// ─────────────────────────────────────────────────────────────────────

/// The top-level orchestrator for the Hybrid Manifold.
#[async_trait]
pub trait HybridEngine: Send + Sync {
    /// Run one full hybrid cycle: Matrix → Rooms → Veto → Execution.
    async fn hybrid_cycle(&self, tick: u64);

    /// Start the main event loop.
    async fn run(&self);

    /// Graceful shutdown.
    async fn shutdown(&self);
}

// ─────────────────────────────────────────────────────────────────────
// HybridEngineImpl — Concrete Implementation
// ─────────────────────────────────────────────────────────────────────

/// Runtime state tracked across cycles.
#[derive(Debug, Clone, Default)]
struct CycleState {
    proposals_received: usize,
    features_received: usize,
    matrix_time_ms: f64,
    rooms_time_ms: f64,
    veto_time_ms: f64,
    total_cycle_ms: f64,
}

/// Configuration for the hybrid engine.
#[derive(Debug, Clone)]
pub struct HybridConfig {
    /// Interval (in ticks) between medium-path cycles.
    pub medium_cycle_interval: u64,
    /// Interval (in ticks) between full-path cycles.
    pub full_cycle_interval: u64,
    /// Maximum concurrent room agents for analysis.
    pub max_concurrent_rooms: usize,
    /// Whether to run feature collection from rooms.
    pub collect_features: bool,
    /// Whether to emit detailed tracing.
    pub enable_tracing: bool,
}

impl Default for HybridConfig {
    fn default() -> Self {
        Self {
            medium_cycle_interval: MEDIUM_CYCLE_INTERVAL,
            full_cycle_interval: FULL_CYCLE_INTERVAL,
            max_concurrent_rooms: MAX_CONCURRENT_ROOMS,
            collect_features: true,
            enable_tracing: true,
        }
    }
}

/// Concrete implementation of the HybridEngine.
///
/// Owns references to the Matrix Engine, Room Agents, and Veto Engine,
/// plus the HybridBridge for communication.
pub struct HybridEngineImpl<M: MatrixEngine, V: VetoEngine> {
    /// The matrix engine (topological feature tensor).
    matrix: Arc<M>,
    /// The bridge — communication backbone.
    bridge: Arc<HybridBridge>,
    /// The veto engine (SAEP constraint resolution).
    veto: Arc<RwLock<V>>,
    /// Room agents — stored in an Arc so spawned tasks can share them.
    rooms: Arc<RwLock<Vec<Box<dyn RoomAgent>>>>,
    /// Engine configuration.
    config: HybridConfig,
    /// Shutdown signal.
    shutdown: Arc<AtomicBool>,
    /// Concurrency limiter for room analysis.
    room_semaphore: Arc<Semaphore>,
    /// Last portfolio (for delta tracking).
    last_portfolio: Arc<RwLock<Option<PortfolioVector>>>,
    /// Current tick counter.
    current_tick: Arc<RwLock<u64>>,
}

impl<M: MatrixEngine + 'static, V: VetoEngine + 'static> HybridEngineImpl<M, V> {
    /// Create a new hybrid engine.
    pub fn new(
        matrix: M,
        bridge: Arc<HybridBridge>,
        veto: V,
        rooms: Vec<Box<dyn RoomAgent>>,
        config: HybridConfig,
    ) -> Self {
        let max_rooms = config.max_concurrent_rooms.max(rooms.len());

        HybridEngineImpl {
            matrix: Arc::new(matrix),
            bridge,
            veto: Arc::new(RwLock::new(veto)),
            rooms: Arc::new(RwLock::new(rooms)),
            config,
            shutdown: Arc::new(AtomicBool::new(false)),
            room_semaphore: Arc::new(Semaphore::new(max_rooms)),
            last_portfolio: Arc::new(RwLock::new(None)),
            current_tick: Arc::new(RwLock::new(0)),
        }
    }

    // ─────────────────────────────────────────────────────────────────
    // Phase 1: Matrix Cycle
    // ─────────────────────────────────────────────────────────────────

    /// Run the appropriate matrix cycle based on tick interval.
    #[instrument(skip(self))]
    async fn phase_matrix(&self, tick: u64) -> MatrixSnapshot {
        let timer = std::time::Instant::now();

        // Always run fast cycle.
        let _metadata = self.matrix.fast_cycle(tick).await;

        // Medium cycle every N ticks.
        if tick.is_multiple_of(self.config.medium_cycle_interval) {
            let _partial = self.matrix.medium_cycle(tick).await;
        }

        // Full cycle every N ticks.
        let snapshot = if tick.is_multiple_of(self.config.full_cycle_interval) {
            info!("Running full matrix cycle at tick {}", tick);
            self.matrix.full_cycle(tick).await
        } else {
            MatrixSnapshot {
                tick,
                n_stocks: 0,
                eigenvalues: vec![],
                eigenvectors: ndarray::Array2::from_shape_vec((0, 0), vec![]).unwrap(),
                topologies: vec![],
                universe_betti: [0, 0, 0],
                regime: "unknown".into(),
                condition_number: 0.0,
            }
        };

        let elapsed_ms = timer.elapsed().as_secs_f64() * 1000.0;
        trace!(elapsed_ms, "Matrix phase complete");
        snapshot
    }

    // ─────────────────────────────────────────────────────────────────
    // Phase 2: Room Analysis
    // ─────────────────────────────────────────────────────────────────

    /// Fan out the snapshot to all room agents and collect proposals.
    /// Uses sequential analysis (not tokio::spawn) to avoid lifetime issues.
    #[instrument(skip(self, snapshot))]
    async fn phase_rooms(&self, snapshot: &MatrixSnapshot) -> Vec<RoomProposal> {
        let timer = std::time::Instant::now();
        let rooms = self.rooms.read().await;
        let room_count = rooms.len();

        if room_count == 0 {
            debug!("No room agents registered, skipping room phase");
            return vec![];
        }

        debug!("Fanning out snapshot to {} room agents", room_count);

        // Broadcast the snapshot to all rooms via the bridge.
        self.bridge.broadcast_snapshot(snapshot.clone());

        // Analyze rooms sequentially, respecting the semaphore concurrency limit.
        // For production, this would use a task pool; for the bridge layer,
        // sequential analysis is correct (rooms are pure computation, not I/O bound).
        let mut proposals = Vec::with_capacity(room_count);
        for room in rooms.iter() {
            let _permit = self.room_semaphore.clone().acquire_owned().await;
            match _permit {
                Ok(_lock) => {
                    let proposal = room.analyze(snapshot, None, None).await;
                    proposals.push(proposal);
                }
                Err(_) => {
                    warn!("Failed to acquire semaphore permit for room");
                }
            }
        }

        let elapsed_ms = timer.elapsed().as_secs_f64() * 1000.0;
        debug!(
            elapsed_ms,
            proposals = proposals.len(),
            "Room phase complete"
        );
        proposals
    }

    // ─────────────────────────────────────────────────────────────────
    // Phase 3: Veto Resolution
    // ─────────────────────────────────────────────────────────────────

    /// Resolve all room proposals through the veto engine.
    #[instrument(skip(self, proposals))]
    async fn phase_veto(&self, proposals: Vec<RoomProposal>) -> PortfolioVector {
        let timer = std::time::Instant::now();

        let veto = self.veto.read().await;
        let last_pf = self.last_portfolio.read().await;

        let portfolio = veto.resolve(&proposals, last_pf.as_ref()).await;

        // Update last portfolio.
        drop(last_pf);
        let mut last_pf_w = self.last_portfolio.write().await;
        *last_pf_w = Some(portfolio.clone());

        let elapsed_ms = timer.elapsed().as_secs_f64() * 1000.0;
        debug!(
            elapsed_ms,
            positions = portfolio.positions.len(),
            net_exposure = portfolio.net_exposure,
            "Veto phase complete"
        );

        portfolio
    }

    // ─────────────────────────────────────────────────────────────────
    // Phase 4: Execution
    // ─────────────────────────────────────────────────────────────────

    /// Execute the portfolio — broadcast results.
    #[instrument(skip(self, portfolio))]
    async fn phase_execution(&self, portfolio: &PortfolioVector) {
        self.bridge.broadcast_portfolio(portfolio.clone());

        // Update current tick for the bridge context.
        let mut ct = self.current_tick.write().await;
        *ct = portfolio.timestamp;

        debug!(
            "Execution phase complete — portfolio with {} positions",
            portfolio.positions.len()
        );
    }

    // ─────────────────────────────────────────────────────────────────
    // Feature Collection (from rooms → matrix)
    // ─────────────────────────────────────────────────────────────────

    /// Collect and forward feature suggestions from rooms.
    /// This runs as a side-task alongside the main cycle.
    #[instrument(skip(self))]
    async fn collect_feature_suggestions(&self) -> Vec<FeatureSuggestion> {
        vec![]
    }

    // ─────────────────────────────────────────────────────────────────
    // Cycle Summary
    // ─────────────────────────────────────────────────────────────────

    /// Log a summary of the hybrid cycle.
    fn log_cycle_summary(&self, tick: u64, state: &CycleState) {
        info!(
            tick,
            proposals = state.proposals_received,
            features = state.features_received,
            matrix_ms = format!("{:.1}", state.matrix_time_ms),
            rooms_ms = format!("{:.1}", state.rooms_time_ms),
            veto_ms = format!("{:.1}", state.veto_time_ms),
            total_ms = format!("{:.1}", state.total_cycle_ms),
            "Hybrid cycle complete"
        );
    }
}

// ─────────────────────────────────────────────────────────────────────
// HybridEngine trait implementation
// ─────────────────────────────────────────────────────────────────────

#[async_trait]
impl<M, V> HybridEngine for HybridEngineImpl<M, V>
where
    M: MatrixEngine + 'static,
    V: VetoEngine + 'static,
{
    /// Run one full hybrid cycle: Matrix → Rooms → Veto → Execution.
    #[instrument(skip(self))]
    async fn hybrid_cycle(&self, tick: u64) {
        let cycle_start = std::time::Instant::now();
        let mut state = CycleState::default();

        // Phase 1: Matrix
        let t0 = std::time::Instant::now();
        let snapshot = self.phase_matrix(tick).await;
        state.matrix_time_ms = t0.elapsed().as_secs_f64() * 1000.0;

        // Phase 2: Rooms
        let t1 = std::time::Instant::now();
        let proposals = self.phase_rooms(&snapshot).await;
        state.rooms_time_ms = t1.elapsed().as_secs_f64() * 1000.0;
        state.proposals_received = proposals.len();

        // Phase 3: Veto
        let t2 = std::time::Instant::now();
        let portfolio = self.phase_veto(proposals).await;
        state.veto_time_ms = t2.elapsed().as_secs_f64() * 1000.0;

        // Phase 4: Execution
        self.phase_execution(&portfolio).await;

        // Collect feature suggestions (non-blocking side task)
        if self.config.collect_features {
            let _features = self.collect_feature_suggestions().await;
            state.features_received = _features.len();
        }

        state.total_cycle_ms = cycle_start.elapsed().as_secs_f64() * 1000.0;

        // Check performance target (end-to-end < 1s for fast path)
        if state.total_cycle_ms > 1000.0 {
            warn!(
                "Hybrid cycle exceeded 1s target: {:.1}ms",
                state.total_cycle_ms
            );
        }

        // Trace
        if self.config.enable_tracing {
            self.log_cycle_summary(tick, &state);
        }
    }

    /// Start the main event loop.
    #[instrument(skip(self))]
    async fn run(&self) {
        info!("HybridEngine event loop started");

        let mut tick = 0u64;

        loop {
            if self.shutdown.load(Ordering::Acquire) || self.bridge.is_shutdown_requested() {
                info!("HybridEngine shutdown signal received, stopping event loop");
                break;
            }

            tick += 1;
            self.hybrid_cycle(tick).await;
            tokio::task::yield_now().await;
        }

        info!("HybridEngine event loop stopped after {} ticks", tick);
    }

    /// Graceful shutdown.
    async fn shutdown(&self) {
        info!("HybridEngine shutting down...");
        self.shutdown.store(true, Ordering::Release);
        self.bridge.request_shutdown();
        self.bridge.emit_system_event(
            "engine_shutdown".into(),
            "HybridEngine graceful shutdown".into(),
        );
    }
}

// ─────────────────────────────────────────────────────────────────────
// Default Veto Engine Implementation
// ─────────────────────────────────────────────────────────────────────

/// A default veto engine that applies basic constraint checking.
pub struct DefaultVetoEngine {
    constraints: Vec<SaepConstraint>,
    frozen: Arc<AtomicBool>,
    freeze_reason: Arc<RwLock<Option<String>>>,
    current_portfolio: Arc<RwLock<Option<PortfolioVector>>>,
    total_proposals_processed: Arc<AtomicU64>,
    total_vetoes: Arc<AtomicU64>,
    total_warnings: Arc<AtomicU64>,
    total_limits: Arc<AtomicU64>,
    total_freezes: Arc<AtomicU64>,
}

impl Default for DefaultVetoEngine {
    fn default() -> Self {
        Self::new()
    }
}

impl DefaultVetoEngine {
    pub fn new() -> Self {
        Self {
            constraints: Vec::new(),
            frozen: Arc::new(AtomicBool::new(false)),
            freeze_reason: Arc::new(RwLock::new(None)),
            current_portfolio: Arc::new(RwLock::new(None)),
            total_proposals_processed: Arc::new(AtomicU64::new(0)),
            total_vetoes: Arc::new(AtomicU64::new(0)),
            total_warnings: Arc::new(AtomicU64::new(0)),
            total_limits: Arc::new(AtomicU64::new(0)),
            total_freezes: Arc::new(AtomicU64::new(0)),
        }
    }
}

#[async_trait]
impl VetoEngine for DefaultVetoEngine {
    async fn register_constraint(&mut self, constraint: SaepConstraint) {
        let id_clone = constraint.id.clone();
        let layer_clone = format!("{:?}", constraint.layer);
        info!(
            "Registered SAEP constraint: {} on {} layer",
            id_clone, layer_clone
        );
        self.constraints.push(constraint);
    }

    async fn resolve(
        &self,
        proposals: &[RoomProposal],
        current_portfolio: Option<&PortfolioVector>,
    ) -> PortfolioVector {
        let timer = std::time::Instant::now();

        if self.frozen.load(Ordering::Acquire) {
            let reason = self.freeze_reason.read().await;
            warn!("Veto engine is frozen: {:?}", reason);
            return current_portfolio.cloned().unwrap_or(PortfolioVector {
                positions: vec![],
                gross_exposure: 0.0,
                net_exposure: 0.0,
                sector_concentrations: HashMap::new(),
                portfolio_var: 0.0,
                timestamp: 0,
            });
        }

        let mut final_positions = Vec::with_capacity(proposals.len());
        let mut gross_exposure: f64 = 0.0;
        let mut net_exposure: f64 = 0.0;

        for proposal in proposals {
            let mut veto_applied: Vec<String> = Vec::new();
            let mut veto_severity: f64 = 0.0;

            // Check each SAEP constraint
            for constraint in &self.constraints {
                // Build sector concentration context (simplified)
                let sector_map = current_portfolio
                    .map(|p| p.sector_concentrations.clone())
                    .unwrap_or_default();

                match (constraint.check_fn)(proposal, &sector_map) {
                    Ok(()) => {}
                    Err(_violation) => {
                        veto_applied.push(constraint.id.clone());
                        match constraint.action {
                            SaepAction::Warn => {
                                self.total_warnings.fetch_add(1, Ordering::Relaxed);
                                veto_severity = veto_severity.max(0.2);
                            }
                            SaepAction::Limit => {
                                self.total_limits.fetch_add(1, Ordering::Relaxed);
                                veto_severity = veto_severity.max(0.5);
                            }
                            SaepAction::Veto => {
                                self.total_vetoes.fetch_add(1, Ordering::Relaxed);
                                veto_severity = 1.0;
                            }
                            SaepAction::Freeze => {
                                self.total_freezes.fetch_add(1, Ordering::Relaxed);
                                veto_severity = 1.0;
                            }
                        }
                    }
                }
            }

            // Compute final weight
            let weight = match proposal.gate {
                TernaryGate::Bullish => proposal.conviction * (1.0 - veto_severity),
                TernaryGate::Bearish => -proposal.conviction * (1.0 - veto_severity),
                TernaryGate::Neutral => 0.0,
            };

            gross_exposure += weight.abs();
            net_exposure += weight;

            final_positions.push(FinalPosition {
                ticker: proposal.ticker.clone(),
                weight: weight.clamp(-1.0, 1.0),
                raw_gate: proposal.gate.clone(),
                veto_applied,
                veto_severity,
            });
        }

        self.total_proposals_processed
            .fetch_add(proposals.len() as u64, Ordering::Relaxed);

        let elapsed_ms = timer.elapsed().as_secs_f64() * 1000.0;
        trace!(elapsed_ms, "Veto resolution complete");

        PortfolioVector {
            positions: final_positions,
            gross_exposure,
            net_exposure,
            sector_concentrations: HashMap::new(),
            portfolio_var: 0.0,
            timestamp: proposals.first().map(|p| p.timestamp).unwrap_or(0),
        }
    }

    async fn get_portfolio(&self) -> PortfolioVector {
        self.current_portfolio
            .read()
            .await
            .clone()
            .unwrap_or(PortfolioVector {
                positions: vec![],
                gross_exposure: 0.0,
                net_exposure: 0.0,
                sector_concentrations: HashMap::new(),
                portfolio_var: 0.0,
                timestamp: 0,
            })
    }

    async fn freeze(&mut self, reason: &str) {
        self.frozen.store(true, Ordering::Release);
        *self.freeze_reason.write().await = Some(reason.to_string());
        warn!("Veto engine frozen: {}", reason);
    }

    async fn unfreeze(&mut self, reason: &str) {
        self.frozen.store(false, Ordering::Release);
        *self.freeze_reason.write().await = None;
        info!("Veto engine unfrozen: {}", reason);
    }
}

// ─────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::TernaryGate;
    use crate::{GovernanceLayer, HybridMessage, Violation};
    use std::sync::Arc;

    // ── Mock Matrix Engine ──────────────────────────────────────────

    struct MockMatrixEngine;

    #[async_trait]
    impl MatrixEngine for MockMatrixEngine {
        async fn ingest(&self, _ticker: &str, _features: &[f64], _tick: u64) {}

        async fn fast_cycle(&self, _tick: u64) -> MatrixMetadata {
            MatrixMetadata {
                tick: _tick,
                n_stocks: 10,
                n_features: 50,
                n_history: 100,
                mean_correlation: 0.35,
                timestamp_ms: 0,
            }
        }

        async fn medium_cycle(&self, _tick: u64) -> PartialSnapshot {
            PartialSnapshot {
                tick: _tick,
                n_stocks: 10,
                correlation_matrix_cond: 2.5,
                top_eigenvalues: vec![0.9, 0.05, 0.03],
                regime: "stable".into(),
                timestamp_ms: 0,
            }
        }

        async fn full_cycle(&self, tick: u64) -> MatrixSnapshot {
            MatrixSnapshot {
                tick,
                n_stocks: 10,
                eigenvalues: vec![0.9, 0.05, 0.03, 0.01, 0.01],
                eigenvectors: ndarray::Array2::from_shape_vec(
                    (5, 2),
                    vec![0.5, 0.5, -0.5, 0.5, 0.5, -0.5, -0.5, -0.5, 0.3, 0.3],
                )
                .unwrap(),
                topologies: vec![],
                universe_betti: [5, 2, 1],
                regime: "stable".into(),
                condition_number: 2.5,
            }
        }

        async fn get_slice(&self, _ticker: &str) -> Option<ndarray::Array2<f32>> {
            Some(ndarray::Array2::zeros((50, 100)))
        }

        async fn get_cross_section(&self, _feature: &str, _time_idx: usize) -> Option<Array1<f32>> {
            Some(Array1::zeros(10))
        }

        async fn add_ticker(&self, _ticker: &str, _initial_features: Option<&[f64]>) {}

        async fn remove_ticker(&self, _ticker: &str) {}
    }

    // ── Mock Room Agent ─────────────────────────────────────────────

    struct MockRoomAgent {
        ticker: String,
    }

    #[async_trait]
    impl RoomAgent for MockRoomAgent {
        async fn analyze(
            &self,
            matrix_snapshot: &MatrixSnapshot,
            _feature_slice: Option<ndarray::Array2<f32>>,
            _cross_section: Option<Array1<f32>>,
        ) -> RoomProposal {
            RoomProposal {
                ticker: self.ticker.clone(),
                gate: TernaryGate::Bullish,
                conviction: 0.8,
                confidence: 0.7,
                narrative_sig: "mock_narrative".into(),
                matrix_agreement: 0.85,
                veto_override: false,
                timestamp: matrix_snapshot.tick,
            }
        }

        async fn on_symmetry_alert(
            &self,
            _peer_ticker: &str,
            _symmetry_score: f64,
            _topology: &TopologicalSignature,
        ) {
        }

        async fn suggest_feature(&self, _suggestion: FeatureSuggestion) {}

        async fn set_regime(&mut self, _label: String) {}

        async fn update_narrative(&mut self, _narrative: String) {}
    }

    #[tokio::test]
    async fn test_default_veto_engine() {
        let mut veto = DefaultVetoEngine::new();

        let constraint = SaepConstraint {
            id: "max_conviction".into(),
            layer: GovernanceLayer::Room,
            check_fn: Arc::new(|proposal: &RoomProposal, _: &HashMap<String, f64>| {
                if proposal.conviction > 1.0 {
                    Err(Violation {
                        constraint_id: "max_conviction".into(),
                        message: "Conviction exceeds 1.0".into(),
                        severity: 0.5,
                    })
                } else {
                    Ok(())
                }
            }),
            action: SaepAction::Limit,
            escalate_to: None,
        };
        veto.register_constraint(constraint).await;

        let proposals = vec![RoomProposal {
            ticker: "AAPL".into(),
            gate: TernaryGate::Bullish,
            conviction: 0.8,
            confidence: 0.7,
            narrative_sig: "test".into(),
            matrix_agreement: 0.9,
            veto_override: false,
            timestamp: 1,
        }];

        let portfolio = veto.resolve(&proposals, None).await;
        assert_eq!(portfolio.positions.len(), 1);
        assert!((portfolio.positions[0].weight - 0.8).abs() < 1e-6);
    }

    #[tokio::test]
    async fn test_default_veto_freeze() {
        let mut veto = DefaultVetoEngine::new();
        veto.freeze("test freeze").await;

        let proposals = vec![RoomProposal {
            ticker: "AAPL".into(),
            gate: TernaryGate::Bullish,
            conviction: 0.8,
            confidence: 0.7,
            narrative_sig: "test".into(),
            matrix_agreement: 0.9,
            veto_override: false,
            timestamp: 1,
        }];

        let portfolio = veto.resolve(&proposals, None).await;
        assert!(portfolio.positions.is_empty());
    }

    #[tokio::test]
    async fn test_hybrid_engine_creation() {
        let matrix = MockMatrixEngine;
        let bridge = Arc::new(HybridBridge::new());
        let veto = DefaultVetoEngine::new();
        let rooms: Vec<Box<dyn RoomAgent>> = vec![
            Box::new(MockRoomAgent {
                ticker: "AAPL".into(),
            }),
            Box::new(MockRoomAgent {
                ticker: "TSLA".into(),
            }),
        ];

        let config = HybridConfig::default();
        let _engine = HybridEngineImpl::new(matrix, bridge, veto, rooms, config);
    }

    #[tokio::test]
    async fn test_hybrid_cycle() {
        let matrix = MockMatrixEngine;
        let bridge = Arc::new(HybridBridge::new());
        let veto = DefaultVetoEngine::new();
        let rooms: Vec<Box<dyn RoomAgent>> = vec![
            Box::new(MockRoomAgent {
                ticker: "AAPL".into(),
            }),
            Box::new(MockRoomAgent {
                ticker: "MSFT".into(),
            }),
        ];

        let config = HybridConfig::default();
        let engine = HybridEngineImpl::new(matrix, bridge.clone(), veto, rooms, config);

        // Subscribe to portfolio results
        let mut pf_rx = bridge.subscribe_portfolio();

        // Run one cycle
        engine.hybrid_cycle(1).await;

        // Verify portfolio was broadcast
        match pf_rx.recv().await.unwrap() {
            HybridMessage::PortfolioVectorBroadcast(portfolio) => {
                assert_eq!(portfolio.positions.len(), 2);
                // Both are bullish with conviction 0.8
                assert!((portfolio.gross_exposure - 1.6).abs() < 1e-6);
            }
            _ => panic!("Expected PortfolioVectorBroadcast"),
        }
    }

    #[tokio::test]
    async fn test_hybrid_cycle_multiple_ticks() {
        let matrix = MockMatrixEngine;
        let bridge = Arc::new(HybridBridge::new());
        let veto = DefaultVetoEngine::new();
        let rooms: Vec<Box<dyn RoomAgent>> = vec![Box::new(MockRoomAgent {
            ticker: "AAPL".into(),
        })];

        let config = HybridConfig::default();
        let engine = HybridEngineImpl::new(matrix, bridge, veto, rooms, config);

        for tick in 1..=3 {
            engine.hybrid_cycle(tick).await;
        }
    }

    #[tokio::test]
    async fn test_saep_constraint_veto() {
        let mut veto = DefaultVetoEngine::new();

        let no_bearish = SaepConstraint {
            id: "no_bearish".into(),
            layer: GovernanceLayer::Room,
            check_fn: Arc::new(|proposal: &RoomProposal, _: &HashMap<String, f64>| {
                if proposal.gate == TernaryGate::Bearish {
                    Err(Violation {
                        constraint_id: "no_bearish".into(),
                        message: "Bearish positions not allowed".into(),
                        severity: 1.0,
                    })
                } else {
                    Ok(())
                }
            }),
            action: SaepAction::Veto,
            escalate_to: None,
        };
        veto.register_constraint(no_bearish).await;

        let proposals = vec![
            RoomProposal {
                ticker: "AAPL".into(),
                gate: TernaryGate::Bullish,
                conviction: 0.8,
                confidence: 0.7,
                narrative_sig: "test".into(),
                matrix_agreement: 0.9,
                veto_override: false,
                timestamp: 1,
            },
            RoomProposal {
                ticker: "TSLA".into(),
                gate: TernaryGate::Bearish,
                conviction: 0.6,
                confidence: 0.5,
                narrative_sig: "test".into(),
                matrix_agreement: 0.2,
                veto_override: false,
                timestamp: 1,
            },
        ];

        let portfolio = veto.resolve(&proposals, None).await;

        // AAPL should have full weight
        assert!((portfolio.positions[0].weight - 0.8).abs() < 1e-6);
        assert!(portfolio.positions[0].veto_applied.is_empty());

        // TSLA should be fully vetoed
        assert!((portfolio.positions[1].weight).abs() < 1e-10);
        assert!(portfolio.positions[1]
            .veto_applied
            .contains(&"no_bearish".to_string()));
        assert!((portfolio.positions[1].veto_severity - 1.0).abs() < 1e-6);
    }
}
