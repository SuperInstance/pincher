//! Core HybridBridge — the communication backbone of the Hybrid Manifold.
//!
//! The bridge connects three layers:
//! - **Matrix Engine** → broadcasts snapshots to all Room Agents
//! - **Room Agents** → submit proposals to the Veto Engine
//! - **Veto Engine** → broadcasts final portfolio to all subscribers
//!
//! # Architecture
//!
//! ```text
//! ┌──────────────┐    broadcast(tx)     ┌──────────────┐
//! │ Matrix Engine│ ──────────────────►  │  Room Agents  │
//! │              │                      │ (N subscribers)│
//! └──────┬───────┘                      └──────┬────────┘
//!        │                                     │
//!        │ feature_tx (mpsc)                   │ proposal_tx (mpsc)
//!        ▼                                     ▼
//! ┌──────────────┐                    ┌──────────────┐
//! │ Feature Coll.│                    │  Veto Engine  │
//! └──────────────┘                    └──────┬────────┘
//!                                            │
//!                                  broadcast(tx)
//!                                            ▼
//!                                   ┌──────────────┐
//!                                   │ Portfolio     │
//!                                   │ Subscribers   │
//!                                   └──────────────┘
//! ```

use crate::error::HybridResult;
use crate::types::{
    FeatureSuggestion, HybridMessage, MatrixSnapshot, PortfolioVector, RoomProposal,
};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use tokio::sync::{broadcast, mpsc, Mutex as AsyncMutex};
use tracing::{debug, info, instrument, trace, warn};

/// Default channel capacities.
const MATRIX_BROADCAST_CAPACITY: usize = 256;
const VETO_BROADCAST_CAPACITY: usize = 256;
const PROPOSAL_CHANNEL_CAPACITY: usize = 8192;
const FEATURE_CHANNEL_CAPACITY: usize = 1024;
const SYSTEM_CHANNEL_CAPACITY: usize = 64;

/// Bridge runtime metrics for observability.
#[derive(Debug, Clone, Default)]
pub struct BridgeMetrics {
    pub total_snapshots: Arc<AtomicU64>,
    pub total_proposals: Arc<AtomicU64>,
    pub total_features: Arc<AtomicU64>,
    pub total_portfolios: Arc<AtomicU64>,
    pub total_system_events: Arc<AtomicU64>,
    pub total_slice_updates: Arc<AtomicU64>,
    pub total_symmetry_alerts: Arc<AtomicU64>,
    pub subscriber_count: Arc<AtomicU64>,
    pub messages_dropped: Arc<AtomicU64>,
}

impl BridgeMetrics {
    fn snapshot_sent(&self) {
        self.total_snapshots.fetch_add(1, Ordering::Relaxed);
    }
    fn proposal_received(&self) {
        self.total_proposals.fetch_add(1, Ordering::Relaxed);
    }
    fn feature_received(&self) {
        self.total_features.fetch_add(1, Ordering::Relaxed);
    }
    fn portfolio_sent(&self) {
        self.total_portfolios.fetch_add(1, Ordering::Relaxed);
    }
    fn system_event(&self) {
        self.total_system_events.fetch_add(1, Ordering::Relaxed);
    }
    fn slice_update(&self) {
        self.total_slice_updates.fetch_add(1, Ordering::Relaxed);
    }
    fn symmetry_alert(&self) {
        self.total_symmetry_alerts.fetch_add(1, Ordering::Relaxed);
    }
    fn message_dropped(&self) {
        self.messages_dropped.fetch_add(1, Ordering::Relaxed);
    }
    fn set_subscriber_count(&self, n: u64) {
        self.subscriber_count.store(n, Ordering::Relaxed);
    }

    /// Snapshot all counters for observability reporting.
    #[allow(dead_code)]
    pub fn snapshot(&self) -> BridgeMetricSnapshot {
        BridgeMetricSnapshot {
            snapshots: self.total_snapshots.load(Ordering::Relaxed),
            proposals: self.total_proposals.load(Ordering::Relaxed),
            features: self.total_features.load(Ordering::Relaxed),
            portfolios: self.total_portfolios.load(Ordering::Relaxed),
            system_events: self.total_system_events.load(Ordering::Relaxed),
            slice_updates: self.total_slice_updates.load(Ordering::Relaxed),
            symmetry_alerts: self.total_symmetry_alerts.load(Ordering::Relaxed),
            subscribers: self.subscriber_count.load(Ordering::Relaxed),
            dropped: self.messages_dropped.load(Ordering::Relaxed),
        }
    }
}

/// A snapshot of bridge metrics at a point in time.
#[derive(Debug, Clone)]
pub struct BridgeMetricSnapshot {
    pub snapshots: u64,
    pub proposals: u64,
    pub features: u64,
    pub portfolios: u64,
    pub system_events: u64,
    pub slice_updates: u64,
    pub symmetry_alerts: u64,
    pub subscribers: u64,
    pub dropped: u64,
}

/// The bridge manager — owns all channels connecting the three layers.
///
/// # Thread Safety
///
/// `HybridBridge` is `Send + Sync` and designed to be shared across tasks
/// via `Arc<HybridBridge>`. All mutable state is behind `RwLock`.
#[derive(Debug)]
pub struct HybridBridge {
    /// Channel to broadcast Matrix snapshots to all room agents.
    matrix_tx: broadcast::Sender<HybridMessage>,

    /// Channel for room agents to submit proposals to the Veto Engine.
    proposal_tx: mpsc::Sender<RoomProposal>,

    /// Channel for room agents to suggest features to the Matrix Engine.
    feature_tx: mpsc::Sender<FeatureSuggestion>,

    /// Channel for Veto Engine to broadcast portfolio vectors.
    veto_tx: broadcast::Sender<HybridMessage>,

    /// Channel for system events.
    system_tx: broadcast::Sender<HybridMessage>,

    /// Receiver side of the proposal channel (consumed by Veto).
    proposal_rx: AsyncMutex<Option<mpsc::Receiver<RoomProposal>>>,

    /// Receiver side of the feature channel (consumed by Matrix).
    feature_rx: AsyncMutex<Option<mpsc::Receiver<FeatureSuggestion>>>,

    /// Runtime metrics.
    metrics: BridgeMetrics,

    /// Shutdown flag.
    shutdown_flag: Arc<AtomicBool>,
}

impl HybridBridge {
    /// Create a new hybrid bridge with default channel capacities.
    pub fn new() -> Self {
        let (matrix_tx, _) = broadcast::channel(MATRIX_BROADCAST_CAPACITY);
        let (proposal_tx, proposal_rx) = mpsc::channel(PROPOSAL_CHANNEL_CAPACITY);
        let (feature_tx, feature_rx) = mpsc::channel(FEATURE_CHANNEL_CAPACITY);
        let (veto_tx, _) = broadcast::channel(VETO_BROADCAST_CAPACITY);
        let (system_tx, _) = broadcast::channel(SYSTEM_CHANNEL_CAPACITY);

        HybridBridge {
            matrix_tx,
            proposal_tx,
            feature_tx,
            veto_tx,
            system_tx,
            proposal_rx: AsyncMutex::new(Some(proposal_rx)),
            feature_rx: AsyncMutex::new(Some(feature_rx)),
            metrics: BridgeMetrics::default(),
            shutdown_flag: Arc::new(AtomicBool::new(false)),
        }
    }

    /// Create a new hybrid bridge with custom channel capacities.
    pub fn with_capacities(
        matrix_cap: usize,
        veto_cap: usize,
        proposal_cap: usize,
        feature_cap: usize,
    ) -> Self {
        let (matrix_tx, _) = broadcast::channel(matrix_cap);
        let (proposal_tx, proposal_rx) = mpsc::channel(proposal_cap);
        let (feature_tx, feature_rx) = mpsc::channel(feature_cap);
        let (veto_tx, _) = broadcast::channel(veto_cap);
        let (system_tx, _) = broadcast::channel(SYSTEM_CHANNEL_CAPACITY);

        HybridBridge {
            matrix_tx,
            proposal_tx,
            feature_tx,
            veto_tx,
            system_tx,
            proposal_rx: AsyncMutex::new(Some(proposal_rx)),
            feature_rx: AsyncMutex::new(Some(feature_rx)),
            metrics: BridgeMetrics::default(),
            shutdown_flag: Arc::new(AtomicBool::new(false)),
        }
    }

    // ─────────────────────────────────────────────────────────────────
    // Subscriptions
    // ─────────────────────────────────────────────────────────────────

    /// Subscribe a room agent to matrix snapshot broadcasts.
    ///
    /// Returns a `broadcast::Receiver` that receives `HybridMessage::SnapshotBroadcast`
    /// messages. Each subscriber gets its own receiver; slow consumers may miss messages.
    pub fn subscribe_matrix(&self) -> broadcast::Receiver<HybridMessage> {
        let rx = self.matrix_tx.subscribe();
        self.metrics
            .set_subscriber_count(self.matrix_tx.receiver_count() as u64);
        rx
    }

    /// Subscribe to portfolio broadcasts from the Veto Engine.
    pub fn subscribe_portfolio(&self) -> broadcast::Receiver<HybridMessage> {
        self.veto_tx.subscribe()
    }

    /// Subscribe to system events (freeze, error, etc.).
    pub fn subscribe_system_events(&self) -> broadcast::Receiver<HybridMessage> {
        self.system_tx.subscribe()
    }

    /// Get the current number of matrix broadcast subscribers.
    pub fn subscriber_count(&self) -> u64 {
        self.matrix_tx.receiver_count() as u64
    }

    /// Access bridge metrics.
    pub fn metrics(&self) -> &BridgeMetrics {
        &self.metrics
    }

    // ─────────────────────────────────────────────────────────────────
    // Take receivers (one-shot, for the consuming engine tasks)
    // ─────────────────────────────────────────────────────────────────

    /// Take the proposal receiver — called once by the Veto Engine consumer.
    pub async fn take_proposal_receiver(&self) -> HybridResult<mpsc::Receiver<RoomProposal>> {
        self.proposal_rx
            .lock()
            .await
            .take()
            .ok_or(crate::error::HybridError::Internal(
                "Proposal receiver already taken".into(),
            ))
    }

    /// Take the feature suggestion receiver — called once by the Matrix Engine consumer.
    pub async fn take_feature_receiver(&self) -> HybridResult<mpsc::Receiver<FeatureSuggestion>> {
        self.feature_rx
            .lock()
            .await
            .take()
            .ok_or(crate::error::HybridError::Internal(
                "Feature receiver already taken".into(),
            ))
    }

    // ─────────────────────────────────────────────────────────────────
    // Sending — Matrix Engine side
    // ─────────────────────────────────────────────────────────────────

    /// Broadcast a matrix snapshot to all subscribed room agents.
    ///
    /// Uses `broadcast::Sender::send` which fans out to all receivers.
    /// Slow consumers that fall behind will miss messages (lagged).
    #[instrument(skip(self, snapshot))]
    pub fn broadcast_snapshot(&self, snapshot: MatrixSnapshot) -> usize {
        let msg = HybridMessage::SnapshotBroadcast(snapshot);
        let recipient_count = match self.matrix_tx.send(msg) {
            Ok(count) => count,
            Err(broadcast::error::SendError(_msg)) => {
                let lagged = self.matrix_tx.receiver_count();
                warn!("Snapshot broadcast dropped: {} receivers lagged", lagged);
                self.metrics.message_dropped();
                0_usize
            }
        };
        self.metrics.snapshot_sent();
        debug!("Snapshot broadcast to {} recipients", recipient_count);
        recipient_count
    }

    /// Send a slice update to a specific room agent (via matrix broadcast).
    /// All rooms receive it but only the target room should process it.
    #[instrument(skip(self))]
    pub fn send_slice_update(&self, ticker: String, data: ndarray::Array2<f32>) {
        let msg = HybridMessage::SliceUpdate { ticker, data };
        let _ = self.matrix_tx.send(msg);
        self.metrics.slice_update();
    }

    /// Broadcast a symmetry alert to all room agents.
    pub fn broadcast_symmetry_alert(&self, tickers: Vec<String>, score: f64) {
        let msg = HybridMessage::SymmetryAlert { tickers, score };
        let _ = self.matrix_tx.send(msg);
        self.metrics.symmetry_alert();
    }

    // ─────────────────────────────────────────────────────────────────
    // Sending — Room Agent side
    // ─────────────────────────────────────────────────────────────────

    /// Submit a proposal from a room agent to the Veto Engine.
    ///
    /// Proposals flow through an mpsc channel to ensure backpressure.
    /// This is a non-blocking send that returns an error if the channel is full.
    #[instrument(skip(self, proposal))]
    pub async fn submit_proposal(&self, proposal: RoomProposal) -> HybridResult<()> {
        self.proposal_tx
            .send(proposal)
            .await
            .map_err(|_| crate::error::HybridError::ChannelClosed)?;
        self.metrics.proposal_received();
        trace!("Proposal submitted");
        Ok(())
    }

    /// Try to submit a proposal without awaiting (non-blocking).
    pub fn try_submit_proposal(&self, proposal: RoomProposal) -> HybridResult<()> {
        self.proposal_tx.try_send(proposal).map_err(|e| match e {
            mpsc::error::TrySendError::Full(_) => {
                self.metrics.message_dropped();
                crate::error::HybridError::Internal("Proposal channel full".into())
            }
            mpsc::error::TrySendError::Closed(_) => crate::error::HybridError::ChannelClosed,
        })?;
        self.metrics.proposal_received();
        Ok(())
    }

    /// Submit a feature suggestion from a room agent to the Matrix Engine.
    #[instrument(skip(self, suggestion))]
    pub async fn submit_feature_suggestion(
        &self,
        suggestion: FeatureSuggestion,
    ) -> HybridResult<()> {
        self.feature_tx
            .send(suggestion)
            .await
            .map_err(|_| crate::error::HybridError::ChannelClosed)?;
        self.metrics.feature_received();
        trace!("Feature suggestion submitted");
        Ok(())
    }

    // ─────────────────────────────────────────────────────────────────
    // Sending — Veto Engine side
    // ─────────────────────────────────────────────────────────────────

    /// Broadcast the final portfolio vector to all subscribers.
    #[instrument(skip(self, portfolio))]
    pub fn broadcast_portfolio(&self, portfolio: PortfolioVector) -> usize {
        let msg = HybridMessage::PortfolioVectorBroadcast(portfolio);
        let count = match self.veto_tx.send(msg) {
            Ok(n) => n,
            Err(_) => {
                self.metrics.message_dropped();
                0_usize
            }
        };
        self.metrics.portfolio_sent();
        debug!("Portfolio broadcast to {} recipients", count);
        count
    }

    // ─────────────────────────────────────────────────────────────────
    // System Events
    // ─────────────────────────────────────────────────────────────────

    /// Emit a system event (freeze, error, etc.).
    pub fn emit_system_event(&self, kind: String, payload: String) {
        let msg = HybridMessage::SystemEvent {
            kind: kind.clone(),
            payload,
        };
        let _ = self.system_tx.send(msg);
        self.metrics.system_event();
        info!("System event emitted: {}", kind);
    }

    // ─────────────────────────────────────────────────────────────────
    // Shutdown
    // ─────────────────────────────────────────────────────────────────

    /// Check if shutdown has been requested.
    pub fn is_shutdown_requested(&self) -> bool {
        self.shutdown_flag.load(Ordering::Acquire)
    }

    /// Request graceful shutdown.
    pub fn request_shutdown(&self) {
        self.shutdown_flag.store(true, Ordering::Release);
        info!("HybridBridge shutdown requested");
        self.emit_system_event("shutdown".into(), "Graceful shutdown requested".into());
    }
}

impl Default for HybridBridge {
    fn default() -> Self {
        Self::new()
    }
}

// ─────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::TernaryGate;
    use ndarray::array;
    use std::f64::consts::FRAC_1_SQRT_2;

    fn sample_proposal(ticker: &str) -> RoomProposal {
        RoomProposal {
            ticker: ticker.to_string(),
            gate: TernaryGate::Bullish,
            conviction: 0.85,
            confidence: 0.72,
            narrative_sig: "abc123".into(),
            matrix_agreement: 0.91,
            veto_override: false,
            timestamp: 1000,
        }
    }

    fn sample_snapshot(tick: u64) -> MatrixSnapshot {
        MatrixSnapshot {
            tick,
            n_stocks: 3,
            eigenvalues: vec![0.95, 0.03, 0.02],
            eigenvectors: array![
                [FRAC_1_SQRT_2, FRAC_1_SQRT_2],
                [FRAC_1_SQRT_2, -FRAC_1_SQRT_2]
            ],
            topologies: vec![],
            universe_betti: [3, 1, 0],
            regime: "stable".into(),
            condition_number: 2.5,
        }
    }

    #[tokio::test]
    async fn test_bridge_create() {
        let bridge = HybridBridge::new();
        assert_eq!(bridge.subscriber_count(), 0);
    }

    #[tokio::test]
    async fn test_bridge_broadcast_snapshot() {
        let bridge = HybridBridge::new();
        let mut rx = bridge.subscribe_matrix();
        let snapshot = sample_snapshot(1);

        let count = bridge.broadcast_snapshot(snapshot);
        // 1 receiver
        assert_eq!(count, 1);

        let received = rx.recv().await.unwrap();
        match received {
            HybridMessage::SnapshotBroadcast(s) => {
                assert_eq!(s.tick, 1);
                assert_eq!(s.regime, "stable");
            }
            _ => panic!("Expected SnapshotBroadcast"),
        }
    }

    #[tokio::test]
    async fn test_bridge_submit_proposal() {
        let bridge = HybridBridge::new();
        let proposal = sample_proposal("AAPL");

        bridge.submit_proposal(proposal).await.unwrap();

        // Take the receiver and verify
        let mut rx = bridge.take_proposal_receiver().await.unwrap();
        let received = rx.recv().await.unwrap();
        assert_eq!(received.ticker, "AAPL");
        assert_eq!(received.gate, TernaryGate::Bullish);
    }

    #[tokio::test]
    async fn test_bridge_proposal_multiple() {
        let bridge = HybridBridge::new();

        let proposals = vec![sample_proposal("AAPL"), sample_proposal("TSLA")];
        for p in proposals {
            bridge.submit_proposal(p).await.unwrap();
        }

        let mut rx = bridge.take_proposal_receiver().await.unwrap();
        let p1 = rx.recv().await.unwrap();
        let p2 = rx.recv().await.unwrap();
        assert_eq!(p1.ticker, "AAPL");
        assert_eq!(p2.ticker, "TSLA");
    }

    #[tokio::test]
    async fn test_bridge_portfolio_broadcast() {
        use std::collections::HashMap;
        let bridge = HybridBridge::new();
        let mut rx = bridge.subscribe_portfolio();

        let portfolio = PortfolioVector {
            positions: vec![],
            gross_exposure: 0.0,
            net_exposure: 0.0,
            sector_concentrations: HashMap::new(),
            portfolio_var: 0.0,
            timestamp: 1,
        };

        bridge.broadcast_portfolio(portfolio);
        match rx.recv().await.unwrap() {
            HybridMessage::PortfolioVectorBroadcast(p) => {
                assert_eq!(p.timestamp, 1);
            }
            _ => panic!("Expected PortfolioVectorBroadcast"),
        }
    }

    #[tokio::test]
    async fn test_bridge_submit_feature_suggestion() {
        let bridge = HybridBridge::new();
        let suggestion = FeatureSuggestion {
            ticker: "AAPL".into(),
            feature_name: "lithium_correlation".into(),
            source: "earnings_call".into(),
            urgency: 0.8,
            sample_data: vec![0.1, 0.2, 0.3],
        };

        bridge.submit_feature_suggestion(suggestion).await.unwrap();

        let mut rx = bridge.take_feature_receiver().await.unwrap();
        let received = rx.recv().await.unwrap();
        assert_eq!(received.feature_name, "lithium_correlation");
    }

    #[tokio::test]
    async fn test_bridge_system_events() {
        let bridge = HybridBridge::new();
        let mut rx = bridge.subscribe_system_events();

        bridge.emit_system_event("freeze".into(), "Market circuit breaker".into());

        match rx.recv().await.unwrap() {
            HybridMessage::SystemEvent { kind, payload } => {
                assert_eq!(kind, "freeze");
                assert_eq!(payload, "Market circuit breaker");
            }
            _ => panic!("Expected SystemEvent"),
        }
    }

    #[tokio::test]
    async fn test_bridge_shutdown_flag() {
        let bridge = HybridBridge::new();
        assert!(!bridge.is_shutdown_requested());
        bridge.request_shutdown();
        assert!(bridge.is_shutdown_requested());
    }

    #[tokio::test]
    async fn test_bridge_symmetry_alert() {
        let bridge = HybridBridge::new();
        let mut rx = bridge.subscribe_matrix();

        bridge.broadcast_symmetry_alert(vec!["AAPL".into(), "MSFT".into()], 0.87);

        match rx.recv().await.unwrap() {
            HybridMessage::SymmetryAlert { tickers, score } => {
                assert_eq!(tickers, vec!["AAPL", "MSFT"]);
                assert!((score - 0.87).abs() < 1e-10);
            }
            _ => panic!("Expected SymmetryAlert"),
        }
    }

    #[tokio::test]
    async fn test_bridge_slice_update() {
        let bridge = HybridBridge::new();
        let mut rx = bridge.subscribe_matrix();

        let data = ndarray::Array2::<f32>::zeros((5, 3));
        bridge.send_slice_update("AAPL".into(), data);

        match rx.recv().await.unwrap() {
            HybridMessage::SliceUpdate { ticker, data } => {
                assert_eq!(ticker, "AAPL");
                assert_eq!(data.shape(), &[5, 3]);
            }
            _ => panic!("Expected SliceUpdate"),
        }
    }

    #[tokio::test]
    async fn test_bridge_try_submit_proposal() {
        let bridge = HybridBridge::new();
        let proposal = sample_proposal("GOOGL");
        bridge.try_submit_proposal(proposal).unwrap();

        let mut rx = bridge.take_proposal_receiver().await.unwrap();
        let received = rx.recv().await.unwrap();
        assert_eq!(received.ticker, "GOOGL");
    }

    #[tokio::test]
    async fn test_bridge_receiver_taken_twice() {
        let bridge = HybridBridge::new();
        let _rx1 = bridge.take_proposal_receiver().await.unwrap();
        let result = bridge.take_proposal_receiver().await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_bridge_metrics() {
        let bridge = HybridBridge::new();
        let _rx = bridge.subscribe_matrix();

        bridge.broadcast_snapshot(sample_snapshot(1));
        bridge.broadcast_portfolio(PortfolioVector {
            positions: vec![],
            gross_exposure: 0.0,
            net_exposure: 0.0,
            sector_concentrations: std::collections::HashMap::new(),
            portfolio_var: 0.0,
            timestamp: 1,
        });

        let s = bridge.metrics().snapshot();
        assert_eq!(s.snapshots, 1);
        assert_eq!(s.portfolios, 1);
        assert_eq!(s.subscribers, 1);
    }
}
