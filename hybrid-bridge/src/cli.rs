//! # Hybrid Bridge CLI — Inspect and control the Hybrid Manifold from the terminal.
//!
//! Provides a `HybridCli` struct that connects to the `HybridBridge` via `Arc` channel
//! endpoints and exposes subcommands for inspecting bridge state, injecting data,
//! taking snapshots, submitting proposals, and toggling freeze state.
//!
//! ## Usage (from pincher-cli or any binary)
//!
//! ```rust,ignore
//! use std::sync::Arc;
//! use hybrid_bridge::HybridBridge;
//! use hybrid_bridge::cli::{HybridCli, HybridCliCommand, run_hybrid_cli};
//!
//! # async fn example() {
//! let bridge = Arc::new(HybridBridge::new());
//!
//! // Parse from clap args
//! let cmd = HybridCliCommand::parse_from(&["hybrid", "status"]);
//! let cli = HybridCli::new(bridge);
//! run_hybrid_cli(&cli, cmd).await.unwrap();
//! # }
//! ```

use crate::bridge::HybridBridge;
use crate::error::{HybridError, HybridResult};
use crate::types::{
    HybridMessage, MatrixSnapshot, PortfolioVector, RoomProposal, TernaryGate, TopologicalSignature,
};
use clap::{Parser, Subcommand};
use std::sync::Arc;
use std::time::Duration;
use tokio::time::timeout;

// ─────────────────────────────────────────────────────────────────────
// CLI Argument Types
// ─────────────────────────────────────────────────────────────────────

/// Top-level hybrid bridge CLI.
#[derive(Parser, Debug, Clone)]
#[command(name = "hybrid", about = "Inspect and control the Hybrid Manifold")]
pub struct HybridCliCommand {
    #[command(subcommand)]
    pub command: HybridSubcommand,
}

/// Available subcommands for the hybrid bridge CLI.
#[derive(Subcommand, Debug, Clone)]
pub enum HybridSubcommand {
    /// Show current bridge status, metrics, and subscriber count
    Status,

    /// Inject a feature slice for a ticker into the bridge
    Inject {
        /// Ticker symbol (e.g., AAPL)
        ticker: String,

        /// Comma-separated feature values (e.g., "0.1,0.5,-0.3")
        #[arg(short, long, value_delimiter = ',', allow_hyphen_values = true)]
        features: Vec<f64>,
    },

    /// Subscribe to the next matrix snapshot and print it
    Snapshot {
        /// Timeout in seconds to wait for a snapshot
        #[arg(short, long, default_value = "10")]
        timeout: u64,
    },

    /// Submit a room proposal for a ticker with a gate direction
    Propose {
        /// Ticker symbol (e.g., AAPL)
        ticker: String,

        /// Gate direction: bullish | bearish | neutral
        gate: GateArg,

        /// Conviction weight [0.0–1.0]
        #[arg(short = 'w', long, default_value = "0.8")]
        conviction: f64,

        /// Confidence [0.0–1.0]
        #[arg(short = 'C', long, default_value = "0.7")]
        confidence: f64,
    },

    /// Freeze the hybrid bridge (halts all actions)
    Freeze {
        /// Reason for the freeze
        reason: String,
    },

    /// Unfreeze the hybrid bridge (resumes operations)
    Unfreeze {
        /// Optional reason for unfreezing
        #[arg(default_value = "Manual unfreeze via CLI")]
        reason: String,
    },
}

/// Gate direction argument (clap-friendly string).
#[derive(Debug, Clone, clap::ValueEnum)]
pub enum GateArg {
    Bullish,
    Bearish,
    Neutral,
}

impl From<GateArg> for TernaryGate {
    fn from(g: GateArg) -> Self {
        match g {
            GateArg::Bullish => TernaryGate::Bullish,
            GateArg::Bearish => TernaryGate::Bearish,
            GateArg::Neutral => TernaryGate::Neutral,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────
// CLI Runtime
// ─────────────────────────────────────────────────────────────────────

/// The hybrid bridge CLI runtime — owns a reference to the bridge and
/// provides human-readable commands for inspecting and controlling it.
#[derive(Debug, Clone)]
pub struct HybridCli {
    /// Shared bridge reference.
    bridge: Arc<HybridBridge>,
    /// Default timeout for blocking operations like snapshot waits.
    default_timeout: Duration,
}

impl HybridCli {
    /// Create a new CLI runtime connected to the given bridge.
    pub fn new(bridge: Arc<HybridBridge>) -> Self {
        Self {
            bridge,
            default_timeout: Duration::from_secs(10),
        }
    }

    /// Set a custom default timeout.
    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.default_timeout = timeout;
        Self { ..self }
    }

    // ── Status ──────────────────────────────────────────────────────

    /// Display bridge health, metrics, and subscriber count.
    pub async fn status(&self) -> HybridResult<()> {
        let metrics = self.bridge.metrics().snapshot();
        let subscribers = self.bridge.subscriber_count();

        println!("┌─ Hybrid Manifold Bridge ──────────────────────┐");
        println!("│                                                │");

        // Subscribers
        println!(
            "│  Subscribers:             {:>6}                │",
            subscribers
        );

        // Metrics
        println!("│  ── Metrics ──                                 │");
        println!(
            "│  Snapshots broadcast:     {:>6}                │",
            metrics.snapshots
        );
        println!(
            "│  Proposals received:      {:>6}                │",
            metrics.proposals
        );
        println!(
            "│  Feature suggestions:     {:>6}                │",
            metrics.features
        );
        println!(
            "│  Portfolios published:    {:>6}                │",
            metrics.portfolios
        );
        println!(
            "│  System events emitted:   {:>6}                │",
            metrics.system_events
        );
        println!(
            "│  Slice updates:           {:>6}                │",
            metrics.slice_updates
        );
        println!(
            "│  Symmetry alerts:         {:>6}                │",
            metrics.symmetry_alerts
        );
        println!(
            "│  Messages dropped:        {:>6}                │",
            metrics.dropped
        );

        // Shutdown status
        let shutdown = self.bridge.is_shutdown_requested();
        println!(
            "│  Shutdown requested:      {:>6}                │",
            if shutdown { "YES" } else { "no" }
        );

        println!("│                                                │");
        println!("└────────────────────────────────────────────────┘");

        Ok(())
    }

    // ── Inject ─────────────────────────────────────────────────────

    /// Inject a feature slice for a ticker into the bridge.
    ///
    /// This sends a `SliceUpdate` message through the bridge's matrix
    /// broadcast channel, simulating data ingestion from the Matrix Engine.
    pub fn inject(&self, ticker: &str, features: &[f64]) -> HybridResult<()> {
        if ticker.is_empty() {
            return Err(HybridError::Internal("Ticker must not be empty".into()));
        }
        if features.is_empty() {
            return Err(HybridError::Internal("Features must not be empty".into()));
        }

        // Convert feature slice to an n x 1 ndarray (single time step, multiple features)
        let n_features = features.len();
        let data = match ndarray::Array2::from_shape_vec((n_features, 1), features.to_vec()) {
            Ok(arr) => arr.mapv(|v| v as f32),
            Err(e) => {
                return Err(HybridError::Internal(format!(
                    "Failed to create feature tensor: {}",
                    e
                )));
            }
        };

        self.bridge.send_slice_update(ticker.to_string(), data);

        println!(
            "✅ Injected {} features for ticker {}",
            features.len(),
            ticker
        );
        println!(
            "   Feature vector: [{}]",
            features
                .iter()
                .map(|v| format!("{:.4}", v))
                .collect::<Vec<_>>()
                .join(", ")
        );

        Ok(())
    }

    // ── Snapshot ───────────────────────────────────────────────────

    /// Subscribe to the next matrix snapshot and print it.
    ///
    /// Waits up to `timeout_secs` for a snapshot broadcast. If no snapshot
    /// arrives within the timeout, returns an error.
    pub async fn snapshot(&self, timeout_secs: u64) -> HybridResult<()> {
        let mut rx = self.bridge.subscribe_matrix();
        let wait = Duration::from_secs(timeout_secs);

        let result = timeout(wait, rx.recv()).await.map_err(|_| {
            HybridError::Internal(format!(
                "Timed out waiting for snapshot after {}s",
                timeout_secs
            ))
        })?;

        let msg = result.map_err(|_| {
            HybridError::Internal("Snapshot channel closed — no more snapshots".into())
        })?;

        match msg {
            HybridMessage::SnapshotBroadcast(snapshot) => {
                print_snapshot(&snapshot);
                Ok(())
            }
            other => {
                println!(
                    "⚠️  Received unexpected message variant: {}",
                    other.variant_name()
                );
                println!("   (expected SnapshotBroadcast)");
                Err(HybridError::Internal(format!(
                    "Expected SnapshotBroadcast, got {}",
                    other.variant_name()
                )))
            }
        }
    }

    // ── Propose ────────────────────────────────────────────────────

    /// Submit a room proposal for a ticker with a gate direction, conviction,
    /// and confidence.
    pub async fn propose(
        &self,
        ticker: &str,
        gate: TernaryGate,
        conviction: f64,
        confidence: f64,
    ) -> HybridResult<()> {
        if ticker.is_empty() {
            return Err(HybridError::Internal("Ticker must not be empty".into()));
        }
        if !(0.0..=1.0).contains(&conviction) {
            return Err(HybridError::Internal(format!(
                "Conviction must be in [0.0, 1.0], got {}",
                conviction
            )));
        }
        if !(0.0..=1.0).contains(&confidence) {
            return Err(HybridError::Internal(format!(
                "Confidence must be in [0.0, 1.0], got {}",
                confidence
            )));
        }

        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64;

        let proposal = RoomProposal {
            ticker: ticker.to_string(),
            gate: gate.clone(),
            conviction,
            confidence,
            narrative_sig: format!("cli_proposal_{}", now),
            matrix_agreement: 0.5, // unknown — midpoint default
            veto_override: false,
            timestamp: now,
        };

        self.bridge
            .submit_proposal(proposal.clone())
            .await
            .map_err(|e| HybridError::Internal(format!("Failed to submit proposal: {}", e)))?;

        println!(
            "✅ Proposal submitted for {} ({:?}) — conviction={:.2}, confidence={:.2}",
            proposal.ticker, proposal.gate, proposal.conviction, proposal.confidence
        );

        Ok(())
    }

    // ── Freeze / Unfreeze ──────────────────────────────────────────

    /// Emit a system freeze event through the bridge.
    ///
    /// This broadcasts a `SystemEvent` with `kind = "freeze"`. Downstream
    /// subscribers (including the Veto Engine) should halt when they see it.
    pub fn freeze(&self, reason: &str) -> HybridResult<()> {
        if reason.trim().is_empty() {
            return Err(HybridError::Internal(
                "Freeze reason must not be empty".into(),
            ));
        }

        self.bridge
            .emit_system_event("freeze".into(), reason.to_string());

        println!("❄️  Bridge frozen — reason: \"{}\"", reason);
        println!("   All downstream components should halt operations.");
        Ok(())
    }

    /// Emit a system unfreeze event through the bridge.
    pub fn unfreeze(&self, reason: &str) -> HybridResult<()> {
        self.bridge
            .emit_system_event("unfreeze".into(), reason.to_string());

        println!("🔥 Bridge unfrozen — reason: \"{}\"", reason);
        println!("   Normal operations may resume.");
        Ok(())
    }
}

// ─────────────────────────────────────────────────────────────────────
// Dispatch
// ─────────────────────────────────────────────────────────────────────

/// Parse and dispatch a hybrid CLI command.
///
/// Convenience function for use in binary entry points.
pub async fn run_hybrid_cli(cli: &HybridCli, cmd: HybridCliCommand) -> HybridResult<()> {
    match cmd.command {
        HybridSubcommand::Status => cli.status().await,
        HybridSubcommand::Inject { ticker, features } => cli.inject(&ticker, &features),
        HybridSubcommand::Snapshot { timeout } => cli.snapshot(timeout).await,
        HybridSubcommand::Propose {
            ticker,
            gate,
            conviction,
            confidence,
        } => {
            cli.propose(&ticker, gate.into(), conviction, confidence)
                .await
        }
        HybridSubcommand::Freeze { reason } => cli.freeze(&reason),
        HybridSubcommand::Unfreeze { reason } => cli.unfreeze(&reason),
    }
}

// ─────────────────────────────────────────────────────────────────────
// Output Helpers
// ─────────────────────────────────────────────────────────────────────

/// Pretty-print a `MatrixSnapshot` to stdout.
fn print_snapshot(snapshot: &MatrixSnapshot) {
    println!("┌─ Matrix Snapshot ─────────────────────────────┐");
    println!(
        "│  Tick:                 {:>12}              │",
        snapshot.tick
    );
    println!(
        "│  Stocks tracked:       {:>12}              │",
        snapshot.n_stocks
    );
    println!(
        "│  Regime:               {:>12}              │",
        snapshot.regime
    );
    println!(
        "│  Condition number:     {:>12.4}            │",
        snapshot.condition_number
    );
    println!(
        "│  Universe Betti:       [{:>2}, {:>2}, {:>2}]               │",
        snapshot.universe_betti[0], snapshot.universe_betti[1], snapshot.universe_betti[2]
    );

    // Eigenvalues (top 5)
    println!("│  ── Top Eigenvalues ──                         │");
    let top_k = snapshot.eigenvalues.len().min(5);
    if top_k > 0 {
        for (i, ev) in snapshot.eigenvalues.iter().enumerate().take(top_k) {
            println!("│    λ{}:  {:.6}                    │", i + 1, ev);
        }
    } else {
        println!("│    (none computed)                            │");
    }

    // Eigenvectors shape
    let eig_shape = snapshot.eigenvectors.shape();
    if eig_shape[0] > 0 && eig_shape[1] > 0 {
        println!(
            "│  Eigenvectors:         {}×{}                │",
            eig_shape[0], eig_shape[1]
        );
    }

    // Topologies
    println!(
        "│  Topologies returned:  {:>12}              │",
        snapshot.topologies.len()
    );
    if !snapshot.topologies.is_empty() {
        // Print first topology as a sample
        print_topology_sample(&snapshot.topologies);
    }

    println!("└────────────────────────────────────────────────┘");
}

/// Pretty-print a sample topology signature.
fn print_topology_sample(topologies: &[TopologicalSignature]) {
    let sample = &topologies[0];
    println!(
        "│  Sample ticker:        {:>12}              │",
        sample.ticker
    );
    println!(
        "│    Betti numbers:      [{:>2}, {:>2}, {:>2}]               │",
        sample.betti_numbers[0],
        sample.betti_numbers.get(1).unwrap_or(&0),
        sample.betti_numbers.get(2).unwrap_or(&0),
    );
    println!(
        "│    Wasserstein dist:   {:>12.4}            │",
        sample.wasserstein_distance_centroid
    );
    println!(
        "│    Regime:             {:>12}              │",
        sample.regime_label
    );
    println!(
        "│    Confidence:         {:>12.2}            │",
        sample.confidence
    );

    // Persistence landscape sample
    if !sample.persistence_landscape.is_empty() {
        let n_pts = sample.persistence_landscape.len().min(5);
        let landscape_snip: Vec<String> = sample.persistence_landscape[..n_pts]
            .iter()
            .map(|v| format!("{:.4}", v))
            .collect();
        println!(
            "│    Landscape (first {}): {}   │",
            n_pts,
            landscape_snip.join(", ")
        );
    }
}

/// Pretty-print a portfolio vector to stdout.
#[allow(dead_code)]
pub(crate) fn print_portfolio(portfolio: &PortfolioVector) {
    println!("┌─ Portfolio Vector ────────────────────────────┐");
    println!(
        "│  Timestamp:            {:>12}              │",
        portfolio.timestamp
    );
    println!(
        "│  Positions:            {:>12}              │",
        portfolio.positions.len()
    );
    println!(
        "│  Gross exposure:       {:>12.4}            │",
        portfolio.gross_exposure
    );
    println!(
        "│  Net exposure:         {:>12.4}            │",
        portfolio.net_exposure
    );
    println!(
        "│  Portfolio VaR:        {:>12.4}            │",
        portfolio.portfolio_var
    );

    if !portfolio.positions.is_empty() {
        println!("│  ── Positions ──                              │");
        let top = portfolio.positions.len().min(10);
        for pos in portfolio.positions.iter().take(top) {
            let weight_sign = if pos.weight > 0.0 { '+' } else { ' ' };
            println!(
                "│    {} {}{:.4} (gate: {:?}, veto_sev: {:.2})      │",
                pos.ticker, weight_sign, pos.weight, pos.raw_gate, pos.veto_severity
            );
        }
        if portfolio.positions.len() > 10 {
            println!(
                "│    ... and {} more                                   │",
                portfolio.positions.len() - 10
            );
        }
    }

    println!("└────────────────────────────────────────────────┘");
}

// ─────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::HybridBridge;

    #[test]
    fn test_gate_arg_bullish() {
        let gate = GateArg::Bullish;
        let ternary: TernaryGate = gate.into();
        assert_eq!(ternary, TernaryGate::Bullish);
    }

    #[test]
    fn test_gate_arg_bearish() {
        let gate = GateArg::Bearish;
        let ternary: TernaryGate = gate.into();
        assert_eq!(ternary, TernaryGate::Bearish);
    }

    #[test]
    fn test_gate_arg_neutral() {
        let gate = GateArg::Neutral;
        let ternary: TernaryGate = gate.into();
        assert_eq!(ternary, TernaryGate::Neutral);
    }

    #[test]
    fn test_hybrid_cli_create() {
        let bridge = Arc::new(HybridBridge::new());
        let cli = HybridCli::new(bridge);
        // Just verify it constructs without panicking
        let _ = cli;
    }

    #[test]
    fn test_hybrid_cli_with_timeout() {
        let bridge = Arc::new(HybridBridge::new());
        let cli = HybridCli::new(bridge.clone()).with_timeout(Duration::from_secs(30));
        let _ = cli;
    }

    #[tokio::test]
    async fn test_status_output() {
        let bridge = Arc::new(HybridBridge::new());
        let cli = HybridCli::new(bridge.clone());

        // Emit a few events so metrics are non-zero
        bridge.emit_system_event("test".into(), "testing 1 2 3".into());
        let _rx = bridge.subscribe_matrix();
        bridge.broadcast_snapshot(crate::types::MatrixSnapshot {
            tick: 1,
            n_stocks: 10,
            eigenvalues: vec![],
            eigenvectors: ndarray::Array2::from_shape_vec((0, 0), vec![]).unwrap(),
            topologies: vec![],
            universe_betti: [0, 0, 0],
            regime: "test".into(),
            condition_number: 0.0,
        });

        // Status should not panic and should return Ok
        let result = cli.status().await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_inject_valid() {
        let bridge = Arc::new(HybridBridge::new());
        let cli = HybridCli::new(bridge);

        let result = cli.inject("AAPL", &[0.1, 0.5, -0.3, 0.8]);
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_inject_empty_ticker() {
        let bridge = Arc::new(HybridBridge::new());
        let cli = HybridCli::new(bridge);

        let result = cli.inject("", &[0.1]);
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_inject_empty_features() {
        let bridge = Arc::new(HybridBridge::new());
        let cli = HybridCli::new(bridge);

        let result = cli.inject("AAPL", &[]);
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_propose_valid() {
        let bridge = Arc::new(HybridBridge::new());
        let cli = HybridCli::new(bridge);

        let result = cli.propose("AAPL", TernaryGate::Bullish, 0.8, 0.7).await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_propose_empty_ticker() {
        let bridge = Arc::new(HybridBridge::new());
        let cli = HybridCli::new(bridge);

        let result = cli.propose("", TernaryGate::Bullish, 0.8, 0.7).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_propose_invalid_conviction() {
        let bridge = Arc::new(HybridBridge::new());
        let cli = HybridCli::new(bridge);

        let result = cli.propose("AAPL", TernaryGate::Bullish, 1.5, 0.7).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_propose_invalid_confidence() {
        let bridge = Arc::new(HybridBridge::new());
        let cli = HybridCli::new(bridge);

        let result = cli.propose("AAPL", TernaryGate::Bullish, 0.8, -0.1).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_freeze_valid() {
        let bridge = Arc::new(HybridBridge::new());
        let cli = HybridCli::new(bridge);

        let result = cli.freeze("Market circuit breaker triggered");
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_freeze_empty_reason() {
        let bridge = Arc::new(HybridBridge::new());
        let cli = HybridCli::new(bridge);

        let result = cli.freeze("");
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_unfreeze_valid() {
        let bridge = Arc::new(HybridBridge::new());
        let cli = HybridCli::new(bridge);

        // freeze first, then unfreeze
        let _ = cli.freeze("test");
        let result = cli.unfreeze("All clear — resuming operations");
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_snapshot_timeout() {
        // Without any snapshot being broadcast, snapshot() should time out
        let bridge = Arc::new(HybridBridge::new());
        let cli = HybridCli::new(bridge);

        let result = cli.snapshot(1).await;
        assert!(result.is_err());
        let err = result.unwrap_err();
        let err_msg = err.to_string();
        assert!(
            err_msg.contains("Timed out"),
            "Expected timeout error, got: {}",
            err_msg
        );
    }

    #[tokio::test]
    async fn test_dispatch_status() {
        let bridge = Arc::new(HybridBridge::new());
        let cli = HybridCli::new(bridge.clone());

        let cmd = HybridCliCommand {
            command: HybridSubcommand::Status,
        };
        let result = run_hybrid_cli(&cli, cmd).await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_dispatch_freeze() {
        let bridge = Arc::new(HybridBridge::new());
        let cli = HybridCli::new(bridge);

        let cmd = HybridCliCommand {
            command: HybridSubcommand::Freeze {
                reason: "CLI test freeze".into(),
            },
        };
        let result = run_hybrid_cli(&cli, cmd).await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_dispatch_inject() {
        let bridge = Arc::new(HybridBridge::new());
        let cli = HybridCli::new(bridge);

        let cmd = HybridCliCommand {
            command: HybridSubcommand::Inject {
                ticker: "MSFT".into(),
                features: vec![0.2, 0.7, -0.1],
            },
        };
        let result = run_hybrid_cli(&cli, cmd).await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_dispatch_propose() {
        let bridge = Arc::new(HybridBridge::new());
        let cli = HybridCli::new(bridge);

        let cmd = HybridCliCommand {
            command: HybridSubcommand::Propose {
                ticker: "TSLA".into(),
                gate: GateArg::Bearish,
                conviction: 0.6,
                confidence: 0.5,
            },
        };
        let result = run_hybrid_cli(&cli, cmd).await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_snapshot_receives_broadcast() {
        let bridge = Arc::new(HybridBridge::new());
        let cli = HybridCli::new(bridge.clone());

        // Spawn a task that broadcasts a snapshot after a short delay
        let bridge_bg = bridge.clone();
        tokio::spawn(async move {
            tokio::time::sleep(Duration::from_millis(100)).await;
            bridge_bg.broadcast_snapshot(crate::types::MatrixSnapshot {
                tick: 42,
                n_stocks: 5,
                eigenvalues: vec![0.95, 0.03, 0.02],
                eigenvectors: ndarray::Array2::from_shape_vec((2, 2), vec![0.7, 0.7, -0.7, 0.7])
                    .unwrap(),
                topologies: vec![],
                universe_betti: [3, 1, 0],
                regime: "rotation".into(),
                condition_number: 3.2,
            });
        });

        let result = cli.snapshot(5).await;
        assert!(
            result.is_ok(),
            "Expected Ok snapshot, got: {:?}",
            result.err()
        );
    }

    #[test]
    fn test_print_portfolio_does_not_panic() {
        use crate::types::{FinalPosition, PortfolioVector, TernaryGate};
        use std::collections::HashMap;

        let pf = PortfolioVector {
            positions: vec![
                FinalPosition {
                    ticker: "AAPL".into(),
                    weight: 0.75,
                    raw_gate: TernaryGate::Bullish,
                    veto_applied: vec![],
                    veto_severity: 0.0,
                },
                FinalPosition {
                    ticker: "TSLA".into(),
                    weight: -0.30,
                    raw_gate: TernaryGate::Bearish,
                    veto_applied: vec!["max_exposure".into()],
                    veto_severity: 0.5,
                },
            ],
            gross_exposure: 1.05,
            net_exposure: 0.45,
            sector_concentrations: HashMap::from([("Tech".into(), 0.7)]),
            portfolio_var: 0.023,
            timestamp: 1000,
        };

        // This should not panic
        print_portfolio(&pf);
    }
}
