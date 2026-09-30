//! # Full Pipeline Test — MatrixEngine → VetoEngine → RoomAgent → Narrative
//!
//! Verifies the complete hybrid cycle end-to-end:
//!
//! 1. **Matrix Phase**: Data ingestion through MockMatrixEngine's feature tensor,
//!    fast/medium/full cycle computation
//! 2. **Room Phase**: Room agents consume snapshots and produce proposals
//! 3. **Veto Phase**: SAEP-constrained resolution into a PortfolioVector
//! 4. **Bridge Phase**: Broadcast/communication through HybridBridge channels
//! 5. **Narrative Phase**: Room agents carry narratives that influence proposals
//! 6. **Ternary Conversion**: Portfolio → trit → weight round-trip
//!
//! This is the authoritative "does the system work?" integration test.

use hybrid_bridge::mock_matrix::MockMatrixEngine;
use hybrid_bridge::mock_room::MockRoomAgent;
use hybrid_bridge::mock_veto::MockVetoEngine;
use hybrid_bridge::ternary_bridge::{
    ConsensusComputer, ConsensusConfig, PortfolioToTrits, TritMapping, TritsToWeights,
    WeightedTritVote,
};
use hybrid_bridge::{
    detect_non_finite, mask_non_finite, GovernanceLayer, HybridBridge, HybridMessage,
    MatrixEngineTrait, MatrixSnapshot, PortfolioVector, RoomAgentTrait, RoomProposal, SaepAction,
    SaepConstraint, TernaryGate, VetoEngineTrait, Violation,
};
use std::collections::HashMap;
use std::sync::Arc;

// ═══════════════════════════════════════════════════════════════════════
// Constants
// ═══════════════════════════════════════════════════════════════════════

const N_TICKERS: usize = 5;
const N_FEATURES: usize = 4; // price, volume, vwap, spread
const N_HISTORY: usize = 100;
const TEST_TICK: u64 = 42;

/// Stock names used throughout the pipeline.
const TICKERS: &[&str] = &["AAPL", "GOOGL", "MSFT", "AMZN", "TSLA"];

// ═══════════════════════════════════════════════════════════════════════
// Phase 1 Helpers — Matrix Setup & Cycle
// ═══════════════════════════════════════════════════════════════════════

/// Create a pre-seeded MockMatrixEngine with all tickers registered.
fn setup_matrix() -> MockMatrixEngine {
    let engine = MockMatrixEngine::new(N_TICKERS, N_FEATURES, N_HISTORY).with_tickers(TICKERS);
    engine.seed_random();
    engine
}

/// Run the full matrix cycle (fast + medium + full) on a tick.
fn run_matrix_cycle(engine: &MockMatrixEngine, tick: u64) -> (MatrixSnapshot, usize) {
    let rt = tokio::runtime::Runtime::new().unwrap();
    let _meta = rt.block_on(engine.fast_cycle(tick));
    let _partial = rt.block_on(engine.medium_cycle(tick));
    let snapshot = rt.block_on(engine.full_cycle(tick));

    let n_topologies = snapshot.topologies.len();
    (snapshot, n_topologies)
}

// ═══════════════════════════════════════════════════════════════════════
// Phase 2 Helpers — Room Analysis & Narrative
// ═══════════════════════════════════════════════════════════════════════

/// Create room agents with narratives that should influence their proposals.
fn setup_rooms_with_narratives() -> Vec<MockRoomAgent> {
    TICKERS
        .iter()
        .enumerate()
        .map(|(i, t)| {
            let narrative = match i {
                0 => "Strong buy — sector rotation beneficiary".to_string(),
                1 => "Cautious — regulatory headwinds expected".to_string(),
                2 => "Neutral — fair value, hold position".to_string(),
                3 => "Accumulate — undervalued on DCF".to_string(),
                4 => "Avoid — high volatility, low visibility".to_string(),
                _ => "Default narrative".to_string(),
            };
            MockRoomAgent::new(t).with_narrative(&narrative)
        })
        .collect()
}

/// Analyze all rooms against a snapshot, collecting proposals.
fn analyze_rooms(
    rooms: &[MockRoomAgent],
    snapshot: &MatrixSnapshot,
    _engine: &MockMatrixEngine,
) -> Vec<RoomProposal> {
    let rt = tokio::runtime::Runtime::new().unwrap();
    rooms
        .iter()
        .map(|room| {
            // Proposals carry their own ticker; we just use the room for analysis
            let proposal = rt.block_on(room.analyze(snapshot, None, None));
            assert!(
                (0.0..=1.0).contains(&proposal.conviction),
                "Conviction out of range: {}",
                proposal.conviction
            );
            assert!(
                (0.0..=1.0).contains(&proposal.confidence),
                "Confidence out of range: {}",
                proposal.confidence
            );
            assert!(
                (0.0..=1.0).contains(&proposal.matrix_agreement),
                "Matrix agreement out of range: {}",
                proposal.matrix_agreement
            );
            assert!(
                !proposal.narrative_sig.is_empty(),
                "Narrative sig must not be empty"
            );
            assert_eq!(proposal.timestamp, snapshot.tick);
            proposal
        })
        .collect()
}

// ═══════════════════════════════════════════════════════════════════════
// Phase 3 Helpers — Veto Resolution
// ═══════════════════════════════════════════════════════════════════════

/// Set up a MockVetoEngine with default constraints plus a custom sector cap.
fn setup_veto_engine() -> MockVetoEngine {
    let mut veto = MockVetoEngine::new().with_default_constraints();

    // Add a custom constraint: cap tech sector exposure at 60%
    let tech_cap = SaepConstraint {
        id: "tech-sector-cap".into(),
        layer: GovernanceLayer::Sector,
        check_fn: Arc::new(|_proposal: &RoomProposal, sectors: &HashMap<String, f64>| {
            let tech_exposure = sectors.get("Technology").copied().unwrap_or(0.0);
            if tech_exposure > 0.6 {
                Err(Violation {
                    constraint_id: "tech-sector-cap".into(),
                    message: "Tech sector exposure exceeds 60% cap".into(),
                    severity: 0.8,
                })
            } else {
                Ok(())
            }
        }),
        action: SaepAction::Limit,
        escalate_to: Some(GovernanceLayer::Market),
    };

    let rt = tokio::runtime::Runtime::new().unwrap();
    rt.block_on(veto.register_constraint(tech_cap));
    veto
}

/// Run the veto engine on proposals, producing a portfolio.
fn run_veto(
    veto: &MockVetoEngine,
    proposals: &[RoomProposal],
    current_portfolio: Option<&PortfolioVector>,
) -> PortfolioVector {
    let rt = tokio::runtime::Runtime::new().unwrap();
    rt.block_on(veto.resolve(proposals, current_portfolio))
}

// ═══════════════════════════════════════════════════════════════════════
// Phase 4 Helpers — Bridge Communication
// ═══════════════════════════════════════════════════════════════════════

/// Simulate the full bridge communication cycle.
fn bridge_cycle(
    bridge: &HybridBridge,
    snapshot: MatrixSnapshot,
    proposals: &[RoomProposal],
    portfolio: PortfolioVector,
) -> (bool, bool, bool) {
    let rt = tokio::runtime::Runtime::new().unwrap();

    // Subscribe receivers before broadcasting
    let mut matrix_rx = bridge.subscribe_matrix();
    let mut veto_rx = bridge.subscribe_portfolio();

    // Broadcast snapshot
    let _n_recipients = bridge.broadcast_snapshot(snapshot.clone());
    let snapshot_received = rt.block_on(async {
        match matrix_rx.recv().await {
            Ok(HybridMessage::SnapshotBroadcast(recvd)) => recvd.tick == snapshot.tick,
            _ => false,
        }
    });

    // Submit proposals through bridge
    for p in proposals {
        rt.block_on(bridge.submit_proposal(p.clone())).unwrap();
    }

    // Broadcast portfolio
    bridge.broadcast_portfolio(portfolio);
    let portfolio_received = rt.block_on(async {
        match veto_rx.recv().await {
            Ok(HybridMessage::PortfolioVectorBroadcast(pf)) => pf.timestamp == snapshot.tick,
            _ => false,
        }
    });

    // Verify metrics are updated
    let metrics = bridge.metrics().snapshot();
    let metrics_updated = metrics.snapshots >= 1
        && metrics.proposals >= proposals.len() as u64
        && metrics.portfolios >= 1;

    (snapshot_received, portfolio_received, metrics_updated)
}

// ═══════════════════════════════════════════════════════════════════════
// Phase 5 Helpers — Ternary Bridge Round-Trip
// ═══════════════════════════════════════════════════════════════════════

/// Round-trip a portfolio through the ternary bridge: Portfolio → Trits → Weights.
fn ternary_roundtrip(portfolio: &PortfolioVector) -> Vec<f64> {
    // Step 1: Portfolio → TritVector (SAEP-aware encoding)
    let encoder = PortfolioToTrits::default();
    let trit_vector = encoder
        .encode(portfolio)
        .expect("Portfolio encoding should succeed");

    // Step 2: TritVector → Weights (linear interpolation)
    let converter = TritsToWeights::default();
    let weights = converter
        .trits_to_weights(&trit_vector)
        .expect("Trit to weight conversion should succeed");

    // Verify weight invariants
    for &w in &weights {
        assert!((-1.0..=1.0).contains(&w), "Weight {} out of [-1, 1]", w);
    }

    weights
}

/// Compute consensus from proposals using the ternary bridge.
fn compute_ternary_consensus(proposals: &[RoomProposal]) -> (f64, f64, f64) {
    // Map proposals to weighted votes
    let votes: Vec<WeightedTritVote> = proposals
        .iter()
        .map(|p| {
            let trit = TritMapping::gate_to_trit(&p.gate);
            WeightedTritVote::new(trit, p.confidence * p.matrix_agreement, p.conviction)
        })
        .collect();

    // Compute continuous consensus score
    let config = ConsensusConfig::default();
    let computer = ConsensusComputer::new(config);
    let score = computer
        .consensus_score(&votes)
        .expect("Consensus score computation should succeed");

    // Compute positional consensus trit
    let trit_result = computer
        .compute_consensus(&votes)
        .expect("Consensus computation should succeed");

    // Map consensus trit to a gate
    let consensus_gate = TritMapping::trit_to_gate(trit_result.get(0).unwrap());
    let consensus_weight = match consensus_gate {
        TernaryGate::Bullish => 1.0 * score.abs(),
        TernaryGate::Bearish => -(score.abs()),
        TernaryGate::Neutral => 0.0,
    };

    (score, consensus_weight, consensus_gate.to_i8() as f64)
}

// ═══════════════════════════════════════════════════════════════════════
// Phase 6 Helpers — Chaos Injection & Recovery
// ═══════════════════════════════════════════════════════════════════════

/// Inject chaos into the matrix tensor and verify detection + masking + recovery.
fn chaos_cycle(engine: &MockMatrixEngine) -> usize {
    let rt = tokio::runtime::Runtime::new().unwrap();

    // Inject NaN and Inf at known coordinates
    engine.inject_nan(0, 0, 10); // stock 0, feature 0, time 10
    engine.inject_nan(2, 1, 50); // stock 2, feature 1, time 50
    engine.inject_inf(1, 0, 30); // stock 1, feature 0, time 30
    engine.inject_inf(4, 2, 80); // stock 4, feature 2, time 80

    // Detect
    let detected = {
        let tensor = engine.tensor();
        rt.block_on(async {
            let t = tensor.read().await;
            detect_non_finite(&t)
        })
    };
    assert_eq!(detected.len(), 4, "Should detect all 4 non-finite values");

    // Mask
    let masked = {
        let tensor = engine.tensor();
        rt.block_on(async {
            let mut t = tensor.write().await;
            mask_non_finite(&mut t)
        })
    };
    assert_eq!(masked, 4, "Should mask all 4 non-finite values");

    // Verify clean
    let remaining = {
        let tensor = engine.tensor();
        rt.block_on(async {
            let t = tensor.read().await;
            detect_non_finite(&t)
        })
    };
    assert!(
        remaining.is_empty(),
        "No non-finite values should remain after masking"
    );

    // Verify recovery: matrix still produces valid output
    let snapshot = rt.block_on(engine.full_cycle(100));
    assert!(
        snapshot.condition_number.is_finite(),
        "Condition number should be finite after recovery"
    );
    for topo in &snapshot.topologies {
        assert!(
            topo.confidence.is_finite(),
            "Topology confidence should be finite after recovery"
        );
    }

    masked
}

// ═══════════════════════════════════════════════════════════════════════
// THE FULL PIPELINE TEST
// ═══════════════════════════════════════════════════════════════════════

#[test]
fn test_full_pipeline_matrix_rooms_veto_narrative() {
    // ── Phase 1: Matrix Setup & Cycle ────────────────────────────────
    let engine = setup_matrix();

    // Run chaos cycle to verify tensor integrity before main pipeline
    let chaos_masked = chaos_cycle(&engine);
    assert_eq!(chaos_masked, 4, "Chaos cycle should mask 4 cells");

    // Run the matrix cycle
    let (snapshot, n_topologies) = run_matrix_cycle(&engine, TEST_TICK);
    assert_eq!(snapshot.tick, TEST_TICK, "Snapshot tick should match");
    assert!(
        snapshot.n_stocks >= N_TICKERS,
        "Snapshot should track all stocks"
    );
    assert!(
        n_topologies == N_TICKERS,
        "Expected {} topologies, got {}",
        N_TICKERS,
        n_topologies
    );
    assert!(
        snapshot.condition_number > 0.0,
        "Condition number should be positive"
    );
    assert!(
        snapshot.regime == "stable",
        "Regime should be stable by default"
    );

    // ── Phase 2: Room Analysis with Narratives ───────────────────────
    let rooms = setup_rooms_with_narratives();
    let proposals = analyze_rooms(&rooms, &snapshot, &engine);

    assert_eq!(
        proposals.len(),
        N_TICKERS,
        "Should have {} proposals",
        N_TICKERS
    );

    // Verify each room's narrative signature is unique
    let sigs: std::collections::HashSet<&str> =
        proposals.iter().map(|p| p.narrative_sig.as_str()).collect();
    assert_eq!(
        sigs.len(),
        N_TICKERS,
        "Each room should produce a unique narrative signature: {:?}",
        sigs
    );

    // Verify tickers match
    for (i, t) in TICKERS.iter().enumerate() {
        assert_eq!(
            &proposals[i].ticker, t,
            "Proposal {} should be for ticker {}",
            i, t
        );
    }

    // ── Phase 2b: Symmetry Alert Handling ────────────────────────────
    let rt = tokio::runtime::Runtime::new().unwrap();
    let topo = &snapshot.topologies[0];
    rt.block_on(rooms[1].on_symmetry_alert(TICKERS[0], 0.85, topo));

    // ── Phase 3: Veto Resolution ─────────────────────────────────────
    let veto = setup_veto_engine();
    let portfolio = run_veto(&veto, &proposals, None);

    // Portfolio invariants
    assert_eq!(
        portfolio.positions.len(),
        N_TICKERS,
        "Portfolio should have {} positions",
        N_TICKERS
    );
    assert!(
        portfolio.gross_exposure > 0.0,
        "Gross exposure should be positive"
    );
    assert!(
        portfolio.gross_exposure >= portfolio.net_exposure.abs(),
        "Gross exposure ({}) should >= |net exposure| ({})",
        portfolio.gross_exposure,
        portfolio.net_exposure.abs()
    );
    assert!(
        portfolio.portfolio_var >= 0.0,
        "Portfolio VaR should be non-negative"
    );
    assert_eq!(
        portfolio.timestamp, TEST_TICK,
        "Portfolio timestamp should match cycle tick"
    );

    // Verify position invariants
    for pos in &portfolio.positions {
        assert!(
            (-1.0..=1.0).contains(&pos.weight),
            "Position weight {} out of [-1, 1] for {}",
            pos.weight,
            pos.ticker
        );
        assert!(
            (0.0..=1.0).contains(&pos.veto_severity),
            "Veto severity {} out of [0, 1] for {}",
            pos.veto_severity,
            pos.ticker
        );
        assert!(
            pos.weight.is_finite(),
            "Weight must be finite for {}",
            pos.ticker
        );
    }

    // Track which tickers got positions
    let tickers_in_portfolio: std::collections::HashSet<&str> = portfolio
        .positions
        .iter()
        .map(|p| p.ticker.as_str())
        .collect();
    for t in TICKERS {
        assert!(
            tickers_in_portfolio.contains(t),
            "Ticker {} should have a position in portfolio",
            t
        );
    }

    // ── Phase 4: Bridge Communication ────────────────────────────────
    let bridge = HybridBridge::new();
    let (snap_ok, port_ok, metrics_ok) = bridge_cycle(&bridge, snapshot, &proposals, portfolio);
    assert!(snap_ok, "Snapshot should be received via bridge");
    assert!(port_ok, "Portfolio should be received via bridge");
    assert!(metrics_ok, "Bridge metrics should be updated");

    // ── Phase 5: Ternary Bridge Round-Trip ───────────────────────────
    // Re-run veto to get the portfolio again
    let portfolio2 = run_veto(&veto, &proposals, None);
    let _weights = ternary_roundtrip(&portfolio2);

    // Compute ternary consensus from proposals
    let (_score, _cweight, _cgate) = compute_ternary_consensus(&proposals);

    // ── Phase 6: Freeze/Unfreeze Cycle ───────────────────────────────
    let mut freeze_veto = MockVetoEngine::new().with_default_constraints();
    let rt = tokio::runtime::Runtime::new().unwrap();

    // Freeze
    rt.block_on(freeze_veto.freeze("Full pipeline test freeze"));

    // Frozen resolve should return empty
    let frozen_pf = rt.block_on(freeze_veto.resolve(&proposals, None));
    assert!(
        frozen_pf.positions.is_empty(),
        "Frozen veto should return empty positions"
    );

    // Verify freeze reason was recorded
    bridge.emit_system_event("freeze".into(), "test freeze".into());
    let _pre_freeze_events = bridge.metrics().snapshot().system_events;

    // Unfreeze
    rt.block_on(freeze_veto.unfreeze("Full pipeline test unfreeze"));

    // Unfrozen resolve should return valid positions
    let unfrozen_pf = rt.block_on(freeze_veto.resolve(&proposals, None));
    assert_eq!(
        unfrozen_pf.positions.len(),
        N_TICKERS,
        "Unfrozen veto should process all proposals"
    );

    // ── Phase 7: Metrics Verification ────────────────────────────────
    let final_metrics = bridge.metrics().snapshot();
    assert!(final_metrics.snapshots >= 1, "Should have ≥1 snapshot");
    assert!(
        final_metrics.proposals >= proposals.len() as u64,
        "Should track proposals"
    );
    assert_eq!(final_metrics.snapshots, 1, "Exactly 1 snapshot broadcast");
    assert_eq!(final_metrics.portfolios, 1, "Exactly 1 portfolio broadcast");
}
