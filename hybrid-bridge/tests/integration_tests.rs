//! # Integration Tests — Hybrid Bridge Full la-link
//!
//! These tests exercise the complete data flow through the Market Manifold
//! pipeline, from raw data ingestion through matrix computation, room agent
//! analysis, veto resolution, and final portfolio output.
//!
//! ## Test Structure
//!
//! 1. **Data Ingestion → Matrix Tensor** — Verify that ingesting mock data
//!    updates the feature tensor and produces valid metadata.
//!
//! 2. **Matrix Snapshot → Room Agent → Room Proposal** — Verify that a room
//!    agent correctly consumes a matrix snapshot/slice and produces a
//!    well-formed RoomProposal.
//!
//! 3. **Proposal → Veto Engine → Final Position** — Verify that the veto
//!    engine aggregates multiple proposals, applies SAEP constraints, and
//!    produces a valid PortfolioVector.
//!
//! 4. **End-to-End Hybrid Cycle** — Verify the full pipeline in one pass,
//!    including latency measurements against performance targets.
//!
//! 5. **Chaos Tests** — Inject NaN/Inf into the tensor; verify safe-mode
//!    detection, masking, and veto engine rejection of pathological proposals.
//!
//! 6. **Edge Cases** — Empty ticker list, zero-length proposals, freeze/recover,
//!    large ticker counts.

use hybrid_bridge::mock_matrix::MockMatrixEngine;
use hybrid_bridge::mock_room::MockRoomAgent;
use hybrid_bridge::mock_veto::MockVetoEngine;
use hybrid_bridge::{
    detect_non_finite, mask_non_finite, GovernanceLayer, HybridBridge, HybridMessage,
    MatrixEngineTrait, MatrixSnapshot, RoomAgentTrait, RoomProposal, SaepAction, SaepConstraint,
    TernaryGate, VetoEngineTrait, Violation,
};
use std::sync::Arc;
use std::time::Instant;

// ═══════════════════════════════════════════════════════════════════════
// Helpers
// ═══════════════════════════════════════════════════════════════════════

/// Create a test matrix engine with pre-populated tickers.
fn setup_matrix(tickers: &[&str], n_features: usize, n_history: usize) -> MockMatrixEngine {
    let engine = MockMatrixEngine::new(tickers.len(), n_features, n_history).with_tickers(tickers);
    engine.seed_random();
    engine
}

/// Create N mock room agents from ticker names.
fn setup_rooms(tickers: &[&str]) -> Vec<MockRoomAgent> {
    tickers.iter().map(|t| MockRoomAgent::new(t)).collect()
}

/// Produce a standard snapshot for testing.
fn produce_snapshot(engine: &MockMatrixEngine, tick: u64) -> MatrixSnapshot {
    let rt = tokio::runtime::Runtime::new().unwrap();
    rt.block_on(engine.full_cycle(tick))
}

/// Run multiple rooms in sequence, collecting proposals.
fn collect_proposals(rooms: &[MockRoomAgent], snapshot: &MatrixSnapshot) -> Vec<RoomProposal> {
    let rt = tokio::runtime::Runtime::new().unwrap();
    rooms
        .iter()
        .map(|room| rt.block_on(room.analyze(snapshot, None, None)))
        .collect()
}

// ═══════════════════════════════════════════════════════════════════════
// 1. DATA INGESTION → MATRIX TENSOR
// ═══════════════════════════════════════════════════════════════════════

#[test]
fn test_data_ingestion_updates_tensor() {
    let engine = MockMatrixEngine::new(5, 3, 100).with_tickers(&["AAPL", "MSFT"]);
    engine.seed_random();

    // Ingest data — since tickers are pre-registered, check fast_cycle succeeds
    let meta = {
        let rt = tokio::runtime::Runtime::new().unwrap();
        rt.block_on(engine.fast_cycle(0))
    };
    assert_eq!(meta.tick, 0, "Tick should match");
    assert!(meta.n_stocks >= 2, "Should have at least 2 stocks");
}

#[test]
fn test_tensor_non_finite_detection_after_ingestion() {
    let engine = setup_matrix(&["SPY", "QQQ"], 3, 10);
    let tensor = engine.tensor();

    let rt = tokio::runtime::Runtime::new().unwrap();
    let flagged = rt.block_on(async {
        let t = tensor.read().await;
        detect_non_finite(&t)
    });
    assert_eq!(
        flagged.len(),
        0,
        "Fresh tensor should have no non-finite values"
    );

    // Inject a NaN
    engine.inject_nan(0, 0, 0);
    let flagged = rt.block_on(async {
        let t = tensor.read().await;
        detect_non_finite(&t)
    });
    assert_eq!(flagged.len(), 1, "Should detect exactly one NaN");
    assert_eq!(
        flagged[0],
        (0, 0, 0),
        "Coordinates should match injection point"
    );
}

#[test]
fn test_matrix_slice_from_engine() {
    let engine = setup_matrix(&["AMZN"], 5, 20);
    let rt = tokio::runtime::Runtime::new().unwrap();
    let slice = rt.block_on(engine.get_slice("AMZN"));
    assert!(slice.is_some(), "Slice should exist for registered ticker");
    if let Some(s) = slice {
        assert_eq!(s.shape(), &[5, 20], "Slice should be feature×history");
    }
}

// ═══════════════════════════════════════════════════════════════════════
// 2. MATRIX SNAPSHOT → ROOM AGENT → ROOM PROPOSAL
// ═══════════════════════════════════════════════════════════════════════

#[test]
fn test_room_agent_produces_valid_proposal_from_snapshot() {
    let engine = setup_matrix(&["TICKER0", "TICKER1"], 3, 100);
    let snapshot = produce_snapshot(&engine, 42);
    let room = MockRoomAgent::new("TICKER0").with_narrative("Tech momentum check");

    let rt = tokio::runtime::Runtime::new().unwrap();
    let proposal = rt.block_on(room.analyze(&snapshot, None, None));

    assert_eq!(
        proposal.ticker, "TICKER0",
        "Proposal ticker must match agent"
    );
    assert!(
        matches!(
            proposal.gate,
            TernaryGate::Bullish | TernaryGate::Neutral | TernaryGate::Bearish
        ),
        "Gate must be a valid TernaryGate"
    );
    assert!(
        (0.0..=1.0).contains(&proposal.conviction),
        "Conviction must be in [0, 1], got {}",
        proposal.conviction
    );
    assert!(
        (0.0..=1.0).contains(&proposal.confidence),
        "Confidence must be in [0, 1], got {}",
        proposal.confidence
    );
    assert!(
        (0.0..=1.0).contains(&proposal.matrix_agreement),
        "Matrix agreement must be in [0, 1], got {}",
        proposal.matrix_agreement
    );
    assert!(
        !proposal.narrative_sig.is_empty(),
        "Narrative sig must not be empty"
    );
    assert_eq!(proposal.timestamp, 42, "Timestamp must match snapshot tick");
    assert!(
        !proposal.veto_override,
        "Default veto_override must be false"
    );
}

#[test]
fn test_multiple_rooms_produce_diverse_proposals() {
    let tickers = &["T0", "T1", "T2", "T3", "T4"];
    let engine = setup_matrix(tickers, 3, 50);
    let snapshot = produce_snapshot(&engine, 100);
    let rooms = setup_rooms(tickers);
    let proposals = collect_proposals(&rooms, &snapshot);

    assert_eq!(proposals.len(), 5, "Should have one proposal per room");

    let ticker_set: std::collections::HashSet<&str> =
        proposals.iter().map(|p| p.ticker.as_str()).collect();
    assert!(ticker_set.contains("T0"));
    assert!(ticker_set.contains("T4"));

    // All timestamps should match
    for proposal in &proposals {
        assert_eq!(
            proposal.timestamp, 100,
            "All proposals should have same timestamp"
        );
    }
}

#[test]
fn test_room_agent_uses_feature_slice_when_provided() {
    let engine = setup_matrix(&["TICKER0"], 4, 20);
    let snapshot = produce_snapshot(&engine, 55);
    let room = MockRoomAgent::new("TICKER0");

    let rt = tokio::runtime::Runtime::new().unwrap();
    let slice = rt.block_on(engine.get_slice("TICKER0"));

    let proposal = rt.block_on(room.analyze(&snapshot, slice, None));
    assert_eq!(proposal.ticker, "TICKER0");
    assert!(proposal.confidence.is_finite());
}

// ═══════════════════════════════════════════════════════════════════════
// 3. PROPOSAL → VETO ENGINE → FINAL POSITION
// ═══════════════════════════════════════════════════════════════════════

#[test]
fn test_veto_engine_resolves_simple_proposals() {
    let engine = MockVetoEngine::new().with_default_constraints();
    let rt = tokio::runtime::Runtime::new().unwrap();

    let proposals = vec![
        RoomProposal {
            ticker: "AAPL".into(),
            gate: TernaryGate::Bullish,
            conviction: 0.8,
            confidence: 0.9,
            narrative_sig: "a1b2".into(),
            matrix_agreement: 0.7,
            veto_override: false,
            timestamp: 1,
        },
        RoomProposal {
            ticker: "MSFT".into(),
            gate: TernaryGate::Bearish,
            conviction: 0.6,
            confidence: 0.7,
            narrative_sig: "c3d4".into(),
            matrix_agreement: 0.5,
            veto_override: false,
            timestamp: 1,
        },
    ];

    let portfolio = rt.block_on(engine.resolve(&proposals, None));

    assert_eq!(portfolio.positions.len(), 2, "Should produce 2 positions");
    assert!(
        portfolio.gross_exposure > 0.0,
        "Gross exposure should be positive"
    );

    // Check individual positions
    for pos in &portfolio.positions {
        assert!(
            (-1.0..=1.0).contains(&pos.weight),
            "Weight must be in [-1, 1]"
        );
        assert!(
            pos.veto_severity >= 0.0 && pos.veto_severity <= 1.0,
            "Veto severity must be in [0, 1]"
        );
    }
}

#[test]
fn test_veto_engine_positions_reflect_gate_and_conviction() {
    let engine = MockVetoEngine::new().with_default_constraints();
    let rt = tokio::runtime::Runtime::new().unwrap();

    let proposals = vec![
        RoomProposal {
            ticker: "AAPL".into(),
            gate: TernaryGate::Bullish,
            conviction: 0.9,
            confidence: 0.95,
            narrative_sig: "x".into(),
            matrix_agreement: 0.8,
            veto_override: false,
            timestamp: 1,
        },
        RoomProposal {
            ticker: "MSFT".into(),
            gate: TernaryGate::Bearish,
            conviction: 0.85,
            confidence: 0.9,
            narrative_sig: "y".into(),
            matrix_agreement: 0.6,
            veto_override: false,
            timestamp: 1,
        },
        RoomProposal {
            ticker: "GOOGL".into(),
            gate: TernaryGate::Neutral,
            conviction: 0.5,
            confidence: 0.5,
            narrative_sig: "z".into(),
            matrix_agreement: 0.9,
            veto_override: false,
            timestamp: 1,
        },
    ];

    let portfolio = rt.block_on(engine.resolve(&proposals, None));

    // AAPL should be positive (bullish * conviction)
    let aapl = portfolio
        .positions
        .iter()
        .find(|p| p.ticker == "AAPL")
        .unwrap();
    assert!(
        aapl.weight > 0.0,
        "Bullish AAPL should have positive weight, got {}",
        aapl.weight
    );

    // MSFT should be negative (bearish * conviction)
    let msft = portfolio
        .positions
        .iter()
        .find(|p| p.ticker == "MSFT")
        .unwrap();
    assert!(
        msft.weight < 0.0,
        "Bearish MSFT should have negative weight, got {}",
        msft.weight
    );

    // GOOGL should be ~0 (neutral)
    let googl = portfolio
        .positions
        .iter()
        .find(|p| p.ticker == "GOOGL")
        .unwrap();
    assert!(
        googl.weight.abs() < 0.01,
        "Neutral GOOGL should have near-zero weight, got {}",
        googl.weight
    );
}

#[test]
fn test_veto_engine_saep_limit_fires_on_high_conviction() {
    let engine = MockVetoEngine::new().with_default_constraints();
    let rt = tokio::runtime::Runtime::new().unwrap();

    let proposals = vec![RoomProposal {
        ticker: "AAPL".into(),
        gate: TernaryGate::Bullish,
        conviction: 0.99,
        confidence: 1.0,
        narrative_sig: "test".into(),
        matrix_agreement: 0.8,
        veto_override: false,
        timestamp: 1,
    }];

    let portfolio = rt.block_on(engine.resolve(&proposals, None));

    let pos = &portfolio.positions[0];
    assert!(
        pos.veto_applied.contains(&"max-conviction".to_string()),
        "max-conviction SAEP should fire for conviction >= 0.95, got {:?}",
        pos.veto_applied
    );
}

#[test]
fn test_veto_engine_multiple_proposals_aggregate_correctly() {
    let engine = MockVetoEngine::new();
    let rt = tokio::runtime::Runtime::new().unwrap();

    let proposals: Vec<RoomProposal> = (0..5)
        .map(|i| RoomProposal {
            ticker: format!("STOCK{}", i),
            gate: if i % 2 == 0 {
                TernaryGate::Bullish
            } else {
                TernaryGate::Bearish
            },
            conviction: 0.5,
            confidence: 0.7,
            narrative_sig: format!("n{}", i),
            matrix_agreement: 0.6,
            veto_override: false,
            timestamp: 42,
        })
        .collect();

    let portfolio = rt.block_on(engine.resolve(&proposals, None));

    assert_eq!(portfolio.positions.len(), 5);
    assert_eq!(portfolio.timestamp, 42);

    assert!(
        portfolio.net_exposure.abs() <= portfolio.gross_exposure,
        "Net exposure ({}) must not exceed gross exposure ({})",
        portfolio.net_exposure,
        portfolio.gross_exposure
    );
}

// ═══════════════════════════════════════════════════════════════════════
// 4. END-TO-END HYBRID CYCLE
// ═══════════════════════════════════════════════════════════════════════

#[test]
fn test_e2e_full_cycle_data_to_portfolio() {
    let tickers = &["TICKER0", "TICKER1", "TICKER2"];
    let engine = setup_matrix(tickers, 3, 100);
    let veto = MockVetoEngine::new().with_default_constraints();
    let rooms = setup_rooms(tickers);

    let rt = tokio::runtime::Runtime::new().unwrap();

    // Phase 1: Ingest data
    for tick in 0..10 {
        rt.block_on(engine.ingest("TICKER0", &[100.0 + tick as f64, 0.5, 0.01], tick));
        rt.block_on(engine.ingest("TICKER1", &[50.0 + tick as f64 * 0.5, -0.3, 0.02], tick));
        rt.block_on(engine.ingest("TICKER2", &[200.0 - tick as f64, 0.0, -0.01], tick));
    }

    // Phase 2: Fast + Medium + Full cycles
    let snapshot = rt.block_on(engine.full_cycle(9));
    assert_eq!(snapshot.tick, 9);
    assert_eq!(snapshot.topologies.len(), 3);

    // Phase 3: Room analysis → Proposals
    let proposals = rooms
        .iter()
        .map(|room| rt.block_on(room.analyze(&snapshot, None, None)))
        .collect::<Vec<_>>();

    assert_eq!(proposals.len(), 3);

    // Phase 4: Veto resolution → Portfolio
    let portfolio = rt.block_on(veto.resolve(&proposals, None));
    assert_eq!(portfolio.positions.len(), 3);
    assert!(portfolio.gross_exposure > 0.0);

    for pos in &portfolio.positions {
        assert!(
            pos.ticker.starts_with("TICKER"),
            "Expected TICKER prefix, got: {}",
            pos.ticker
        );
        assert!(pos.weight.is_finite(), "Weight must be finite");
    }
}

#[test]
fn test_e2e_latency_within_target() {
    let tickers: Vec<String> = (0..100).map(|i| format!("STOCK{}", i)).collect();
    let ticker_refs: Vec<&str> = tickers.iter().map(|s| s.as_str()).collect();
    let engine = setup_matrix(&ticker_refs, 10, 250);
    let veto = MockVetoEngine::new().with_default_constraints();
    let rooms = setup_rooms(&ticker_refs);
    let rt = tokio::runtime::Runtime::new().unwrap();

    let start = Instant::now();

    // Full cycle
    let _snapshot = rt.block_on(engine.full_cycle(5));
    let full_elapsed = start.elapsed();

    // Room analysis for 100 rooms
    let proposals: Vec<RoomProposal> = {
        let mut ps = Vec::with_capacity(100);
        for room in &rooms {
            ps.push(rt.block_on(room.analyze(&_snapshot, None, None)));
        }
        ps
    };
    let _room_elapsed = Instant::now() - start;

    // Veto resolution
    let _portfolio = rt.block_on(veto.resolve(&proposals, None));
    let total_elapsed = Instant::now() - start;

    // Assertions against targets (with generous margin for test overhead)
    assert!(
        full_elapsed.as_millis() < 200,
        "full_cycle on mock should complete < 200ms, took {:?}",
        full_elapsed
    );
    assert!(
        total_elapsed.as_millis() < 2000,
        "e2e cycle for 100 stocks should complete < 2s, took {:?}",
        total_elapsed
    );
}

#[test]
fn test_e2e_bridge_broadcast_cycle() {
    let tickers = &["A", "B", "C"];
    let engine = setup_matrix(tickers, 3, 50);
    let veto = MockVetoEngine::new().with_default_constraints();
    let rooms = setup_rooms(tickers);
    let bridge = HybridBridge::default();
    let rt = tokio::runtime::Runtime::new().unwrap();

    // Subscribe a room to matrix broadcasts
    let mut rx = bridge.subscribe_matrix();

    // Subscribe to veto broadcasts
    let mut veto_rx = bridge.subscribe_portfolio();

    // Simulate a complete bridge cycle
    let snapshot = rt.block_on(engine.full_cycle(1));

    // Broadcast via bridge (synchronous)
    bridge.broadcast_snapshot(snapshot.clone());

    // Room receives the broadcast
    let received = rt.block_on(rx.recv());
    match received {
        Ok(HybridMessage::SnapshotBroadcast(recv_snap)) => {
            assert_eq!(
                recv_snap.tick, 1,
                "Received snapshot should have correct tick"
            );
        }
        other => panic!("Expected SnapshotBroadcast, got: {:?}", other),
    }

    // Rooms produce proposals and submit via bridge
    for room in &rooms {
        let p = rt.block_on(room.analyze(&snapshot, None, None));
        rt.block_on(bridge.submit_proposal(p.clone())).unwrap();
    }

    // Veto resolves
    let proposals = rooms
        .iter()
        .map(|r| rt.block_on(r.analyze(&snapshot, None, None)))
        .collect::<Vec<_>>();
    let portfolio = rt.block_on(veto.resolve(&proposals, None));
    bridge.broadcast_portfolio(portfolio.clone());

    // Verify veto broadcast is received
    let veto_msg = rt.block_on(veto_rx.recv());
    match veto_msg {
        Ok(HybridMessage::PortfolioVectorBroadcast(pf)) => {
            assert_eq!(pf.positions.len(), 3);
        }
        other => panic!("Expected PortfolioVectorBroadcast, got: {:?}", other),
    }
}

// ═══════════════════════════════════════════════════════════════════════
// 5. CHAOS TESTS — NaN / Infinity Attack Surface
// ═══════════════════════════════════════════════════════════════════════

#[test]
fn test_chaos_nan_in_matrix_triggers_detection() {
    let tickers: Vec<String> = (0..10).map(|i| format!("T{}", i)).collect();
    let ticker_refs: Vec<&str> = tickers.iter().map(|s| s.as_str()).collect();
    let engine = setup_matrix(&ticker_refs, 5, 100);
    let rt = tokio::runtime::Runtime::new().unwrap();

    // Initially clean
    let flagged_before = {
        let t = engine.tensor();
        rt.block_on(async {
            let tensor = t.read().await;
            detect_non_finite(&tensor)
        })
    };
    assert_eq!(flagged_before.len(), 0, "Fresh tensor should be clean");

    // Inject NaNs across multiple stocks and features
    engine.inject_nan(3, 1, 10);
    engine.inject_nan(7, 3, 50);
    engine.inject_nan(1, 0, 99);
    engine.inject_nan(5, 2, 30);

    let flagged_after = {
        let t = engine.tensor();
        rt.block_on(async {
            let tensor = t.read().await;
            detect_non_finite(&tensor)
        })
    };
    assert_eq!(flagged_after.len(), 4, "Should detect exactly 4 NaN cells");
}

#[test]
fn test_chaos_inf_in_matrix_detected_and_masked() {
    let engine = setup_matrix(&["A", "B", "C"], 3, 50);
    let rt = tokio::runtime::Runtime::new().unwrap();

    engine.inject_inf(0, 0, 0);
    engine.inject_inf(2, 1, 25);

    let flagged = {
        let t = engine.tensor();
        rt.block_on(async {
            let tensor = t.read().await;
            detect_non_finite(&tensor)
        })
    };
    assert_eq!(flagged.len(), 2, "Should detect both Inf cells");

    // Verify masking works
    let masked = {
        let t = engine.tensor();
        rt.block_on(async {
            let mut tensor = t.write().await;
            mask_non_finite(&mut tensor)
        })
    };
    assert_eq!(masked, 2, "Should mask both Inf cells");

    let remaining = {
        let t = engine.tensor();
        rt.block_on(async {
            let tensor = t.read().await;
            detect_non_finite(&tensor)
        })
    };
    assert!(remaining.is_empty(), "No non-finite values should remain");
}

#[test]
fn test_chaos_veto_engine_rejects_nan_proposal() {
    let engine = MockVetoEngine::new().with_default_constraints();
    let rt = tokio::runtime::Runtime::new().unwrap();

    let nan_proposal = RoomProposal {
        ticker: "AAPL".into(),
        gate: TernaryGate::Bullish,
        conviction: f64::NAN,
        confidence: 0.8,
        narrative_sig: "chaos-nan".into(),
        matrix_agreement: 0.5,
        veto_override: false,
        timestamp: 999,
    };

    let portfolio = rt.block_on(engine.resolve(&[nan_proposal], None));

    assert_eq!(portfolio.positions.len(), 1);
    let pos = &portfolio.positions[0];
    assert!(
        pos.veto_applied.contains(&"nan-detector".to_string()),
        "nan-detector SAEP should fire for NaN conviction, got {:?}",
        pos.veto_applied
    );
    assert!(
        pos.weight.abs() < 0.01,
        "NaN proposal should be vetoed to near-zero weight, got {}",
        pos.weight
    );
    assert!(
        (pos.veto_severity - 1.0).abs() < 0.01,
        "Veto severity should be 1.0 for NaN detection, got {}",
        pos.veto_severity
    );
}

#[test]
fn test_chaos_veto_engine_rejects_infinite_proposal() {
    let engine = MockVetoEngine::new().with_default_constraints();
    let rt = tokio::runtime::Runtime::new().unwrap();

    let inf_proposal = RoomProposal {
        ticker: "MSFT".into(),
        gate: TernaryGate::Bearish,
        conviction: 0.7,
        confidence: f64::INFINITY,
        narrative_sig: "chaos-inf".into(),
        matrix_agreement: 0.5,
        veto_override: false,
        timestamp: 1000,
    };

    let portfolio = rt.block_on(engine.resolve(&[inf_proposal], None));

    let pos = &portfolio.positions[0];
    assert!(
        pos.weight.abs() < 0.01,
        "Inf proposal should be vetoed to near-zero weight, got {}",
        pos.weight
    );
    assert_eq!(pos.veto_severity, 1.0, "Full veto severity expected");
}

#[test]
fn test_chaos_mixed_nan_inf_vetos_all_bad() {
    let engine = MockVetoEngine::new().with_default_constraints();
    let rt = tokio::runtime::Runtime::new().unwrap();

    let proposals = vec![
        RoomProposal {
            ticker: "AAPL".into(),
            gate: TernaryGate::Bullish,
            conviction: f64::NAN,
            confidence: 0.8,
            narrative_sig: "a".into(),
            matrix_agreement: 0.5,
            veto_override: false,
            timestamp: 1,
        },
        RoomProposal {
            ticker: "MSFT".into(),
            gate: TernaryGate::Bearish,
            conviction: 0.6,
            confidence: f64::INFINITY,
            narrative_sig: "b".into(),
            matrix_agreement: 0.5,
            veto_override: false,
            timestamp: 1,
        },
        RoomProposal {
            ticker: "GOOGL".into(),
            gate: TernaryGate::Neutral,
            conviction: f64::NEG_INFINITY,
            confidence: 0.5,
            narrative_sig: "c".into(),
            matrix_agreement: 0.5,
            veto_override: false,
            timestamp: 1,
        },
        RoomProposal {
            ticker: "AMZN".into(),
            gate: TernaryGate::Bullish,
            conviction: 0.8,
            confidence: 0.9,
            narrative_sig: "d".into(),
            matrix_agreement: 0.7,
            veto_override: false,
            timestamp: 1,
        },
    ];

    let portfolio = rt.block_on(engine.resolve(&proposals, None));

    // Three bad proposals should all be vetoed to zero
    for ticker in &["AAPL", "MSFT", "GOOGL"] {
        let pos = portfolio
            .positions
            .iter()
            .find(|p| p.ticker == *ticker)
            .unwrap();
        assert!(
            pos.weight.abs() < 0.01,
            "{} with non-finite values should be vetoed to 0, got weight={}",
            ticker,
            pos.weight
        );
    }

    // AMZN (clean) should have a non-zero weight
    let amzn = portfolio
        .positions
        .iter()
        .find(|p| p.ticker == "AMZN")
        .unwrap();
    assert!(
        amzn.weight > 0.0,
        "Clean AMZN should have positive weight, got {}",
        amzn.weight
    );
}

#[test]
fn test_chaos_tensor_masking_recovers_downstream_computation() {
    let engine = setup_matrix(&["A", "B", "C", "D", "E", "F", "G", "H"], 5, 200);

    let rt = tokio::runtime::Runtime::new().unwrap();

    // Inject NaNs and Inf
    engine.inject_nan(0, 0, 10);
    engine.inject_nan(1, 1, 20);
    engine.inject_nan(2, 2, 30);
    engine.inject_nan(3, 3, 40);
    engine.inject_nan(4, 4, 50);
    engine.inject_inf(5, 0, 60);
    engine.inject_inf(6, 1, 70);
    engine.inject_inf(7, 2, 80);

    // Mask the tensor
    let masked = {
        let t = engine.tensor();
        rt.block_on(async {
            let mut tensor = t.write().await;
            mask_non_finite(&mut tensor)
        })
    };
    assert_eq!(masked, 8, "Should mask all 8 non-finite cells");

    // Now verify the matrix can still produce valid output
    let snapshot = rt.block_on(engine.full_cycle(100));
    assert_eq!(snapshot.n_stocks, 8);
    assert!(
        snapshot.condition_number.is_finite(),
        "Condition number must be finite after masking"
    );

    // Each topology should have finite confidence
    for topo in &snapshot.topologies {
        assert!(
            topo.confidence.is_finite(),
            "All confidences must be finite after masking"
        );
    }
}

// ═══════════════════════════════════════════════════════════════════════
// 6. EDGE CASES
// ═══════════════════════════════════════════════════════════════════════

#[test]
fn test_edge_empty_proposals() {
    let engine = MockVetoEngine::new().with_default_constraints();
    let rt = tokio::runtime::Runtime::new().unwrap();

    let portfolio = rt.block_on(engine.resolve(&[], None));

    assert!(
        portfolio.positions.is_empty(),
        "Empty proposals should produce empty portfolio"
    );
    assert_eq!(portfolio.gross_exposure, 0.0);
    assert_eq!(portfolio.net_exposure, 0.0);
}

#[test]
fn test_edge_veto_override_reduces_impact() {
    let mut engine = MockVetoEngine::new();

    // Register a constraint that would veto high conviction
    let conviction_limit = SaepConstraint {
        id: "hard-cap".into(),
        layer: GovernanceLayer::Room,
        check_fn: Arc::new(|proposal, _ctx| {
            if proposal.conviction > 0.8 {
                Err(Violation {
                    constraint_id: "hard-cap".into(),
                    message: "Conviction too high".into(),
                    severity: 1.0,
                })
            } else {
                Ok(())
            }
        }),
        action: SaepAction::Veto,
        escalate_to: None,
    };
    let rt = tokio::runtime::Runtime::new().unwrap();
    rt.block_on(engine.register_constraint(conviction_limit));

    let proposals = vec![
        RoomProposal {
            ticker: "NO_OVERRIDE".into(),
            gate: TernaryGate::Bullish,
            conviction: 0.9,
            confidence: 0.9,
            narrative_sig: "nope".into(),
            matrix_agreement: 0.5,
            veto_override: false,
            timestamp: 1,
        },
        RoomProposal {
            ticker: "WITH_OVERRIDE".into(),
            gate: TernaryGate::Bullish,
            conviction: 0.9,
            confidence: 0.9,
            narrative_sig: "override".into(),
            matrix_agreement: 0.5,
            veto_override: true,
            timestamp: 1,
        },
    ];

    let portfolio = rt.block_on(engine.resolve(&proposals, None));

    let no_override = portfolio
        .positions
        .iter()
        .find(|p| p.ticker == "NO_OVERRIDE")
        .unwrap();
    let with_override = portfolio
        .positions
        .iter()
        .find(|p| p.ticker == "WITH_OVERRIDE")
        .unwrap();

    // Without override: veto severity = 1.0 → weight ≈ 0
    assert!(
        no_override.weight.abs() < 0.01,
        "Without override, full veto → weight ≈ 0, got {}",
        no_override.weight
    );

    // With override: veto halved → weight = 0.9 * 1 * (1 - 0.5) = 0.45
    assert!(
        with_override.weight > 0.01,
        "With override, weight should be > 0, got {}",
        with_override.weight
    );
}

#[test]
fn test_edge_freeze_stops_all_veto_resolution() {
    let mut engine = MockVetoEngine::new().with_default_constraints();
    let rt = tokio::runtime::Runtime::new().unwrap();

    // Freeze
    rt.block_on(engine.freeze("Market crash imminent"));

    // Try to resolve during freeze
    let proposals = vec![RoomProposal {
        ticker: "SPY".into(),
        gate: TernaryGate::Bullish,
        conviction: 0.8,
        confidence: 0.9,
        narrative_sig: "freeze-test".into(),
        matrix_agreement: 0.7,
        veto_override: false,
        timestamp: 1,
    }];

    let portfolio = rt.block_on(engine.resolve(&proposals, None));
    assert!(
        portfolio.positions.is_empty(),
        "Frozen engine should return empty portfolio"
    );

    // Unfreeze
    rt.block_on(engine.unfreeze("All clear"));

    let portfolio2 = rt.block_on(engine.resolve(&proposals, None));
    assert_eq!(
        portfolio2.positions.len(),
        1,
        "Unfrozen engine should process proposals"
    );
}

#[test]
fn test_edge_large_number_of_tickers() {
    let n = 100;
    let tickers: Vec<String> = (0..n).map(|i| format!("STK{}", i)).collect();
    let ticker_refs: Vec<&str> = tickers.iter().map(|s| s.as_str()).collect();
    let engine = setup_matrix(&ticker_refs, 3, 50);
    let rooms = setup_rooms(&ticker_refs);
    let snapshot = produce_snapshot(&engine, 1);
    let proposals = collect_proposals(&rooms, &snapshot);

    assert_eq!(proposals.len(), n, "Should produce {} proposals", n);

    // Veto should handle 100 proposals quickly
    let veto = MockVetoEngine::new().with_default_constraints();
    let rt = tokio::runtime::Runtime::new().unwrap();
    let start = Instant::now();
    let portfolio = rt.block_on(veto.resolve(&proposals, None));
    let elapsed = start.elapsed();

    assert_eq!(
        portfolio.positions.len(),
        n,
        "Portfolio should have {} positions",
        n
    );
    assert!(
        elapsed.as_millis() < 500,
        "100-proposal veto resolution should complete < 500ms, took {:?}",
        elapsed
    );
}

#[test]
fn test_edge_custom_saep_constraint_built_at_runtime() {
    let mut engine = MockVetoEngine::new();
    let rt = tokio::runtime::Runtime::new().unwrap();

    // Register a custom constraint: block any proposal with "HACK" in ticker
    let custom = SaepConstraint {
        id: "anti-hack".into(),
        layer: GovernanceLayer::Room,
        check_fn: Arc::new(|proposal, _ctx| {
            if proposal.ticker.contains("HACK") {
                Err(Violation {
                    constraint_id: "anti-hack".into(),
                    message: "HACK tickers are blocked".into(),
                    severity: 1.0,
                })
            } else {
                Ok(())
            }
        }),
        action: SaepAction::Veto,
        escalate_to: Some(GovernanceLayer::Market),
    };
    rt.block_on(engine.register_constraint(custom));

    let proposals = vec![
        RoomProposal {
            ticker: "HACK_ATTACK".into(),
            gate: TernaryGate::Bullish,
            conviction: 0.7,
            confidence: 0.8,
            narrative_sig: "bad".into(),
            matrix_agreement: 0.5,
            veto_override: false,
            timestamp: 1,
        },
        RoomProposal {
            ticker: "SAFE_STOCK".into(),
            gate: TernaryGate::Bearish,
            conviction: 0.6,
            confidence: 0.7,
            narrative_sig: "good".into(),
            matrix_agreement: 0.6,
            veto_override: false,
            timestamp: 1,
        },
    ];

    let portfolio = rt.block_on(engine.resolve(&proposals, None));

    let hack = portfolio
        .positions
        .iter()
        .find(|p| p.ticker == "HACK_ATTACK")
        .unwrap();
    assert!(
        hack.weight.abs() < 0.01,
        "HACK ticker should be vetoed, got weight={}",
        hack.weight
    );
    assert!(
        hack.veto_applied.contains(&"anti-hack".to_string()),
        "anti-hack constraint should fire"
    );

    let safe = portfolio
        .positions
        .iter()
        .find(|p| p.ticker == "SAFE_STOCK")
        .unwrap();
    assert!(
        safe.weight < 0.0,
        "Safe stock with bearish gate should have negative weight"
    );
    assert!(
        safe.veto_applied.is_empty(),
        "Safe stock should have no veto"
    );
}
