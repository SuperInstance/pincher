//! Integration tests for the Hybrid Bridge CLI.
//!
//! These tests exercise the `HybridCli` methods against a real `HybridBridge`
//! instance, verifying that commands produce correct output and handle error
//! conditions gracefully.

use std::sync::Arc;
use std::time::Duration;

use clap::Parser;
use hybrid_bridge::cli::{run_hybrid_cli, GateArg, HybridCli, HybridCliCommand, HybridSubcommand};
use hybrid_bridge::HybridBridge;
use hybrid_bridge::TernaryGate;

// ─────────────────────────────────────────────────────────────────────
// Smoke Tests — Basic Construction
// ─────────────────────────────────────────────────────────────────────

#[test]
fn test_cli_construction_default() {
    let bridge = Arc::new(HybridBridge::new());
    let cli = HybridCli::new(bridge);
    let _ = cli; // does not panic
}

#[test]
fn test_cli_construction_with_custom_timeout() {
    let bridge = Arc::new(HybridBridge::new());
    let cli = HybridCli::new(bridge).with_timeout(Duration::from_secs(60));
    let _ = cli;
}

#[test]
fn test_cli_command_parser_status() {
    // Simulate command-line parsing: `hybrid status`
    let cmd = HybridCliCommand::try_parse_from(["hybrid", "status"])
        .expect("`hybrid status` should parse");
    assert!(matches!(cmd.command, HybridSubcommand::Status));
}

#[test]
fn test_cli_command_parser_inject() {
    // `hybrid inject AAPL --features 0.1,0.5,-0.3`
    let cmd = HybridCliCommand::try_parse_from([
        "hybrid",
        "inject",
        "AAPL",
        "--features",
        "0.1,0.5,-0.3",
    ])
    .expect("`hybrid inject` should parse");
    match cmd.command {
        HybridSubcommand::Inject { ticker, features } => {
            assert_eq!(ticker, "AAPL");
            assert_eq!(features, vec![0.1, 0.5, -0.3]);
        }
        _ => panic!("Expected Inject subcommand"),
    }
}

#[test]
fn test_cli_command_parser_inject_short_args() {
    // `hybrid inject AAPL -f 0.0,1.0,-1.0` with short form
    let cmd = HybridCliCommand::try_parse_from(["hybrid", "inject", "AAPL", "-f", "0.0,1.0,-1.0"])
        .expect("`hybrid inject` with short args should parse");
    match cmd.command {
        HybridSubcommand::Inject { ticker, features } => {
            assert_eq!(ticker, "AAPL");
            assert_eq!(features, vec![0.0, 1.0, -1.0]);
        }
        _ => panic!("Expected Inject subcommand"),
    }
}

#[test]
fn test_cli_command_parser_snapshot() {
    // `hybrid snapshot`
    let cmd = HybridCliCommand::try_parse_from(["hybrid", "snapshot"])
        .expect("`hybrid snapshot` should parse");
    assert!(matches!(cmd.command, HybridSubcommand::Snapshot { .. }));

    // `hybrid snapshot --timeout 30`
    let cmd = HybridCliCommand::try_parse_from(["hybrid", "snapshot", "--timeout", "30"])
        .expect("`hybrid snapshot --timeout 30` should parse");
    match cmd.command {
        HybridSubcommand::Snapshot { timeout } => {
            assert_eq!(timeout, 30);
        }
        _ => panic!("Expected Snapshot subcommand"),
    }

    // `hybrid snapshot -t 5`
    let cmd = HybridCliCommand::try_parse_from(["hybrid", "snapshot", "-t", "5"])
        .expect("`hybrid snapshot -t 5` should parse");
    match cmd.command {
        HybridSubcommand::Snapshot { timeout } => {
            assert_eq!(timeout, 5);
        }
        _ => panic!("Expected Snapshot subcommand"),
    }
}

#[test]
fn test_cli_command_parser_propose() {
    // `hybrid propose AAPL bullish`
    let cmd = HybridCliCommand::try_parse_from(["hybrid", "propose", "AAPL", "bullish"])
        .expect("`hybrid propose` should parse");
    match cmd.command {
        HybridSubcommand::Propose { ticker, gate, .. } => {
            assert_eq!(ticker, "AAPL");
            assert!(matches!(gate, GateArg::Bullish));
        }
        _ => panic!("Expected Propose subcommand"),
    }
}

#[test]
fn test_cli_command_parser_propose_full() {
    // `hybrid propose TSLA bearish --conviction 0.6 --confidence 0.5`
    let cmd = HybridCliCommand::try_parse_from([
        "hybrid", "propose", "TSLA", "bearish", "-w", "0.6", "-C", "0.5",
    ])
    .expect("`hybrid propose` with all flags should parse");
    match cmd.command {
        HybridSubcommand::Propose {
            ticker,
            gate,
            conviction,
            confidence,
        } => {
            assert_eq!(ticker, "TSLA");
            assert!(matches!(gate, GateArg::Bearish));
            assert!((conviction - 0.6).abs() < 1e-6);
            assert!((confidence - 0.5).abs() < 1e-6);
        }
        _ => panic!("Expected Propose subcommand"),
    }
}

#[test]
fn test_cli_command_parser_freeze() {
    // `hybrid freeze "Market circuit breaker"`
    let cmd = HybridCliCommand::try_parse_from(["hybrid", "freeze", "Market circuit breaker"])
        .expect("`hybrid freeze` should parse");
    match cmd.command {
        HybridSubcommand::Freeze { reason } => {
            assert_eq!(reason, "Market circuit breaker");
        }
        _ => panic!("Expected Freeze subcommand"),
    }
}

#[test]
fn test_cli_command_parser_unfreeze() {
    // `hybrid unfreeze`
    let cmd = HybridCliCommand::try_parse_from(["hybrid", "unfreeze"])
        .expect("`hybrid unfreeze` should parse");
    assert!(matches!(cmd.command, HybridSubcommand::Unfreeze { .. }));

    // `hybrid unfreeze "All clear"`
    let cmd = HybridCliCommand::try_parse_from(["hybrid", "unfreeze", "All clear"])
        .expect("`hybrid unfreeze` with reason should parse");
    match cmd.command {
        HybridSubcommand::Unfreeze { reason } => {
            assert_eq!(reason, "All clear");
        }
        _ => panic!("Expected Unfreeze subcommand"),
    }
}

// ─────────────────────────────────────────────────────────────────────
// Operation Tests — Against Real Bridge
// ─────────────────────────────────────────────────────────────────────

#[tokio::test]
async fn test_status_after_events() {
    let bridge = Arc::new(HybridBridge::new());
    let cli = HybridCli::new(bridge.clone());

    // Emit some activity to populate metrics
    bridge.emit_system_event("test".into(), "integration test".into());
    let _rx = bridge.subscribe_matrix();
    bridge.broadcast_snapshot(dummy_snapshot(1, 10));

    let result = cli.status().await;
    assert!(result.is_ok(), "Status should succeed: {:?}", result.err());
}

#[tokio::test]
async fn test_inject_and_verify_bridge_receives() {
    let bridge = Arc::new(HybridBridge::new());
    let cli = HybridCli::new(bridge.clone());

    // Subscribe before inject so we catch the message
    let mut rx = bridge.subscribe_matrix();

    cli.inject("GOOGL", &[0.5, -0.2, 0.8]).unwrap();

    // Receive the slice update from the bridge
    let msg = tokio::time::timeout(Duration::from_secs(2), rx.recv())
        .await
        .expect("Should receive slice update within timeout")
        .expect("Channel should not be closed");

    match msg {
        HybridMessage::SliceUpdate { ticker, data } => {
            assert_eq!(ticker, "GOOGL");
            assert_eq!(data.shape(), &[3, 1]); // 3 features, 1 time step
        }
        other => {
            panic!("Expected SliceUpdate, got: {:?}", other.variant_name());
        }
    }

    // Verify metrics reflect the update
    let metrics = bridge.metrics().snapshot();
    assert_eq!(metrics.slice_updates, 1);
}

#[tokio::test]
async fn test_propose_and_verify_bridge_receives() {
    let bridge = Arc::new(HybridBridge::new());
    let cli = HybridCli::new(bridge.clone());

    cli.propose("MSFT", TernaryGate::Bullish, 0.9, 0.85)
        .await
        .unwrap();

    // Take the proposal receiver and check
    let mut rx = bridge
        .take_proposal_receiver()
        .await
        .expect("Should take proposal receiver");

    let proposal = tokio::time::timeout(Duration::from_secs(2), rx.recv())
        .await
        .expect("Should receive proposal within timeout")
        .expect("Channel should not be closed");

    assert_eq!(proposal.ticker, "MSFT");
    assert_eq!(proposal.gate, TernaryGate::Bullish);
    assert!((proposal.conviction - 0.9).abs() < 1e-6);
    assert!((proposal.confidence - 0.85).abs() < 1e-6);
    assert!(!proposal.narrative_sig.is_empty());
    assert!(proposal.timestamp > 0);
}

#[tokio::test]
async fn test_freeze_emits_system_event() {
    let bridge = Arc::new(HybridBridge::new());
    let cli = HybridCli::new(bridge.clone());

    let mut event_rx = bridge.subscribe_system_events();

    cli.freeze("Integration test freeze").unwrap();

    let msg = tokio::time::timeout(Duration::from_secs(2), event_rx.recv())
        .await
        .expect("Should receive system event within timeout")
        .expect("Channel should not be closed");

    match msg {
        HybridMessage::SystemEvent { kind, payload } => {
            assert_eq!(kind, "freeze");
            assert_eq!(payload, "Integration test freeze");
        }
        other => {
            panic!("Expected SystemEvent, got: {:?}", other.variant_name());
        }
    }
}

#[tokio::test]
async fn test_unfreeze_emits_system_event() {
    let bridge = Arc::new(HybridBridge::new());
    let cli = HybridCli::new(bridge.clone());

    let mut event_rx = bridge.subscribe_system_events();

    // Freeze first, then unfreeze
    let _ = cli.freeze("pre-freeze");
    // Consume the freeze event
    let _ = tokio::time::timeout(Duration::from_secs(2), event_rx.recv()).await;

    cli.unfreeze("Integration test unfreeze").unwrap();

    let msg = tokio::time::timeout(Duration::from_secs(2), event_rx.recv())
        .await
        .expect("Should receive unfreeze event within timeout")
        .expect("Channel should not be closed");

    match msg {
        HybridMessage::SystemEvent { kind, payload } => {
            assert_eq!(kind, "unfreeze");
            assert_eq!(payload, "Integration test unfreeze");
        }
        other => {
            panic!(
                "Expected SystemEvent(unfreeze), got: {:?}",
                other.variant_name()
            );
        }
    }
}

#[tokio::test]
async fn test_snapshot_broadcast_received() {
    let bridge = Arc::new(HybridBridge::new());
    let cli = HybridCli::new(bridge.clone());

    // Spawn a background task that broadcasts a snapshot after a short delay
    let bg_bridge = bridge.clone();
    tokio::spawn(async move {
        tokio::time::sleep(Duration::from_millis(50)).await;
        bg_bridge.broadcast_snapshot(MatrixSnapshot {
            tick: 100,
            n_stocks: 50,
            eigenvalues: vec![0.80, 0.12, 0.05, 0.02, 0.01],
            eigenvectors: ndarray::Array2::from_shape_vec(
                (2, 3),
                vec![0.5, 0.5, 0.5, -0.5, 0.5, -0.5],
            )
            .unwrap(),
            topologies: vec![],
            universe_betti: [7, 3, 1],
            regime: "fragmentation".into(),
            condition_number: 4.7,
        });
    });

    let result = cli.snapshot(5).await;
    assert!(
        result.is_ok(),
        "Snapshot should succeed when broadcast arrives: {:?}",
        result.err()
    );
}

#[tokio::test]
async fn test_dispatch_all_subcommands_ok() {
    let bridge = Arc::new(HybridBridge::new());
    let cli = HybridCli::new(bridge.clone());

    // Status
    let result = run_hybrid_cli(
        &cli,
        HybridCliCommand {
            command: HybridSubcommand::Status,
        },
    )
    .await;
    assert!(result.is_ok(), "Status dispatch: {:?}", result.err());

    // Inject
    let result = run_hybrid_cli(
        &cli,
        HybridCliCommand {
            command: HybridSubcommand::Inject {
                ticker: "NVDA".into(),
                features: vec![0.9, -0.1, 0.4],
            },
        },
    )
    .await;
    assert!(result.is_ok(), "Inject dispatch: {:?}", result.err());

    // Propose
    let result = run_hybrid_cli(
        &cli,
        HybridCliCommand {
            command: HybridSubcommand::Propose {
                ticker: "AMD".into(),
                gate: GateArg::Bullish,
                conviction: 0.7,
                confidence: 0.6,
            },
        },
    )
    .await;
    assert!(result.is_ok(), "Propose dispatch: {:?}", result.err());

    // Freeze
    let result = run_hybrid_cli(
        &cli,
        HybridCliCommand {
            command: HybridSubcommand::Freeze {
                reason: "Integration test freeze".into(),
            },
        },
    )
    .await;
    assert!(result.is_ok(), "Freeze dispatch: {:?}", result.err());

    // Unfreeze
    let result = run_hybrid_cli(
        &cli,
        HybridCliCommand {
            command: HybridSubcommand::Unfreeze {
                reason: "Integration test unfreeze".into(),
            },
        },
    )
    .await;
    assert!(result.is_ok(), "Unfreeze dispatch: {:?}", result.err());
}

#[tokio::test]
async fn test_inject_negative_values() {
    let bridge = Arc::new(HybridBridge::new());
    let cli = HybridCli::new(bridge.clone());

    let mut rx = bridge.subscribe_matrix();

    // Inject with negative and zero values
    let features = vec![-0.5, 0.0, 1.0, -1.0, 0.333];
    cli.inject("TEST", &features).unwrap();

    let msg = tokio::time::timeout(Duration::from_secs(2), rx.recv())
        .await
        .expect("Should receive slice update")
        .expect("Channel should not be closed");

    match msg {
        HybridMessage::SliceUpdate { ticker, data } => {
            assert_eq!(ticker, "TEST");
            assert_eq!(data.shape(), &[5, 1]); // 5 features, 1 time step
            assert!((data[[0, 0]] - (-0.5_f32)).abs() < 1e-6);
            assert!((data[[1, 0]] - 0.0_f32).abs() < 1e-6);
            assert!((data[[2, 0]] - 1.0_f32).abs() < 1e-6);
            assert!((data[[3, 0]] - (-1.0_f32)).abs() < 1e-6);
            assert!((data[[4, 0]] - 0.333_f32).abs() < 1e-6);
        }
        other => {
            panic!("Expected SliceUpdate, got: {:?}", other.variant_name());
        }
    }
}

// ─────────────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────────────

use hybrid_bridge::HybridMessage;
use hybrid_bridge::MatrixSnapshot;

/// Create a minimal dummy snapshot for test use.
fn dummy_snapshot(tick: u64, n_stocks: usize) -> MatrixSnapshot {
    MatrixSnapshot {
        tick,
        n_stocks,
        eigenvalues: vec![],
        eigenvectors: ndarray::Array2::from_shape_vec((0, 0), vec![]).unwrap(),
        topologies: vec![],
        universe_betti: [0, 0, 0],
        regime: "test".into(),
        condition_number: 0.0,
    }
}
