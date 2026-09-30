//! Mock Veto Engine for integration tests.
//!
//! Provides an in-memory implementation of the `VetoEngine` trait that
//! applies configurable SAEP constraints for testing.

use crate::engine::VetoEngine;
use crate::types::{
    FinalPosition, GovernanceLayer, PortfolioVector, RoomProposal, SaepAction, SaepConstraint,
    TernaryGate, Violation,
};
use async_trait::async_trait;
use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{info, warn};

/// A mock veto engine for testing.
///
/// Supports registering SAEP constraints, freeze/unfreeze, and
/// proposal resolution with built-in NaN/Inf detection.
pub struct MockVetoEngine {
    constraints: Vec<SaepConstraint>,
    frozen: Arc<AtomicBool>,
    freeze_reason: Arc<RwLock<Option<String>>>,
    total_proposals_processed: Arc<AtomicU64>,
    total_vetoes: Arc<AtomicU64>,
}

impl MockVetoEngine {
    /// Create a new mock veto engine with no constraints.
    pub fn new() -> Self {
        Self {
            constraints: Vec::new(),
            frozen: Arc::new(AtomicBool::new(false)),
            freeze_reason: Arc::new(RwLock::new(None)),
            total_proposals_processed: Arc::new(AtomicU64::new(0)),
            total_vetoes: Arc::new(AtomicU64::new(0)),
        }
    }

    /// Pre-register the standard set of SAEP constraints used in integration tests.
    ///
    /// Registered constraints:
    /// - `max-conviction`: Limits proposals with conviction >= 0.95 (Limit action)
    /// - `nan-detector`: Vetos proposals with NaN/Inf values (Veto action)
    pub fn with_default_constraints(mut self) -> Self {
        // Constraint: max conviction limit
        let max_conviction = SaepConstraint {
            id: "max-conviction".into(),
            layer: GovernanceLayer::Room,
            check_fn: Arc::new(|proposal: &RoomProposal, _: &HashMap<String, f64>| {
                if proposal.conviction >= 0.95 {
                    Err(Violation {
                        constraint_id: "max-conviction".into(),
                        message: "Conviction exceeds limit".into(),
                        severity: 0.5,
                    })
                } else {
                    Ok(())
                }
            }),
            action: SaepAction::Limit,
            escalate_to: None,
        };

        // Constraint: NaN/Inf detector
        let nan_detector = SaepConstraint {
            id: "nan-detector".into(),
            layer: GovernanceLayer::Room,
            check_fn: Arc::new(|proposal: &RoomProposal, _: &HashMap<String, f64>| {
                if !proposal.conviction.is_finite()
                    || !proposal.confidence.is_finite()
                    || !proposal.matrix_agreement.is_finite()
                {
                    Err(Violation {
                        constraint_id: "nan-detector".into(),
                        message: "Non-finite value in proposal".into(),
                        severity: 1.0,
                    })
                } else {
                    Ok(())
                }
            }),
            action: SaepAction::Veto,
            escalate_to: None,
        };

        self.constraints.push(max_conviction);
        self.constraints.push(nan_detector);
        self
    }
}

impl Default for MockVetoEngine {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl VetoEngine for MockVetoEngine {
    async fn register_constraint(&mut self, constraint: SaepConstraint) {
        info!(
            "MockVetoEngine: registered constraint '{}' on {:?} layer",
            constraint.id, constraint.layer
        );
        self.constraints.push(constraint);
    }

    async fn resolve(
        &self,
        proposals: &[RoomProposal],
        current_portfolio: Option<&PortfolioVector>,
    ) -> PortfolioVector {
        if self.frozen.load(Ordering::Acquire) {
            let reason = self.freeze_reason.read().await;
            warn!("MockVetoEngine is frozen: {:?}", reason);
            return PortfolioVector {
                positions: vec![],
                gross_exposure: 0.0,
                net_exposure: 0.0,
                sector_concentrations: HashMap::new(),
                portfolio_var: 0.0,
                timestamp: proposals.first().map(|p| p.timestamp).unwrap_or(0),
            };
        }

        let mut final_positions = Vec::with_capacity(proposals.len());
        let mut gross_exposure: f64 = 0.0;
        let mut net_exposure: f64 = 0.0;

        for proposal in proposals {
            let mut veto_applied: Vec<String> = Vec::new();
            let mut veto_severity: f64 = 0.0;

            let sector_map = current_portfolio
                .map(|p| p.sector_concentrations.clone())
                .unwrap_or_default();

            // Check constraints
            for constraint in &self.constraints {
                match (constraint.check_fn)(proposal, &sector_map) {
                    Ok(()) => {}
                    Err(_violation) => {
                        veto_applied.push(constraint.id.clone());
                        match constraint.action {
                            SaepAction::Warn => {
                                veto_severity = veto_severity.max(0.2);
                            }
                            SaepAction::Limit => {
                                veto_severity = veto_severity.max(0.5);
                            }
                            SaepAction::Veto => {
                                veto_severity = 1.0;
                            }
                            SaepAction::Freeze => {
                                veto_severity = 1.0;
                            }
                        }
                    }
                }
            }

            // Apply veto_override: halve the veto severity
            if proposal.veto_override && veto_severity > 0.0 {
                veto_severity *= 0.5;
            }

            // Compute final weight — guard against NaN/Inf from non-finite conviction
            let effective_conviction = if proposal.conviction.is_finite() {
                proposal.conviction
            } else {
                0.0
            };
            let weight = match proposal.gate {
                TernaryGate::Bullish => effective_conviction * (1.0 - veto_severity),
                TernaryGate::Bearish => -effective_conviction * (1.0 - veto_severity),
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
        self.total_vetoes.fetch_add(
            veto_applied_count(&final_positions) as u64,
            Ordering::Relaxed,
        );

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
        PortfolioVector {
            positions: vec![],
            gross_exposure: 0.0,
            net_exposure: 0.0,
            sector_concentrations: HashMap::new(),
            portfolio_var: 0.0,
            timestamp: 0,
        }
    }

    async fn freeze(&mut self, reason: &str) {
        self.frozen.store(true, Ordering::Release);
        *self.freeze_reason.write().await = Some(reason.to_string());
        warn!("MockVetoEngine frozen: {}", reason);
    }

    async fn unfreeze(&mut self, reason: &str) {
        self.frozen.store(false, Ordering::Release);
        *self.freeze_reason.write().await = None;
        info!("MockVetoEngine unfrozen: {}", reason);
    }
}

/// Count position entries that had any veto applied.
fn veto_applied_count(positions: &[FinalPosition]) -> usize {
    positions
        .iter()
        .filter(|p| !p.veto_applied.is_empty())
        .count()
}
