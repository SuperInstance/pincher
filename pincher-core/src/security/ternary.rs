//! Ternary logic adapter for the veto system.
//!
//! Bridges [`ternary_types::Ternary`] to [`VetoDecision`] with Kleene logic
//! operations and confidence-threshold bridging.
//!
//! # Kleene Logic
//!
//! Ternary values form a three-valued (Kleene) logic where:
//!
//! | a | b | a ∧ b | a ∨ b | ¬a |
//! |---|---|---|---|---|
//! | Pos | Pos | Pos | Pos | Neg |
//! | Pos | Zero | Zero | Pos | Neg |
//! | Pos | Neg | Neg | Pos | Neg |
//! | Zero | Zero | Zero | Zero | Zero |
//! | Zero | Neg | Neg | Zero | Zero |
//! | Neg | Neg | Neg | Neg | Pos |
//!
//! This matches the veto system's three decision states.

use ternary_types::Ternary;
use crate::security::VetoDecision;

/// Kleene conjunction (AND) on ternary values.
pub fn kleene_and(a: Ternary, b: Ternary) -> Ternary {
    use Ternary::{Neg, Neutral as Zero, Pos};
    match (a, b) {
        (Neg, _) | (_, Neg) => Neg,
        (Zero, _) | (_, Zero) => Zero,
        (Pos, Pos) => Pos,
    }
}

/// Kleene disjunction (OR) on ternary values.
pub fn kleene_or(a: Ternary, b: Ternary) -> Ternary {
    use Ternary::{Neg, Neutral as Zero, Pos};
    match (a, b) {
        (Pos, _) | (_, Pos) => Pos,
        (Zero, _) | (_, Zero) => Zero,
        (Neg, Neg) => Neg,
    }
}

/// Kleene negation (NOT) on a ternary value.
pub fn kleene_not(a: Ternary) -> Ternary {
    use Ternary::{Neg, Neutral as Zero, Pos};
    match a {
        Pos => Neg,
        Zero => Zero,
        Neg => Pos,
    }
}

/// Convert a confidence value in [0.0, 1.0] to a Ternary decision.
///
/// - `confidence >= high_threshold` → `Pos`
/// - `confidence <= low_threshold` → `Neg`
/// - otherwise → `Neutral`
pub fn from_confidence(confidence: f64, high_threshold: f64, low_threshold: f64) -> Ternary {
    if confidence >= high_threshold {
        Ternary::Pos
    } else if confidence <= low_threshold {
        Ternary::Neg
    } else {
        Ternary::Neutral
    }
}

/// Convert a `Ternary` to a `VetoDecision`.
///
/// - `Pos` → `Allow`
/// - `Neutral` → `RequireConfirmation(reason)` with the given reason
/// - `Neg` → `Deny(reason)` with the given reason
pub fn ternary_to_veto(value: Ternary, reason: impl Into<String>) -> VetoDecision {
    let reason = reason.into();
    match value {
        Ternary::Pos => VetoDecision::Allow,
        Ternary::Neutral => VetoDecision::RequireConfirmation(reason),
        Ternary::Neg => VetoDecision::Deny(reason),
    }
}

/// Convert a `VetoDecision` to a `Ternary`.
///
/// - `Allow` → `Pos`
/// - `RequireConfirmation(_)` → `Neutral`
/// - `Deny(_)` → `Neg`
pub fn veto_to_ternary(decision: &VetoDecision) -> Ternary {
    match decision {
        VetoDecision::Allow => Ternary::Pos,
        VetoDecision::RequireConfirmation(_) => Ternary::Neutral,
        VetoDecision::Deny(_) => Ternary::Neg,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ternary_types::Ternary::{Neg, Neutral, Pos};

    #[test]
    fn test_kleene_and() {
        assert_eq!(kleene_and(Pos, Pos), Pos);
        assert_eq!(kleene_and(Pos, Neutral), Neutral);
        assert_eq!(kleene_and(Pos, Neg), Neg);
        assert_eq!(kleene_and(Neutral, Neg), Neg);
        assert_eq!(kleene_and(Neutral, Neutral), Neutral);
        assert_eq!(kleene_and(Neg, Neg), Neg);
    }

    #[test]
    fn test_kleene_or() {
        assert_eq!(kleene_or(Pos, Pos), Pos);
        assert_eq!(kleene_or(Pos, Neutral), Pos);
        assert_eq!(kleene_or(Pos, Neg), Pos);
        assert_eq!(kleene_or(Neutral, Neg), Neutral);
        assert_eq!(kleene_or(Neutral, Neutral), Neutral);
        assert_eq!(kleene_or(Neg, Neg), Neg);
    }

    #[test]
    fn test_kleene_not() {
        assert_eq!(kleene_not(Pos), Neg);
        assert_eq!(kleene_not(Neg), Pos);
        assert_eq!(kleene_not(Neutral), Neutral);
    }

    #[test]
    fn test_from_confidence() {
        assert_eq!(from_confidence(0.9, 0.7, 0.3), Pos);
        assert_eq!(from_confidence(0.1, 0.7, 0.3), Neg);
        assert_eq!(from_confidence(0.5, 0.7, 0.3), Neutral);
    }

    #[test]
    fn test_ternary_to_veto_allow() {
        let v = ternary_to_veto(Pos, "should not appear");
        assert_eq!(v, VetoDecision::Allow);
    }

    #[test]
    fn test_ternary_to_veto_require_confirmation() {
        let v = ternary_to_veto(Neutral, "needs review");
        assert_eq!(v, VetoDecision::RequireConfirmation("needs review".into()));
    }

    #[test]
    fn test_ternary_to_veto_deny() {
        let v = ternary_to_veto(Neg, "blocked");
        assert_eq!(v, VetoDecision::Deny("blocked".into()));
    }

    #[test]
    fn test_veto_to_ternary_roundtrip() {
        let decisions = vec![
            VetoDecision::Allow,
            VetoDecision::RequireConfirmation("test".into()),
            VetoDecision::Deny("test".into()),
        ];
        for d in &decisions {
            let t = veto_to_ternary(d);
            let back = ternary_to_veto(t, "test");
            assert_eq!(d.clone(), back, "roundtrip failed for {d:?}");
        }
    }

    #[test]
    fn test_kleene_associativity() {
        // (a ∧ b) ∧ c == a ∧ (b ∧ c)
        let values = [Pos, Neutral, Neg];
        for &a in &values {
            for &b in &values {
                for &c in &values {
                    let left = kleene_and(kleene_and(a, b), c);
                    let right = kleene_and(a, kleene_and(b, c));
                    assert_eq!(left, right, "AND not associative for {a:?} {b:?} {c:?}");
                }
            }
        }
    }

    #[test]
    fn test_kleene_de_morgan() {
        // ¬(a ∨ b) == ¬a ∧ ¬b
        let values = [Pos, Neutral, Neg];
        for &a in &values {
            for &b in &values {
                let left = kleene_not(kleene_or(a, b));
                let right = kleene_and(kleene_not(a), kleene_not(b));
                assert_eq!(left, right, "De Morgan failed for {a:?} {b:?}");
            }
        }
    }

    #[test]
    fn test_kleene_commutativity() {
        let values = [Pos, Neutral, Neg];
        for &a in &values {
            for &b in &values {
                assert_eq!(kleene_and(a, b), kleene_and(b, a), "AND not commutative");
                assert_eq!(kleene_or(a, b), kleene_or(b, a), "OR not commutative");
            }
        }
    }

    #[test]
    fn test_veto_to_ternary_direct() {
        assert_eq!(veto_to_ternary(&VetoDecision::Allow), Pos);
        assert_eq!(
            veto_to_ternary(&VetoDecision::RequireConfirmation("r".into())),
            Neutral
        );
        assert_eq!(veto_to_ternary(&VetoDecision::Deny("r".into())), Neg);
    }
}
