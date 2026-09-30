//! Chaos testing utilities for the hybrid bridge.
//!
//! These utilities deliberately inject pathological data into the feature
//! tensor to verify that downstream components (the veto engine, matrix
//! masking logic, safe-mode protocols) handle them gracefully.

/// The number of non-finite cells that triggers safe mode.
pub const SAFE_MODE_THRESHOLD: usize = 1;

/// Result of a chaos injection + recovery cycle.
#[derive(Debug, Clone)]
pub struct ChaosTestResult {
    /// Number of non-finite values injected.
    pub injected: usize,
    /// Number of non-finite values detected.
    pub detected: usize,
    /// Number of non-finite values masked.
    pub masked: usize,
    /// Whether safe mode would have been triggered.
    pub safe_mode_triggered: bool,
    /// Any failures encountered during the cycle.
    pub failures: Vec<String>,
}

/// Inject NaN cells into a tensor at random positions and test recovery.
pub fn inject_nan_random(
    tensor: &mut ndarray::Array3<f32>,
    count: usize,
) -> Vec<(usize, usize, usize)> {
    let mut rng = fastrand::Rng::new();
    let mut coords = Vec::with_capacity(count);
    let shape = tensor.shape().to_vec();
    for _ in 0..count {
        let s = rng.usize(0..shape[0]);
        let f = rng.usize(0..shape[1]);
        let t = rng.usize(0..shape[2]);
        tensor[[s, f, t]] = f32::NAN;
        coords.push((s, f, t));
    }
    coords
}

/// Inject Inf cells into a tensor at random positions.
pub fn inject_inf_random(
    tensor: &mut ndarray::Array3<f32>,
    count: usize,
) -> Vec<(usize, usize, usize)> {
    let mut rng = fastrand::Rng::new();
    let mut coords = Vec::with_capacity(count);
    let shape = tensor.shape().to_vec();
    for _ in 0..count {
        let s = rng.usize(0..shape[0]);
        let f = rng.usize(0..shape[1]);
        let t = rng.usize(0..shape[2]);
        tensor[[s, f, t]] = f32::INFINITY;
        coords.push((s, f, t));
    }
    coords
}

/// Run a full chaos cycle: inject → detect → mask → verify.
pub fn run_chaos_cycle(tensor: &mut ndarray::Array3<f32>, n_inject: usize) -> ChaosTestResult {
    let mut failures = Vec::new();

    // 1. Inject NaNs
    let _coords = inject_nan_random(tensor, n_inject);

    // 2. Detect using the crate-level detection function
    let detected = crate::types::detect_non_finite(tensor);
    let n_detected = detected.len();
    let safe_mode = n_detected >= SAFE_MODE_THRESHOLD;

    if n_detected == 0 {
        failures.push("Expected non-finite detection but found none".to_string());
    }

    // 3. Mask
    let n_masked = crate::types::mask_non_finite(tensor);

    // 4. Verify
    let remaining = crate::types::detect_non_finite(tensor);
    if !remaining.is_empty() {
        failures.push(format!(
            "{} non-finite values remained after masking",
            remaining.len()
        ));
    }

    // 5. Verify the tensor can be meaningfully used
    let total: f32 = tensor.iter().copied().sum();
    if !total.is_finite() {
        failures.push(format!("Tensor sum is non-finite after masking: {}", total));
    }

    ChaosTestResult {
        injected: n_inject,
        detected: n_detected,
        masked: n_masked,
        safe_mode_triggered: safe_mode,
        failures,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{detect_non_finite, mask_non_finite};
    use ndarray::Array3;

    #[test]
    fn test_inject_nan_random_injects_expected_count() {
        let mut tensor = Array3::<f32>::zeros((5, 10, 20));
        let coords = inject_nan_random(&mut tensor, 7);
        assert_eq!(coords.len(), 7);
        for &(s, f, t) in &coords {
            assert!(tensor[[s, f, t]].is_nan());
        }
    }

    #[test]
    fn test_run_chaos_cycle_clean() {
        let mut tensor = Array3::<f32>::zeros((3, 4, 5));
        let result = run_chaos_cycle(&mut tensor, 5);
        assert!(
            result.failures.is_empty(),
            "Failures: {:?}",
            result.failures
        );
        assert_eq!(result.detected, result.masked);
        assert!(result.safe_mode_triggered);
    }

    #[test]
    fn test_run_chaos_cycle_large_injection() {
        let mut tensor = Array3::<f32>::ones((10, 20, 30));
        let result = run_chaos_cycle(&mut tensor, 50);
        assert!(
            result.failures.is_empty(),
            "Failures: {:?}",
            result.failures
        );
        assert_eq!(result.detected, result.masked);
        assert_eq!(result.injected, 50);
    }

    #[test]
    fn test_run_chaos_cycle_mixed_nan_inf() {
        let mut tensor = Array3::<f32>::zeros((4, 4, 4));
        tensor[[0, 0, 0]] = f32::NAN;
        tensor[[1, 1, 1]] = f32::INFINITY;
        tensor[[2, 2, 2]] = f32::NEG_INFINITY;

        assert_eq!(detect_non_finite(&tensor).len(), 3);
        assert_eq!(mask_non_finite(&mut tensor), 3);
        assert!(detect_non_finite(&tensor).is_empty());
    }
}
