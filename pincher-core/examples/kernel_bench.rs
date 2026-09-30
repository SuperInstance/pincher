//! Benchmark: SIMD vs scalar kernel performance.
//!
//! Measures cosine similarity, L2 normalization, and vector scaling
//! across various vector sizes. Demonstrates the `ternary-kernel`
//! feature gate.
//!
//! ## Run
//!
//! ```sh
//! # Scalar (default)
//! cargo run --example kernel_bench
//!
//! # NEON SIMD (aarch64)
//! cargo run --example kernel_bench --features ternary-kernel
//! ```

use std::time::{Duration, Instant};

use pincher_core::kernel::{fast_cosine_similarity, fast_l2_normalize, fast_scale};
use pincher_core::EMBEDDING_DIM;

fn mean_duration(durations: &[Duration]) -> Duration {
    let total: Duration = durations.iter().sum();
    total / durations.len() as u32
}

fn main() {
    let dims = [64, 128, EMBEDDING_DIM, 768];

    #[cfg(all(feature = "ternary-kernel", target_arch = "aarch64"))]
    println!("=== Kernel Benchmark (NEON SIMD — aarch64) ===");
    #[cfg(not(all(feature = "ternary-kernel", target_arch = "aarch64")))]
    println!("=== Kernel Benchmark (Scalar) ===");

    println!(
        "{:<10} {:<25} {:<25} {:<25} {:<25}",
        "Dim", "Cosine Sim (mean)", "L2 Norm (mean)", "Scale (mean)", "Sim+Norm Combined"
    );
    println!(
        "{:-<10} {:-<25} {:-<25} {:-<25} {:-<25}",
        "", "", "", "", ""
    );

    for &dim in &dims {
        let n = 1000;

        // Generate random vectors
        let mut a: Vec<f32> = (0..dim).map(|i| (i as f32 + 1.0) / dim as f32).collect();
        let b: Vec<f32> = (0..dim)
            .map(|i| ((i * 7 + 3) as f32) / dim as f32)
            .collect();
        let original_a = a.clone();

        // Warmup
        for _ in 0..100 {
            let _ = fast_cosine_similarity(&a, &b);
        }

        // Cosine similarity
        let mut cos_durations = Vec::with_capacity(n);
        for _ in 0..n {
            let start = Instant::now();
            let _sim = fast_cosine_similarity(&a, &b);
            cos_durations.push(start.elapsed());
        }

        // L2 normalization
        let mut l2_durations = Vec::with_capacity(n);
        for _ in 0..n {
            a.copy_from_slice(&original_a);
            let start = Instant::now();
            fast_l2_normalize(&mut a);
            l2_durations.push(start.elapsed());
        }

        // Scale
        let mut scale_durations = Vec::with_capacity(n);
        for _ in 0..n {
            a.copy_from_slice(&original_a);
            let start = Instant::now();
            fast_scale(&mut a, 0.5);
            scale_durations.push(start.elapsed());
        }

        // Combined: cosine + normalize
        let mut combined_durations = Vec::with_capacity(n);
        for _ in 0..n {
            a.copy_from_slice(&original_a);
            let start = Instant::now();
            let _sim = fast_cosine_similarity(&a, &b);
            fast_l2_normalize(&mut a);
            combined_durations.push(start.elapsed());
        }

        let cos_mean = mean_duration(&cos_durations);
        let l2_mean = mean_duration(&l2_durations);
        let scale_mean = mean_duration(&scale_durations);
        let combined_mean = mean_duration(&combined_durations);

        println!(
            "{:<10} {:<25?} {:<25?} {:<25?} {:<25?}",
            dim, cos_mean, l2_mean, scale_mean, combined_mean
        );
    }
}
