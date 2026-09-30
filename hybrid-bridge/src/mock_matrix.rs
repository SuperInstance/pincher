//! Mock Matrix Engine for integration tests.
//!
//! Provides an in-memory `Array3<f32>`-backed implementation of the
//! `MatrixEngine` trait with seed/inject capabilities for testing.

use crate::engine::MatrixEngine;
use crate::types::{MatrixMetadata, MatrixSnapshot, PartialSnapshot, TopologicalSignature};
use async_trait::async_trait;
use ndarray::{Array1, Array2, Array3};
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::debug;

pub type MockTensor = Arc<RwLock<Array3<f32>>>;

pub struct MockMatrixEngine {
    tensor: MockTensor,
    tickers: Vec<String>,
}

impl MockMatrixEngine {
    pub fn new(n_stocks: usize, n_features: usize, n_history: usize) -> Self {
        let data = Array3::<f32>::zeros((n_stocks, n_features, n_history));
        Self {
            tensor: Arc::new(RwLock::new(data)),
            tickers: Vec::with_capacity(n_stocks),
        }
    }

    /// Pre-register tickers by name.
    pub fn with_tickers(mut self, tickers: &[&str]) -> Self {
        for &t in tickers {
            self.tickers.push(t.to_string());
        }
        self
    }

    /// Seed the tensor with random data in [-1, 1].
    pub fn seed_random(&self) {
        let mut tensor = self.tensor.blocking_write();
        let mut rng = fastrand::Rng::new();
        for val in tensor.iter_mut() {
            *val = rng.f32() * 2.0 - 1.0;
        }
    }

    /// Inject a single NaN at (stock, feature, time).
    pub fn inject_nan(&self, stock: usize, feature: usize, time: usize) {
        let mut tensor = self.tensor.blocking_write();
        tensor[[stock, feature, time]] = f32::NAN;
    }

    /// Inject a single Inf at (stock, feature, time).
    pub fn inject_inf(&self, stock: usize, feature: usize, time: usize) {
        let mut tensor = self.tensor.blocking_write();
        tensor[[stock, feature, time]] = f32::INFINITY;
    }

    pub fn tensor(&self) -> MockTensor {
        self.tensor.clone()
    }
}

#[async_trait]
impl MatrixEngine for MockMatrixEngine {
    async fn ingest(&self, _ticker: &str, _features: &[f64], _tick: u64) {
        debug!("MockMatrixEngine::ingest — pre-registered ticker");
    }

    async fn fast_cycle(&self, tick: u64) -> MatrixMetadata {
        let tensor = self.tensor.read().await;
        let shape = tensor.shape();
        MatrixMetadata {
            tick,
            n_stocks: shape[0],
            n_features: shape[1],
            n_history: shape[2],
            mean_correlation: 0.0,
            timestamp_ms: 0,
        }
    }

    async fn medium_cycle(&self, tick: u64) -> PartialSnapshot {
        let n_stocks = self.tensor.read().await.shape()[0];
        PartialSnapshot {
            tick,
            n_stocks,
            correlation_matrix_cond: (n_stocks as f64).sqrt() * 2.0,
            top_eigenvalues: vec![0.9, 0.05, 0.03],
            regime: "stable".into(),
            timestamp_ms: 0,
        }
    }

    async fn full_cycle(&self, tick: u64) -> MatrixSnapshot {
        let tensor = self.tensor.read().await;
        let n_stocks = tensor.shape()[0];
        let topologies: Vec<TopologicalSignature> = self
            .tickers
            .iter()
            .enumerate()
            .map(|(i, t)| {
                let slice: Vec<f32> = tensor.slice(ndarray::s![i, 0, ..]).to_vec();
                let mean = slice.iter().copied().sum::<f32>() / slice.len() as f32;
                let var =
                    slice.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / slice.len() as f32;
                let conf = if var.is_finite() {
                    (1.0 - var.min(0.99)) as f64
                } else {
                    0.0
                };
                TopologicalSignature {
                    ticker: t.clone(),
                    betti_numbers: vec![1, 0, 0],
                    persistence_landscape: vec![mean as f64, var as f64],
                    wasserstein_distance_centroid: 0.0,
                    regime_label: "stable".into(),
                    confidence: conf.clamp(0.0, 1.0),
                }
            })
            .collect();

        let cond = (n_stocks as f64).sqrt() * 2.0;
        let ev = Array2::<f64>::zeros((n_stocks, 3));
        MatrixSnapshot {
            tick,
            n_stocks,
            eigenvalues: vec![cond, cond * 0.4, cond * 0.15],
            eigenvectors: ev,
            topologies,
            universe_betti: [n_stocks.min(3), 1, 0],
            regime: "stable".into(),
            condition_number: cond,
        }
    }

    async fn get_slice(&self, ticker: &str) -> Option<Array2<f32>> {
        let idx = self.tickers.iter().position(|t| t == ticker)?;
        let tensor = self.tensor.read().await;
        Some(tensor.index_axis(ndarray::Axis(0), idx).to_owned())
    }

    async fn get_cross_section(&self, _feature: &str, _time_idx: usize) -> Option<Array1<f32>> {
        None
    }

    async fn add_ticker(&self, ticker: &str, _initial_features: Option<&[f64]>) {
        debug!("MockMatrixEngine::add_ticker({})", ticker);
    }

    async fn remove_ticker(&self, _ticker: &str) {}
}
