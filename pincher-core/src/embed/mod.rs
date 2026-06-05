//! Embedding module for PincherOS
//!
//! Provides text embedding capabilities using ONNX Runtime (when available)
//! or a hash-based fallback.

pub mod onnx;

// Re-export key types from the onnx module
pub use onnx::{
    cosine_similarity, download_model, EmbedError, EmbedResult, Embedder, EMBEDDING_DIM,
};
