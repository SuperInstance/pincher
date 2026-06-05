//! ONNX embedding module for PincherOS
//!
//! Provides text embedding using the all-MiniLM-L6-v2 model via the
//! `ort` ONNX Runtime crate. Falls back to deterministic hash embeddings if the
//! model is not found (with a warning log).
//!
//! When the `onnx` feature is disabled, only the fallback (deterministic hash)
//! embedding and hash-based similarity are available.

use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use thiserror::Error;
use tracing::{debug, info, instrument, warn};

#[cfg(feature = "onnx")]
use std::sync::Arc;

#[cfg(feature = "onnx")]
use ndarray::Array2;
#[cfg(feature = "onnx")]
use ort::session::Session;
#[cfg(feature = "onnx")]
use ort::value::Value;

/// Embedding dimensionality for all-MiniLM-L6-v2.
pub const EMBEDDING_DIM: usize = 384;

/// Default model URL for all-MiniLM-L6-v2 ONNX INT8.
pub const DEFAULT_MODEL_URL: &str =
    "https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main/onnx/model_int8.onnx";

/// Default directory for storing the ONNX model.
pub const DEFAULT_MODEL_DIR: &str = ".pincher/models";

/// Default model filename.
pub const DEFAULT_MODEL_FILENAME: &str = "all-MiniLM-L6-v2-int8.onnx";

/// Embedding errors.
#[derive(Debug, Error)]
pub enum EmbedError {
    #[error("ONNX Runtime error: {0}")]
    #[cfg(feature = "onnx")]
    Ort(#[from] ort::Error),

    #[error("ONNX Runtime not available (compile with --features onnx)")]
    OrtNotAvailable,

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Tokenization error: {0}")]
    Tokenization(String),

    #[error("Model not found at path: {0}")]
    ModelNotFound(PathBuf),

    #[error("HTTP download error: {0}")]
    Download(String),

    #[error("Dimension mismatch: expected {expected}, got {actual}")]
    DimensionMismatch { expected: usize, actual: usize },
}

/// Result type for embedding operations.
pub type EmbedResult<T> = Result<T, EmbedError>;

/// The embedding model state.
#[derive(Clone)]
pub enum EmbedderState {
    /// ONNX model is loaded and ready.
    #[cfg(feature = "onnx")]
    Loaded(Arc<Session>),
    /// Fallback mode: model not available, using deterministic hash embeddings.
    Fallback,
}

/// ONNX-based text embedder using all-MiniLM-L6-v2.
///
/// If the ONNX model cannot be loaded, falls back to deterministic hash embeddings
/// with a warning. This allows the system to function in degraded mode.
pub struct Embedder {
    state: EmbedderState,
    #[cfg(feature = "onnx")]
    tokenizer: SimpleTokenizer,
}

/// A minimal whitespace + punctuation tokenizer for embedding.
pub struct SimpleTokenizer {
    vocab_size: usize,
    max_length: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TokenOutput {
    input_ids: Vec<i64>,
    attention_mask: Vec<i64>,
    token_type_ids: Vec<i64>,
}

impl SimpleTokenizer {
    /// Create a new simple tokenizer.
    pub fn new() -> Self {
        Self {
            vocab_size: 30522, // Standard BERT vocab size
            max_length: 128,
        }
    }

    /// Tokenize input text into model inputs.
    pub fn tokenize(&self, text: &str) -> TokenOutput {
        let text = text.to_lowercase();
        let text = text.replace(|c: char| !c.is_alphanumeric() && c != ' ', " ");
        let words: Vec<&str> = text.split_whitespace().collect();

        let mut input_ids: Vec<i64> = Vec::new();
        // [CLS] token
        input_ids.push(101);

        for word in words.iter() {
            if input_ids.len() >= self.max_length - 1 {
                break;
            }
            let subwords = self.word_to_subwords(word);
            for sw in subwords {
                if input_ids.len() >= self.max_length - 1 {
                    break;
                }
                let token_id = self.hash_token(&sw) % (self.vocab_size as u64 - 1000);
                input_ids.push(token_id as i64 + 1000);
            }
        }

        // [SEP] token
        input_ids.push(102);

        // Pad to max_length
        let seq_len = input_ids.len();
        let padding_len = self.max_length.saturating_sub(seq_len);
        input_ids.extend(vec![0i64; padding_len]);

        let attention_mask: Vec<i64> = input_ids
            .iter()
            .map(|&id| if id != 0 { 1 } else { 0 })
            .collect();

        let token_type_ids: Vec<i64> = vec![0i64; self.max_length];

        TokenOutput {
            input_ids,
            attention_mask,
            token_type_ids,
        }
    }

    fn word_to_subwords(&self, word: &str) -> Vec<String> {
        if word.len() <= 4 {
            return vec![format!("##{}", word)];
        }
        let mut subwords = Vec::new();
        let chars: Vec<char> = word.chars().collect();
        let mut i = 0;
        let mut first = true;
        while i < chars.len() {
            let end = std::cmp::min(i + 4, chars.len());
            let chunk: String = chars[i..end].iter().collect();
            if first {
                subwords.push(chunk);
                first = false;
            } else {
                subwords.push(format!("##{}", chunk));
            }
            i = end;
        }
        subwords
    }

    fn hash_token(&self, token: &str) -> u64 {
        use std::hash::{Hash, Hasher};
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        token.hash(&mut hasher);
        hasher.finish()
    }

    /// Get the max sequence length.
    pub fn max_length(&self) -> usize {
        self.max_length
    }
}

impl Default for SimpleTokenizer {
    fn default() -> Self {
        Self::new()
    }
}

impl Embedder {
    /// Create a new embedder, attempting to load the ONNX model.
    #[instrument]
    pub fn new(model_path: Option<&Path>) -> EmbedResult<Self> {
        #[cfg(feature = "onnx")]
        {
            let tokenizer = SimpleTokenizer::new();

            let model_path = model_path
                .map(|p| p.to_path_buf())
                .or_else(|| default_model_path());

            match model_path {
                Some(path) if path.exists() => {
                    info!(path = ?path, "Loading ONNX embedding model");
                    match Self::load_model(&path) {
                        Ok(session) => {
                            info!("ONNX model loaded successfully");
                            Ok(Self {
                                state: EmbedderState::Loaded(Arc::new(session)),
                                tokenizer,
                            })
                        }
                        Err(e) => {
                            warn!(
                                error = %e,
                                path = ?path,
                                "Failed to load ONNX model, falling back to deterministic hash embeddings"
                            );
                            Ok(Self {
                                state: EmbedderState::Fallback,
                                tokenizer,
                            })
                        }
                    }
                }
                Some(path) => {
                    warn!(
                        path = ?path,
                        "ONNX model not found, falling back to deterministic hash embeddings."
                    );
                    Ok(Self {
                        state: EmbedderState::Fallback,
                        tokenizer,
                    })
                }
                None => {
                    warn!("No model path specified, falling back to deterministic hash embeddings");
                    Ok(Self {
                        state: EmbedderState::Fallback,
                        tokenizer,
                    })
                }
            }
        }

        #[cfg(not(feature = "onnx"))]
        {
            let _ = model_path;
            warn!("ONNX feature not enabled, using deterministic hash embeddings.");
            Ok(Self {
                state: EmbedderState::Fallback,
            })
        }
    }

    /// Load the ONNX session from a file path.
    #[cfg(feature = "onnx")]
    fn load_model(path: &Path) -> EmbedResult<Session> {
        let session = Session::builder()?
            .with_optimization_level(ort::session::GraphOptimizationLevel::Level3)?
            .with_intra_threads(2)?
            .commit_from_file(path)?;
        Ok(session)
    }

    /// Check if the embedder is using the ONNX model (not fallback).
    pub fn is_loaded(&self) -> bool {
        match &self.state {
            #[cfg(feature = "onnx")]
            EmbedderState::Loaded(_) => true,
            EmbedderState::Fallback => false,
        }
    }

    /// Embed a single text string into a vector.
    #[instrument(skip(self, text), fields(text_len = text.len()))]
    pub fn embed(&self, text: &str) -> EmbedResult<Vec<f32>> {
        match &self.state {
            #[cfg(feature = "onnx")]
            EmbedderState::Loaded(session) => self.embed_onnx(session, text),
            EmbedderState::Fallback => {
                debug!(
                    text_preview = &text[..text.len().min(50)],
                    "Using deterministic hash embedding (model not loaded)"
                );
                Ok(deterministic_embedding(text))
            }
        }
    }

    /// Embed a batch of text strings.
    #[instrument(skip(self, texts), fields(batch_size = texts.len()))]
    pub fn batch_embed(&self, texts: &[&str]) -> EmbedResult<Vec<Vec<f32>>> {
        texts.iter().map(|text| self.embed(text)).collect()
    }

    /// Run the ONNX model inference for a single text.
    #[cfg(feature = "onnx")]
    fn embed_onnx(&self, session: &Session, text: &str) -> EmbedResult<Vec<f32>> {
        debug!(text_len = text.len(), "Running ONNX embedding inference");

        let tokens = self.tokenizer.tokenize(text);
        let seq_len = self.tokenizer.max_length;

        // Create input tensors
        let input_ids_tensor = Value::from_array(
            Array2::from_shape_vec((1, seq_len), tokens.input_ids).map_err(|e| {
                EmbedError::Tokenization(format!("Failed to create input_ids tensor: {}", e))
            })?,
        )?;

        let attention_mask_tensor = Value::from_array(
            Array2::from_shape_vec((1, seq_len), tokens.attention_mask).map_err(|e| {
                EmbedError::Tokenization(format!("Failed to create attention_mask tensor: {}", e))
            })?,
        )?;

        let token_type_ids_tensor = Value::from_array(
            Array2::from_shape_vec((1, seq_len), tokens.token_type_ids).map_err(|e| {
                EmbedError::Tokenization(format!("Failed to create token_type_ids tensor: {}", e))
            })?,
        )?;

        // Run inference
        let outputs = session.run(ort::inputs![
            input_ids_tensor,
            attention_mask_tensor,
            token_type_ids_tensor,
        ]?)?;

        // Extract the last_hidden_state output (index 0)
        let output = outputs[0].try_extract_tensor::<f32>()?;

        // Mean pooling over sequence dimension: output shape is [1, seq_len, 384]
        let shape = output.shape();
        if shape.len() != 3 || shape[2] != EMBEDDING_DIM {
            return Err(EmbedError::DimensionMismatch {
                expected: EMBEDDING_DIM,
                actual: if shape.len() == 3 { shape[2] } else { 0 },
            });
        }

        // Mean pooling: average over non-padding tokens
        let attention_weights: Vec<f32> = tokens.attention_mask.iter().map(|&m| m as f32).collect();
        let total_weight: f32 = attention_weights.iter().sum();

        let mut pooled = vec![0.0f32; EMBEDDING_DIM];
        for (t, &weight) in attention_weights.iter().enumerate() {
            for d in 0..EMBEDDING_DIM {
                pooled[d] += output[[0, t, d]] * weight;
            }
        }

        if total_weight > 0.0 {
            for v in pooled.iter_mut() {
                *v /= total_weight;
            }
        }

        // L2 normalize
        let norm: f32 = pooled.iter().map(|v| v * v).sum::<f32>().sqrt();
        if norm > 0.0 {
            for v in pooled.iter_mut() {
                *v /= norm;
            }
        }

        debug!("Embedding computed successfully");
        Ok(pooled)
    }

    /// Update embeddings for built-in reflexes in the database.
    #[instrument(skip(self, conn))]
    pub fn seed_embeddings(&self, conn: &rusqlite::Connection) -> EmbedResult<()> {
        use crate::db::schema;

        let reflexes = schema::get_all_reflexes(conn)
            .map_err(|e| EmbedError::Tokenization(format!("DB error: {}", e)))?;

        for reflex in reflexes {
            if reflex.embedding.iter().all(|&v| v == 0.0) {
                let embedding = self.embed(&reflex.intent)?;
                schema::update_reflex_embedding(conn, &reflex.id, &embedding)
                    .map_err(|e| EmbedError::Tokenization(format!("DB error: {}", e)))?;
                debug!(reflex_id = %reflex.id, intent = %reflex.intent, "Updated embedding for built-in reflex");
            }
        }

        info!("Seeded embeddings for built-in reflexes");
        Ok(())
    }
}

/// Compute cosine similarity between two embedding vectors.
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }

    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();

    if norm_a == 0.0 || norm_b == 0.0 {
        return 0.0;
    }

    dot / (norm_a * norm_b)
}

/// Generate a deterministic embedding (fallback mode).
///
/// Uses SHA-256 trigram hashing to produce a consistent 384-dimensional
/// vector for the same input text. This ensures teach-then-match works
/// even without an ONNX model loaded.
fn deterministic_embedding(text: &str) -> Vec<f32> {
    use sha2::Digest;

    let mut vec = vec![0.0f32; EMBEDDING_DIM];

    // Trigram-based hashing for local structure
    let chars: Vec<char> = text.chars().collect();
    for i in 0..chars.len().saturating_sub(2) {
        let trigram: String = chars[i..i + 3].iter().collect();
        let hash = sha2::Sha256::digest(trigram.as_bytes());
        for j in 0..8 {
            let idx = (hash[j] as usize) % EMBEDDING_DIM;
            let sign = if hash[(j + 8) % 32] % 2 == 0 {
                1.0f32
            } else {
                -1.0f32
            };
            vec[idx] += sign * 0.1;
        }
    }

    // Whole-word hashing for semantic content
    for word in text.split_whitespace() {
        let word_lower = word.to_lowercase();
        let hash = sha2::Sha256::digest(word_lower.as_bytes());
        for j in 0..6 {
            let idx = (hash[j] as usize + j * 64) % EMBEDDING_DIM;
            let sign = if hash[(j + 8) % 32] % 2 == 0 {
                1.0f32
            } else {
                -1.0f32
            };
            vec[idx] += sign * 0.05;
        }
    }

    // Global text hash for overall similarity
    let global_hash = sha2::Sha256::digest(text.as_bytes());
    for j in 0..12 {
        let idx = (global_hash[j] as usize) % EMBEDDING_DIM;
        let sign = if global_hash[(j + 12) % 32] % 2 == 0 {
            1.0f32
        } else {
            -1.0f32
        };
        vec[idx] += sign * 0.08;
    }

    // L2 normalize
    let norm: f32 = vec.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        for x in vec.iter_mut() {
            *x /= norm;
        }
    }

    vec
}

/// Get the default model path.
fn default_model_path() -> Option<PathBuf> {
    dirs::home_dir().map(|home| home.join(DEFAULT_MODEL_DIR).join(DEFAULT_MODEL_FILENAME))
}

/// Download the ONNX model to the default location.
#[instrument]
pub fn download_model(output_path: Option<&Path>) -> EmbedResult<PathBuf> {
    let output_path = output_path
        .map(|p| p.to_path_buf())
        .or_else(default_model_path)
        .ok_or_else(|| EmbedError::Download("Cannot determine model output path".into()))?;

    if let Some(parent) = output_path.parent() {
        std::fs::create_dir_all(parent)?;
    }

    info!(path = ?output_path, url = DEFAULT_MODEL_URL, "Downloading ONNX model");

    let status = std::process::Command::new("curl")
        .args([
            "-L",
            "-o",
            output_path
                .to_str()
                .ok_or_else(|| EmbedError::Download("Invalid output path encoding".into()))?,
            DEFAULT_MODEL_URL,
        ])
        .status()
        .map_err(|e| EmbedError::Download(format!("Failed to run curl: {}", e)))?;

    if !status.success() {
        return Err(EmbedError::Download(format!(
            "curl exited with status: {}",
            status
        )));
    }

    info!(path = ?output_path, "Model downloaded successfully");
    Ok(output_path)
}

/// Helper module for getting home directory (minimal implementation).
mod dirs {
    use std::path::PathBuf;

    pub fn home_dir() -> Option<PathBuf> {
        std::env::var("HOME")
            .or_else(|_| std::env::var("USERPROFILE"))
            .ok()
            .map(PathBuf::from)
    }
}
