//! Reflex matching logic for PincherOS
//!
//! Matches incoming intents against stored reflexes using vector similarity
//! search (sqlite-vec) with cosine similarity re-ranking.

use crate::db::schema::{self, ReflexRow};
use crate::embed::{cosine_similarity, Embedder};
use rusqlite::Connection;
use serde::{Deserialize, Serialize};
use thiserror::Error;
use tracing::{debug, info, instrument, warn};

/// Matching errors.
#[derive(Debug, Error)]
pub enum MatchError {
    #[error("Database error: {0}")]
    Db(#[from] rusqlite::Error),

    #[error("Embedding error: {0}")]
    Embed(#[from] crate::embed::EmbedError),

    #[error("No reflexes found in database")]
    NoReflexes,
}

/// Result type for matching operations.
pub type MatchOpResult<T> = Result<T, MatchError>;

/// The result of matching an intent against stored reflexes.
///
/// Uses a three-tier classification:
/// - **Exact**: High-confidence match (>0.90 similarity) — short-circuit execution
/// - **Similar**: Moderate match (>0.70) — route through LLM for refinement
/// - **Novel**: No good match (<0.70) — new reflex territory, needs LLM guidance
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MatchResult {
    /// High-confidence match — can short-circuit directly.
    Exact {
        /// Cosine similarity score.
        similarity: f32,
        /// The matched reflex.
        reflex: ReflexRow,
    },
    /// Moderate match — needs LLM refinement but has a starting point.
    Similar {
        /// Cosine similarity score.
        similarity: f32,
        /// The best-matching reflex.
        reflex: ReflexRow,
    },
    /// No good match — this is novel territory.
    Novel {
        /// Best similarity score found (below threshold).
        best_similarity: f32,
    },
}

impl MatchResult {
    /// Get the similarity score from the match result.
    pub fn similarity(&self) -> f32 {
        match self {
            MatchResult::Exact { similarity, .. } => *similarity,
            MatchResult::Similar { similarity, .. } => *similarity,
            MatchResult::Novel { best_similarity } => *best_similarity,
        }
    }

    /// Get a reference to the matched reflex, if any.
    pub fn reflex(&self) -> Option<&ReflexRow> {
        match self {
            MatchResult::Exact { reflex, .. } => Some(reflex),
            MatchResult::Similar { reflex, .. } => Some(reflex),
            MatchResult::Novel { .. } => None,
        }
    }

    /// Check if this is an exact match.
    pub fn is_exact(&self) -> bool {
        matches!(self, MatchResult::Exact { .. })
    }

    /// Check if this is a similar match.
    pub fn is_similar(&self) -> bool {
        matches!(self, MatchResult::Similar { .. })
    }

    /// Check if this is a novel (no match) result.
    pub fn is_novel(&self) -> bool {
        matches!(self, MatchResult::Novel { .. })
    }
}

/// Similarity thresholds for match classification.
///
/// Calibrated for all-MiniLM-L6-v2 based on empirical STS benchmark data:
/// - Exact: 0.80 (very similar paraphrases typically score 0.80-0.95)
/// - Similar: 0.55 (semantically related intents score 0.55-0.80)
pub struct MatchThresholds {
    /// Above this = Exact match. Default: 0.80 (calibrated for MiniLM-L6-v2)
    pub exact: f32,
    /// Above this = Similar match. Default: 0.55
    pub similar: f32,
}

impl Default for MatchThresholds {
    fn default() -> Self {
        Self {
            exact: 0.80,
            similar: 0.55,
        }
    }
}

/// Match an intent string against stored reflexes.
#[instrument(skip(conn, embedder))]
pub fn match_reflex(
    conn: &Connection,
    embedder: &Embedder,
    intent: &str,
) -> MatchOpResult<MatchResult> {
    match_reflex_with_thresholds(conn, embedder, intent, &MatchThresholds::default())
}

/// Match an intent with custom thresholds.
#[instrument(skip(conn, embedder, thresholds))]
pub fn match_reflex_with_thresholds(
    conn: &Connection,
    embedder: &Embedder,
    intent: &str,
    thresholds: &MatchThresholds,
) -> MatchOpResult<MatchResult> {
    info!(intent = intent, "Matching reflex for intent");

    // Step 1: Embed the intent
    let query_embedding = embedder.embed(intent)?;

    // Step 2: Try exact string match first (fast path)
    if let Some(exact_reflex) = schema::get_reflex_by_intent(conn, intent)? {
        let similarity = if exact_reflex.embedding.iter().all(|&v| v == 0.0) {
            let re_embedded = embedder.embed(&exact_reflex.intent)?;
            cosine_similarity(&query_embedding, &re_embedded)
        } else {
            cosine_similarity(&query_embedding, &exact_reflex.embedding)
        };

        info!(
            intent = intent,
            match_type = "exact_string",
            similarity = similarity,
            "Found exact string match for intent"
        );

        return Ok(MatchResult::Exact {
            similarity, // Use actual similarity, not inflated
            reflex: exact_reflex,
        });
    }

    // Step 3: Vector similarity search via sqlite-vec
    let nearest = schema::search_nearest(conn, &query_embedding, 5)?;

    if nearest.is_empty() {
        info!(
            intent = intent,
            match_type = "novel",
            "No reflexes found — novel territory"
        );
        return Ok(MatchResult::Novel {
            best_similarity: 0.0,
        });
    }

    // Step 4: Re-rank with cosine similarity and pick the best
    let mut best_match: Option<(f32, ReflexRow)> = None;

    for (id, _vec_distance, reflex) in nearest {
        let similarity = if reflex.embedding.iter().all(|&v| v == 0.0) {
            match embedder.embed(&reflex.intent) {
                Ok(re_embedded) => cosine_similarity(&query_embedding, &re_embedded),
                Err(e) => {
                    warn!(reflex_id = id, error = %e, "Failed to re-embed reflex, skipping");
                    continue;
                }
            }
        } else {
            cosine_similarity(&query_embedding, &reflex.embedding)
        };

        debug!(
            reflex_id = id,
            intent = %reflex.intent,
            similarity = similarity,
            "Candidate match"
        );

        match &best_match {
            Some((best_sim, _)) if similarity <= *best_sim => {}
            _ => best_match = Some((similarity, reflex)),
        }
    }

    let (best_similarity, best_reflex) = match best_match {
        Some(m) => m,
        None => {
            return Ok(MatchResult::Novel {
                best_similarity: 0.0,
            });
        }
    };

    // Step 5: Classify based on thresholds
    let result = if best_similarity >= thresholds.exact {
        MatchResult::Exact {
            similarity: best_similarity,
            reflex: best_reflex,
        }
    } else if best_similarity >= thresholds.similar {
        MatchResult::Similar {
            similarity: best_similarity,
            reflex: best_reflex,
        }
    } else {
        MatchResult::Novel { best_similarity }
    };

    Ok(result)
}

/// Find all reflexes similar to the given intent above the given threshold.
#[instrument(skip(conn, embedder))]
pub fn find_similar_reflexes(
    conn: &Connection,
    embedder: &Embedder,
    intent: &str,
    min_similarity: f32,
    limit: usize,
) -> MatchOpResult<Vec<(f32, ReflexRow)>> {
    let query_embedding = embedder.embed(intent)?;
    let nearest = schema::search_nearest(conn, &query_embedding, limit)?;

    let mut results = Vec::new();
    for (_id, _vec_distance, reflex) in nearest {
        let similarity = if reflex.embedding.iter().all(|&v| v == 0.0) {
            match embedder.embed(&reflex.intent) {
                Ok(re_embedded) => cosine_similarity(&query_embedding, &re_embedded),
                Err(_) => continue,
            }
        } else {
            cosine_similarity(&query_embedding, &reflex.embedding)
        };

        if similarity >= min_similarity {
            results.push((similarity, reflex));
        }
    }

    results.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
    Ok(results)
}
