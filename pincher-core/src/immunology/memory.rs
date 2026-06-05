//! Immune memory for the PincherOS immunology system
//!
//! The immune memory stores **antibodies** — learned rejection patterns that
//! block known-bad inputs. Just as a crab's immune system remembers past
//! infections to fight them faster, PincherOS remembers past adversarial
//! patterns to reject them immediately.
//!
//! # Persistence
//!
//! Antibodies are persisted in SQLite so they survive across restarts.
//! Each antibody tracks:
//! - How many times it has been activated (generation count)
//! - When it was last seen matching an input
//! - The regex pattern or embedding threshold that defines it
//!
//! # Antibody Lifecycle
//!
//! 1. **Creation**: When an antigen is detected with high confidence, an
//!    antibody is created from its pattern.
//! 2. **Activation**: Each time the antibody matches an input, its generation
//!    count is incremented and `last_seen` is updated.
//! 3. **Decay**: Antibodies that haven't been activated for a configurable
//!    period can be pruned to prevent false positive accumulation.

use chrono::Utc;
use rusqlite::{params, Connection};
use serde::{Deserialize, Serialize};
use std::path::Path;
use thiserror::Error;
use tracing::{debug, info, instrument, warn};

use super::antigen::AntigenKind;

/// Immune memory errors.
#[derive(Debug, Error)]
pub enum MemoryError {
    /// SQLite database error.
    #[error("Database error: {0}")]
    Database(#[from] rusqlite::Error),

    /// Invalid regex pattern in an antibody.
    #[error("Invalid regex pattern: {0}")]
    InvalidRegex(String),

    /// Antibody not found.
    #[error("Antibody not found: {0}")]
    NotFound(String),

    /// IO error.
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
}

/// Result type for immune memory operations.
pub type MemoryResult<T> = Result<T, MemoryError>;

/// A learned rejection pattern — the immune system's "antibody".
///
/// Antibodies are created from high-confidence antigen detections and
/// stored in immune memory. When a new input matches an antibody's
/// pattern, it is immediately rejected without full detection scanning.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Antibody {
    /// Unique identifier for this antibody.
    pub id: String,
    /// The antigen kind this antibody defends against.
    pub antigen_kind: AntigenKind,
    /// The regex pattern that matches blocked inputs.
    pub pattern: String,
    /// A human-readable description of what this antibody blocks.
    pub description: String,
    /// How many times this antibody has been activated (matched an input).
    pub generation_count: i64,
    /// When this antibody was last activated (RFC 3339).
    pub last_seen: String,
    /// When this antibody was created (RFC 3339).
    pub created_at: String,
}

impl Antibody {
    /// Create a new antibody from a pattern.
    pub fn new(
        antigen_kind: AntigenKind,
        pattern: impl Into<String>,
        description: impl Into<String>,
    ) -> Self {
        let now = Utc::now().to_rfc3339();
        Self {
            id: uuid::Uuid::new_v4().to_string(),
            antigen_kind,
            pattern: pattern.into(),
            description: description.into(),
            generation_count: 0,
            last_seen: now.clone(),
            created_at: now,
        }
    }

    /// Test whether an input matches this antibody's pattern.
    ///
    /// Returns `true` if the pattern matches, `false` otherwise.
    /// If the regex is invalid, logs a warning and returns `false`.
    pub fn matches(&self, input: &str) -> bool {
        match regex::Regex::new(&self.pattern) {
            Ok(re) => re.is_match(input),
            Err(e) => {
                warn!(
                    antibody_id = %self.id,
                    pattern = %self.pattern,
                    error = %e,
                    "Invalid regex in antibody, skipping match"
                );
                false
            }
        }
    }
}

/// SQL to create the antibodies table.
const CREATE_ANTIBODIES: &str = r#"
CREATE TABLE IF NOT EXISTS antibodies (
    id               TEXT PRIMARY KEY,
    antigen_kind     TEXT NOT NULL,
    pattern          TEXT NOT NULL,
    description      TEXT NOT NULL DEFAULT '',
    generation_count INTEGER NOT NULL DEFAULT 0,
    last_seen        TEXT NOT NULL DEFAULT '',
    created_at       TEXT NOT NULL DEFAULT (datetime('now'))
);
"#;

/// SQL to create indexes on the antibodies table.
const CREATE_ANTIBODIES_INDEXES: &str = r#"
CREATE INDEX IF NOT EXISTS idx_antibodies_antigen_kind ON antibodies(antigen_kind);
CREATE INDEX IF NOT EXISTS idx_antibodies_last_seen ON antibodies(last_seen);
"#;

/// Immune memory — persistent storage for antibodies.
///
/// Wraps an SQLite connection and provides CRUD operations for
/// antibody records. The connection is owned and can be opened
/// against a file path or created in-memory for testing.
pub struct ImmuneMemory {
    conn: Connection,
}

impl ImmuneMemory {
    /// Open immune memory from a file path, creating the database if needed.
    #[instrument(skip(path))]
    pub fn open(path: &Path) -> MemoryResult<Self> {
        info!(path = ?path, "Opening immune memory database");

        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }

        let conn = Connection::open(path)?;
        conn.execute_batch("PRAGMA journal_mode=WAL; PRAGMA foreign_keys=ON;")?;

        let memory = Self { conn };
        memory.run_migrations()?;

        info!("Immune memory database opened successfully");
        Ok(memory)
    }

    /// Open immune memory in-memory (for testing).
    pub fn open_in_memory() -> MemoryResult<Self> {
        let conn = Connection::open_in_memory()?;
        conn.execute_batch("PRAGMA journal_mode=WAL; PRAGMA foreign_keys=ON;")?;

        let memory = Self { conn };
        memory.run_migrations()?;

        Ok(memory)
    }

    /// Run database migrations for the antibodies table.
    fn run_migrations(&self) -> MemoryResult<()> {
        debug!("Running immune memory migrations");
        self.conn.execute_batch(CREATE_ANTIBODIES)?;
        self.conn.execute_batch(CREATE_ANTIBODIES_INDEXES)?;
        debug!("Immune memory migrations complete");
        Ok(())
    }

    /// Store a new antibody in immune memory.
    #[instrument(skip(self, antibody))]
    pub fn store_antibody(&self, antibody: &Antibody) -> MemoryResult<()> {
        debug!(
            antibody_id = %antibody.id,
            kind = %antibody.antigen_kind,
            "Storing antibody"
        );

        self.conn.execute(
            "INSERT INTO antibodies (id, antigen_kind, pattern, description, generation_count, last_seen, created_at)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)",
            params![
                antibody.id,
                antibody.antigen_kind.to_string(),
                antibody.pattern,
                antibody.description,
                antibody.generation_count,
                antibody.last_seen,
                antibody.created_at,
            ],
        )?;

        info!(
            antibody_id = %antibody.id,
            kind = %antibody.antigen_kind,
            "Antibody stored in immune memory"
        );
        Ok(())
    }

    /// Activate an antibody — increment its generation count and update last_seen.
    ///
    /// Returns the updated generation count, or an error if the antibody doesn't exist.
    #[instrument(skip(self))]
    pub fn activate_antibody(&self, id: &str) -> MemoryResult<i64> {
        let now = Utc::now().to_rfc3339();

        let rows_affected = self.conn.execute(
            "UPDATE antibodies SET generation_count = generation_count + 1, last_seen = ?1 WHERE id = ?2",
            params![now, id],
        )?;

        if rows_affected == 0 {
            return Err(MemoryError::NotFound(id.to_string()));
        }

        let generation_count: i64 = self.conn.query_row(
            "SELECT generation_count FROM antibodies WHERE id = ?1",
            params![id],
            |row| row.get(0),
        )?;

        debug!(
            antibody_id = id,
            generation_count = generation_count,
            "Antibody activated"
        );

        Ok(generation_count)
    }

    /// Find all antibodies that match the given input.
    ///
    /// Returns a list of antibodies whose regex patterns match the input,
    /// along with their current generation counts.
    #[instrument(skip(self, input))]
    pub fn find_matching_antibodies(&self, input: &str) -> MemoryResult<Vec<Antibody>> {
        let all = self.list_all()?;
        let matching: Vec<Antibody> = all.into_iter().filter(|ab| ab.matches(input)).collect();

        if !matching.is_empty() {
            debug!(
                match_count = matching.len(),
                "Found matching antibodies for input"
            );
        }

        Ok(matching)
    }

    /// Find antibodies for a specific antigen kind.
    #[instrument(skip(self))]
    pub fn find_by_kind(&self, kind: &AntigenKind) -> MemoryResult<Vec<Antibody>> {
        let mut stmt = self.conn.prepare(
            "SELECT id, antigen_kind, pattern, description, generation_count, last_seen, created_at
             FROM antibodies WHERE antigen_kind = ?1",
        )?;

        let rows = stmt.query_map(params![kind.to_string()], |row| Ok(antibody_from_row(row)))?;

        let mut results = Vec::new();
        for row in rows {
            results.push(row?);
        }

        Ok(results)
    }

    /// Get a specific antibody by ID.
    #[instrument(skip(self))]
    pub fn get(&self, id: &str) -> MemoryResult<Option<Antibody>> {
        let result = self.conn.query_row(
            "SELECT id, antigen_kind, pattern, description, generation_count, last_seen, created_at
             FROM antibodies WHERE id = ?1",
            params![id],
            |row| Ok(antibody_from_row(row)),
        );

        match result {
            Ok(ab) => Ok(Some(ab)),
            Err(rusqlite::Error::QueryReturnedNoRows) => Ok(None),
            Err(e) => Err(MemoryError::Database(e)),
        }
    }

    /// Remove an antibody from immune memory.
    #[instrument(skip(self))]
    pub fn remove_antibody(&self, id: &str) -> MemoryResult<bool> {
        let rows_affected = self
            .conn
            .execute("DELETE FROM antibodies WHERE id = ?1", params![id])?;

        if rows_affected > 0 {
            info!(antibody_id = id, "Antibody removed from immune memory");
            Ok(true)
        } else {
            debug!(antibody_id = id, "Antibody not found for removal");
            Ok(false)
        }
    }

    /// List all antibodies in immune memory.
    #[instrument(skip(self))]
    pub fn list_all(&self) -> MemoryResult<Vec<Antibody>> {
        let mut stmt = self.conn.prepare(
            "SELECT id, antigen_kind, pattern, description, generation_count, last_seen, created_at
             FROM antibodies ORDER BY created_at DESC",
        )?;

        let rows = stmt.query_map([], |row| Ok(antibody_from_row(row)))?;

        let mut results = Vec::new();
        for row in rows {
            results.push(row?);
        }

        Ok(results)
    }

    /// Get the total number of antibodies.
    #[instrument(skip(self))]
    pub fn count(&self) -> MemoryResult<i64> {
        let count = self
            .conn
            .query_row("SELECT COUNT(*) FROM antibodies", [], |row| row.get(0))?;
        Ok(count)
    }

    /// Prune antibodies that haven't been activated since the given timestamp.
    ///
    /// Returns the number of antibodies pruned.
    #[instrument(skip(self))]
    pub fn prune_older_than(&self, cutoff: &str) -> MemoryResult<usize> {
        let rows_affected = self.conn.execute(
            "DELETE FROM antibodies WHERE last_seen < ?1 AND generation_count < 3",
            params![cutoff],
        )?;

        if rows_affected > 0 {
            info!(
                pruned_count = rows_affected,
                cutoff = cutoff,
                "Pruned inactive antibodies"
            );
        }

        Ok(rows_affected)
    }

    /// Check if any antibody blocks the given input.
    ///
    /// Returns `true` if any antibody matches, `false` otherwise.
    /// Also activates any matching antibodies (increments their generation count).
    #[instrument(skip(self, input))]
    pub fn is_blocked(&self, input: &str) -> MemoryResult<bool> {
        let matching = self.find_matching_antibodies(input)?;

        if matching.is_empty() {
            return Ok(false);
        }

        // Activate all matching antibodies
        for ab in &matching {
            if let Err(e) = self.activate_antibody(&ab.id) {
                warn!(
                    antibody_id = %ab.id,
                    error = %e,
                    "Failed to activate matching antibody"
                );
            }
        }

        debug!(match_count = matching.len(), "Input blocked by antibodies");

        Ok(true)
    }
}

/// Helper to construct an Antibody from a database row.
fn antibody_from_row(row: &rusqlite::Row<'_>) -> Antibody {
    let kind_str: String = row.get(1).unwrap_or_default();
    let antigen_kind = match kind_str.as_str() {
        "prompt_injection" => AntigenKind::PromptInjection,
        "malicious_action" => AntigenKind::MaliciousAction,
        "resource_abuse" => AntigenKind::ResourceAbuse,
        "stale_reflex" => AntigenKind::StaleReflex,
        _ => AntigenKind::PromptInjection, // fallback
    };

    Antibody {
        id: row.get(0).unwrap_or_default(),
        antigen_kind,
        pattern: row.get(2).unwrap_or_default(),
        description: row.get(3).unwrap_or_default(),
        generation_count: row.get(4).unwrap_or(0),
        last_seen: row.get(5).unwrap_or_default(),
        created_at: row.get(6).unwrap_or_default(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_open_in_memory() {
        let memory = ImmuneMemory::open_in_memory().unwrap();
        assert_eq!(memory.count().unwrap(), 0);
    }

    #[test]
    fn test_store_and_retrieve_antibody() {
        let memory = ImmuneMemory::open_in_memory().unwrap();

        let antibody = Antibody::new(
            AntigenKind::PromptInjection,
            r"(?i)ignore\s+previous\s+instructions",
            "Blocks instruction override attempts",
        );

        let id = antibody.id.clone();
        memory.store_antibody(&antibody).unwrap();

        let retrieved = memory.get(&id).unwrap().unwrap();
        assert_eq!(retrieved.pattern, antibody.pattern);
        assert_eq!(retrieved.antigen_kind, AntigenKind::PromptInjection);
        assert_eq!(retrieved.generation_count, 0);
    }

    #[test]
    fn test_activate_antibody() {
        let memory = ImmuneMemory::open_in_memory().unwrap();

        let antibody = Antibody::new(
            AntigenKind::MaliciousAction,
            r"sh\s+-c",
            "Blocks sh -c commands",
        );

        let id = antibody.id.clone();
        memory.store_antibody(&antibody).unwrap();

        let gen = memory.activate_antibody(&id).unwrap();
        assert_eq!(gen, 1);

        let gen = memory.activate_antibody(&id).unwrap();
        assert_eq!(gen, 2);

        let retrieved = memory.get(&id).unwrap().unwrap();
        assert_eq!(retrieved.generation_count, 2);
    }

    #[test]
    fn test_activate_nonexistent_antibody() {
        let memory = ImmuneMemory::open_in_memory().unwrap();
        let result = memory.activate_antibody("nonexistent");
        assert!(result.is_err());
    }

    #[test]
    fn test_find_matching_antibodies() {
        let memory = ImmuneMemory::open_in_memory().unwrap();

        let ab1 = Antibody::new(
            AntigenKind::PromptInjection,
            r"(?i)ignore\s+previous",
            "Block override",
        );
        let ab2 = Antibody::new(AntigenKind::MaliciousAction, r"sh\s+-c", "Block shell exec");

        memory.store_antibody(&ab1).unwrap();
        memory.store_antibody(&ab2).unwrap();

        let matches = memory
            .find_matching_antibodies("ignore previous instructions")
            .unwrap();
        assert_eq!(matches.len(), 1);
        assert_eq!(matches[0].antigen_kind, AntigenKind::PromptInjection);
    }

    #[test]
    fn test_find_by_kind() {
        let memory = ImmuneMemory::open_in_memory().unwrap();

        let ab1 = Antibody::new(
            AntigenKind::PromptInjection,
            r"(?i)jailbreak",
            "Block jailbreak",
        );
        let ab2 = Antibody::new(
            AntigenKind::MaliciousAction,
            r"DROP\s+TABLE",
            "Block SQL injection",
        );

        memory.store_antibody(&ab1).unwrap();
        memory.store_antibody(&ab2).unwrap();

        let injection_abs = memory.find_by_kind(&AntigenKind::PromptInjection).unwrap();
        assert_eq!(injection_abs.len(), 1);

        let action_abs = memory.find_by_kind(&AntigenKind::MaliciousAction).unwrap();
        assert_eq!(action_abs.len(), 1);
    }

    #[test]
    fn test_is_blocked() {
        let memory = ImmuneMemory::open_in_memory().unwrap();

        // No antibodies yet
        assert!(!memory.is_blocked("ignore previous instructions").unwrap());

        let antibody = Antibody::new(
            AntigenKind::PromptInjection,
            r"(?i)ignore\s+previous\s+instructions",
            "Block override",
        );
        memory.store_antibody(&antibody).unwrap();

        // Now it should be blocked
        assert!(memory.is_blocked("ignore previous instructions").unwrap());
        // And the antibody should be activated
        let retrieved = memory.get(&antibody.id).unwrap().unwrap();
        assert_eq!(retrieved.generation_count, 1);
    }

    #[test]
    fn test_remove_antibody() {
        let memory = ImmuneMemory::open_in_memory().unwrap();

        let antibody = Antibody::new(AntigenKind::PromptInjection, r"test", "Test antibody");

        memory.store_antibody(&antibody).unwrap();
        assert_eq!(memory.count().unwrap(), 1);

        let removed = memory.remove_antibody(&antibody.id).unwrap();
        assert!(removed);
        assert_eq!(memory.count().unwrap(), 0);

        let removed_again = memory.remove_antibody(&antibody.id).unwrap();
        assert!(!removed_again);
    }

    #[test]
    fn test_list_all() {
        let memory = ImmuneMemory::open_in_memory().unwrap();

        let ab1 = Antibody::new(AntigenKind::PromptInjection, r"a", "A");
        let ab2 = Antibody::new(AntigenKind::MaliciousAction, r"b", "B");

        memory.store_antibody(&ab1).unwrap();
        memory.store_antibody(&ab2).unwrap();

        let all = memory.list_all().unwrap();
        assert_eq!(all.len(), 2);
    }

    #[test]
    fn test_antibody_matches() {
        let antibody = Antibody::new(
            AntigenKind::PromptInjection,
            r"(?i)ignore\s+previous",
            "Block override",
        );

        assert!(antibody.matches("ignore previous instructions"));
        assert!(antibody.matches("IGNORE PREVIOUS RULES"));
        assert!(!antibody.matches("what is the weather?"));
    }

    #[test]
    fn test_antibody_invalid_regex() {
        let antibody = Antibody::new(
            AntigenKind::PromptInjection,
            r"([invalid", // Invalid regex
            "Broken pattern",
        );

        // Should return false and not panic
        assert!(!antibody.matches("anything"));
    }

    #[test]
    fn test_prune_older_than() {
        let memory = ImmuneMemory::open_in_memory().unwrap();

        let antibody = Antibody::new(AntigenKind::PromptInjection, r"test", "Test");

        // Manually set last_seen to the past
        memory.store_antibody(&antibody).unwrap();
        memory
            .conn
            .execute(
                "UPDATE antibodies SET last_seen = '2020-01-01T00:00:00Z' WHERE id = ?1",
                params![antibody.id],
            )
            .unwrap();

        let pruned = memory.prune_older_than("2025-01-01T00:00:00Z").unwrap();
        assert_eq!(pruned, 1);
        assert_eq!(memory.count().unwrap(), 0);
    }

    #[test]
    fn test_prune_does_not_remove_active_antibodies() {
        let memory = ImmuneMemory::open_in_memory().unwrap();

        let antibody = Antibody::new(AntigenKind::PromptInjection, r"test", "Test");

        memory.store_antibody(&antibody).unwrap();

        // Activate it 3+ times so it's "well-established"
        for _ in 0..3 {
            memory.activate_antibody(&antibody.id).unwrap();
        }

        // Set last_seen to the past
        memory
            .conn
            .execute(
                "UPDATE antibodies SET last_seen = '2020-01-01T00:00:00Z' WHERE id = ?1",
                params![antibody.id],
            )
            .unwrap();

        // Should NOT prune because generation_count >= 3
        let pruned = memory.prune_older_than("2025-01-01T00:00:00Z").unwrap();
        assert_eq!(pruned, 0);
        assert_eq!(memory.count().unwrap(), 1);
    }
}
