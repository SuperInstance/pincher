//! Database module for PincherOS

pub mod schema;

use rusqlite::Connection;
use schema::*;
use std::path::Path;
use thiserror::Error;
use tracing::instrument;

/// Database errors.
#[derive(Debug, Error)]
pub enum DbError {
    #[error("SQLite error: {0}")]
    Sqlite(#[from] rusqlite::Error),

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Serialization error: {0}")]
    Serialization(#[from] serde_json::Error),
}

/// Result type for database operations.
pub type DbResult<T> = Result<T, DbError>;

/// The main database wrapper for PincherOS.
pub struct Database {
    conn: Connection,
}

impl Database {
    /// Open and initialize the database at the given path.
    #[instrument(skip(path))]
    pub fn open(path: &Path) -> DbResult<Self> {
        let conn = schema::init_db(path)?;
        Ok(Self { conn })
    }

    /// Open an in-memory database.
    pub fn open_in_memory() -> DbResult<Self> {
        let conn = Connection::open_in_memory()?;
        conn.execute_batch("PRAGMA journal_mode=WAL; PRAGMA foreign_keys=ON;")?;
        schema::run_migrations(&conn)?;
        Ok(Self { conn })
    }

    /// Get a reference to the underlying SQLite connection.
    pub fn connection(&self) -> &Connection {
        &self.conn
    }

    /// Get a mutable reference to the underlying SQLite connection.
    pub fn connection_mut(&mut self) -> &mut Connection {
        &mut self.conn
    }

    // ── Reflex CRUD ───────────────────────────────────────────────

    pub fn insert_reflex(
        &self,
        id: &str,
        intent: &str,
        action_sql: &str,
        embedding: &[f32],
        confidence: f64,
    ) -> DbResult<()> {
        schema::insert_reflex(&self.conn, id, intent, action_sql, embedding, confidence)?;
        Ok(())
    }

    pub fn get_reflex_by_intent(&self, intent: &str) -> DbResult<Option<ReflexRow>> {
        Ok(schema::get_reflex_by_intent(&self.conn, intent)?)
    }

    pub fn get_reflex_by_id(&self, id: &str) -> DbResult<Option<ReflexRow>> {
        Ok(schema::get_reflex_by_id(&self.conn, id)?)
    }

    pub fn search_nearest(
        &self,
        query_embedding: &[f32],
        limit: usize,
    ) -> DbResult<Vec<(String, f32, ReflexRow)>> {
        Ok(schema::search_nearest(&self.conn, query_embedding, limit)?)
    }

    pub fn update_reflex_embedding(&self, id: &str, embedding: &[f32]) -> DbResult<()> {
        schema::update_reflex_embedding(&self.conn, id, embedding)?;
        Ok(())
    }

    pub fn update_reflex_confidence(&self, id: &str, confidence: f64) -> DbResult<()> {
        schema::update_reflex_confidence(&self.conn, id, confidence)?;
        Ok(())
    }

    pub fn increment_reflex_invoke(&self, id: &str) -> DbResult<()> {
        schema::increment_reflex_invoke(&self.conn, id)?;
        Ok(())
    }

    pub fn get_all_reflexes(&self) -> DbResult<Vec<ReflexRow>> {
        Ok(schema::get_all_reflexes(&self.conn)?)
    }

    pub fn get_reflex_count(&self) -> DbResult<i64> {
        Ok(schema::get_reflex_count(&self.conn)?)
    }

    // ── Action Log ────────────────────────────────────────────────

    pub fn log_action(
        &self,
        reflex_id: &str,
        input: &str,
        output: &str,
        latency_ms: i64,
        confidence: f64,
    ) -> DbResult<()> {
        schema::log_action(&self.conn, reflex_id, input, output, latency_ms, confidence)?;
        Ok(())
    }

    pub fn get_recent_actions(&self, limit: usize) -> DbResult<Vec<ActionLogRow>> {
        Ok(schema::get_recent_actions(&self.conn, limit)?)
    }

    pub fn get_action_log_count(&self) -> DbResult<i64> {
        Ok(schema::get_action_log_count(&self.conn)?)
    }

    // ── Shell ─────────────────────────────────────────────────────

    pub fn upsert_shell(
        &self,
        fingerprint: &str,
        hostname: &str,
        os: &str,
        cpu_count: i64,
        ram_mb: i64,
        gpu: &str,
    ) -> DbResult<()> {
        schema::upsert_shell(
            &self.conn,
            fingerprint,
            hostname,
            os,
            cpu_count,
            ram_mb,
            gpu,
        )?;
        Ok(())
    }

    // ── Session ───────────────────────────────────────────────────

    pub fn create_session(&self, shell_fingerprint: &str) -> DbResult<String> {
        Ok(schema::create_session(&self.conn, shell_fingerprint)?)
    }

    pub fn update_session_state(&self, id: &str, state: &str) -> DbResult<()> {
        schema::update_session_state(&self.conn, id, state)?;
        Ok(())
    }

    /// Checkpoint the WAL.
    pub fn checkpoint_wal(&self) -> DbResult<()> {
        self.conn
            .execute_batch("PRAGMA wal_checkpoint(TRUNCATE);")?;
        Ok(())
    }

    /// End all active sessions.
    pub fn end_all_sessions(&self) -> DbResult<()> {
        self.conn.execute(
            "UPDATE sessions SET state = 'ended' WHERE state = 'active'",
            [],
        )?;
        Ok(())
    }

    /// Get a raw connection reference (for shell profile storage).
    pub fn conn(&self) -> &Connection {
        &self.conn
    }

    /// Start a new session with the given shell fingerprint.
    pub fn start_session(&self, shell_fingerprint: &str) -> DbResult<String> {
        self.create_session(shell_fingerprint)
    }
}
