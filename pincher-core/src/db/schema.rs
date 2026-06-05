//! SQLite + sqlite-vec schema management for PincherOS
//!
//! Manages the persistent store for reflexes, sessions, action logs,
//! and shell fingerprints. Includes vector search via sqlite-vec.

use rusqlite::{params, Connection, Result as SqlResult};
use serde::{Deserialize, Serialize};
use std::path::Path;
use tracing::{debug, info, instrument, warn};

/// Row representation of a reflex from the database.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReflexRow {
    pub id: String,
    pub intent: String,
    pub action_sql: String,
    pub embedding: Vec<f32>,
    pub confidence: f64,
    pub invoke_count: i64,
    pub last_invoked: String,
    pub created_at: String,
}

/// Row representation of a session.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionRow {
    pub id: String,
    pub shell_fingerprint: String,
    pub state: String,
    pub started_at: String,
}

/// Row representation of an action log entry.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ActionLogRow {
    pub id: String,
    pub reflex_id: String,
    pub input: String,
    pub output: String,
    pub latency_ms: i64,
    pub confidence: f64,
    pub created_at: String,
}

/// Row representation of a shell fingerprint.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ShellRow {
    pub fingerprint: String,
    pub hostname: String,
    pub os: String,
    pub cpu_count: i64,
    pub ram_mb: i64,
    pub gpu: String,
    pub last_seen: String,
}

/// Embedding dimensionality for all-MiniLM-L6-v2.
pub const EMBEDDING_DIM: usize = 384;

/// SQL to create the reflexes table.
const CREATE_REFLEXES: &str = r#"
CREATE TABLE IF NOT EXISTS reflexes (
    id         TEXT PRIMARY KEY,
    intent     TEXT NOT NULL,
    action_sql TEXT NOT NULL,
    embedding  BLOB NOT NULL,
    confidence REAL NOT NULL DEFAULT 0.5,
    invoke_count INTEGER NOT NULL DEFAULT 0,
    last_invoked TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);
"#;

/// SQL to create the sessions table.
const CREATE_SESSIONS: &str = r#"
CREATE TABLE IF NOT EXISTS sessions (
    id                TEXT PRIMARY KEY,
    shell_fingerprint TEXT NOT NULL,
    state             TEXT NOT NULL DEFAULT 'active',
    started_at        TEXT NOT NULL DEFAULT (datetime('now'))
);
"#;

/// SQL to create the action_log table.
const CREATE_ACTION_LOG: &str = r#"
CREATE TABLE IF NOT EXISTS action_log (
    id          TEXT PRIMARY KEY,
    reflex_id   TEXT NOT NULL,
    input       TEXT NOT NULL DEFAULT '',
    output      TEXT NOT NULL DEFAULT '',
    latency_ms  INTEGER NOT NULL DEFAULT 0,
    confidence  REAL NOT NULL DEFAULT 0.0,
    created_at  TEXT NOT NULL DEFAULT (datetime('now')),
    FOREIGN KEY (reflex_id) REFERENCES reflexes(id)
);
"#;

/// SQL to create the shells table.
const CREATE_SHELLS: &str = r#"
CREATE TABLE IF NOT EXISTS shells (
    fingerprint TEXT PRIMARY KEY,
    hostname    TEXT NOT NULL DEFAULT '',
    os          TEXT NOT NULL DEFAULT '',
    cpu_count   INTEGER NOT NULL DEFAULT 0,
    ram_mb      INTEGER NOT NULL DEFAULT 0,
    gpu         TEXT NOT NULL DEFAULT '',
    last_seen   TEXT NOT NULL DEFAULT (datetime('now'))
);
"#;

/// SQL to create the sqlite-vec virtual table for vector search.
const CREATE_VEC_REFLEXES: &str = r#"
CREATE VIRTUAL TABLE IF NOT EXISTS vec_reflexes
USING vec0(
    id      TEXT PRIMARY KEY,
    embedding float[384]
);
"#;

/// Migration tracking table.
const CREATE_MIGRATIONS: &str = r#"
CREATE TABLE IF NOT EXISTS _migrations (
    id   INTEGER PRIMARY KEY,
    name TEXT NOT NULL UNIQUE,
    applied_at TEXT NOT NULL DEFAULT (datetime('now'))
);
"#;

/// Built-in reflex definitions: (intent, action_sql, description).
const BUILTIN_REFLEXES: &[(&str, &str, &str)] = &[
    (
        "system.info",
        "SELECT hostname, os, cpu_count, ram_mb, gpu, last_seen FROM shells ORDER BY last_seen DESC LIMIT 1",
        "Show current system information",
    ),
    (
        "file.read",
        "SELECT read_file('{{path}}')",
        "Read contents of a file at the given path",
    ),
    (
        "file.write",
        "SELECT write_file('{{path}}', '{{content}}')",
        "Write content to a file at the given path",
    ),
    (
        "process.list",
        "SELECT pid, name, cpu_usage, memory FROM processes ORDER BY cpu_usage DESC LIMIT 20",
        "List running processes sorted by CPU usage",
    ),
    (
        "process.kill",
        "SELECT kill_process({{pid}})",
        "Kill a process by PID",
    ),
    (
        "network.ping",
        "SELECT ping('{{host}}', {{timeout_ms}})",
        "Ping a network host",
    ),
    (
        "git.status",
        "SELECT git_status('{{repo_path}}')",
        "Show git working tree status",
    ),
    (
        "git.diff",
        "SELECT git_diff('{{repo_path}}', '{{revision}}')",
        "Show git diff for a repository",
    ),
    (
        "docker.ps",
        "SELECT container_id, image, status, names FROM docker_containers LIMIT 20",
        "List running Docker containers",
    ),
    (
        "env.get",
        "SELECT env_var('{{name}}')",
        "Get an environment variable value",
    ),
];

/// Initialize the database at the given path, creating tables and running migrations.
#[instrument(skip(path))]
pub fn init_db(path: &Path) -> SqlResult<Connection> {
    info!("Initializing PincherOS database at {:?}", path);

    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).ok();
    }

    // Register sqlite-vec extension BEFORE opening the connection.
    // sqlite3_auto_extension must be called before Connection::open so
    // that the vec0 virtual table module is available when tables are created.
    register_sqlite_vec();

    let conn = Connection::open(path)?;
    conn.execute_batch("PRAGMA journal_mode=WAL; PRAGMA foreign_keys=ON;")?;

    run_migrations(&conn)?;
    seed_builtins(&conn)?;

    info!("Database initialized successfully");
    Ok(conn)
}

/// Register the sqlite-vec extension globally via `sqlite3_auto_extension`.
///
/// This must be called BEFORE any SQLite connections are opened. The
/// registration is process-wide and idempotent — calling it multiple times
/// is safe but unnecessary.
fn register_sqlite_vec() {
    static REGISTERED: std::sync::Once = std::sync::Once::new();
    REGISTERED.call_once(|| {
        debug!("Registering sqlite-vec extension (static auto-extension)");
        unsafe {
            #[allow(clippy::missing_transmute_annotations)]
            rusqlite::ffi::sqlite3_auto_extension(Some(std::mem::transmute(
                sqlite_vec::sqlite3_vec_init as *const (),
            )));
        }
        debug!("sqlite-vec extension registered successfully");
    });
}

/// Load the sqlite-vec extension into an already-opened connection.
///
/// This is only needed if the connection was opened before `register_sqlite_vec`
/// was called. Prefer calling `register_sqlite_vec()` before opening connections.
#[allow(dead_code)]
fn load_sqlite_vec(_conn: &Connection) -> SqlResult<()> {
    // No-op: the extension should already be registered via auto_extension.
    // Keeping this function for API compatibility.
    Ok(())
}

/// Run idempotent migrations. Each migration is tracked in _migrations table.
#[instrument(skip(conn))]
pub fn run_migrations(conn: &Connection) -> SqlResult<()> {
    info!("Running database migrations");

    conn.execute_batch(CREATE_MIGRATIONS)?;

    // Migration 1: Create core tables
    run_migration(conn, 1, "create_core_tables", || {
        conn.execute_batch(&format!(
            "{}\n{}\n{}\n{}\n{}",
            CREATE_REFLEXES, CREATE_SESSIONS, CREATE_ACTION_LOG, CREATE_SHELLS, CREATE_VEC_REFLEXES
        ))?;
        Ok(())
    })?;

    // Migration 2: Create indexes
    run_migration(conn, 2, "create_indexes", || {
        conn.execute_batch(
            "CREATE INDEX IF NOT EXISTS idx_reflexes_intent ON reflexes(intent);
             CREATE INDEX IF NOT EXISTS idx_reflexes_confidence ON reflexes(confidence);
             CREATE INDEX IF NOT EXISTS idx_action_log_reflex_id ON action_log(reflex_id);
             CREATE INDEX IF NOT EXISTS idx_action_log_created_at ON action_log(created_at);
             CREATE INDEX IF NOT EXISTS idx_sessions_shell_fingerprint ON sessions(shell_fingerprint);",
        )?;
        Ok(())
    })?;

    // Migration 3: Add invoke_count column if not exists (idempotent via try-catch pattern)
    run_migration(conn, 3, "add_reflex_invoke_count", || {
        // Try adding the column; if it already exists, sqlite will error, which we handle
        let result = conn.execute_batch(
            "ALTER TABLE reflexes ADD COLUMN invoke_count INTEGER NOT NULL DEFAULT 0;",
        );
        match result {
            Ok(()) => Ok(()),
            Err(e) => {
                // Column already exists — this is fine
                if e.to_string().contains("duplicate column name") {
                    debug!("invoke_count column already exists, skipping");
                    Ok(())
                } else {
                    Err(e)
                }
            }
        }
    })?;

    info!("Migrations complete");
    Ok(())
}

/// Run a single named migration idempotently.
fn run_migration<F>(conn: &Connection, id: i64, name: &str, f: F) -> SqlResult<()>
where
    F: FnOnce() -> SqlResult<()>,
{
    let applied: bool = conn
        .query_row(
            "SELECT COUNT(*) FROM _migrations WHERE id = ?1",
            params![id],
            |row| row.get::<_, i64>(0),
        )
        .map(|count| count > 0)?;

    if applied {
        debug!(migration = name, "Migration already applied, skipping");
        return Ok(());
    }

    debug!(migration = name, "Applying migration");
    f()?;

    conn.execute(
        "INSERT INTO _migrations (id, name) VALUES (?1, ?2)",
        params![id, name],
    )?;

    info!(migration = name, "Migration applied successfully");
    Ok(())
}

/// Seed built-in reflexes into the database if they don't already exist.
#[instrument(skip(conn))]
pub fn seed_builtins(conn: &Connection) -> SqlResult<()> {
    info!("Seeding built-in reflexes");

    for (intent, action_sql, _description) in BUILTIN_REFLEXES {
        // Check if this reflex already exists
        let exists: bool = conn
            .query_row(
                "SELECT COUNT(*) FROM reflexes WHERE intent = ?1",
                params![intent],
                |row| row.get::<_, i64>(0),
            )
            .map(|count| count > 0)?;

        if exists {
            debug!(intent = intent, "Built-in reflex already exists, skipping");
            continue;
        }

        let id = uuid::Uuid::new_v4().to_string();
        let now = chrono::Utc::now().to_rfc3339();

        // Use zero embedding for built-ins — they'll be populated later by the embedder
        let zero_embedding = vec![0.0f32; EMBEDDING_DIM];
        let embedding_bytes = embed_to_bytes(&zero_embedding);

        conn.execute(
            "INSERT INTO reflexes (id, intent, action_sql, embedding, confidence, invoke_count, last_invoked, created_at)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)",
            params![id, intent, action_sql, embedding_bytes, 1.0f64, 0i64, "", now],
        )?;

        // Also insert into vec_reflexes for vector search
        conn.execute(
            "INSERT INTO vec_reflexes (id, embedding) VALUES (?1, ?2)",
            params![id, embedding_bytes_to_vec_f32(&zero_embedding)],
        )?;

        debug!(intent = intent, id = id, "Seeded built-in reflex");
    }

    info!("Built-in reflexes seeded");
    Ok(())
}

/// Convert a float32 embedding vector to bytes for BLOB storage.
pub fn embed_to_bytes(embedding: &[f32]) -> Vec<u8> {
    embedding.iter().flat_map(|f| f.to_le_bytes()).collect()
}

/// Convert bytes from BLOB storage back to a float32 embedding vector.
pub fn bytes_to_embed(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(4)
        .map(|chunk| {
            let arr: [u8; 4] = [chunk[0], chunk[1], chunk[2], chunk[3]];
            f32::from_le_bytes(arr)
        })
        .collect()
}

/// Convert embedding bytes to the format expected by sqlite-vec (serialized f32 vector).
fn embedding_bytes_to_vec_f32(embedding: &[f32]) -> Vec<u8> {
    embed_to_bytes(embedding)
}

/// Query a reflex by its intent string.
#[instrument(skip(conn))]
pub fn get_reflex_by_intent(conn: &Connection, intent: &str) -> SqlResult<Option<ReflexRow>> {
    let mut stmt = conn.prepare(
        "SELECT id, intent, action_sql, embedding, confidence, invoke_count, last_invoked, created_at
         FROM reflexes WHERE intent = ?1",
    )?;

    let result = stmt.query_row(params![intent], |row| {
        let embedding_blob: Vec<u8> = row.get(3)?;
        Ok(ReflexRow {
            id: row.get(0)?,
            intent: row.get(1)?,
            action_sql: row.get(2)?,
            embedding: bytes_to_embed(&embedding_blob),
            confidence: row.get(4)?,
            invoke_count: row.get(5)?,
            last_invoked: row.get(6)?,
            created_at: row.get(7)?,
        })
    });

    match result {
        Ok(row) => Ok(Some(row)),
        Err(rusqlite::Error::QueryReturnedNoRows) => Ok(None),
        Err(e) => Err(e),
    }
}

/// Query a reflex by its ID.
#[instrument(skip(conn))]
pub fn get_reflex_by_id(conn: &Connection, id: &str) -> SqlResult<Option<ReflexRow>> {
    let mut stmt = conn.prepare(
        "SELECT id, intent, action_sql, embedding, confidence, invoke_count, last_invoked, created_at
         FROM reflexes WHERE id = ?1",
    )?;

    let result = stmt.query_row(params![id], |row| {
        let embedding_blob: Vec<u8> = row.get(3)?;
        Ok(ReflexRow {
            id: row.get(0)?,
            intent: row.get(1)?,
            action_sql: row.get(2)?,
            embedding: bytes_to_embed(&embedding_blob),
            confidence: row.get(4)?,
            invoke_count: row.get(5)?,
            last_invoked: row.get(6)?,
            created_at: row.get(7)?,
        })
    });

    match result {
        Ok(row) => Ok(Some(row)),
        Err(rusqlite::Error::QueryReturnedNoRows) => Ok(None),
        Err(e) => Err(e),
    }
}

/// Search for the nearest reflexes by embedding using sqlite-vec.
#[instrument(skip(conn, query_embedding))]
pub fn search_nearest(
    conn: &Connection,
    query_embedding: &[f32],
    limit: usize,
) -> SqlResult<Vec<(String, f32, ReflexRow)>> {
    let query_bytes = embed_to_bytes(query_embedding);

    let mut stmt = conn.prepare(&format!(
        "SELECT v.id, v.distance, r.intent, r.action_sql, r.embedding, r.confidence, r.invoke_count, r.last_invoked, r.created_at
         FROM vec_reflexes v
         JOIN reflexes r ON v.id = r.id
         WHERE v.embedding MATCH ?1
         ORDER BY v.distance
         LIMIT {}",
        limit
    ))?;

    let rows = stmt.query_map(params![query_bytes], |row| {
        let id: String = row.get(0)?;
        let distance: f32 = row.get(1)?;
        let intent: String = row.get(2)?;
        let action_sql: String = row.get(3)?;
        let embedding_blob: Vec<u8> = row.get(4)?;
        let confidence: f64 = row.get(5)?;
        let invoke_count: i64 = row.get(6)?;
        let last_invoked: String = row.get(7)?;
        let created_at: String = row.get(8)?;

        let reflex = ReflexRow {
            id: id.clone(),
            intent,
            action_sql,
            embedding: bytes_to_embed(&embedding_blob),
            confidence,
            invoke_count,
            last_invoked,
            created_at,
        };

        Ok((id, distance, reflex))
    })?;

    let mut results = Vec::new();
    for row in rows {
        results.push(row?);
    }

    Ok(results)
}

/// Insert a new reflex into the database.
#[instrument(skip(conn, embedding))]
pub fn insert_reflex(
    conn: &Connection,
    id: &str,
    intent: &str,
    action_sql: &str,
    embedding: &[f32],
    confidence: f64,
) -> SqlResult<()> {
    let embedding_bytes = embed_to_bytes(embedding);
    let now = chrono::Utc::now().to_rfc3339();

    conn.execute(
        "INSERT INTO reflexes (id, intent, action_sql, embedding, confidence, invoke_count, last_invoked, created_at)
         VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)",
        params![id, intent, action_sql, embedding_bytes, confidence, 0i64, "", now],
    )?;

    conn.execute(
        "INSERT INTO vec_reflexes (id, embedding) VALUES (?1, ?2)",
        params![id, embed_to_bytes(embedding)],
    )?;

    Ok(())
}

/// Update a reflex's embedding in both tables.
#[instrument(skip(conn, embedding))]
pub fn update_reflex_embedding(conn: &Connection, id: &str, embedding: &[f32]) -> SqlResult<()> {
    let embedding_bytes = embed_to_bytes(embedding);

    conn.execute(
        "UPDATE reflexes SET embedding = ?1 WHERE id = ?2",
        params![embedding_bytes, id],
    )?;

    // Delete and re-insert in vec_reflexes (virtual tables don't support UPDATE on match columns)
    conn.execute("DELETE FROM vec_reflexes WHERE id = ?1", params![id])?;
    conn.execute(
        "INSERT INTO vec_reflexes (id, embedding) VALUES (?1, ?2)",
        params![id, embed_to_bytes(embedding)],
    )?;

    Ok(())
}

/// Update a reflex's confidence score (Bayesian update).
#[instrument(skip(conn))]
pub fn update_reflex_confidence(conn: &Connection, id: &str, new_confidence: f64) -> SqlResult<()> {
    conn.execute(
        "UPDATE reflexes SET confidence = ?1 WHERE id = ?2",
        params![new_confidence, id],
    )?;
    Ok(())
}

/// Increment a reflex's invoke count and update last_invoked timestamp.
#[instrument(skip(conn))]
pub fn increment_reflex_invoke(conn: &Connection, id: &str) -> SqlResult<()> {
    let now = chrono::Utc::now().to_rfc3339();
    conn.execute(
        "UPDATE reflexes SET invoke_count = invoke_count + 1, last_invoked = ?1 WHERE id = ?2",
        params![now, id],
    )?;
    Ok(())
}

/// Log an action execution.
#[instrument(skip(conn, input, output))]
pub fn log_action(
    conn: &Connection,
    reflex_id: &str,
    input: &str,
    output: &str,
    latency_ms: i64,
    confidence: f64,
) -> SqlResult<()> {
    let id = uuid::Uuid::new_v4().to_string();
    let now = chrono::Utc::now().to_rfc3339();

    conn.execute(
        "INSERT INTO action_log (id, reflex_id, input, output, latency_ms, confidence, created_at)
         VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)",
        params![id, reflex_id, input, output, latency_ms, confidence, now],
    )?;

    Ok(())
}

/// Get recent action log entries.
#[instrument(skip(conn))]
pub fn get_recent_actions(conn: &Connection, limit: usize) -> SqlResult<Vec<ActionLogRow>> {
    let mut stmt = conn.prepare(&format!(
        "SELECT id, reflex_id, input, output, latency_ms, confidence, created_at
         FROM action_log ORDER BY created_at DESC LIMIT {}",
        limit
    ))?;

    let rows = stmt.query_map([], |row| {
        Ok(ActionLogRow {
            id: row.get(0)?,
            reflex_id: row.get(1)?,
            input: row.get(2)?,
            output: row.get(3)?,
            latency_ms: row.get(4)?,
            confidence: row.get(5)?,
            created_at: row.get(6)?,
        })
    })?;

    let mut results = Vec::new();
    for row in rows {
        results.push(row?);
    }

    Ok(results)
}

/// Insert or update a shell fingerprint.
#[instrument(skip(conn))]
pub fn upsert_shell(
    conn: &Connection,
    fingerprint: &str,
    hostname: &str,
    os: &str,
    cpu_count: i64,
    ram_mb: i64,
    gpu: &str,
) -> SqlResult<()> {
    let now = chrono::Utc::now().to_rfc3339();

    conn.execute(
        "INSERT INTO shells (fingerprint, hostname, os, cpu_count, ram_mb, gpu, last_seen)
         VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)
         ON CONFLICT(fingerprint) DO UPDATE SET
            hostname = excluded.hostname,
            os = excluded.os,
            cpu_count = excluded.cpu_count,
            ram_mb = excluded.ram_mb,
            gpu = excluded.gpu,
            last_seen = excluded.last_seen",
        params![fingerprint, hostname, os, cpu_count, ram_mb, gpu, now],
    )?;

    Ok(())
}

/// Create a new session.
#[instrument(skip(conn))]
pub fn create_session(conn: &Connection, shell_fingerprint: &str) -> SqlResult<String> {
    let id = uuid::Uuid::new_v4().to_string();
    let now = chrono::Utc::now().to_rfc3339();

    conn.execute(
        "INSERT INTO sessions (id, shell_fingerprint, state, started_at) VALUES (?1, ?2, ?3, ?4)",
        params![id, shell_fingerprint, "active", now],
    )?;

    Ok(id)
}

/// Update session state.
#[instrument(skip(conn))]
pub fn update_session_state(conn: &Connection, id: &str, state: &str) -> SqlResult<()> {
    conn.execute(
        "UPDATE sessions SET state = ?1 WHERE id = ?2",
        params![state, id],
    )?;
    Ok(())
}

/// Get total reflex count.
#[instrument(skip(conn))]
pub fn get_reflex_count(conn: &Connection) -> SqlResult<i64> {
    conn.query_row("SELECT COUNT(*) FROM reflexes", [], |row| row.get(0))
}

/// Get total action log count.
#[instrument(skip(conn))]
pub fn get_action_log_count(conn: &Connection) -> SqlResult<i64> {
    conn.query_row("SELECT COUNT(*) FROM action_log", [], |row| row.get(0))
}

/// Get all reflexes (for migration/pack operations).
#[instrument(skip(conn))]
pub fn get_all_reflexes(conn: &Connection) -> SqlResult<Vec<ReflexRow>> {
    let mut stmt = conn.prepare(
        "SELECT id, intent, action_sql, embedding, confidence, invoke_count, last_invoked, created_at FROM reflexes",
    )?;

    let rows = stmt.query_map([], |row| {
        let embedding_blob: Vec<u8> = row.get(3)?;
        Ok(ReflexRow {
            id: row.get(0)?,
            intent: row.get(1)?,
            action_sql: row.get(2)?,
            embedding: bytes_to_embed(&embedding_blob),
            confidence: row.get(4)?,
            invoke_count: row.get(5)?,
            last_invoked: row.get(6)?,
            created_at: row.get(7)?,
        })
    })?;

    let mut results = Vec::new();
    for row in rows {
        results.push(row?);
    }

    Ok(results)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_embed_to_bytes_roundtrip() {
        let original: Vec<f32> = vec![0.1, 0.2, 0.3, -0.4, 0.5];
        let bytes = embed_to_bytes(&original);
        let restored = bytes_to_embed(&bytes);
        assert_eq!(original, restored);
    }

    #[test]
    fn test_init_db_in_memory() {
        let conn = Connection::open_in_memory().unwrap();
        conn.execute_batch("PRAGMA journal_mode=WAL; PRAGMA foreign_keys=ON;")
            .unwrap();

        // Just test table creation without sqlite-vec (which requires the extension)
        conn.execute_batch(CREATE_MIGRATIONS).unwrap();
        conn.execute_batch(&format!(
            "{}\n{}\n{}\n{}",
            CREATE_REFLEXES, CREATE_SESSIONS, CREATE_ACTION_LOG, CREATE_SHELLS
        ))
        .unwrap();

        let count: i64 = conn
            .query_row("SELECT COUNT(*) FROM reflexes", [], |row| row.get(0))
            .unwrap();
        assert_eq!(count, 0);
    }
}
