//! Reflex engine — the heart of PincherOS
//!
//! The ReflexEngine coordinates intent matching, reflex execution,
//! confidence tracking, and the teach/learn cycle.

use crate::db::schema::{self, ReflexRow};
use crate::embed::Embedder;
use crate::reflex::matcher::{match_reflex, MatchResult};
use rusqlite::Connection;
use serde::{Deserialize, Serialize};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Instant;
use thiserror::Error;
use tracing::{debug, info, instrument, warn};

/// Reflex engine errors.
#[derive(Debug, Error)]
pub enum EngineError {
    #[error("Database error: {0}")]
    Db(#[from] rusqlite::Error),

    #[error("Embedding error: {0}")]
    Embed(#[from] crate::embed::EmbedError),

    #[error("Match error: {0}")]
    Match(#[from] crate::reflex::matcher::MatchError),

    #[error("Execution error: {0}")]
    Execution(String),

    #[error("Reflex not found: {0}")]
    ReflexNotFound(String),

    #[error("Vetoed: {0}")]
    Vetoed(String),

    #[error("Resource constraint: {0}")]
    ResourceConstraint(String),
}

/// Result type for engine operations.
pub type EngineResult<T> = Result<T, EngineError>;

/// Interval between automatic WAL checkpoints (seconds).
const WAL_CHECKPOINT_INTERVAL_SECS: u64 = 30;

/// Number of write operations after which a WAL checkpoint is triggered.
const WAL_CHECKPOINT_OPS_THRESHOLD: u64 = 100;

/// A learned reflex with its metadata.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Reflex {
    pub id: String,
    pub intent: String,
    pub action: String,
    pub confidence: f64,
    pub invoke_count: i64,
}

impl From<ReflexRow> for Reflex {
    fn from(row: ReflexRow) -> Self {
        Reflex {
            id: row.id,
            intent: row.intent,
            action: row.action_sql,
            confidence: row.confidence,
            invoke_count: row.invoke_count,
        }
    }
}

/// The result of executing a reflex or processing an intent.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Execution {
    pub output: String,
    pub latency_ms: i64,
    pub confidence: f64,
    pub match_type: MatchType,
    pub reflex_id: Option<String>,
    pub intent: String,
}

/// How an intent was matched to a reflex.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MatchType {
    Exact,
    Similar,
    Novel,
    Builtin,
}

impl std::fmt::Display for MatchType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            MatchType::Exact => write!(f, "exact"),
            MatchType::Similar => write!(f, "similar"),
            MatchType::Novel => write!(f, "novel"),
            MatchType::Builtin => write!(f, "builtin"),
        }
    }
}

/// Confidence update delegates to the proper multiplicative model in confidence.rs.
/// See [`crate::reflex::confidence::update_confidence`] for details.
fn update_confidence(current: f64, success: bool) -> f64 {
    crate::reflex::confidence::update_confidence(current, success)
}

/// Built-in reflex dispatch table.
fn dispatch_builtin(intent: &str, action: &str, input: &str) -> EngineResult<String> {
    match intent {
        "system.info" => builtin_system_info(),
        "file.read" => builtin_file_read(action, input),
        "file.write" => builtin_file_write(action, input),
        "process.list" => builtin_process_list(),
        "process.kill" => builtin_process_kill(action, input),
        "network.ping" => builtin_network_ping(action, input),
        "git.status" => builtin_git_status(action, input),
        "git.diff" => builtin_git_diff(action, input),
        "docker.ps" => builtin_docker_ps(),
        "env.get" => builtin_env_get(action, input),
        _ => Err(EngineError::Execution(format!(
            "Unknown built-in reflex: {}",
            intent
        ))),
    }
}

/// The main reflex engine.
pub struct ReflexEngine {
    conn: Connection,
    embedder: Embedder,
    /// Timestamp (in seconds since a reference epoch) of the last WAL checkpoint.
    last_checkpoint_epoch: Arc<AtomicU64>,
    /// Approximate count of write operations since the last WAL checkpoint.
    ops_since_checkpoint: Arc<AtomicU64>,
}

impl ReflexEngine {
    /// Create a new ReflexEngine with the given database connection and embedder.
    pub fn new(conn: Connection, embedder: Embedder) -> Self {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        Self {
            conn,
            embedder,
            last_checkpoint_epoch: Arc::new(AtomicU64::new(now)),
            ops_since_checkpoint: Arc::new(AtomicU64::new(0)),
        }
    }

    /// Create a new ReflexEngine by opening the database at the given path.
    ///
    /// The underlying SQLite connection is opened with WAL journal mode
    /// and a 5-second busy timeout (see [`schema::init_db`]).
    ///
    /// Periodic WAL checkpoint management runs automatically on write
    /// operations: `PRAGMA wal_checkpoint(TRUNCATE)` is invoked every
    /// ~30 seconds or after 100 write operations, whichever comes first.
    /// If the DB is busy, it falls back to PASSIVE mode.
    pub fn open(
        db_path: &std::path::Path,
        model_path: Option<&std::path::Path>,
    ) -> EngineResult<Self> {
        let conn = schema::init_db(db_path)?;
        let embedder = Embedder::new(model_path)?;

        let engine = Self::new(conn, embedder);

        // Seed embeddings for built-in reflexes (non-fatal)
        // We can't access conn after moving it, so we'll skip this for now
        // In a real implementation, we'd pass the conn separately or use interior mutability

        Ok(engine)
    }

    /// Teach the engine a new reflex.
    #[instrument(skip(self, action))]
    pub fn teach(&mut self, intent: &str, action: &str) -> EngineResult<Reflex> {
        info!(intent = intent, "Teaching new reflex");

        if let Some(existing) = schema::get_reflex_by_intent(&self.conn, intent)? {
            info!(
                intent = intent,
                existing_id = %existing.id,
                "Reflex with this intent already exists — updating"
            );

            let embedding = self.embedder.embed(intent)?;
            schema::update_reflex_embedding(&self.conn, &existing.id, &embedding)?;
            self.record_write_and_checkpoint();

            return Ok(Reflex::from(existing));
        }

        let embedding = self.embedder.embed(intent)?;
        let id = uuid::Uuid::new_v4().to_string();

        schema::insert_reflex(&self.conn, &id, intent, action, &embedding, 0.5)?;
        self.record_write_and_checkpoint();

        info!(
            reflex_id = id,
            intent = intent,
            "New reflex taught successfully"
        );

        Ok(Reflex {
            id,
            intent: intent.to_string(),
            action: action.to_string(),
            confidence: 0.5,
            invoke_count: 0,
        })
    }

    /// Process a command intent through the reflex engine.
    #[instrument(skip(self))]
    pub fn do_command(&mut self, intent: &str) -> EngineResult<Execution> {
        info!(intent = intent, "Processing command");
        let start = Instant::now();

        let match_result = match_reflex(&self.conn, &self.embedder, intent)?;

        let execution = match match_result {
            MatchResult::Exact { similarity, reflex } => {
                info!(
                    intent = intent,
                    similarity = similarity,
                    reflex_id = %reflex.id,
                    "Exact match — short-circuiting to reflex execution"
                );

                self.execute_reflex(&reflex, intent, MatchType::Exact)?
            }

            MatchResult::Similar { similarity, reflex } => {
                info!(
                    intent = intent,
                    similarity = similarity,
                    reflex_id = %reflex.id,
                    "Similar match — executing with potential LLM refinement"
                );

                let mut exec = self.execute_reflex(&reflex, intent, MatchType::Similar)?;
                exec.output = format!(
                    "[SIMILAR MATCH: {:.2}%] {}\n---\nThis response may need LLM refinement.",
                    similarity * 100.0,
                    exec.output
                );
                exec
            }

            MatchResult::Novel { best_similarity } => {
                info!(
                    intent = intent,
                    best_similarity = best_similarity,
                    "Novel intent — no matching reflex found"
                );

                Execution {
                    output: format!(
                        "[NOVEL INTENT] No matching reflex found (best similarity: {:.2}%).",
                        best_similarity * 100.0,
                    ),
                    latency_ms: start.elapsed().as_millis() as i64,
                    confidence: best_similarity as f64,
                    match_type: MatchType::Novel,
                    reflex_id: None,
                    intent: intent.to_string(),
                }
            }
        };

        Ok(execution)
    }

    /// Execute a specific reflex.
    #[instrument(skip(self, reflex))]
    pub fn execute(&mut self, reflex: &Reflex, input: &str) -> EngineResult<Execution> {
        let start = Instant::now();

        // SECURITY: Run veto check before any execution
        self.check_veto(&reflex.intent, &reflex.action)?;

        let is_builtin = is_builtin_intent(&reflex.intent);

        let output = if is_builtin {
            dispatch_builtin(&reflex.intent, &reflex.action, input)?
        } else {
            self.execute_action_sql(&reflex.action, input)?
        };

        let latency_ms = start.elapsed().as_millis() as i64;

        schema::increment_reflex_invoke(&self.conn, &reflex.id)?;
        self.confidence_update(&reflex.id, true)?;

        schema::log_action(
            &self.conn,
            &reflex.id,
            input,
            &output,
            latency_ms,
            reflex.confidence,
        )?;
        self.record_write_and_checkpoint();

        Ok(Execution {
            output,
            latency_ms,
            confidence: reflex.confidence,
            match_type: if is_builtin {
                MatchType::Builtin
            } else {
                MatchType::Exact
            },
            reflex_id: Some(reflex.id.clone()),
            intent: reflex.intent.clone(),
        })
    }

    /// Execute a reflex row (internal helper).
    fn execute_reflex(
        &mut self,
        reflex: &ReflexRow,
        input: &str,
        match_type: MatchType,
    ) -> EngineResult<Execution> {
        let start = Instant::now();

        // SECURITY: Run veto check before any execution
        self.check_veto(&reflex.intent, &reflex.action_sql)?;

        let is_builtin = is_builtin_intent(&reflex.intent);

        let output = if is_builtin {
            dispatch_builtin(&reflex.intent, &reflex.action_sql, input)?
        } else {
            self.execute_action_sql(&reflex.action_sql, input)?
        };

        let latency_ms = start.elapsed().as_millis() as i64;

        schema::increment_reflex_invoke(&self.conn, &reflex.id)?;

        schema::log_action(
            &self.conn,
            &reflex.id,
            input,
            &output,
            latency_ms,
            reflex.confidence,
        )?;
        self.record_write_and_checkpoint();

        Ok(Execution {
            output,
            latency_ms,
            confidence: reflex.confidence,
            match_type,
            reflex_id: Some(reflex.id.clone()),
            intent: reflex.intent.clone(),
        })
    }

    /// Check veto policy for the given intent and action.
    ///
    /// Returns `Ok(())` if the action is allowed or requires only confirmation;
    /// returns `Err(EngineError::Vetoed)` if the action is denied.
    fn check_veto(&self, intent: &str, action: &str) -> EngineResult<()> {
        // SECURITY: Run veto check before any execution
        let action_to_check = if is_builtin_intent(intent) {
            intent
        } else {
            action
        };
        let veto_context = crate::security::veto::ExecutionContext::for_command(action_to_check);
        let veto_engine = crate::security::veto::VetoEngine::default();
        let veto_result = veto_engine
            .check(action_to_check, &veto_context)
            .map_err(|e| EngineError::Vetoed(format!("Veto check failed: {}", e)))?;
        match &veto_result {
            crate::security::veto::VetoDecision::Deny(reason) => {
                return Err(EngineError::Vetoed(format!("Action denied: {}", reason)));
            }
            crate::security::veto::VetoDecision::RequireConfirmation(reason) => {
                debug!("Action requires confirmation (proceeding): {}", reason);
            }
            crate::security::veto::VetoDecision::Allow => {}
        }
        Ok(())
    }

    /// Execute an action SQL string.
    ///
    /// Supports SELECT, INSERT, UPDATE, and DELETE queries.
    /// Dynamic SQL interpolation via `{{...}}` is **prohibited** for security.
    /// Shell commands are routed through the sandboxed executor.
    fn execute_action_sql(&self, action_sql: &str, input: &str) -> EngineResult<String> {
        debug!(action_sql = action_sql, "Executing action SQL");

        // SECURITY: Reject any action that interpolates user input into SQL or shell.
        if action_sql.contains("{{input}}") || action_sql.contains("{{") {
            return Err(EngineError::Execution(
                "Dynamic SQL/shell interpolation is prohibited. Use parameterized builtins instead.".into()
            ));
        }

        let normalized = action_sql.trim().to_uppercase();

        if normalized.starts_with("SELECT") {
            match self
                .conn
                .query_row(action_sql, [], |row| row.get::<_, String>(0))
            {
                Ok(result) => Ok(result),
                Err(e) => {
                    debug!(error = %e, "SQL execution failed — returning raw action");
                    Ok(format!("Action: {}", action_sql))
                }
            }
        } else if normalized.starts_with("INSERT")
            || normalized.starts_with("UPDATE")
            || normalized.starts_with("DELETE")
        {
            match self.conn.execute(action_sql, []) {
                Ok(rows_affected) => Ok(format!("OK: {} row(s) affected", rows_affected)),
                Err(e) => {
                    debug!(error = %e, "SQL mutation failed");
                    Err(EngineError::Execution(format!(
                        "SQL mutation failed: {}",
                        e
                    )))
                }
            }
        } else if action_sql.trim().starts_with("$") {
            // Shell command: strip the leading '$' and execute via sandbox
            let shell_cmd = action_sql.trim().strip_prefix('$').unwrap().trim();
            self.execute_action_shell(shell_cmd, input)
        } else {
            // Unknown action type — attempt sandboxed shell execution
            self.execute_action_shell(action_sql, input)
        }
    }

    /// Execute a shell command through the sandbox.
    ///
    /// Uses the capability-based sandbox with restricted defaults.
    /// Falls back to restricted `Command::new` if sandbox is unavailable.
    fn execute_action_shell(&self, command: &str, _input: &str) -> EngineResult<String> {
        debug!(command = command, "Executing action via sandbox");

        // Build a restricted sandbox manifest
        let manifest = crate::security::sandbox::CapabilityManifest::new("reflex-shell")
            .with_capability(crate::security::sandbox::Capability::FilesystemRead)
            .with_capability(crate::security::sandbox::Capability::Subprocess)
            .with_read_path("/usr")
            .with_read_path("/bin")
            .with_read_path("/lib")
            .with_read_path("/tmp")
            .with_read_path(".");

        match crate::security::sandbox::build_sandbox(&manifest) {
            Ok(config) => {
                match crate::security::sandbox::execute_sandboxed(command, &config) {
                    Ok(_status) => Ok(format!("Sandboxed command executed: {}", command)),
                    Err(e) => {
                        // Sandbox unavailable — fall back to restricted direct execution
                        debug!(error = %e, "Sandbox execution failed, falling back to restricted Command");
                        self.execute_shell_fallback(command)
                    }
                }
            }
            Err(e) => {
                debug!(error = %e, "Sandbox build failed, falling back to restricted Command");
                self.execute_shell_fallback(command)
            }
        }
    }

    /// Fallback shell execution with restricted environment.
    ///
    /// Used when the sandbox (bwrap/landlock) is not available.
    /// Runs the command with a minimal PATH and no inherited env vars.
    fn execute_shell_fallback(&self, command: &str) -> EngineResult<String> {
        // Parse command into binary + args (simple split on whitespace)
        let parts: Vec<&str> = command.split_whitespace().collect();
        if parts.is_empty() {
            return Err(EngineError::Execution("Empty shell command".into()));
        }

        let binary = parts[0];
        let args = &parts[1..];

        let output = std::process::Command::new(binary)
            .args(args)
            .env_clear()
            .env("PATH", "/usr/bin:/bin")
            .env("HOME", "/tmp")
            .current_dir("/tmp")
            .output()
            .map_err(|e| {
                EngineError::Execution(format!("Failed to execute '{}': {}", binary, e))
            })?;

        let stdout = String::from_utf8_lossy(&output.stdout).to_string();
        let stderr = String::from_utf8_lossy(&output.stderr).to_string();

        if output.status.success() {
            // Truncate to 4 KiB
            if stdout.len() > 4096 {
                Ok(stdout[..4096].to_string())
            } else {
                Ok(stdout)
            }
        } else {
            Err(EngineError::Execution(format!(
                "Command '{}' failed (exit {:?}): {}",
                binary,
                output.status.code(),
                stderr.trim()
            )))
        }
    }

    /// Update a reflex's confidence: +0.05 on success, -0.10 on failure, clamped to [0.0, 1.0].
    #[instrument(skip(self))]
    pub fn confidence_update(&mut self, reflex_id: &str, success: bool) -> EngineResult<()> {
        let reflex = schema::get_reflex_by_id(&self.conn, reflex_id)?
            .ok_or_else(|| EngineError::ReflexNotFound(reflex_id.to_string()))?;

        let new_confidence = update_confidence(reflex.confidence, success);

        debug!(
            reflex_id = reflex_id,
            old_confidence = reflex.confidence,
            new_confidence = new_confidence,
            success = success,
            "Updated reflex confidence"
        );

        schema::update_reflex_confidence(&self.conn, reflex_id, new_confidence)?;
        Ok(())
    }

    /// Perform a WAL checkpoint if enough time has elapsed or enough write
    /// operations have accumulated since the last checkpoint.
    ///
    /// Runs `PRAGMA wal_checkpoint(TRUNCATE)` to prevent unbounded WAL growth.
    /// If TRUNCATE fails (e.g. the DB is busy), falls back to PASSIVE mode
    /// which only checkpoints what it can without blocking readers or writers.
    ///
    /// Called automatically by write operations (`teach`, `execute`, etc.).
    pub fn checkpoint_wal_if_needed(&self) {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        let last = self.last_checkpoint_epoch.load(Ordering::Relaxed);
        let ops = self.ops_since_checkpoint.load(Ordering::Relaxed);

        let elapsed = now.saturating_sub(last);
        if elapsed < WAL_CHECKPOINT_INTERVAL_SECS && ops < WAL_CHECKPOINT_OPS_THRESHOLD {
            return;
        }

        // Reset counters early so concurrent calls don't both attempt a checkpoint
        if self
            .last_checkpoint_epoch
            .compare_exchange(last, now, Ordering::Release, Ordering::Relaxed)
            .is_err()
        {
            return; // another thread already started a checkpoint
        }
        self.ops_since_checkpoint.store(0, Ordering::Relaxed);

        debug!(
            elapsed_secs = elapsed,
            ops_since_last = ops,
            "Performing WAL checkpoint (TRUNCATE)"
        );

        // TRUNCATE is ideal — resets the WAL to minimal size — but requires
        // that no other readers/writers hold the WAL open.
        // PRAGMA statements return rows; use query_row and discard the result.
        if let Err(e) = self
            .conn
            .query_row("PRAGMA wal_checkpoint(TRUNCATE)", [], |_| Ok(()))
        {
            debug!(
                error = %e,
                "WAL checkpoint TRUNCATE failed, retrying with PASSIVE"
            );
            if let Err(e2) = self
                .conn
                .query_row("PRAGMA wal_checkpoint(PASSIVE)", [], |_| Ok(()))
            {
                warn!("WAL checkpoint (PASSIVE) also failed: {}", e2);
            }
        }
    }

    /// Record a write operation for checkpoint accounting.
    ///
    /// Each call increments the operation counter and triggers a WAL
    /// checkpoint if the threshold has been reached.
    fn record_write_and_checkpoint(&self) {
        self.ops_since_checkpoint.fetch_add(1, Ordering::Relaxed);
        self.checkpoint_wal_if_needed();
    }

    /// Get the database connection reference.
    pub fn connection(&self) -> &Connection {
        &self.conn
    }

    /// Get the embedder reference.
    pub fn embedder(&self) -> &Embedder {
        &self.embedder
    }

    /// Get status information about the engine.
    pub fn get_status(&self) -> EngineResult<EngineStatus> {
        let reflex_count = schema::get_reflex_count(&self.conn)?;
        let action_count = schema::get_action_log_count(&self.conn)?;

        Ok(EngineStatus {
            reflex_count,
            action_log_count: action_count,
            embedder_loaded: self.embedder.is_loaded(),
        })
    }
}

/// Engine status summary.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EngineStatus {
    pub reflex_count: i64,
    pub action_log_count: i64,
    pub embedder_loaded: bool,
}

/// Check if an intent corresponds to a built-in reflex.
fn is_builtin_intent(intent: &str) -> bool {
    matches!(
        intent,
        "system.info"
            | "file.read"
            | "file.write"
            | "process.list"
            | "process.kill"
            | "network.ping"
            | "git.status"
            | "git.diff"
            | "docker.ps"
            | "env.get"
    )
}

// ── Built-in Reflex Implementations ──────────────────────────────────

fn builtin_system_info() -> EngineResult<String> {
    use sysinfo::System;
    let mut sys = System::new_all();
    sys.refresh_all();

    let info = serde_json::json!({
        "hostname": System::host_name().unwrap_or_default(),
        "os": System::name().unwrap_or_default(),
        "os_version": System::os_version().unwrap_or_default(),
        "cpu_count": sys.cpus().len(),
        "ram_total_mb": sys.total_memory() / 1024 / 1024,
        "ram_used_mb": sys.used_memory() / 1024 / 1024,
        "uptime_secs": System::uptime(),
    });

    Ok(serde_json::to_string_pretty(&info).unwrap_or_else(|_| format!("{:?}", info)))
}

fn builtin_file_read(action: &str, input: &str) -> EngineResult<String> {
    let path = extract_template_var(action, "path").unwrap_or_else(|| input.to_string());

    // SECURITY: Validate path against traversal and sensitive locations
    let canonical = std::path::Path::new(&path).canonicalize().map_err(|e| {
        EngineError::Execution(format!("Path resolution failed for '{}': {}", path, e))
    })?;

    let path_str = canonical.to_string_lossy();

    // Block sensitive paths
    const BLOCKED_PREFIXES: &[&str] = &[
        "/etc/shadow",
        "/etc/ssh",
        "/root/.ssh",
        "/proc/self/environ",
        "/etc/gshadow",
        "/etc/passwd-",
        "/etc/shadow-",
    ];
    for prefix in BLOCKED_PREFIXES {
        if path_str.starts_with(prefix) {
            return Err(EngineError::Execution(format!(
                "Access to '{}' is forbidden by security policy",
                prefix
            )));
        }
    }

    let content = std::fs::read_to_string(&canonical)
        .map_err(|e| EngineError::Execution(format!("Failed to read file '{}': {}", path, e)))?;

    Ok(content)
}

fn builtin_file_write(action: &str, input: &str) -> EngineResult<String> {
    let path = extract_template_var(action, "path").unwrap_or_else(|| "output.txt".to_string());

    let content = extract_template_var(action, "content").unwrap_or_else(|| input.to_string());

    // SECURITY: Restrict writes to safe directories only
    let parent = std::path::Path::new(&path)
        .parent()
        .unwrap_or(std::path::Path::new("."));
    let canonical_dir = parent
        .canonicalize()
        .unwrap_or_else(|_| std::path::PathBuf::from("."));
    let dir_str = canonical_dir.to_string_lossy();

    const SAFE_DIRS: &[&str] = &["/tmp", "/var/tmp"];
    if !SAFE_DIRS.iter().any(|d| dir_str.starts_with(d))
        && !path.starts_with("./")
        && !path.starts_with("output")
    {
        return Err(EngineError::Execution(
            "File writes are restricted to /tmp, /var/tmp, and relative paths".into(),
        ));
    }

    std::fs::write(&path, content)
        .map_err(|e| EngineError::Execution(format!("Failed to write file '{}': {}", path, e)))?;

    Ok(format!("Successfully wrote to {}", path))
}

fn builtin_process_list() -> EngineResult<String> {
    use sysinfo::System;
    let mut sys = System::new_all();
    sys.refresh_all();

    let mut processes: Vec<_> = sys
        .processes()
        .iter()
        .map(|(pid, proc_info)| {
            serde_json::json!({
                "pid": pid.as_u32(),
                "name": proc_info.name().to_string_lossy(),
                "cpu_usage": format!("{:.1}%", proc_info.cpu_usage()),
                "memory_mb": proc_info.memory() / 1024 / 1024,
            })
        })
        .collect();

    processes.truncate(20);

    Ok(serde_json::to_string_pretty(&processes)
        .unwrap_or_else(|_| "No processes found".to_string()))
}

fn builtin_process_kill(action: &str, input: &str) -> EngineResult<String> {
    let pid_str = extract_template_var(action, "pid").unwrap_or_else(|| input.to_string());

    let pid: u32 = pid_str
        .trim()
        .parse()
        .map_err(|_| EngineError::Execution(format!("Invalid PID: '{}'", pid_str)))?;

    // SECURITY: Block killing system processes and self
    let self_pid = std::process::id();
    if pid <= 100 {
        return Err(EngineError::Execution(format!(
            "Cannot kill system process (PID {} <= 100)",
            pid
        )));
    }
    if pid == self_pid {
        return Err(EngineError::Execution(
            "Cannot kill self (PincherOS process)".into(),
        ));
    }

    use sysinfo::{Pid, System};
    let mut sys = System::new_all();
    sys.refresh_all();

    let pid_obj = Pid::from_u32(pid);
    if let Some(process) = sys.process(pid_obj) {
        process.kill();
        Ok(format!(
            "Killed process {} ({})",
            pid,
            process.name().to_string_lossy()
        ))
    } else {
        Err(EngineError::Execution(format!("Process {} not found", pid)))
    }
}

fn builtin_network_ping(action: &str, input: &str) -> EngineResult<String> {
    let host = extract_template_var(action, "host").unwrap_or_else(|| input.to_string());

    let timeout = extract_template_var(action, "timeout_ms")
        .and_then(|t| t.parse::<u64>().ok())
        .unwrap_or(5000);

    let output = std::process::Command::new("ping")
        .args(["-c", "1", "-W", &format!("{}", timeout / 1000), &host])
        .output()
        .map_err(|e| EngineError::Execution(format!("Failed to ping '{}': {}", host, e)))?;

    let stdout = String::from_utf8_lossy(&output.stdout).to_string();
    let stderr = String::from_utf8_lossy(&output.stderr).to_string();

    if output.status.success() {
        Ok(stdout)
    } else {
        Ok(format!("Ping failed: {}", stderr))
    }
}

fn builtin_git_status(action: &str, input: &str) -> EngineResult<String> {
    let repo_path = extract_template_var(action, "repo_path").unwrap_or_else(|| input.to_string());

    let output = std::process::Command::new("git")
        .args(["status", "--porcelain"])
        .current_dir(&repo_path)
        .output()
        .map_err(|e| EngineError::Execution(format!("Failed to run git status: {}", e)))?;

    Ok(String::from_utf8_lossy(&output.stdout).to_string())
}

fn builtin_git_diff(action: &str, _input: &str) -> EngineResult<String> {
    let repo_path = extract_template_var(action, "repo_path").unwrap_or_else(|| ".".to_string());

    let revision = extract_template_var(action, "revision").unwrap_or_else(|| "HEAD".to_string());

    let output = std::process::Command::new("git")
        .args(["diff", &revision])
        .current_dir(&repo_path)
        .output()
        .map_err(|e| EngineError::Execution(format!("Failed to run git diff: {}", e)))?;

    Ok(String::from_utf8_lossy(&output.stdout).to_string())
}

fn builtin_docker_ps() -> EngineResult<String> {
    let output = std::process::Command::new("docker")
        .args([
            "ps",
            "--format",
            "{{.ID}}\\t{{.Image}}\\t{{.Status}}\\t{{.Names}}",
        ])
        .output()
        .map_err(|e| EngineError::Execution(format!("Failed to run docker ps: {}", e)))?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(EngineError::Execution(format!(
            "docker ps failed: {}",
            stderr
        )));
    }

    Ok(String::from_utf8_lossy(&output.stdout).to_string())
}

fn builtin_env_get(action: &str, input: &str) -> EngineResult<String> {
    let var_name = extract_template_var(action, "name").unwrap_or_else(|| input.to_string());

    // SECURITY: Only allow reading from a safe allowlist
    const SAFE_VARS: &[&str] = &[
        "HOME",
        "USER",
        "SHELL",
        "LANG",
        "PATH",
        "TERM",
        "EDITOR",
        "PWD",
        "OLDPWD",
        "HOSTNAME",
        "LOGNAME",
        "COLORTERM",
    ];
    if !SAFE_VARS.contains(&var_name.as_str()) {
        return Err(EngineError::Execution(format!(
            "Reading environment variable '{}' is not permitted — only safe variables are allowed",
            var_name
        )));
    }

    match std::env::var(&var_name) {
        Ok(value) => Ok(format!("{}={}", var_name, value)),
        Err(_) => Ok(format!("Environment variable '{}' not set", var_name)),
    }
}

/// Extract a template variable like {{path}} from an action string.
///
/// Parses `{{var_name}}` patterns from the action template and returns
/// Some if the pattern exists, None if it doesn't. The actual value
/// substitution is handled by the builtin functions using the input.
fn extract_template_var(action: &str, var_name: &str) -> Option<String> {
    let pattern = format!("{{{{{}}}}}", var_name); // e.g. "{{path}}"
    if action.contains(&pattern) {
        // The template var exists — return the pattern to signal it was found.
        // The actual value comes from the input parameter in the caller.
        Some(pattern)
    } else {
        None
    }
}
