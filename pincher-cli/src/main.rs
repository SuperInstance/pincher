//! PincherOS CLI — the post-model operating system command-line interface.
//!
//! A hermit crab finds the right shell for every situation.
//! PincherOS finds the right reflex for every intent.

use anyhow::Result;
use clap::{Parser, Subcommand};
use colored::*;
use pincher_core::{
    ReflexEngine, Reflex, EngineStatus,
    ShellFingerprint, EMBEDDING_DIM,
    pack_nail, unpack_nail, fingerprint as capture_fingerprint,
    db::schema::get_all_reflexes,
};
use std::io::{self, Write};
use std::path::PathBuf;
use std::time::Instant;

const VERSION: &str = env!("CARGO_PKG_VERSION");

// ─── CLI Definition ──────────────────────────────────────────────────────────

#[derive(Parser)]
#[command(
    name = "pincher",
    about = "PincherOS — the post-model operating system",
    version = VERSION,
    after_help = "A hermit crab finds the right shell for every situation."
)]
struct Cli {
    #[command(subcommand)]
    command: Commands,

    /// Database path (default: ~/.pincher/reflexes.db)
    #[arg(long, env = "PINCHER_DB", global = true)]
    db: Option<String>,

    /// Log level (trace, debug, info, warn, error)
    #[arg(long, default_value = "warn", global = true)]
    log_level: String,
}

#[derive(Subcommand)]
enum Commands {
    /// Show current engine status, reflex count, resource state
    Status,

    /// Interactive teach flow: prompt for intent + action, store reflex
    Teach {
        /// Intent to teach (skip interactive prompt)
        #[arg(short, long)]
        intent: Option<String>,

        /// Action to associate (skip interactive prompt)
        #[arg(short, long)]
        action: Option<String>,
    },

    /// Execute a natural language intent through the reflex engine
    Do {
        /// Natural language intent to execute
        intent: String,
    },

    /// Pack current state into .nail file for migration
    Pack {
        /// Output file path (default: pincher-state.nail)
        output: Option<PathBuf>,
    },

    /// Unpack .nail file and merge state
    Unpack {
        /// Path to .nail file to import
        nail: PathBuf,
    },

    /// Run benchmark: embed latency, teach latency, match latency
    Bench,

    /// Detailed hardware fingerprint
    ShellInfo,

    /// Health check: verify ONNX model, SQLite, reflexes, embedding, disk
    Doctor,

    /// List all stored reflexes with confidence scores
    Reflexes {
        /// Show detailed information for each reflex
        #[arg(long)]
        verbose: bool,
    },
}

// ─── Helpers ─────────────────────────────────────────────────────────────────

fn expand_tilde(path: &str) -> PathBuf {
    if path.starts_with("~/") {
        if let Ok(home) = std::env::var("HOME") {
            return PathBuf::from(format!("{}{}", home, &path[1..]));
        }
    }
    PathBuf::from(path)
}

fn default_db_path() -> String {
    format!(
        "{}/.pincher/reflexes.db",
        std::env::var("HOME").unwrap_or_else(|_| ".".to_string())
    )
}

fn init_tracing(level: &str) {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new(level)),
        )
        .with_target(false)
        .init();
}

fn create_engine(db_path: &str) -> Result<ReflexEngine> {
    let path = expand_tilde(db_path);
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let engine = ReflexEngine::open(&path, None)?;
    Ok(engine)
}

fn prompt(prompt_text: &str) -> Result<String> {
    print!("{}", prompt_text);
    io::stdout().flush()?;
    let mut input = String::new();
    io::stdin().read_line(&mut input)?;
    Ok(input.trim().to_string())
}

fn get_current_fingerprint() -> ShellFingerprint {
    capture_fingerprint().unwrap_or_else(|_| ShellFingerprint {
        hostname: "unknown".to_string(),
        os: std::env::consts::OS.to_string(),
        os_version: "unknown".to_string(),
        cpu_count: 1,
        ram_mb: 0,
        gpu: "unknown".to_string(),
        mac_hash: "unknown".to_string(),
    })
}

fn print_crab_status(fp: &ShellFingerprint, status: &EngineStatus) {
    println!();
    println!(
        "{}",
        format!("   PincherOS v{}", VERSION).bright_red().bold()
    );
    println!("{}", "  ////////////////////////".yellow());
    println!(
        "{} {}",
        " //".yellow(),
        format!("Shell: {}  \\\\", fp.hostname).green()
    );
    println!(
        "{} {}",
        "//".yellow(),
        format!("   Reflexes: {}    \\\\", status.reflex_count).green()
    );
    println!(
        "{} {}",
        "\\\\".yellow(),
        "   State: Normal        //".green()
    );
    println!(
        "{} {}",
        " \\\\".yellow(),
        format!("  RAM: {}MB       //", fp.ram_mb).green()
    );
    println!("{}", "  ////////////////////////".yellow());
    println!();
}

fn print_timing(label: &str, duration: std::time::Duration) {
    let ms = duration.as_secs_f64() * 1000.0;
    println!(
        "  {} {}",
        format!("{}:", label).dimmed(),
        format!("{:.2}ms", ms).cyan()
    );
}

// ─── Command Implementations ─────────────────────────────────────────────────

fn cmd_status(engine: &mut ReflexEngine) -> Result<()> {
    let fp = get_current_fingerprint();
    let status = engine.get_status()?;

    print_crab_status(&fp, &status);

    println!("  {} {}", "OS:".dimmed(), fp.os);
    println!("  {} {}", "OS Version:".dimmed(), fp.os_version);
    println!("  {} {}", "CPU Cores:".dimmed(), fp.cpu_count);
    println!(
        "  {} {}",
        "Total RAM:".dimmed(),
        format!("{:.1} GB", fp.ram_mb as f64 / 1024.0)
    );
    println!(
        "  {} {}",
        "GPU:".dimmed(),
        fp.gpu
    );
    println!(
        "  {} {}",
        "Reflexes:".dimmed(),
        status.reflex_count
    );
    println!(
        "  {} {}",
        "Action Log:".dimmed(),
        status.action_log_count
    );
    println!(
        "  {} {}",
        "Embed Dim:".dimmed(),
        EMBEDDING_DIM
    );
    println!(
        "  {} {}",
        "Model Loaded:".dimmed(),
        if status.embedder_loaded { "yes".green() } else { "no (fallback mode)".yellow() }
    );

    Ok(())
}

fn cmd_teach(engine: &mut ReflexEngine, intent: Option<String>, action: Option<String>) -> Result<()> {
    println!("\n{}", "Teach PincherOS a new reflex".bright_red().bold());
    println!(
        "{}",
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━".dimmed()
    );

    let intent = match intent {
        Some(i) => i,
        None => {
            let i = prompt(&format!(
                "{} ",
                "What intent should PincherOS learn?".green()
            ))?;
            if i.is_empty() {
                anyhow::bail!("Intent cannot be empty");
            }
            i
        }
    };

    let action = match action {
        Some(a) => a,
        None => {
            let a = prompt(&format!(
                "{} ",
                "What action should be taken?".green()
            ))?;
            if a.is_empty() {
                anyhow::bail!("Action cannot be empty");
            }
            a
        }
    };

    println!(
        "\n  {} Storing reflex...",
        "...".yellow()
    );

    let start = Instant::now();
    let reflex = engine.teach(&intent, &action)?;
    let elapsed = start.elapsed();

    println!(
        "\n  {} {}",
        "OK".green(),
        "Reflex stored!".green().bold()
    );
    println!("  {} {}", "  ID:".dimmed(), reflex.id);
    println!("  {} {}", "  Intent:".dimmed(), reflex.intent);
    println!("  {} {}", "  Action:".dimmed(), reflex.action);
    println!(
        "  {} {}",
        "  Confidence:".dimmed(),
        format!("{:.2}", reflex.confidence)
    );
    print_timing("  Time", elapsed);

    Ok(())
}

fn cmd_do(engine: &mut ReflexEngine, intent: &str) -> Result<()> {
    println!("\n{}", format!("Executing: \"{}\"", intent).bright_red().bold());
    println!(
        "{}",
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━".dimmed()
    );

    let start = Instant::now();
    let execution = engine.do_command(intent)?;
    let elapsed = start.elapsed();

    if let Some(reflex_id) = &execution.reflex_id {
        println!(
            "\n  {} {}",
            "OK".green(),
            "Matched reflex:".green().bold()
        );
        println!("  {} {}", "  Reflex ID:".dimmed(), reflex_id.cyan());
        println!("  {} {}", "  Match Type:".dimmed(), format!("{:?}", execution.match_type).cyan());
        println!(
            "  {} {}",
            "  Confidence:".dimmed(),
            format!("{:.4}", execution.confidence)
        );
        println!("  {} {}", "  Output:".dimmed(), truncate(&execution.output, 120));
    } else {
        println!(
            "\n  {} {}",
            "X".red(),
            "No matching reflex found".red().bold()
        );
        println!(
            "  {}",
            "  Try 'pincher teach' to add one.".dimmed()
        );
    }
    print_timing("  Time", elapsed);

    Ok(())
}

fn cmd_pack(_engine: &ReflexEngine, output: Option<PathBuf>) -> Result<()> {
    let output_path = output.unwrap_or_else(|| PathBuf::from("pincher-state.nail"));

    println!(
        "\n{}",
        format!("Packing state to {}", output_path.display()).bright_red().bold()
    );
    println!(
        "{}",
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━".dimmed()
    );

    let db_path = expand_tilde(&default_db_path());

    let start = Instant::now();
    pack_nail(&db_path, &output_path)?;
    let elapsed = start.elapsed();

    let file_size = std::fs::metadata(&output_path)
        .map(|m| m.len())
        .unwrap_or(0);

    println!(
        "\n  {} {}",
        "OK".green(),
        "State packed!".green().bold()
    );
    println!("  {} {}", "  File:".dimmed(), output_path.display());
    println!(
        "  {} {}",
        "  Size:".dimmed(),
        format!("{:.2} KB", file_size as f64 / 1024.0)
    );
    print_timing("  Time", elapsed);

    Ok(())
}

fn cmd_unpack(nail_path: &PathBuf, db_path: &str) -> Result<()> {
    println!(
        "\n{}",
        format!("Unpacking from {}", nail_path.display()).bright_red().bold()
    );
    println!(
        "{}",
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━".dimmed()
    );

    let output_dir = expand_tilde(db_path).parent()
        .map(|p| p.to_path_buf())
        .unwrap_or_else(|| PathBuf::from("."));

    let start = Instant::now();
    unpack_nail(nail_path, &output_dir)?;
    let elapsed = start.elapsed();

    println!(
        "\n  {} {}",
        "OK".green(),
        "State unpacked!".green().bold()
    );
    print_timing("  Time", elapsed);

    Ok(())
}

fn cmd_bench(engine: &mut ReflexEngine) -> Result<()> {
    println!(
        "\n{}",
        "PincherOS Benchmark Suite".bright_red().bold()
    );
    println!(
        "{}",
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━".dimmed()
    );

    let embedder = engine.embedder();

    // 1. Single embed latency
    print!("  {} Embedding single text...", "...".yellow());
    let start = Instant::now();
    embedder.embed("hello world")?;
    let single_embed = start.elapsed();
    println!(
        " {}",
        format!("{:.2}ms", single_embed.as_secs_f64() * 1000.0).cyan()
    );

    // 2. Batch embed (10x) latency
    print!(
        "  {} Embedding batch (10x)...",
        "...".yellow()
    );
    let texts: Vec<&str> = [
        "open the browser",
        "show me the files",
        "connect to server",
        "list all processes",
        "kill the daemon",
        "start the service",
        "check disk usage",
        "display network stats",
        "update the system",
        "clean temp files",
    ].to_vec();
    let start = Instant::now();
    embedder.batch_embed(&texts)?;
    let batch_embed = start.elapsed();
    println!(
        " {}",
        format!("{:.2}ms", batch_embed.as_secs_f64() * 1000.0).cyan()
    );

    // 3. Teach latency (with seeding)
    print!(
        "  {} Seeding 10 reflexes...",
        "...".yellow()
    );
    let seed_intents = [
        ("open browser", "xdg-open https://example.com"),
        ("list files", "ls -la"),
        ("show processes", "ps aux"),
        ("check memory", "free -h"),
        ("disk usage", "df -h"),
        ("network status", "ip addr show"),
        ("git status", "git status"),
        ("build project", "cargo build --release"),
        ("run tests", "cargo test"),
        ("deploy app", "kubectl apply -f deploy.yaml"),
    ];
    let start = Instant::now();
    for (intent, action) in &seed_intents {
        engine.teach(intent, action)?;
    }
    let teach_total = start.elapsed();
    let teach_avg = teach_total / 10;
    println!(
        " {}",
        format!("{:.2}ms avg", teach_avg.as_secs_f64() * 1000.0).cyan()
    );

    // 4. Full do_command latency
    print!(
        "  {} Full do_command...",
        "...".yellow()
    );
    let start = Instant::now();
    let do_count = 50;
    for _ in 0..do_count {
        let _ = engine.do_command("check memory");
    }
    let do_total = start.elapsed();
    let do_avg = do_total / do_count;
    println!(
        " {}",
        format!(
            "{:.2}ms avg ({} iterations)",
            do_avg.as_secs_f64() * 1000.0,
            do_count
        )
        .cyan()
    );

    // Summary table
    println!();
    println!(
        "{}",
        "  +-------------------------+--------------+-------+".dimmed()
    );
    println!(
        "{}",
        "  | Benchmark               | Latency      | Status|".dimmed()
    );
    println!(
        "{}",
        "  +-------------------------+--------------+-------+".dimmed()
    );

    let single_pass = single_embed.as_secs_f64() * 1000.0 < 50.0;
    let batch_pass = batch_embed.as_secs_f64() * 1000.0 < 500.0;
    let teach_pass = teach_avg.as_secs_f64() * 1000.0 < 50.0;
    let do_pass = do_avg.as_secs_f64() * 1000.0 < 50.0;

    for (label, latency_ms, pass) in [
        ("Single Embed", single_embed.as_secs_f64() * 1000.0, single_pass),
        ("Batch Embed (10x)", batch_embed.as_secs_f64() * 1000.0, batch_pass),
        ("Teach (avg)", teach_avg.as_secs_f64() * 1000.0, teach_pass),
        ("Do Command (avg)", do_avg.as_secs_f64() * 1000.0, do_pass),
    ] {
        let status = if pass {
            "  PASS".green()
        } else {
            "  FAIL".red()
        };
        println!(
            "  | {:<23} {:>10.2}ms |{} |",
            label,
            latency_ms,
            status,
        );
    }

    println!(
        "{}",
        "  +-------------------------+--------------+-------+".dimmed()
    );

    let all_pass = single_pass && batch_pass && teach_pass && do_pass;
    if all_pass {
        println!(
            "\n  {} {}",
            "OK".green(),
            "All benchmarks passed!".green().bold()
        );
    } else {
        println!(
            "\n  {} {}",
            "!!".yellow(),
            "Some benchmarks exceeded thresholds".yellow().bold()
        );
    }

    Ok(())
}

fn cmd_shell_info(engine: &ReflexEngine) -> Result<()> {
    let fp = get_current_fingerprint();

    println!("\n{}", "Shell Fingerprint".bright_red().bold());
    println!(
        "{}",
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━".dimmed()
    );

    println!("\n  {} Hardware", "==".cyan().bold());
    println!("  {} {}", "Hostname:".dimmed(), fp.hostname);
    println!("  {} {}", "OS:".dimmed(), fp.os);
    println!("  {} {}", "OS Version:".dimmed(), fp.os_version);
    println!("  {} {}", "Architecture:".dimmed(), std::env::consts::ARCH);
    println!("  {} {}", "CPU Cores:".dimmed(), fp.cpu_count);
    println!(
        "  {} {}",
        "Total RAM:".dimmed(),
        format!("{:.2} GB", fp.ram_mb as f64 / 1024.0)
    );
    println!(
        "  {} {}",
        "GPU:".dimmed(),
        fp.gpu
    );

    println!("\n  {} Runtime", "==".cyan().bold());
    println!("  {} {}", "Embedding Dim:".dimmed(), EMBEDDING_DIM);
    println!("  {} {}", "Model Loaded:".dimmed(), engine.embedder().is_loaded());
    println!("  {} {}", "DB Path:".dimmed(), default_db_path());

    let status = engine.get_status()?;
    println!("  {} {}", "Reflexes:".dimmed(), status.reflex_count);
    println!("  {} {}", "Action Log:".dimmed(), status.action_log_count);

    println!("\n  {} Environment", "==".cyan().bold());
    println!("  {} {}", "PID:".dimmed(), std::process::id());
    println!(
        "  {} {}",
        "Working Dir:".dimmed(),
        std::env::current_dir()
            .map(|p| p.display().to_string())
            .unwrap_or_else(|_| "unknown".to_string())
    );

    Ok(())
}

// ─── Doctor Command ──────────────────────────────────────────────────────────

struct DoctorCheck {
    name: String,
    passed: bool,
    message: String,
}

fn doctor_check(name: &str, passed: bool, message: &str) -> DoctorCheck {
    DoctorCheck {
        name: name.to_string(),
        passed,
        message: message.to_string(),
    }
}

fn cmd_doctor(engine: &ReflexEngine) -> Result<()> {
    println!("\n{}", "PincherOS Doctor — Health Check".bright_red().bold());
    println!(
        "{}",
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━".dimmed()
    );

    let mut checks: Vec<DoctorCheck> = Vec::new();

    // 1. Embedding model check
    let embedder = engine.embedder();
    let model_loaded = embedder.is_loaded();
    checks.push(doctor_check(
        "ONNX Model",
        model_loaded,
        if model_loaded {
            "Model loaded successfully"
        } else {
            "Model not loaded — running in fallback/hash mode"
        },
    ));

    // 2. Embedding dimension check
    let embed_dim = EMBEDDING_DIM;
    checks.push(doctor_check(
        "Embedding Dim",
        embed_dim > 0,
        &format!("Dimension: {}", embed_dim),
    ));

    // 3. SQLite accessibility
    let conn = engine.connection();
    let sqlite_ok = conn
        .query_row("SELECT 1", [], |row| row.get::<_, i64>(0))
        .is_ok();
    checks.push(doctor_check(
        "SQLite",
        sqlite_ok,
        if sqlite_ok {
            "Database accessible and responsive"
        } else {
            "FAILED — database not responding"
        },
    ));

    // 4. Reflex count
    let status = engine.get_status()?;
    let has_reflexes = status.reflex_count > 0;
    checks.push(doctor_check(
        "Reflexes",
        has_reflexes,
        &format!(
            "{} reflex(es) stored, {} action log entries",
            status.reflex_count, status.action_log_count
        ),
    ));

    // 5. Disk space check
    let db_path_str = default_db_path();
    let db_path = expand_tilde(&db_path_str);
    let home_dir = db_path.parent().unwrap_or_else(|| std::path::Path::new("/tmp"));
    let disk_info = std::fs::metadata(home_dir).ok().and_then(|_| {
        let output = std::process::Command::new("df")
            .args(["-h", home_dir.to_str().unwrap_or("/")])
            .output()
            .ok();
        output.map(|o| String::from_utf8_lossy(&o.stdout).to_string())
    });
    let disk_ok = disk_info.is_some();
    checks.push(doctor_check(
        "Disk Space",
        disk_ok,
        &disk_info
            .map(|info| {
                info.lines()
                    .nth(1)
                    .map(|l| {
                        let parts: Vec<&str> = l.split_whitespace().collect();
                        if parts.len() >= 4 {
                            format!(
                                "Total: {}, Used: {}, Avail: {}",
                                parts.get(1).unwrap_or(&"?"),
                                parts.get(2).unwrap_or(&"?"),
                                parts.get(3).unwrap_or(&"?")
                            )
                        } else {
                            "Disk space info available".to_string()
                        }
                    })
                    .unwrap_or_else(|| "Disk space info available".to_string())
            })
            .unwrap_or_else(|| "Could not determine disk space".to_string()),
    ));

    // 6. bwrap availability
    let bwrap_available = std::process::Command::new("which")
        .arg("bwrap")
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false);
    checks.push(doctor_check(
        "Sandbox (bwrap)",
        bwrap_available,
        if bwrap_available {
            "bubblewrap (bwrap) available — sandbox ready"
        } else {
            "bwrap not found — sandboxed execution unavailable"
        },
    ));

    // Print report
    println!();
    let mut all_passed = true;
    for check in &checks {
        let status_icon = if check.passed {
            "PASS".green()
        } else {
            all_passed = false;
            "FAIL".red()
        };
        println!(
            "  [{}] {:<20} {}",
            status_icon,
            check.name.cyan(),
            check.message.dimmed()
        );
    }

    println!();
    if all_passed {
        println!(
            "  {} {}",
            "OK".green(),
            "All checks passed — PincherOS is healthy!".green().bold()
        );
    } else {
        println!(
            "  {} {}",
            "!!".yellow(),
            "Some checks failed — see above for details".yellow().bold()
        );
    }

    Ok(())
}

fn cmd_reflexes(engine: &ReflexEngine, verbose: bool) -> Result<()> {
    let conn = engine.connection();
    let reflexes = get_all_reflexes(conn)?;

    println!("\n{}", "Stored Reflexes".bright_red().bold());
    println!(
        "{}",
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━".dimmed()
    );

    if reflexes.is_empty() {
        println!(
            "\n  {} No reflexes stored yet.",
            "-".dimmed()
        );
        println!(
            "  {}",
            "  Use 'pincher teach' to add one.".dimmed()
        );
        return Ok(());
    }

    println!(
        "\n  {} reflexes found\n",
        reflexes.len()
    );

    // Convert to Reflex for display
    let display_reflexes: Vec<Reflex> = reflexes.into_iter().map(|r| r.into()).collect();

    if verbose {
        for reflex in &display_reflexes {
            print_reflex_detail(reflex);
        }
    } else {
        // Compact table
        println!(
            "  {} {} {} {} {}",
            "ID".dimmed().to_string().pad_to_width(8),
            "Intent".dimmed().to_string().pad_to_width(25),
            "Action".dimmed().to_string().pad_to_width(25),
            "Confidence".dimmed().to_string().pad_to_width(12),
            "Invoked".dimmed()
        );
        println!(
            "  {}",
            "-------------------------------------------------------------".dimmed()
        );

        for reflex in &display_reflexes {
            let confidence_bar = confidence_bar(reflex.confidence);
            println!(
                "  {:<8} {:<25} {:<25} {} {:>6}",
                truncate(&reflex.id, 8),
                truncate(&reflex.intent, 25),
                truncate(&reflex.action, 25),
                confidence_bar,
                reflex.invoke_count,
            );
        }
    }

    Ok(())
}

fn print_reflex_detail(reflex: &Reflex) {
    println!("  {} {}", "-- Reflex #".dimmed(), reflex.id.to_string().cyan());
    println!("  {} {}", "    Intent:".dimmed(), reflex.intent);
    println!("  {} {}", "    Action:".dimmed(), reflex.action.cyan());
    println!(
        "  {} {}",
        "    Confidence:".dimmed(),
        format!("{:.4}", reflex.confidence)
    );
    println!("  {} {}", "    Invoked:".dimmed(), reflex.invoke_count);
    println!();
}

fn confidence_bar(confidence: f64) -> String {
    let width = 10;
    let filled = (confidence * width as f64).round() as usize;
    let empty = width - filled;

    let bar: String = "#".repeat(filled);
    let space: String = "-".repeat(empty);

    if confidence >= 0.8 {
        format!("{}{} {}", bar.green(), space.dimmed(), format!("{:.2}", confidence).green())
    } else if confidence >= 0.5 {
        format!("{}{} {}", bar.yellow(), space.dimmed(), format!("{:.2}", confidence).yellow())
    } else {
        format!("{}{} {}", bar.red(), space.dimmed(), format!("{:.2}", confidence).red())
    }
}

fn truncate(s: &str, max_len: usize) -> String {
    if s.len() <= max_len {
        s.to_string()
    } else {
        format!("{}~", &s[..max_len - 1])
    }
}

// ─── Trait helper for padding ────────────────────────────────────────────────

trait PadToWidth {
    fn pad_to_width(&self, width: usize) -> String;
}

impl PadToWidth for String {
    fn pad_to_width(&self, width: usize) -> String {
        if self.len() >= width {
            self[..width].to_string()
        } else {
            let padding = width - self.chars().count();
            format!("{}{}", self, " ".repeat(padding))
        }
    }
}

// ─── Main ────────────────────────────────────────────────────────────────────

fn main() -> Result<()> {
    let cli = Cli::parse();

    init_tracing(&cli.log_level);

    let db_path = cli.db.unwrap_or_else(default_db_path);
    let overall_start = Instant::now();

    let result = match cli.command {
        Commands::Status => {
            let mut engine = create_engine(&db_path)?;
            cmd_status(&mut engine)
        }
        Commands::Teach { intent, action } => {
            let mut engine = create_engine(&db_path)?;
            cmd_teach(&mut engine, intent, action)
        }
        Commands::Do { intent } => {
            let mut engine = create_engine(&db_path)?;
            cmd_do(&mut engine, &intent)
        }
        Commands::Pack { output } => {
            let engine = create_engine(&db_path)?;
            cmd_pack(&engine, output)
        }
        Commands::Unpack { nail } => {
            cmd_unpack(&nail, &db_path)
        }
        Commands::Bench => {
            let mut engine = create_engine(&db_path)?;
            cmd_bench(&mut engine)
        }
        Commands::ShellInfo => {
            let engine = create_engine(&db_path)?;
            cmd_shell_info(&engine)
        }
        Commands::Doctor => {
            let engine = create_engine(&db_path)?;
            cmd_doctor(&engine)
        }
        Commands::Reflexes { verbose } => {
            let engine = create_engine(&db_path)?;
            cmd_reflexes(&engine, verbose)
        }
    };

    let elapsed = overall_start.elapsed();

    if result.is_ok() {
        println!(
            "\n  {} {}",
            "Time:".dimmed(),
            format!("Completed in {:.2}ms", elapsed.as_secs_f64() * 1000.0).dimmed()
        );
    }

    result
}
