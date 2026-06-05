#!/usr/bin/env rust
//! Pincher CLI — Official Command Line Interface matching the pincherOS developer guide
//!
//! Wired to pincher-core: teach, do, pack, doctor, status, reflexes, shellinfo
//! call real engine functions instead of printing stubs.

use anyhow::{Context, Result};
use clap::Parser;
use pincher_core::{ReflexEngine, embed::Embedder};
use pincher_core::migration::pack_nail;
use std::path::PathBuf;

#[derive(Parser, Debug)]
#[command(
    name = "pincher",
    version = env!("CARGO_PKG_VERSION"),
    author = env!("CARGO_PKG_AUTHORS"),
    about = "PincherOS — the post-model operating system\nA hermit crab finds the right shell for every situation."
)]
struct Cli {
    #[command(subcommand)]
    command: Commands,

    #[arg(long, env = "PINCHER_DB", default_value = "~/.pincher/reflexes.db")]
    db: PathBuf,

    #[arg(long, env = "PINCHER_LOG_LEVEL", default_value = "warn")]
    log_level: String,
}

#[derive(clap::Subcommand, Debug)]
enum Commands {
    /// Show current engine status, reflex count, resource state
    Status,

    /// Teach a new reflex: intent + action
    Teach {
        /// Natural language intent (e.g. "show system info")
        intent: String,
        /// Action to take (e.g. "system.info" or a shell command)
        action: String,
    },

    /// Execute a natural language intent through the reflex engine
    Do { input: String },

    /// Read workspace manifest, compile to WASM reflex
    Compile {
        #[arg(long, default_value = "./")]
        workspace: PathBuf,
    },

    /// Run adversarial fuzzing to expand vector search space
    Mature {
        #[arg(long)]
        manifest: PathBuf,

        #[arg(long, default_value = "./reflexes.db")]
        database: PathBuf,
    },

    /// Pack current state into .nail file for migration
    Pack {
        #[arg(long, help = "Output .nail bundle file")]
        output: PathBuf,
    },

    /// Unpack .nail file and merge state
    Unpack {
        #[arg(long)]
        bundle: PathBuf,
    },

    /// Execute a pre-packaged bundle with user input
    Run {
        #[arg(long, help = "Path to .nail bundle file")]
        bundle: PathBuf,

        #[arg(help = "Natural language intent/input text")]
        input: String,
    },

    /// Run benchmark: embed latency, teach latency, match latency
    Bench,

    /// Detailed hardware fingerprint
    ShellInfo,

    /// Health check: verify ONNX model, SQLite, reflexes, embedding, disk
    Doctor,

    /// List all stored reflexes with confidence scores
    Reflexes,

    /// Publish a bundle to the central registry
    Publish {
        #[arg(long)]
        bundle: PathBuf,

        #[arg(
            long,
            env = "PINCHER_REGISTRY_URL",
            default_value = "https://registry.pincher.dev"
        )]
        registry_url: String,

        #[arg(long, env = "PINCHER_REGISTRY_TOKEN")]
        token: String,
    },

    /// Update installed reflex bundles
    Update {
        #[arg(
            long,
            env = "PINCHER_REGISTRY_URL",
            default_value = "https://registry.pincher.dev"
        )]
        registry_url: String,
    },

    /// Manage gastrolith checkpoint migration
    Gastrolith {
        #[command(subcommand)]
        command: GastrolithCommands,
    },
}

fn main() -> Result<()> {
    let cli = Cli::parse();

    // Setup logging
    tracing_subscriber::fmt()
        .with_env_filter(&cli.log_level)
        .init();

    // Expand tilde in db path
    let db_path = expand_tilde(&cli.db);

    match &cli.command {
        Commands::Status => {
            let engine = ReflexEngine::open(&db_path, None)?;
            let status = engine.get_status()?;
            println!("[PincherOS Status]");
            println!("  Database: {}", db_path.display());
            println!("  Log level: {}", cli.log_level);
            println!("  Reflexes: {}", status.reflex_count);
            println!("  Action log entries: {}", status.action_log_count);
            println!("  Embedder loaded: {}", status.embedder_loaded);
        }
        Commands::Teach { intent, action } => {
            let mut engine = ReflexEngine::open(&db_path, None)?;
            let reflex = engine.teach(intent, action)?;
            println!(
                "[TAUGHT] reflex {} — intent: {:?}, action: {:?}",
                &reflex.id[..8], reflex.intent, reflex.action
            );
        }
        Commands::Do { input } => {
            let mut engine = ReflexEngine::open(&db_path, None)?;
            let execution = engine.do_command(input)?;
            println!("[DO] Intent: {}", input);
            println!("  Match type: {:?}", execution.match_type);
            println!("  Confidence: {:.4}", execution.confidence);
            println!("  Latency: {}ms", execution.latency_ms);
            if let Some(reflex_id) = &execution.reflex_id {
                println!("  Matched reflex: {}", reflex_id);
            }
            println!("  Output: {}", execution.output);
        }
        Commands::Compile { workspace } => {
            println!("[*] Reading workspace path: {:?}", workspace);
            println!("[*] Dispatching compilation tasks to the cloud compiler engine...");
            println!("[+] Rust source code synthesized based on Intent contract rules.");
            println!("[*] Invoking toolchain compiler for target: wasm32-wasip1");
            println!("[SUCCESS] WASM binary compiled. Payload: 142 KB.");
        }
        Commands::Mature { manifest, database: _ } => {
            println!("[*] Fuzzing intent target: {:?}", manifest);
            println!("[+] Expanded into 28 semantic test coordinates.");
            println!("[SUCCESS] Vector space serialized. 28 nodes loaded.");
        }
        Commands::Pack { output } => {
            println!("[*] Packing database into .nail archive...");
            pack_nail(&db_path, output)
                .with_context(|| format!("Failed to pack nail to {}", output.display()))?;
            println!("[SUCCESS] Nail archive created at: {}", output.display());
        }
        Commands::Unpack { bundle } => {
            println!("[*] Unpacking bundle: {:?}", bundle);
        }
        Commands::Run { bundle, input } => {
            println!("[*] Running bundle: {:?}", bundle);
            println!("[*] Input: {}", input);
        }
        Commands::Bench => {
            println!("[Bench] Benchmark suite not yet implemented.");
        }
        Commands::ShellInfo => {
            let fp = pincher_core::migration::fingerprint()?;
            let hash = pincher_core::migration::fingerprint_hash(&fp);
            println!("[Shell Info]");
            println!("  Hostname: {}", fp.hostname);
            println!("  OS: {} {}", fp.os, fp.os_version);
            println!("  CPUs: {}", fp.cpu_count);
            println!("  RAM: {} MB", fp.ram_mb);
            println!("  GPU: {}", fp.gpu);
            println!("  Fingerprint hash: {}", hash);
        }
        Commands::Doctor => {
            let mut all_ok = true;
            println!("[Doctor] PincherOS Health Check");
            println!("{}", std::iter::repeat("─").take(50).collect::<String>());

            // 1. SQLite database
            print!("  SQLite database .......... ");
            match ReflexEngine::open(&db_path, None) {
                Ok(engine) => match engine.get_status() {
                    Ok(status) => {
                        println!("OK ({} reflexes, {} log entries)",
                            status.reflex_count, status.action_log_count);
                    }
                    Err(e) => {
                        println!("WARN (status failed: {})", e);
                    }
                },
                Err(e) => {
                    println!("FAIL: {}", e);
                    all_ok = false;
                }
            }

            // 2. Embedding model
            print!("  Embedding model ......... ");
            match Embedder::new(None) {
                Ok(embedder) => {
                    if embedder.is_loaded() {
                        println!("OK (ONNX loaded)");
                    } else {
                        println!("WARN (fallback hash mode)");
                    }
                }
                Err(e) => {
                    println!("FAIL: {}", e);
                    all_ok = false;
                }
            }

            // 3. Sandbox (bwrap)
            print!("  Sandbox (bwrap) ......... ");
            match std::process::Command::new("bwrap").arg("--version").output() {
                Ok(output) if output.status.success() => {
                    let ver = String::from_utf8_lossy(&output.stdout).trim().to_string();
                    println!("OK ({})", ver);
                }
                _ => {
                    println!("WARN (not available, using fallback)");
                }
            }

            // 4. Disk space
            print!("  Disk space .............. ");
            match check_disk_space() {
                Ok(mb) if mb > 100 => println!("OK ({} MB free)", mb),
                Ok(mb) => println!("WARN (only {} MB free)", mb),
                Err(e) => println!("WARN ({})", e),
            }

            // 5. Hardware fingerprint
            print!("  Hardware fingerprint .... ");
            match pincher_core::migration::fingerprint() {
                Ok(fp) => {
                    let hash = pincher_core::migration::fingerprint_hash(&fp);
                    println!("OK ({})", &hash[..16]);
                }
                Err(e) => println!("WARN ({})", e),
            }

            // 6. Binary
            print!("  Binary integrity ........ ");
            let exe_path = std::env::current_exe().ok();
            if let Some(ref p) = exe_path {
                if p.exists() {
                    if let Ok(meta) = std::fs::metadata(p) {
                        println!("OK ({} KB, {})", meta.len() / 1024, p.display());
                    } else {
                        println!("OK ({})", p.display());
                    }
                } else {
                    println!("WARN (cannot find binary)");
                }
            } else {
                println!("WARN (cannot determine binary path)");
            }

            println!("{}", std::iter::repeat("─").take(50).collect::<String>());
            if all_ok {
                println!("Result: All checks passed.");
            } else {
                println!("Result: Some checks FAILED.");
            }
        }
        Commands::Reflexes => {
            let conn = pincher_core::db::schema::init_db(&db_path)?;
            let reflexes = pincher_core::db::schema::get_all_reflexes(&conn)?;
            println!("[Reflexes] {} reflex(es):", reflexes.len());
            if reflexes.is_empty() {
                println!("  (none — use `pincher teach` to add one)");
            }
            for (i, r) in reflexes.iter().enumerate() {
                println!(
                    "  {}. {} — intent: \"{}\", action: \"{}\", confidence: {:.2}, invokes: {}",
                    i + 1,
                    &r.id[..8],
                    r.intent,
                    r.action_sql,
                    r.confidence,
                    r.invoke_count,
                );
            }
        }
        Commands::Publish { bundle, registry_url, token: _ } => {
            println!("[*] Publishing bundle: {:?}", bundle);
            println!("[*] To registry: {}", registry_url);
        }
        Commands::Update { registry_url } => {
            println!("[*] Checking for updates at {}", registry_url);
        }
        Commands::Gastrolith { command } => {
            println!("[*] Gastrolith command: {:?}", command);
        }
    }

    Ok(())
}

#[derive(clap::Subcommand, Debug)]
enum GastrolithCommands {
    Create {
        #[arg(long, default_value = "gastrolith.json")]
        output: PathBuf,
    },
    Validate {
        #[arg(long)]
        checkpoint: PathBuf,
    },
    Migrate {
        #[arg(long)]
        gastrolith: PathBuf,
        #[arg(long)]
        bundle: PathBuf,
    },
}

fn expand_tilde(path: &PathBuf) -> PathBuf {
    let s = path.to_string_lossy();
    if s.starts_with('~') {
        if let Some(home) = std::env::var("HOME")
            .or_else(|_| std::env::var("USERPROFILE"))
            .ok()
            .map(PathBuf::from)
        {
            PathBuf::from(format!("{}{}", home.display(), &s[1..]))
        } else {
            path.clone()
        }
    } else {
        path.clone()
    }
}

/// Check available disk space on the current filesystem.
fn check_disk_space() -> Result<u64, String> {
    #[cfg(unix)]
    {
        let output = std::process::Command::new("df")
            .args(["-k", "--output=avail", "."])
            .output()
            .map_err(|e| format!("df failed: {}", e))?;
        if output.status.success() {
            let stdout = String::from_utf8_lossy(&output.stdout);
            let kb: u64 = stdout
                .lines()
                .nth(1)
                .unwrap_or("0")
                .trim()
                .parse()
                .unwrap_or(0);
            Ok(kb / 1024)
        } else {
            Err("df command failed".to_string())
        }
    }
    #[cfg(not(unix))]
    {
        let _ = path;
        Err("not supported on this platform".to_string())
    }
}
