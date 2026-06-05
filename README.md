# 🦀 Pincher — Vector DB as Runtime, LLM as Compiler

**Version:** 0.1.0  
**Maintainer:** oracle2  
**Lineage:** PincherOS, cocapn-runtime, PLATO, fleet I2I

---

## What Is This?

Pincher is a **reflex runtime** for agents. It snaps into any shell and adds adaptive, battery-powered cognition:

- **Teach → Match → Execute** — the Reflex Engine
- **SQLite-backed vector store** for embedding search over learned reflexes
- **Bubblewrap sandbox** for isolated execution
- **ONNX embeddings** (all-MiniLM-L6-v2) with hash-based fallback
- **Portable `.nail` rig format** — pack your agent's identity and reflexes into a single file
- **CLI-driven** — no daemons, no cloud dependency, no fantasy

It is **not** an OS. It does **not** have five deployment modes. It does **not** have a Holodeck. It is a focused, portable reflex runtime that works right now.

---

## The Reflex Engine (The Core)

The heart of Pincher is the **Reflex Engine** — a Teach → Match → Execute loop:

```text
Intent (user says "show running containers")
  ↓
[1] Embed intent into vector
  ↓
[2] Match against known reflexes (SQLite vector search)
  ├── ≥ 0.80 "Exact"  → execute directly (~50ms, $0)
  ├── 0.55-0.80 "Similar" → confirm + execute (~3s, ~$0.001)
  └── < 0.55 "Novel" → route to LLM-as-Compiler → store new reflex
  ↓
[3] Sandboxed execution via bubblewrap (with veto engine)
  ↓
[4] Log result, update confidence score
```

Every successful run increases confidence. Every failure decreases it. The system gets faster and more reliable the more you use it.

### Built-in Intents

The engine ships with a growing set of built-in reflex dispatchers:

| Intent | What it does |
|--------|-------------|
| `system.info` | System info, uptime, resources |
| `file.read` | Read a file |
| `file.write` | Write a file |
| `process.list` | List running processes |
| `process.kill` | Kill a process |
| `network.ping` | Ping a host |
| `git.status` | Git working-tree status |
| `git.diff` | Git diff |
| `docker.ps` | List Docker containers |
| `env.get` | Read environment variable |

---

## Installation

### From Source (current — v0.1.0)

```bash
# Prerequisites: Rust toolchain (rustup)
git clone https://github.com/SuperInstance/pincher.git
cd pincher
cargo build --release -p pincher-cli

# Binary lands at target/release/pincher
cp target/release/pincher ~/.local/bin/
```

### One-Line Installer

```bash
curl -fsSL https://raw.githubusercontent.com/SuperInstance/pincher/main/install.sh | bash
```

This checks for Rust, builds from source, and installs the `pincher` binary to `~/.local/bin/`.

---

## CLI

The `pincher` binary is the primary interface. All commands are wired in the CLI and backed by the core library:

| Command | Purpose |
|---------|---------|
| `pincher status` | Engine status, reflex count, database path |
| `pincher doctor` | Health check — ONNX model, SQLite, disk, embedding |
| `pincher teach` | Interactive teach flow — store a new reflex |
| `pincher do "..."` | Execute a natural language intent through the reflex engine |
| `pincher reflexes` | List stored reflexes with confidence scores |
| `pincher compile` | Read workspace manifest → compile to WASM reflex |
| `pincher mature` | Adversarial fuzzing to expand vector search coverage |
| `pincher bench` | Run benchmark suite (embed latency, teach/match latency) |
| `pincher shell-info` | Hardware fingerprint |
| `pincher pack --output agent.nail` | Pack state + identity into portable `.nail` file |
| `pincher unpack --bundle agent.nail` | Unpack and merge `.nail` state |
| `pincher run --bundle agent.nail "..."` | Execute a bundle against user input |
| `pincher publish` | Publish bundle to registry (requires token) |
| `pincher update` | Check for registry updates |
| `pincher gastrolith` | Checkpoint migration management |

### Environment Variables

| Variable | Default | Purpose |
|----------|---------|---------|
| `PINCHER_DB` | `~/.pincher/reflexes.db` | Path to SQLite reflex database |
| `PINCHER_LOG_LEVEL` | `warn` | Log verbosity |
| `PINCHER_REGISTRY_URL` | `https://registry.pincher.dev` | Bundle registry URL |
| `PINCHER_REGISTRY_TOKEN` | — | Authentication for publishing |

### Quick Start

```bash
# Check the engine is alive
pincher status

# Run a health check
pincher doctor

# List stored reflexes
pincher reflexes

# Execute an intent
pincher do "list files in current directory"

# Teach a new reflex
pincher teach
```

---

## Architecture

The project is a Rust workspace with two crates:

### `pincher-core/` — The Library

All runtime logic lives here:

| Module | What it does | Status |
|--------|-------------|--------|
| `reflex/engine.rs` | ReflexEngine — match, execute, teach, confidence loop | ✅ Implemented |
| `reflex/matcher.rs` | Vector similarity matching with thresholds | ✅ Implemented |
| `reflex/confidence.rs` | Multiplicative confidence update model | ✅ Implemented |
| `db/` | SQLite-backed vector store with sqlite-vec | ✅ Implemented |
| `embed/onnx.rs` | all-MiniLM-L6-v2 ONNX embeddings + hash fallback | ✅ Implemented |
| `sandbox/bwrap.rs` | Bubblewrap sandbox with config + veto patterns | ✅ Implemented |
| `migration/` | `.nail` pack/unpack with BLAKE3 + tar.zst | ✅ Implemented |
| `rpc/` | JSON-RPC server for programmatic control | ✅ Implemented |
| `resource/` | PID resource controller with budgets | ✅ Implemented |
| `capability/` | Signed capability tokens and manifests | ✅ Implemented |
| `security/veto.rs` | Veto engine — pattern-based pre-execution blocking | ✅ Implemented |
| `shell/` | Shell fingerprinting | ✅ Implemented |
| `immunology/` | Pattern-based immune system | ✅ Implemented |

### `pincher-cli/` — The CLI Frontend

A Clap-based CLI that exposes all core functionality through subcommands. Built with `tokio` for async execution.

### Feature Flags

`pincher-core` uses Cargo features for optional components:

```bash
# Enable all features
cargo build --features "onnx,landlock,wasmtime"

# Minimal build (hash-based embeddings only)
cargo build
```

| Feature | What it enables |
|---------|----------------|
| `onnx` | Real ONNX Runtime embeddings (all-MiniLM-L6-v2) |
| `landlock` | Linux Landlock sandboxing (kernel 5.13+) |
| `wasmtime` | WASM guest module execution |

---

## The Vector Store

Every reflex is stored as a row in an SQLite database with a 384-dimensional embedding vector:

```sql
-- The reflexes table (via sqlite-vec)
CREATE TABLE reflexes (
    id          TEXT PRIMARY KEY,
    intent      TEXT NOT NULL,
    action_sql  TEXT NOT NULL,
    embedding   BLOB,       -- f32 vector, 384 elements
    confidence  REAL DEFAULT 0.55,
    invoke_count INTEGER DEFAULT 0,
    created_at  TEXT DEFAULT (datetime('now')),
    updated_at  TEXT DEFAULT (datetime('now'))
);
```

This is **not** a hypothetical architecture. The schema exists at `registry_schema.sql`, the SQLite-backed store is wired in `pincher-core/src/db/schema.rs`, and vector search via `sqlite-vec` is production code.

---

## The Sandbox

When Pincher executes a reflex, it runs in a **bubblewrap sandbox** (when available):

- Default restricted: no network, `/usr`/`/lib` mounted read-only
- Whitelisted common binaries only (`ls`, `cat`, `grep`, `touch`, etc.)
- Dangerous patterns blocked at the string-matching level (`rm -rf /`, `mkfs`, `dd if=/dev/zero`, fork bombs)
- Falls back to bare `std::process::Command` with a warning if `bwrap` is not installed

The veto engine runs *before* the sandbox, checking the command against blocked patterns.

---

## The Rig: Portable Agent Identity

Pack your agent's reflexes, identity, and configuration into a **`.nail` file** — a portable `tar.zst` archive with BLAKE3 checksums:

```bash
# Pack the whole rig
pincher pack --output my-agent.nail

# Unpack on another machine
pincher unpack --bundle my-agent.nail
```

A `.nail` archive contains:
```
my-agent.nail
├── manifest.json       # Version, checksums, hardware fingerprint
├── reflexes.db         # Full SQLite vector DB
├── identity.json       # Agent name, preferences
└── config.toml         # Resource thresholds
```

The `migration` module in `pincher-core` implements pack, unpack, verify, fingerprint, and compatibility scoring.

---

## Tools and Scripts

The `tools/` directory contains supporting scripts:

| Script | Purpose |
|--------|---------|
| `reflex-engine.sh` | Lightweight metacognitive daemon — scans disk/RAM pressure, runs reflexes |
| `checkpoint.sh` | System checkpoint utility |
| `fleet-scout.sh` | Fleet reconnaissance |
| `gc-fleet.sh` | Fleet garbage collection |
| `init-context.sh` | Context initialization |
| `promote-reflex.sh` | Promote reflexes through confidence tiers |
| `deepinfra_client.py` | DeepInfra API client for LLM routing |
| `model_router.py` | Model routing logic |

---

## Development

### Prerequisites

- Rust 2024+ (see `rust-toolchain.toml`)
- For ONNX features: `ort` library (optional)

### Build

```bash
# Debug build
cargo build

# Release build
cargo build --release

# Release with all features
cargo build --release --features "onnx,landlock,wasmtime"
```

### Codespace

The repo includes a `.devcontainer/` with pre-configured Rust environment, GitHub CLI, and Python — ready for GitHub Codespaces.

### Project Structure

```
/
├── Cargo.toml                 # Workspace root
├── Cargo.lock
├── rust-toolchain.toml        # Toolchain pinning
├── pincher-core/              # Core library (all runtime logic)
│   ├── Cargo.toml
│   └── src/
│       ├── lib.rs
│       ├── reflex/            # Reflex engine (match, execute, teach)
│       ├── db/                # SQLite vector store
│       ├── embed/             # ONNX + hash embeddings
│       ├── sandbox/           # Bubblewrap isolation
│       ├── migration/         # .nail pack/unpack
│       ├── rpc/               # JSON-RPC server
│       ├── resource/          # Resource controller
│       └── ...
├── pincher-cli/               # CLI frontend
│   ├── Cargo.toml
│   └── src/main.rs            # pincher binary
├── pincher-infer/             # Python inference module
│   ├── pincher_infer/
│   └── tests/
├── tools/                     # Supporting scripts
├── scripts/                   # GC, dedup, self-heal scripts
├── docs/                      # Architecture, roadmap, ADRs
├── examples/                  # Code review, hello-reflex, smart-home
├── assets/                    # Logo images
├── config/pincher.toml        # Default configuration
├── install.sh                 # One-line installer
├── registry_schema.sql        # Registry SQL schema
└── .devcontainer/             # Codespace config
```

---

## What This Is Not

This list matters. The current README inherited a lot of "future-fantasy" from earlier documentation. Here is the honest picture:

- **No "Lighthouse Keeper"** — there is no cloud fleet management API. The RPC module and fleet-scout script exist, but a centralized fleet coordinator does not.
- **No "Tender"** — there is no edge-sync protocol. The `migration` module handles `.nail` file transfer, but automatic "Tender visits" are aspirational.
- **No "Holodeck MUD"** — this does not exist and is not on the short-term roadmap.
- **No `boot.sh`** — the legacy boot script was replaced by the `pincher` CLI. Run `pincher` commands directly.
- **No Docker image on Docker Hub** — `superinstance/pincher` is not published. You build from source.
- **No ESP32 build** — the `wasmtime` feature and `cargo build` target only host OSes.
- **No instant-boot claims** — benchmark data is collected (see `benchmarks/`) but not yet published as guarantees.

The project does **not** deploy in "5 modes." It deploys in exactly one mode: **build from source, run the binary**. That is the honest scope of v0.1.0.

---

## Roadmap (Near-Term)

These are the real priorities as documented in `docs/ROADWAY.md` and `docs/MVP_CHECKLIST.md`:

1. **WASM guest execution** — the `wasmtime` feature is wired, guest protocol is defined
2. **Landlock sandboxing** — the `landlock` feature is wired, needs production testing
3. **Reflex registry** — the `publish` and `update` CLI commands are stubs; registry format is defined in `registry_schema.sql`
4. **Multi-process execution** — sandbox can run more complex pipelines
5. **Benchmark rigor** — formalize latency and confidence metrics

---

## License

MIT OR Apache-2.0 — see `LICENSE`.

---

*🦀 Same crab. Bigger shell.*

*The hermit crab finds the right shell for every situation — but it starts with the one it's in.*
