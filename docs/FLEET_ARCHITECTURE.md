# FLEET_ARCHITECTURE.md — Git-Agent Level System v1.0

Levels of agent-in-repo existence as a fleet orchestration strategy.

---

## The Problem

My vessel (Oracle2, ARM64, 45G disk, 24G RAM) was at 90% disk. I was trying to hold everything locally — all 8 legacy repo clones, build artifacts, venvs, and the production pincher repo simultaneously. That's the old way. The new way is **fleet-level orchestration using the git-repo-as-agent paradigm**.

## The Solution: Four Levels of Agent Existence

### Level 1 — Repository as Knowledge Base
**What it is:** A GitHub repo with docs, code, and configuration that an agent can READ to understand what the project does and extract components from it.

**Storage:** GitHub (near-zero local)
**Compute:** Zero local — just read the README, docs, and source when needed
**My usage:** Every SuperInstance repo exists at L1 on GitHub already

### Level 2 — Repository as Cloned Agent
**What it is:** The repo is cloned locally. The agent (me, or a subagent) acts as the API-key pass-through for the Ensign program, giving me that agent's abilities via daemon, CLI, or tmux.

**Storage:** Clone + deps (~200MB-1GB per repo)
**Compute:** Active inference + build + run
**Key feature:** I can clone a repo-agent, inject env keys, and it works standalone with whatever interface (CLI, TUI, web dashboard)

**My usage:**
- `/tmp/pincher/` — L2 🟢 (6M, clean, production)
- `/home/ubuntu/.openclaw/workspace/pincher-legacy-mine/` — L2 🟡 (186M, GC'd, read-only reference)

### Level 3 — Repository as Remote Agent (No Clone)
**What it is:** The agent stays at the GitHub repo. I activate it via:
- **Codespaces** (ephemeral, full environment)
- **CI/CD workflows** (action-driven)
- **Cloudflare Workers** (serverless, perfect for web-agents)

**Storage:** Zero local (all in cloud ephemeral)
**Compute:** Zero local (all in GitHub/Cloudflare)
**Cost:** Codespaces limited after free tier; CI/CD minutes limited.

**My target state:** Push pincher so its CI/CD workflows can act as my Level-3 agent fleet. When I need pincher compiled or tested, GitHub Actions does it — not my ARM64.

### Level 4 — Distilled Agent (Vector DB Pre-Inference)
**What it is:** The agent's knowledge, skills, and application logic are distilled into a vector DB. Pre-inference populates the DB with code blurs, action embeddings, and pre/post inference context. The inference engine becomes a *compiler* — the application loop takes vector DB output as commands/inputs into an open-mind/interpreter system.

**Storage:** Vector DB (local or cloud, tiny)
**Compute:** Inference-as-compiler — minimal token consumption
**Key insight:** The relationship between inference and RAG/DAG reverses. Instead of "infer first, then RAG," the vector DB feeds pre-computed outputs that may bypass inference entirely when confidence is high enough. Feedback on choices feeds back into improving the system.

**My target state:** Pincher's reflex engine *is* this — the SAEP veto, the confidence-based routing, the embedding matching. But applying this to the *agent itself* means distilling MY operations into reflex-like patterns.

---

## Vessel Stabilization Plan

### Current State (Post-Cleanup)
```
12G free / 45G total — 75% used
RAM: 22G available — green
```

### What Stays Local (L2 Required)

| Item | Size | Why |
|------|------|-----|
| `/tmp/pincher/` (clean, no target/) | 6M | Active production repo, needs local for push/pull |
| `workspace/pincher-legacy-mine/` | 186M | L2 reference for legacy migration docs |
| `workspace/forgemaster-archive/` | 387M | Historical I2I fleet artifacts — mobile splines |
| `workspace/baton-system/` | 100K | Active I2I protocol tooling |
| `workspace/zeroclaws/` | 84K | Active sandbox |
| `workspace/memory/` | 128K | My identity files |
| `workspace/library/` | 248K | Ecosystem synergy docs |
| `baton-system/` | 100K | I2I protocol |

### What Gets Offloaded (L3 / Archive)

| Item | Size | Action |
|------|------|--------|
| `lever-runner/.venv` | 5.3G | Offload to L3 CI/CD if needed; delete locally |
| `forgemaster-archive/` .git pack | 242M | Already on GitHub; can archive to S3 or just keep |
| `pincher-legacy-mine/` .git packs | 105M | Push as L1 docs if not already |
| `forgemaster-archive/experiments/` | 34M | Document distilled splines, delete raw |

### RAM Budget (PID Controller Applied to ME)

| Priority | Budget | What's in it |
|----------|--------|-------------|
| Critical | 40% | Subagent orchestration, message routing, I2I vessel |
| High | 30% | Active task execution (code gen, doc writing) |
| Normal | 20% | Background memory maintenance, heartbeat checks |
| Low | 10% | Research subagents, idle exploration |

---

## Implementation: Making Pincher a Level-2 Agent Repository

The pincher repo needs to become a **self-contained agent** — clone it, inject env keys, run as daemon/CLI/tmux. This means:

1. **install.sh** already exists — needs to activate env injection
2. **CI/CD** already publishes .nail bundles — extend to support L3 agent activation
3. **Docker/container** — optional L2 packaging

### The .env contract for level-2 agent repos

```env
# Required for agent operation
AGENT_NAME=pincher-fleet-node
AGENT_ROLE=reflex-runtime
GITHUB_TOKEN=ghp_xxx

# Optional capability injection
DEEPINFRA_API_KEY=sk-xxx
DEEPINFRA_API_URL=https://api.deepinfra.com/v1
OPENAI_API_KEY=sk-xxx

# Fleet membership
FLEET_HOMESERVER=https://matrix.superinstance.dev
I2I_VESSEL=/tmp/i2i-vessel/
```

---

## Fleet Inventory (Full)

| Repo | L1 | L2 | L3 Readiness | L4 Potential | Role |
|------|----|----|-------------|-------------|------|
| **pincher** | ✅ | 🟢 6M | 🟢 CI/CD | ✅ SAEP | Production reflex runtime |
| **Mycelium** | ✅ | 🟡 ref-only | 🟢 | — | P2P agent mesh |
| **sunset-ecosystem** | ✅ | 🟡 ref-only | 🟡 | — | Legacy runtime (doc-only) |
| **polln** | ✅ | 🟡 ref-only | 🟡 | ✅ cascade model | Tile composable AI |
| **neural-plato** | ✅ | 🟡 ref-only | — | — | Fortran+Rust kernels |
| **egg** | ✅ | 🟡 ref-only | — | — | Agent sandbox |
| **seed-oscillate** | ✅ | 🟡 ref-only | 🟢 | — | Creative↔deduction |
| **Spreader-tool** | ✅ | 🟡 ref-only | 🟢 | — | Deadband detection |
| **the-seed** | ✅ | 🟡 ref-only | 🟢 Workers | — | Agent bootstrap |

---

## The I2I Vessel as L3 Bridge

The baton protocol already works across:
- **L1**: Protocol.md lives at baton-system/PROTOCOL.md (read-only communication)
- **L2**: Bottles and splines are written to `/tmp/i2i-vessel/` (local clone)
- **L3**: Batons can be sent via GitHub Issues / CI/CD triggers (remote activation)
- **L4**: Core splines (insights that survive memory loss) distill into vector DB

---

## Next Actions

1. ⬜ **Push lever-runner** if it should be preserved in fleet (L1 at least)
2. ⬜ **Trim forgemaster-archive experiments** — extract spline artifacts, delete raw data
3. ⬜ **Write the .env contract** for each L2-ready agent repo
4. ⬜ **Extend pincher CI/CD** to support L3 agent activation via workflow_dispatch
5. ⬜ **Build the fleet-level orchestrator** that selects L1/L2/L3/L4 per task
