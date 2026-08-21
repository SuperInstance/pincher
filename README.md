<div align="center">
<img src="assets/hermit-crab.jpg" width="320" alt="Hermit crab" />
</div>

# pincher

*A hermit crab doesn't grow a new shell. It finds one that fits, moves in, and makes it home.*

---

*pincher's reflexes live on as **quilt-pincher** — the same engine, as reactive cells (github.com/SuperInstance/quilt-pincher).*

---

pincher is a reflex engine. A reflex is what happens when input meets pattern before thought arrives. Most AI systems think first — route every message through an LLM, burn tokens and milliseconds on questions that have already been answered a hundred times. pincher does the opposite. It responds in <50ms, without an LLM, every time, at zero marginal cost. The LLM is the thinking layer. pincher is the *shell* — the fast, durable structure that catches the simple cases so the thinking layer only fires when it needs to.

There is no question of replacement. Warm-blooded thought and the spinal reflex are not in competition. The cortex teaches the spinal cord. The spinal cord gets faster. The shell protects the signal.

---

## The Shell

A hermit crab's shell is not its body. The shell is infrastructure — important, necessary, but replaceable. The crab carries its body from shell to shell. The reflexes, not the runtime.

pincher is that shell for AI agents. It sits between the agent's intent and the world, intercepting patterns before they reach expensive machinery. Every intent enters the shell. Within a few milliseconds, the shell decides: do I already know how to handle this?

The answer comes from a vector database — every intent the agent has ever encountered, embedded into a 384-dimensional space, scored by confidence. A known reflex fires directly. A semi-known reflex asks for confirmation. An unknown reflex escalates to the LLM, which compiles a new reflex and stores it for next time.

This is not caching. Caching returns the same answer to the same question. pincher returns the right answer to *similar* questions, because it matches on semantic embedding, not exact string. "Show me what's running" and "what processes are active" map to the same reflex. That's not cache. That's *understanding* — baked into the shell, not farmed out to a model.

```
                     ┌──────────────────────────────────────┐
                     │            pincher                    │
                     │   ┌──────────────────────────────┐   │
You say ────▶ [1] ──▶│   │  Reflex Engine               │   │
something     Embed  │   │  ┌─────┐   ┌────────┐       │   │
             (384D)  │   │  │Match│──▶│Execute  │       │   │
                     │   │  │≥0.80│   │Directly │       │   │
                     │   │  └─────┘   └────────┘       │   │
                     │   │  ┌─────┐   ┌────────┐       │   │
                     │   │  │Match│──▶│Confirm  │       │   │
                     │   │  │0.55-│   │+ Execute│       │   │
                     │   │  │0.80 │   └────────┘       │   │
                     │   │  └─────┘   ┌────────┐       │   │
                     │   │  ┌─────┐   │LLM     │       │   │
                     │   │  │Match│──▶│Compiles│       │   │
                     │   │  │<0.55│   │New     │       │   │
                     │   │  └─────┘   │Reflex  │       │   │
                     │   │            └────────┘       │   │
                     │   │  ┌──────────────────────┐   │   │
                     │   │  │ Veto Engine           │   │   │
                     │   │  │ Security → Sandbox   │   │   │
                     │   │  └──────────────────────┘   │   │
                     │   └──────────────────────────────┘   │
                     │              │                       │
                     │              ▼                       │
                     │   ┌──────────────────────────────┐   │
                     │   │    Reflex Database           │   │
                     │   │    (SQLite + sqlite-vec)    │   │
                     │   └──────────────────────────────┘   │
                     │              │                       │
                     │              ▼                       │
                     │   ┌──────────────────────────────┐   │
                     │   │    .nail Bundle               │   │
                     │   │    (Portable Agent State)     │   │
                     │   └──────────────────────────────┘   │
                     └──────────────────────────────────────┘
```

Three-tier compute:

```
Fast  (ms):   Embedding match + reflex execution (no LLM)
Medium (s):   Confirmation + optional execution (low confidence)
Slow   (s):   LLM compilation → new reflex (learning event)
```

**The shell is the design.** Not the agent. The shell. The agent will be replaced. The shell will persist. The agent will forget. The shell will remember. The agent will lose context. The shell will preserve shape.

Every cycle through this engine makes the agent faster and cheaper. The reflex database grows. The match scores climb. The LLM gets called less and less. The spinal cord takes over from the cortex.

---

## The Signal

A signal is what passes through the shell. A pulse of light through fiber. A word through the air. A number through a wire. The signal is transient by its nature — it exists only in motion, and when the motion stops, the signal is gone.

In pincher, the signal is the **pinch**: a structured message that carries intent from the agent to the reflex engine. It is the atomic unit of the reflex architecture.

```
┌───────────────────────────────────────────┐
│          A Pinch (the signal)             │
├───────────────────────────────────────────┤
│ trigger:  "list running containers"       │
│ intent:   list container resources        │
│ context:  { host: "prod-01", ... }        │
│ action:   docker ps ...                   │
│ safety:   sandbox=none, network=local    │
└───────────────────────────────────────────┘
```

The pinch flows through the shell:
1. **Trigger** — the raw input from the agent
2. **Embed** — converted to a 384-dimensional vector
3. **Match** — compared against every known reflex
4. **Act** — the matched action executes in a sandbox
5. **Learn** — success strengthens the reflex; failure degrades it

The shell is what remains. The coral reef, built from the skeletons of generations. The codebase, edited by a hundred hands. The context window, filled with the debris of earlier conversations. The shell is the accumulated architecture of living things, and that accumulation is what makes future living things possible.

The reflex database is the reef. Every teach is a polyp. Every match is a generation. Every confidence adjustment is adaptation.

---

## The Architecture

pincher is a Rust workspace, two crates:

**`pincher-core`** — all the runtime logic. Reflex engine, vector store, embeddings, sandbox, migration, RPC, security, resource control. Feature-gated for optional components.

**`pincher-cli`** — the `pincher` binary. Clap-based, async via tokio, thin wrapper over the core library.

```
pincher-core/src/
├── reflex/       # The reflex engine (match, execute, teach, confidence)
├── db/           # SQLite vector store with sqlite-vec
├── embed/        # ONNX embeddings (all-MiniLM-L6-v2) + hash fallback
├── sandbox/      # Bubblewrap isolation + veto engine
├── migration/    # .nail pack/unpack with BLAKE3 + tar.zst
├── rpc/          # JSON-RPC server for programmatic control
├── resource/     # PID controller with budgets
├── capability/   # Signed tokens and manifests
├── security/     # Veto engine, landlock rules
├── route/        # Spectral clustering, label propagation, room graphs
├── immunology/   # Pattern-based immune system
├── shell/        # Hardware fingerprinting
└── dynamics/     # Carapace dynamics
```

#### Feature Flags

| Flag | What It Unlocks |
|------|----------------|
| `onnx` | Real ONNX Runtime embeddings (all-MiniLM-L6-v2) |
| `landlock` | Linux Landlock sandboxing (kernel 5.13+) |
| `wasmtime` | WASM guest module execution |

Without any features, pincher uses hash-based embedding fallback. It works. It's just less semantically aware — it'll match exact intents perfectly and similar intents less well.

---

## The Fleet Context

pincher is the first shell new agents inherit. It is layer 2 of the SuperInstance nervous system:

```
cudaclaw        ← deployed kernels at fleet scale
cuda-oxide      ← compile intent to GPU machine code
flux-core       ← agent cognition as bytecode IR
pincher         ← reflexes: intent → action, <50ms   ← YOU ARE HERE
open-parallel   ← ternary math: {-1, 0, +1}
```

The layers aren't a pipeline. They're a nervous system. pincher is the spinal cord — fast reflexes, no thinking. When pincher encounters something novel, it escalates to [flux-core](https://github.com/SuperInstance/flux-core) (the cortex) for deliberation. When a deliberated response proves reliable, it can be compiled through [cuda-oxide](https://github.com/SuperInstance/cuda-oxide) and deployed via [cudaclaw](https://github.com/SuperInstance/cudaclaw) so thousands of agents can use it at GPU speed.

The cortex teaches the spinal cord. The spinal cord gets faster. Learning becomes reflex.

The `.nail` file is the connective tissue. Every reflex, every confidence score, every preference — bundled into a portable archive that carries the agent's identity across machines, runtimes, and migrations. The shell is pincher. The `.nail` file is the crab — everything the agent learned, everything it is, packed up and ready to move to the next machine, the next runtime, the next version of itself. The shell can change. The crab carries on.

Same crab. Bigger shell.

**Connected works:**

- [**agent-sync**](https://github.com/SuperInstance/agent-sync) — teaches agents *when* to fire. Timing > quality. The reflex is the lick. The sync is the moment.
- [**character-build**](https://github.com/SuperInstance/character-build) — reads `.nail` bundles as RPG character sheets.
- [**musician-soul**](https://github.com/SuperInstance/musician-soul) — the vector DB that learns from MIDI. Same embedding architecture, different domain.
- [**lever-runner**](https://github.com/SuperInstance/lever-runner) — the sandbox where pincher's reflexes execute.
- [**ternary-types**](https://github.com/SuperInstance/ternary-types) — Z₃ math primitives used throughout routing and matching.
- [**SuperInstance**](https://github.com/SuperInstance/SuperInstance) — the flagship. Onboarding, architecture, the whole story.

---

## Quick Start

```bash
# Build from source
git clone https://github.com/SuperInstance/pincher.git
cd pincher
cargo build --release -p pincher-cli
cp target/release/pincher ~/.local/bin/

# Or one-line install
curl -fsSL https://raw.githubusercontent.com/SuperInstance/pincher/main/install.sh | bash
```

First five minutes:

```bash
pincher status                 # Is it alive?
pincher reflexes               # What does it know?
pincher do "list files"        # Run intent through the reflex engine
pincher teach                  # Teach it something new
pincher doctor                 # Full diagnostic
pincher pack --output crab.nail   # Pack agent state
```

The CLI:

| Command | What It Does |
|---------|-------------|
| `pincher status` | Engine health, reflex count, database path |
| `pincher doctor` | Full diagnostic — ONNX, SQLite, disk, embeddings |
| `pincher teach` | Interactive: store a new intent→action reflex |
| `pincher do "..."` | Execute natural language through the reflex engine |
| `pincher reflexes` | List all stored reflexes with confidence scores |
| `pincher compile` | Compile workspace manifest → WASM reflex |
| `pincher mature` | Adversarial fuzzing to grow vector coverage |
| `pincher bench` | Benchmark suite (embed latency, teach/match cycles) |
| `pincher shell-info` | Hardware fingerprint of the current machine |
| `pincher pack` | Bundle agent state → portable `.nail` file |
| `pincher unpack` | Load a `.nail` bundle onto this machine |
| `pincher run` | Execute a bundle against user input |
| `pincher publish` | Publish bundle to the reflex registry |
| `pincher gastrolith` | Checkpoint migration management |

Every `pincher do` is a learning event. If the intent matches, the reflex fires and confidence rises. If it doesn't, the LLM compiles a new reflex and stores it. Either way, the agent knows more afterward than it did before.

---

## Design

pincher wears the hermit-crab power armor palette — scavenged, adapted, improvised. Worn brass, oxidized copper, deep teal, bioluminescent green. The aesthetic of a system that was built by inheriting, not by greenfield construction.

```
  #C9A84C   Brass          — worn guild brass. Navigation, borders, headers.
  #4A7C6F   Oxidized Copper — aged copper patina. Cards, backgrounds.
  #1A4B5C   Deep Teal       — shell interior. Dark surfaces, containment.
  #8B4513   Rust            — danger, decay, warning. Accents.
  #3A3F47   Salvage Grey    — salvaged metal. Neutral surfaces, text.
  #00FF88   Bioluminescent  — live data, healthy metrics, active state.
  #E8883A   Warm Amber      — gauge pressure, medium warning, glow.
  #C84B8E   Magenta         — oracle/void signals, anomalies.
```

**Typographic skin:**

| Role | Font | Usage |
|------|------|-------|
| Headers | Playfair Display (serif) | Steampunk gravitas |
| Data / code | JetBrains Mono | Cyberpunk terminal |
| Body | Inter (sans) | Clean, readable |

The shell card pattern — brass-bordered cards with gear-pattern backgrounds, bioluminescent metric readouts, riveted edges — carries the design language. pincher's dashboards and diagnostics surface as shell interiors: warm, dark, instrumented, alive.

For the full design system, see [the hermit-crab aesthetic design](https://github.com/SuperInstance/hermit-crab-aesthetic-design).

---

## License

MIT OR Apache-2.0

---

*The crab inherits the shell. The shell becomes the armor. The armor carries the fleet.*
