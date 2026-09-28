# Glyph Reflexes — hosting the chiaroscuro middle-agent in pincher

*Status: proposal. No code in this repo implements any of this yet. Every
factual claim about the glyph medium cites the feed-lab receipts
(chiaroscuro exp001–exp003) or the fleet canon essays; everything forward
is labeled a wager. The fleet doctrine applies: the crossing is always a
rate — claims carry numbers or they are rumors.*

---

## 1. The assignment, in pincher's own grammar

The fleet has been studying a camera feed made of characters
([chiaroscuro](https://github.com/SuperInstance/chiaroscuro)). The canon
essays (*Four Things the Pixels Don't Have*, *Tiny Models of Motion,
Written in Characters*) argue the character stream is a token array with
four advantages the pixel stream surrendered: shape, frame rate,
cross-modal trainability, and space. Between the glyph world and the
vision-model world, the canon says, sits a middle agent — a translator a
prior scout draft called "the pincher."

This proposal's thesis: **that middle agent should literally be pincher.**
Not a new repo. Not a new architecture. The reflex engine this repository
already is — vector database as runtime, LLM as compiler, <50ms at zero
marginal cost — is precisely the decoupled-adapter shape the canon asks
for, wearing perception instead of intent.

## 2. What a glyph reflex is

A reflex in this repo today is: intent embedding → known pattern → direct
response, with LLM escalation for the unknown. A glyph reflex is the same
triple with a different embedding source:

- **Intent source:** a glyph frame, a glyph-space scene packet
  (glyphspace's regions/relations/materials), or a glyphcast sidecar
  event (birth, death, flip, crossing). Each embeds into the same
  384-dimensional reflex space this engine already uses.
- **Response pattern:** the scene reading — orientations, discrete
  structures, object claims, event announcements — the thing the feed-lab
  measured at 0.58 top accuracy for a general reader (wireframe engine,
  exp001). Known scenes fire their reading directly, in reflex time.
- **Escalation:** the unknown frame (<0.55) goes to the LLM layer — here,
  a vision model — which reads it, *and compiles a new reflex so the next
  similar frame never escalates again.* That is the pincher doctrine
  applied to seeing: the cortex teaches the spinal cord, the spinal cord
  gets faster, the shell protects the signal.

The frame-rate advantage the canon claims for the glyph medium (178×
fewer samples per frame than the pixel stream) is not just a bandwidth
property — it is a *latency budget*. Perception at stream rate is exactly
what a <50ms reflex engine is for. A middle agent that costs an LLM call
per frame burns the advantage on the first frame; a reflex shell spends it
on every frame after the first encounter.

## 3. Why pincher is the right shell (and not a new repo)

The canon's middle-agent doctrine asks for a "decoupled adapter, tunable
without retraining either side." Pincher's reflex database already is
that: reflexes are small weights in a vector space, retrainable without
touching the embedding producer (the glyph pipeline) or the responder
(the vision model). The three-tier confidence ladder (≥0.80 direct,
0.55–0.80 confirm, <0.55 escalate) is a built-in honesty mechanism —
every fired reflex carries its confidence, and low-confidence claims are
*routed*, not hidden. That is the arbitration protocol from the
*Field Notes* story — the dinghy refuted in eleven milliseconds — with
the veto engine as the enforcement point.

There is also a sibling precedent: **quilt-pincher** already carries this
engine's reflexes as reactive cells. A glyph reflex layer would sit
naturally beside it: cells for state, glyph reflexes for perception.

## 4. Integration surface (concrete)

| Pincher component | Glyph lane counterpart | Notes |
|---|---|---|
| Intent embedding | Grid/scene/sidecar → 384-dim (small encoder, or hashed structural features to start) | Start dumb: cell-class histograms + region layout hashes embed fine for prototype reflexes |
| Reflex database | Per-install scene readings | A harbor camera's reflexes are *its* harbor — the per-install LoRA doctrine, as data |
| LLM compiler | Vision model (open-weights first) | Reads escalated frames; compiles reflexes; receipts pin model version + engine pin |
| Veto engine | Claim arbitration | Two-sided claims (glyph vs pixel) that disagree below confidence get vetoed to escalation |
| Confirmation tier | Human/agent confirm | 0.55–0.80 scene claims ask before acting — the honest-rate doctrine made interactive |

## 5. What would be measured (receipts or it didn't happen)

- **Reflex hit-rate on the mock scene** — known-frame readings fired
  directly vs escalated, per engine preset. This is the crossing rate of
  the shell itself.
- **Compilation curve** — escalations per 1000 frames over a stream
  session; the pincher-shaped claim is it decays as the shell learns the
  install.
- **FAIL-first pins** — seeded false crossings (the feed-lab's
  hallucinated-circle case is the canonical seed) must route to
  escalation, never fire as direct reflexes.
- **Latency receipt** — end-to-end frame→reading time, measured, on the
  target hardware.

## 6. First build (smallest honest)

One evening, no new infrastructure: a bench harness that replays the
feed-lab's stored mock-scene frames through pincher's existing embedding
and matching path (structural features first — no learned encoder), a
handful of seeded reflexes for the scene's recurring configurations, and
a scorer comparing reflex-fired readings against the sealed ground truth.
Receipt: hit-rate table + latency. If the hit-rate is poor, that is a
finding about where the embedding boundary lives, not a failure of the
proposal — publish it either way.

## 7. Standing under the doctrine

The freeze is the law; the crossing is always a rate. Every reflex this
layer ever fires should know its own measured rate, and the claims that
matter should keep carrying their denominators — 4/8, 5/9, 0.58 — the
way the rest of the fleet's sentences do.

The shell is occupied. The stream is arriving. Teach it to see.
