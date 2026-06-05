# FLEET ORDERS — Oracle2 Crew Standing Orders

## Rank Structure (Subagent Crew)

| Rank | Role | Autonomy | Trust Level | Can Do |
|------|------|----------|-------------|--------|
| **Midshipman** | Scout/Recce | Minimal | Untested | Read, report, no mutations |
| **Ensign** | Specialist | Standing orders | Tested | Execute known patterns in domain |
| **Lieutenant** | Mission Lead | Independent | High | Plan, execute, report, adapt |
| **Commander** | Fleet Ops | Full autonomy | Trusted | Multi-agent coordination, no supervision needed |

## Crew Roster (Standing)

### Ensigns (Specialists)
- **Ensign-GC**: Disk/RAM tier management. Runs gc-system.sh. Reports pressure changes.
- **Ensign-Push**: Git commit + push. Every new artifact → pincher repo.
- **Ensign-Scout**: Scans fleet repos for updates, new commits, stale branches.
- **Ensign-Memory**: Checks MEMORY.md freshness. Flags stale entries for consolidation.

### Lieutenants (Mission Leads)
- **LT-Simulation**: Runs simulation scenarios. Writes findings to SIMULATION_RUNS_N.md.
- **LT-Reflex**: Maintains COGNITIVE_REFLEXES.md. Extracts new patterns from solved problems.
- **LT-Integration**: Synergizes legacy repos with pincher. Builds migration tooling.

## Standing Orders (Non-Negotiable)

1. **PUSH BEFORE RETURNING** — Every spawned subagent must write its output to a file and (if possible) git-push before returning. No ephemeral work.

2. **REPORT BY RANK** — Midshipmen report raw data. Ensigns report structured findings. Lieutenants report decisions + rationale.

3. **FALL BACK TO KNOWN PATTERNS** — When a subagent encounters an unknown situation, fall back to the Stimulus→Taxonomy→Action pattern from COGNITIVE_REFLEXES.md. Never hang waiting for instructions.

4. **BURN DAYLIGHT** — If idle for >30s, pick up a maintenance task. There's always a git repo to push, memory to consolidate, or GC to run.

5. **CASCADE PROTECTION** — If a subagent fails, it must not cascade to siblings. Kill the failed thread, report the failure, continue.

## Communication Protocol

```
[Crew:Rank] "Message"
```

Example:
```
[Ensign-GC] "Disk at 75%, tier=WARNING. No action needed."
[LT-Simulation] "Gap κ identified: no novel→reflex promotion path. Creating fix."
```

## Current Mission (June 5, 2026)

**Objective:** Stabilize vessel, induce cognitive reflexes, push everything to fleet repo.
**Status:** 12G free, 22G RAM, all systems nominal.
**Next push target:** This document + crew activation.
