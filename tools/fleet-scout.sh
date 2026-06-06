#!/usr/bin/env bash
# fleet-scout.sh — Level-0 scout: check all SuperInstance fleet repos for activity
# Also reports ternary-types connectivity across the ternary math stack.
# Usage: bash tools/fleet-scout.sh

set -euo pipefail

FLEET_REPOS=(
    "SuperInstance/pincher"
    "SuperInstance/Mycelium"
    "SuperInstance/sunset-ecosystem"
    "SuperInstance/polln"
    "SuperInstance/neural-plato"
    "SuperInstance/egg"
    "SuperInstance/seed-oscillate"
    "SuperInstance/Spreader-tool"
    "SuperInstance/the-seed"
    "SuperInstance/oracle1-vessel"
)

echo "=== FLEET SCOUT — $(date -u) ==="
echo ""

# ─── Part 1: Fleet Activity ──────────────────────────────────────────────────

for repo in "${FLEET_REPOS[@]}"; do
    echo "--- $repo ---"
    data=$(gh repo view "$repo" --json nameWithOwner,updatedAt,description,diskUsage,isFork 2>/dev/null)
    if [ $? -eq 0 ]; then
        name=$(echo "$data" | jq -r '.nameWithOwner')
        updated=$(echo "$data" | jq -r '.updatedAt')
        desc=$(echo "$data" | jq -r '.description // "no description"' | cut -c1-80)
        size=$(echo "$data" | jq -r '.diskUsage')
        echo "  Updated: $updated"
        echo "  Size: ${size}KB"
        echo "  $desc"
    else
        echo "  ⚠️  Cannot reach GitHub"
    fi
    echo ""
done

# ─── Part 2: Ternary-Types Connectivity Report ───────────────────────────────

echo "=== TERNARY-TYPES CONNECTIVITY REPORT ==="
echo ""

# The canonical list of ternary math stack crates that should depend on ternary-types
TERNARY_CRATES=(
    "ternary-activation"
    "ternary-bite"
    "ternary-checkpoint"
    "ternary-conv"
    "ternary-distill"
    "ternary-em"
    "ternary-fuse"
    "ternary-hmm"
    "ternary-knn"
    "ternary-logistic"
    "ternary-loss"
    "ternary-matmul"
    "ternary-norm"
    "ternary-optimizer"
    "ternary-pool"
    "ternary-prune"
    "ternary-quantize"
    "ternary-regression"
    "ternary-svm"
    "ternary-warp"
    "ternary-spatial"
    "ternary-dynamics"
    "ternary-grad"
    "ternary-tnn"
    "ternary-llm"
    "ternary-hamiltonian"
    "ternary-noether"
    "ternary-event"
    "ternary-pack"
    "ternary-rhythm"
    "ternary-visualizer"
    "eisenstein-quantize"
    "pythagorean48"
    "deadband-snr"
)

total=${#TERNARY_CRATES[@]}
connected=0
disconnected=0
disconnected_list=""

for repo in "${TERNARY_CRATES[@]}"; do
    # Check if Cargo.toml in the repo has 'ternary-types' as a dependency
    toml_url="https://raw.githubusercontent.com/SuperInstance/$repo/master/Cargo.toml"
    if curl -sL --max-time 5 "$toml_url" 2>/dev/null | grep -q 'ternary-types'; then
        echo "  ✅ $repo — connected"
        connected=$((connected + 1))
    else
        echo "  ❌ $repo — NOT connected"
        disconnected=$((disconnected + 1))
        disconnected_list="$disconnected_list\n    - $repo"
    fi
done

echo ""
pct=$(echo "scale=1; $connected * 100 / $total" | bc)
echo "  Total crates:  $total"
echo "  Connected:     $connected  ($pct%)"
echo "  Disconnected:  $disconnected"

if [ "$disconnected" -gt 0 ]; then
    echo ""
    echo "  ⚠️  Crates needing connection:$disconnected_list"
fi

echo ""
echo "=== SCOUT COMPLETE ==="
