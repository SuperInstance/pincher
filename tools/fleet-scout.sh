#!/usr/bin/env bash
# fleet-scout.sh — Level-0 scout: check all SuperInstance fleet repos for activity
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

for repo in "${FLEET_REPOS[@]}"; do
    echo "--- $repo ---"
    # Get latest commit
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
