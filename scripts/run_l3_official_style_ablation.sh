#!/usr/bin/env bash
# One-group follow-up: official-Prompt-Guide expression of the same L3 facts.
#
# Only --memory_prompt_mode changes (l3_official_style_only). The 88-Episode
# manifest, checkpoint, seed, Habitat config, L1/L3 weights, memory backends,
# token budget and action decoder are the same as the historical runs. The
# completed l3_original_only and l3_organized_only groups are loaded from
# --historical-root for the paired comparison and are never rerun.
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
exec bash "$REPO_ROOT/scripts/run_l1_l3_prompt_ablation.sh" --official-style "$@"
