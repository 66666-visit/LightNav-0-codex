#!/usr/bin/env bash
# Independent entry for the two supplemental L1-original + L3-organized groups.
#
# This wrapper only changes the mode list. It reuses the existing 88-Episode
# manifest, checkpoint, seed, memory/action configuration and evaluation
# protocol from run_l1_l3_prompt_ablation.sh, and never reruns the historical
# seven groups. With --historical-root DIR, the completed seven groups are
# loaded from DIR for the final paired comparison.
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
exec bash "$REPO_ROOT/scripts/run_l1_l3_prompt_ablation.sh" --supplemental "$@"
