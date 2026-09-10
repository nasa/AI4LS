#!/bin/bash
# Wrapper script for TFM Battery v2-final
# Sets all environment variables and launches the battery.
# Lives on shared storage so it survives GPU-machine swaps.
#
# On a fresh machine, this script will:
#   1. Set the TABPFN_TOKEN (licensed, authenticated access)
#   2. Point TABPFN_CKPT / TABPFN_CLF_CKPT to shared weights
#   3. Cache the token to ~/.cache/tabpfn/auth_token for TabPFN's internal use
#   4. Set determinism env vars
#   5. Launch the requested battery component

set -euo pipefail

# ---- Token (licensed path, registered PriorLabs account) ----
# Set your licensed TABPFN_TOKEN before running (do NOT hardcode it here):
#   export TABPFN_TOKEN="your_token_here"
if [ -z "${TABPFN_TOKEN:-}" ]; then echo "ERROR: TABPFN_TOKEN not set. Export a licensed token first."; exit 1; fi
export TABPFN_NO_BROWSER=1

# Cache token for TabPFN's get_cached_token() fallback
mkdir -p ~/.cache/tabpfn
printf '%s' "$TABPFN_TOKEN" > ~/.cache/tabpfn/auth_token
chmod 600 ~/.cache/tabpfn/auth_token

# ---- Checkpoint paths (shared storage, persistent) ----
SHARED=/mnt/shared-workspace/tfm_battery_v2
export TABPFN_CKPT="$SHARED/weights/tabpfn-v3-regressor-v3_default.ckpt"
export TABPFN_CLF_CKPT="$SHARED/weights/tabpfn-v3-classifier-v3_default.ckpt"

# ---- Determinism ----
export CUBLAS_WORKSPACE_CONFIG=:4096:8

# ---- Library path (conda libstdc++ has GLIBCXX_3.4.32 needed by pyarrow) ----
export LD_LIBRARY_PATH="/opt/conda/lib:${LD_LIBRARY_PATH:-}"

# ---- Launch ----
BATTERY_SCRIPT="$SHARED/run_battery.py"
COMPONENT="${1:-all}"

echo "[wrapper] TABPFN_CKPT=$TABPFN_CKPT"
echo "[wrapper] TABPFN_CLF_CKPT=$TABPFN_CLF_CKPT"
echo "[wrapper] Token cached at ~/.cache/tabpfn/auth_token"
echo "[wrapper] Launching: python3 $BATTERY_SCRIPT $COMPONENT"
echo ""

cd /workspace
python3 "$BATTERY_SCRIPT" "$COMPONENT"
