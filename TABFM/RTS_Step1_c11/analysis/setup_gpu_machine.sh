#!/bin/bash
# One-shot setup for a fresh GPU machine to run the TFM battery.
# Installs torch 2.13.0+cu130 (must match original environment for the
# prerequisite gate to reproduce) and tabpfn. The wrapper script
# (run_with_env.sh) handles LD_LIBRARY_PATH and launches the battery.
#
# Usage on a fresh GPU machine:
#   bash /mnt/shared-workspace/tfm_battery_v2/setup_gpu_machine.sh
#   bash /mnt/shared-workspace/tfm_battery_v2/run_with_env.sh all
set -euo pipefail

echo "[setup] Installing torch 2.13.0+cu130..."
uv pip install torch==2.13.0 --index-url https://download.pytorch.org/whl/cu130

echo "[setup] Installing tabpfn 8.3.0 (pinned: matches the version executing on gpu-battery-2/3; gate reproduced 0.3762039 exactly under it)..."
uv pip install tabpfn==8.3.0

echo "[setup] Verifying environment..."
LD_LIBRARY_PATH=/opt/conda/lib:${LD_LIBRARY_PATH:-} python3 -c "
import torch, tabpfn
assert torch.__version__.startswith('2.13.0'), f'torch version mismatch: {torch.__version__}'
assert torch.cuda.is_available(), 'CUDA not available'
print(f'[setup] torch={torch.__version__} cuda=True gpu={torch.cuda.get_device_name(0)} tabpfn={tabpfn.__version__}')
print('[setup] OK')
"
echo "[setup] Done. Launch with: bash /mnt/shared-workspace/tfm_battery_v2/run_with_env.sh all"
