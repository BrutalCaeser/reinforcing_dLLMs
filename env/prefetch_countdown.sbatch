#!/bin/bash
# =============================================================================
# prefetch_countdown.sbatch — cache the Countdown training dataset on the
# internet-enabled `sharing` partition, so the GPU smoke/RL jobs can run offline.
# (Baseline eval used d1's local cd3 test file; *training* pulls the full HF set.)
# Submit: sbatch env/prefetch_countdown.sbatch
# =============================================================================
#SBATCH --job-name=cd-fetch
#SBATCH --partition=sharing
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=/scratch/gupta.yashv/diffusion-rl/logs/cd_fetch_%j.log

set -uo pipefail
P=/scratch/gupta.yashv/diffusion-rl   # cluster scratch path kept (conda-env-prefix safety)
export HF_HOME=$P/hf
mkdir -p "$P/logs"

echo "node:$(hostname) date:$(date)"
module load anaconda3/2024.06
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$P/envs/d1"

echo "=== download + build arrow cache for Jiayi-Pan/Countdown-Tasks-3to4 (train) ==="
python - <<'PY'
from datasets import load_dataset
ds = load_dataset("Jiayi-Pan/Countdown-Tasks-3to4", split="train")
print("rows:", len(ds), "| example:", ds[0])
PY
echo "DONE countdown prefetch: $(date)"
