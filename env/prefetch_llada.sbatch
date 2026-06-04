#!/bin/bash
# =============================================================================
# prefetch_llada.sbatch  —  pre-download LLaDA-8B-Instruct into the d1 HF cache,
# in parallel with the env build (CPU `sharing` partition has internet).
# Uses huggingface-cli (no fragile python -c quoting). Reuses the existing
# blockpareto env (has huggingface_hub) so it doesn't depend on the d1 env.
# Submit: sbatch env/prefetch_llada.sbatch
# =============================================================================
#SBATCH --job-name=llada-fetch
#SBATCH --partition=sharing
#SBATCH --time=01:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --output=/scratch/gupta.yashv/diffusion-rl/logs/llada_fetch_%j.log

set -uo pipefail
export HF_HOME=/scratch/gupta.yashv/diffusion-rl/hf
mkdir -p "$HF_HOME"
echo "node:$(hostname) date:$(date)"
module load anaconda3/2024.06
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate /scratch/gupta.yashv/block-pareto/envs/blockpareto

huggingface-cli download GSAI-ML/LLaDA-8B-Instruct --quiet
rc=$?
echo "download rc=$rc"
du -sh "$HF_HOME" 2>/dev/null
echo "DONE llada prefetch: $(date)"
