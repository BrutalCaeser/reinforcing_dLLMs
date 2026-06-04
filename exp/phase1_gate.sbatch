#!/bin/bash
# =============================================================================
# phase1_gate.sbatch — Gate G1-RL, checks 1 & 2 (the no-training validations).
#   Check 2: Countdown reward unit tests   (pure Python, instant)
#   Check 1: d1 one-step log-prob estimator vs brute-force ELBO on real LLaDA
# Smoke RL loop (check 3) is a separate job (exp/phase1_tiny_rl.sbatch).
# Submit: sbatch exp/phase1_gate.sbatch
# =============================================================================
#SBATCH --job-name=p1-gate
#SBATCH --partition=gpu-short
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=/scratch/gupta.yashv/diffusion-rl/logs/p1_gate_%j.log

set -uo pipefail
P=/scratch/gupta.yashv/diffusion-rl
REPO=$P/repo
export HF_HOME=$P/hf
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HUB_OFFLINE=1   # model is cached; don't hit the network on a GPU node
MODEL=${MODEL:-GSAI-ML/LLaDA-8B-Instruct}
ELBO_SAMPLES=${ELBO_SAMPLES:-512}
mkdir -p "$P/logs" "$REPO/results"

echo "node:$(hostname) date:$(date) model=$MODEL elbo_samples=$ELBO_SAMPLES"
module load anaconda3/2024.06
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$P/envs/d1"

python -c "import torch,transformers;assert torch.cuda.is_available();print('OK gpu',torch.cuda.get_device_name(0),'torch',torch.__version__)" || { echo "ENV CHECK FAILED"; exit 1; }

cd "$REPO"
echo "===================== CHECK 2: reward unit tests ====================="
python src/test_rewards.py
rc_rewards=$?
echo "reward tests rc=$rc_rewards"

echo "===================== CHECK 1: estimator vs ELBO ====================="
export MODEL
python src/elbo_vs_onestep.py --model_path "$MODEL" --elbo_samples "$ELBO_SAMPLES" \
    --out "$REPO/results/phase1_estimator.json"
rc_est=$?
echo "estimator test rc=$rc_est"

echo "=== SUMMARY ==="
echo "reward rc=$rc_rewards  estimator rc=$rc_est"
if [ "$rc_rewards" -eq 0 ] && [ "$rc_est" -eq 0 ]; then
    echo "GATE G1-RL (checks 1&2): PASS"
else
    echo "GATE G1-RL (checks 1&2): NEEDS REVIEW"
fi
echo "DONE phase1_gate: $(date)"
