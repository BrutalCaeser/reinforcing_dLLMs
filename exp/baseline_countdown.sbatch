#!/bin/bash
# =============================================================================
# baseline_countdown.sbatch  —  Gate G0-RL: baseline (no-RL) Countdown accuracy
# for LLaDA-8B-Instruct = the number to beat with diffu-GRPO.
#
# Runs d1's own eval (eval/eval.py → saves generations; parse_and_get_acc.py →
# accuracy) on the 256-example cd3 test set, single GPU. Validates the env +
# model load + generation path AND yields the baseline in one job (eval.py loads
# the model early, so a broken env/flash-attn fails fast, not 30 min in).
#
# gen_length=128 ⇒ eval.py sets diffusion_steps=gen_length//2=64, block_length=32.
# Submit: sbatch exp/baseline_countdown.sbatch
# =============================================================================
#SBATCH --job-name=cd-baseline
#SBATCH --partition=gpu-short
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=/scratch/gupta.yashv/diffusion-rl/logs/cd_baseline_%j.log

set -uo pipefail
P=/scratch/gupta.yashv/diffusion-rl
D1=$P/d1
export HF_HOME=$P/hf
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
MODEL=${MODEL:-GSAI-ML/LLaDA-8B-Instruct}
GEN=${GEN:-128}
BATCH=${BATCH:-8}
OUT=$P/results/baseline_countdown
mkdir -p "$OUT" "$P/logs"

echo "node:$(hostname) date:$(date) model=$MODEL gen=$GEN batch=$BATCH"
module load anaconda3/2024.06
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$P/envs/d1"

# fail-fast env check
python -c "import torch,transformers,trl,peft;assert torch.cuda.is_available();print('OK gpu',torch.cuda.get_device_name(0),'torch',torch.__version__,'trl',trl.__version__)" || { echo "ENV CHECK FAILED"; exit 1; }

cd "$D1/eval"
echo "=== generate (LLaDA-8B-Instruct, countdown, 256 ex) ==="
torchrun --nproc_per_node=1 --master_port=29577 eval.py \
    --dataset countdown \
    --model_path "$MODEL" \
    --gen_length "$GEN" \
    --block_length 32 \
    --batch_size "$BATCH" \
    --output_dir "$OUT" 2>&1 | tail -45
rc=${PIPESTATUS[0]}
if [ "$rc" -ne 0 ]; then echo "eval.py FAILED rc=$rc"; exit "$rc"; fi

echo "=== parse accuracy ==="
python -c "from parse_and_get_acc import aggregate_results; aggregate_results('$OUT')" 2>&1 | tail -25
echo "DONE baseline_countdown: $(date)"
