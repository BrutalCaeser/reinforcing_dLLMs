#!/bin/bash
# =============================================================================
# eval_adapter_countdown.sbatch — eval a diffu-GRPO LoRA adapter on HELD-OUT Countdown.
# Identical to baseline_countdown.sbatch (same eval.py, gen_length 128 -> diffusion_steps 64,
# block 32, 256-example countdown_cd3_test.jsonl) EXCEPT it loads a LoRA adapter via
# eval.py --checkpoint_path. This is the apples-to-apples comparison vs the 21.48% baseline:
# does the train-reward lift translate to held-out (generalizing) accuracy?
#
# eval.py loads the base in full bf16 and applies the adapter with PeftModel.from_pretrained
# (d1's own eval path; the adapter was LoRA-trained on a 4-bit base — same as d1).
# Submit: sbatch --gres=gpu:v100-sxm2:1 --export=ALL,CKPT=<adapter_dir>,RUN=<name> exp/eval_adapter_countdown.sbatch
# =============================================================================
#SBATCH --job-name=cd-eval-ad
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100-sxm2:1
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=/scratch/gupta.yashv/diffusion-rl/logs/cd_eval_adapter_%j.log

set -uo pipefail
P=/scratch/gupta.yashv/diffusion-rl
D1=$P/d1
export HF_HOME=$P/hf
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
MODEL=${MODEL:-GSAI-ML/LLaDA-8B-Instruct}
CKPT=${CKPT:?set CKPT=<adapter dir with adapter_config.json + adapter_model.safetensors>}
GEN=${GEN:-128}
BATCH=${BATCH:-8}
RUN=${RUN:-eval_run2_adapter}
OUT=$P/results/$RUN
mkdir -p "$OUT" "$P/logs"

echo "node:$(hostname) date:$(date) | model=$MODEL ckpt=$CKPT gen=$GEN batch=$BATCH run=$RUN"
module load anaconda3/2024.06
module load cuda/12.8.0
export CUDA_HOME=${CUDA_HOME:-/shared/EL9/explorer/cuda/12.8.0}
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$P/envs/d1"
python -c "import torch;print('gpu',torch.cuda.get_device_name(0))"

cd "$D1/eval"
echo "=== generate (LLaDA-8B-Instruct + LoRA adapter, countdown held-out 256) ==="
torchrun --nproc_per_node=1 --master_port=29588 eval.py \
    --dataset countdown \
    --model_path "$MODEL" \
    --checkpoint_path "$CKPT" \
    --gen_length "$GEN" \
    --block_length 32 \
    --batch_size "$BATCH" \
    --output_dir "$OUT"
rc=$?
if [ "$rc" -ne 0 ]; then echo "eval.py FAILED rc=$rc"; exit "$rc"; fi

echo "=== parse accuracy (compare vs baseline 21.48%) ==="
python -c "from parse_and_get_acc import aggregate_results; aggregate_results('$OUT')" 2>&1 | tail -25
echo "DONE eval_adapter_countdown: $(date)"
