#!/bin/bash
# =============================================================================
# rungA_subset.sbatch — Phase 2 Rung-A run #2: MECHANISM DEMO (Gate G-RL).
# Trains diffu-GRPO on a SMALL FIXED Countdown subset, revisited over many epochs, so the policy
# can actually learn it. Run #1 (full set, 1500 steps) saw each prompt once -> flat reward; this caps
# the train set (via exp/rungA_train.py wrapper) and revisits it. Question: does mean reward on the
# fixed set rise toward 1.0? If yes -> diffu-GRPO moves reward (mechanism proven) -> eval vs 21.48%.
#
# Changes vs run #1: small fixed subset (SUBSET=32), G 4->6 (stronger advantage), LR 3e-6->1e-5.
# Submit: sbatch --partition=gpu --gres=gpu:h200:1 --export=ALL exp/rungA_subset.sbatch
#   (h200/v100-sxm2 per --test-only availability; max_completion 128 fits 32GB)
# =============================================================================
#SBATCH --job-name=rungA2
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h200:1
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --output=/scratch/gupta.yashv/diffusion-rl/logs/rungA2_%j.log

set -uo pipefail
P=/scratch/gupta.yashv/diffusion-rl
D1=$P/d1
REPO=$P/repo
export HF_HOME=$P/hf
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export WANDB_MODE=offline
export D1_DIR=$D1/diffu-grpo
MODEL=${MODEL:-GSAI-ML/LLaDA-8B-Instruct}

# --- knobs (mechanism-demo defaults) ---
SUBSET=${SUBSET:-32}            # fixed #prompts to revisit
G=${G:-6}                       # num_generations (d1=6; stronger advantage than run #1's 4)
PERDEV=${PERDEV:-6}             # per_device_train_batch_size (=G -> 1 prompt/round)
DIFFSTEPS=${DIFFSTEPS:-64}
MAXCOMP=${MAXCOMP:-128}
NUMITER=${NUMITER:-6}
LR=${LR:-1e-5}                  # raised from d1's 3e-6 to move faster on a small set
MAXSTEPS=${MAXSTEPS:-4000}      # ~ (4000/NUMITER)/SUBSET ≈ 20 epochs over the subset
SAVESTEPS=${SAVESTEPS:-192}     # ~1 epoch (SUBSET*NUMITER); divisible by NUMITER
RESUME=${RESUME:-False}
RUN=${RUN:-rungA_cd_subset}
OUT=$P/results/$RUN
mkdir -p "$P/logs" "$OUT"

echo "node:$(hostname) date:$(date) | SUBSET=$SUBSET G=$G perdev=$PERDEV diffsteps=$DIFFSTEPS maxcomp=$MAXCOMP numiter=$NUMITER lr=$LR maxsteps=$MAXSTEPS run=$RUN"
module load anaconda3/2024.06
module load cuda/12.8.0
export CUDA_HOME=${CUDA_HOME:-/shared/EL9/explorer/cuda/12.8.0}
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$P/envs/d1"
echo "CUDA_HOME=$CUDA_HOME"; python -c "import torch;print('gpu',torch.cuda.get_device_name(0))"

cd "$REPO"
export MAX_TRAIN_SAMPLES=$SUBSET
START=$(date +%s)
accelerate launch --num_processes 1 --num_machines 1 --mixed_precision bf16 --dynamo_backend no \
    exp/rungA_train.py \
    --config "$D1/diffu-grpo/slurm_scripts/train.yaml" \
    --model_path "$MODEL" --dataset countdown \
    --num_generations "$G" --per_device_train_batch_size "$PERDEV" --generation_batch_size "$PERDEV" \
    --gradient_accumulation_steps 1 \
    --diffusion_steps "$DIFFSTEPS" --block_length 32 --max_completion_length "$MAXCOMP" --max_prompt_length 200 \
    --num_iterations "$NUMITER" --max_steps "$MAXSTEPS" \
    --learning_rate "$LR" --beta 0.04 --epsilon 0.5 \
    --save_strategy steps --save_steps "$SAVESTEPS" --save_total_limit 3 \
    --resume_from_checkpoint "$RESUME" \
    --report_to none --logging_steps 1 \
    --run_name "$RUN" --output_dir "$OUT" 2>&1 | tail -60
rc=${PIPESTATUS[0]}
END=$(date +%s)
echo "rungA2 rc=$rc | wall=$((END-START))s for max_steps=$MAXSTEPS"
echo "checkpoints:"; ls -1 "$OUT" 2>/dev/null | grep -i checkpoint | tail -4
echo "DONE rungA_subset: $(date)"
