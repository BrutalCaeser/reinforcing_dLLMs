#!/bin/bash
# =============================================================================
# rungA.sbatch — Phase 2, Rung-A diffu-GRPO RL run on Countdown (Gate G-RL).
# RL from LLaDA-8B-Instruct (LoRA r128 + 4-bit), NfePareto-cut rollout (diffusion_steps 64).
# The headline: does held-out Countdown accuracy rise above the 21.48% baseline?
#
# Parameterized via env vars (sbatch --export=ALL,VAR=val ...). Defaults = a short
# CALIBRATION run (MAXSTEPS=12) to get a real A100 per-step time + early reward signal
# before committing to long 8h jobs. Full run: override MAXSTEPS (e.g. 150) + RESUME=True
# on resubmits. Checkpoint-resume built in (save_steps; resume_from_checkpoint).
#
# GPU ALLOCATION (smart scheduling): requests `gpu:1` (ANY type) so SLURM backfills onto the
#   FIRST free GPU. a100/h200 are usually 100% allocated -> pinning them queues ~1 day; `gpu:1`
#   lands on whatever frees first (often h200/a100/v100) within the hour (verify with
#   `sbatch --test-only ...`). Pin a fast type ONLY when `sinfo` shows it idle (e.g. add
#   `--gres=gpu:h200:1`). For runs >8h, `--partition=courses-gpu` (1-day limit, idle p100s, 16GB).
# Submit (calibration):  sbatch --time=01:00:00 --export=ALL,MAXSTEPS=12 exp/rungA.sbatch
# Submit (full run):     sbatch --export=ALL,MAXSTEPS=150,SAVESTEPS=25,RUN=rungA_cd_full exp/rungA.sbatch
# Resume:                sbatch --export=ALL,MAXSTEPS=150,RESUME=True,RUN=rungA_cd_full exp/rungA.sbatch
# =============================================================================
#SBATCH --job-name=rungA
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100-sxm2:1
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --output=/scratch/gupta.yashv/diffusion-rl/logs/rungA_%j.log

set -uo pipefail
P=/scratch/gupta.yashv/diffusion-rl   # cluster scratch path kept (conda-env-prefix safety)
D1=$P/d1
export HF_HOME=$P/hf
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export WANDB_MODE=offline
MODEL=${MODEL:-GSAI-ML/LLaDA-8B-Instruct}

# --- knobs (Rung-A defaults; calibration unless overridden) ---
G=${G:-4}                       # num_generations (group size); d1=6
DIFFSTEPS=${DIFFSTEPS:-64}      # rollout NFE (NfePareto-cut from d1's 128)
MAXCOMP=${MAXCOMP:-128}         # max completion length: 128 fits a 32GB V100 AND matches eval gen_length
                                #   (d1 used 256 on 8xA100; 256 OOMs a V100 -> we fit the common GPU)
PERDEV=${PERDEV:-4}             # per_device_train_batch_size (=G -> 1 prompt/step)
NUMITER=${NUMITER:-6}           # mu, GRPO inner updates per generation; d1=12
# NOTE: LLaDAModelLM does NOT support gradient_checkpointing (verified) -> rely on max_completion 128
#       fitting a 32GB V100; if it OOMs, drop MAXCOMP to 96. Do NOT pass --gradient_checkpointing.
MAXSTEPS=${MAXSTEPS:-12}        # optimizer steps; calibration=12, full run >=150
SAVESTEPS=${SAVESTEPS:-24}      # divisible by NUMITER (6) per d1's resume note
RESUME=${RESUME:-False}
RUN=${RUN:-rungA_cd_calib}
OUT=$P/results/$RUN
mkdir -p "$P/logs" "$OUT"

echo "node:$(hostname) date:$(date) | G=$G diffsteps=$DIFFSTEPS maxcomp=$MAXCOMP perdev=$PERDEV numiter=$NUMITER maxsteps=$MAXSTEPS resume=$RESUME run=$RUN"
module load anaconda3/2024.06
module load cuda/12.8.0
export CUDA_HOME=${CUDA_HOME:-/shared/EL9/explorer/cuda/12.8.0}
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$P/envs/d1"
echo "CUDA_HOME=$CUDA_HOME"; python -c "import torch;print('gpu',torch.cuda.get_device_name(0))"

cd "$D1/diffu-grpo"
START=$(date +%s)
accelerate launch --num_processes 1 --num_machines 1 --mixed_precision bf16 --dynamo_backend no \
    diffu_grpo_train.py \
    --config slurm_scripts/train.yaml \
    --model_path "$MODEL" --dataset countdown \
    --num_generations "$G" --per_device_train_batch_size "$PERDEV" --generation_batch_size "$PERDEV" \
    --gradient_accumulation_steps 1 \
    --diffusion_steps "$DIFFSTEPS" --block_length 32 --max_completion_length "$MAXCOMP" --max_prompt_length 200 \
    --num_iterations "$NUMITER" --max_steps "$MAXSTEPS" \
    --learning_rate 3e-6 --beta 0.04 --epsilon 0.5 \
    --save_strategy steps --save_steps "$SAVESTEPS" --save_total_limit 3 \
    --resume_from_checkpoint "$RESUME" \
    --report_to none --logging_steps 1 \
    --run_name "$RUN" --output_dir "$OUT" 2>&1 | tail -80
rc=${PIPESTATUS[0]}
END=$(date +%s)
echo "rungA rc=$rc | wall=$((END-START))s for max_steps=$MAXSTEPS (=> ~$(( (END-START) / (MAXSTEPS>0?MAXSTEPS:1) ))s/step incl. amortized generation)"
echo "checkpoints:"; ls -1 "$OUT" 2>/dev/null | grep -i checkpoint | tail -5
echo "DONE rungA: $(date)"
