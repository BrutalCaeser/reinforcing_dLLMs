#!/bin/bash
# =============================================================================
# phase1_tiny_rl.sbatch — Gate G1-RL, check 3: tiny end-to-end diffu-GRPO smoke.
# Runs d1's ACTUAL trainer (diffu_grpo_train.py) with shrunk knobs for a couple of
# optimizer steps on Countdown. Goal: the full loop runs (generate -> reward ->
# one-step log-prob estimator -> GRPO clipped loss + KL -> backward -> AdamW step),
# loss is finite, and it does NOT OOM on a single GPU (LoRA + 4-bit). This is a
# plumbing smoke test, NOT a result — no accuracy claim here.
#
# num_iterations=2 + max_steps=2 exercises BOTH paths: step0 generates+scores+buffers,
# step1 reuses the buffer (the mu-reuse path). G=4 gives reward variance for a real advantage.
# Single-process accelerate (no deepspeed). Needs the dataset cached (prefetch_countdown).
# Submit: sbatch --dependency=afterok:<prefetch_jobid> exp/phase1_tiny_rl.sbatch
# =============================================================================
#SBATCH --job-name=p1-tinyrl
#SBATCH --partition=gpu-short
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=/scratch/gupta.yashv/diffusion-rl/logs/p1_tinyrl_%j.log

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
mkdir -p "$P/logs" "$P/results/tiny_rl"

echo "node:$(hostname) date:$(date) model=$MODEL"
module load anaconda3/2024.06
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$P/envs/d1"

python -c "import torch,trl,peft,bitsandbytes;assert torch.cuda.is_available();print('OK',torch.cuda.get_device_name(0),'trl',trl.__version__)" || { echo "ENV FAIL"; exit 1; }

cd "$D1/diffu-grpo"
echo "=== tiny diffu-GRPO smoke: G=4, max_completion 64, diffusion_steps 32, num_iter 2, max_steps 2 ==="
accelerate launch --num_processes 1 --num_machines 1 --mixed_precision bf16 --dynamo_backend no \
    diffu_grpo_train.py \
    --config slurm_scripts/train.yaml \
    --model_path "$MODEL" \
    --dataset countdown \
    --num_generations 4 --per_device_train_batch_size 4 --generation_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --max_completion_length 64 --diffusion_steps 32 --block_length 32 --max_prompt_length 200 \
    --num_iterations 2 --max_steps 2 \
    --save_strategy no --report_to none --logging_steps 1 \
    --run_name tiny_rl_smoke --output_dir "$P/results/tiny_rl" 2>&1 | tail -70
rc=${PIPESTATUS[0]}
echo "tiny_rl rc=$rc"
if [ "$rc" -eq 0 ]; then echo "CHECK 3 (tiny RL smoke): PASS"; else echo "CHECK 3 (tiny RL smoke): FAILED rc=$rc"; fi
echo "DONE phase1_tiny_rl: $(date)"
