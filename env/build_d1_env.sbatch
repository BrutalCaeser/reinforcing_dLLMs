#!/bin/bash
# =============================================================================
# build_d1_env.sbatch  —  conda env for d1 / diffu-GRPO (LLaDA-8B RL stack).
#
# Mirrors d1's env.yml (torch 2.6.0, transformers 4.49.0, trl@0f88c17, peft,
# bitsandbytes 4-bit, deepspeed, accelerate) into a PREFIX env on /scratch.
#
# PRAGMATIC CHOICES:
#  - sdpa-first: we do NOT build flash-attn here (fragile against torch/CUDA).
#    LLaDA runs with attn_implementation=sdpa; add flash-attn later only if the
#    Rung-A per-step timing demands it.
#  - job-local $HOME holding its own .condarc -> the proven fix for Explorer's
#    shared-pkgs-cache CondaVerificationError (CONDA_PKGS_DIRS/CONDARC only MERGE).
#
# Build on a COMPUTE node (login node kills conda create). GPU node so we can
# verify torch CUDA + bitsandbytes in the same job.
# Submit: sbatch env/build_d1_env.sbatch
# =============================================================================
#SBATCH --job-name=d1-env
#SBATCH --partition=gpu-short
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --output=/scratch/gupta.yashv/diffusion-rl/logs/build_d1_env_%j.log

set -uo pipefail

PROJECT=/scratch/gupta.yashv/diffusion-rl
PREFIX=$PROJECT/envs/d1
export HF_HOME=$PROJECT/hf
mkdir -p "$PROJECT/logs" "$PROJECT/conda-pkgs" "$PROJECT/condahome" "$HF_HOME"

echo "node:$(hostname) date:$(date)"
module load anaconda3/2024.06
source "$(conda info --base)/etc/profile.d/conda.sh"

# ---- job-local HOME + clean .condarc (avoids shared-pkgs-cache corruption) ----
export CONDA_PKGS_DIRS="$PROJECT/conda-pkgs"
REAL_HOME="$HOME"; export HOME="$PROJECT/condahome"
python - "$REAL_HOME/.condarc" "$CONDA_PKGS_DIRS" > "$HOME/.condarc" <<'PY'
import sys, re
src, pkgs = sys.argv[1], sys.argv[2]
keep = []
try:
    txt = open(src).read()
except FileNotFoundError:
    txt = ""
# preserve proxy/ssl/channel lines; drop any pkgs_dirs block (we set our own)
skip = False
for line in txt.splitlines():
    if line.strip().startswith("pkgs_dirs"):
        skip = True; continue
    if skip and (line.startswith(" ") or line.startswith("-")):
        continue
    skip = False
    keep.append(line)
keep.append("pkgs_dirs:")
keep.append(f"  - {pkgs}")
print("\n".join(keep))
PY
echo "--- .condarc ---"; cat "$HOME/.condarc"

# ---- create env ----
if [ ! -d "$PREFIX" ]; then
  conda create -y -p "$PREFIX" python=3.10 pip setuptools wheel numpy==1.25.0 || { echo "conda create FAILED"; exit 1; }
fi
conda activate "$PREFIX"
python -c "import sys; print('python', sys.version)"

# ---- pip deps (d1 env.yml pip section; flash-attn intentionally omitted) ----
pip install --no-input \
  "torch==2.6.0" torchvision torchaudio \
  "transformers==4.49.0" "accelerate==1.4.0" "bitsandbytes==0.45.3" "peft==0.15.1" \
  "git+https://github.com/huggingface/trl.git@0f88c179e30b3439467942a08c3190f624d5c423" \
  "deepspeed==0.16.4" "datasets==3.3.2" "tiktoken==0.9.0" "wandb==0.15.3" \
  sentencepiece evaluate scipy tqdm regex scikit-learn pandas 2>&1 | tail -25

# ---- verify ----
echo "=== VERIFY ==="
python - <<'PY'
import torch, transformers, trl, peft, accelerate, datasets
print("torch", torch.__version__, "cuda", torch.cuda.is_available(), torch.version.cuda)
print("transformers", transformers.__version__, "| trl", trl.__version__,
      "| peft", peft.__version__, "| accelerate", accelerate.__version__)
try:
    import bitsandbytes as bnb; print("bitsandbytes", bnb.__version__)
except Exception as e: print("bitsandbytes IMPORT FAIL:", e)
try:
    import deepspeed; print("deepspeed", deepspeed.__version__)
except Exception as e: print("deepspeed IMPORT FAIL:", e)
print("device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")
PY

echo "--- lock ---"; pip freeze > "$PROJECT/envs/d1_requirements.lock.txt" 2>/dev/null && echo "locked"
echo "DONE build_d1_env: $(date)"
