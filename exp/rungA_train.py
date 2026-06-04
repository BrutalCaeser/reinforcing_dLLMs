#!/usr/bin/env python3
"""
Rung-A run #2 — mechanism demo. Thin wrapper over d1's diffu-GRPO that trains on a SMALL FIXED
Countdown subset, revisited over many epochs, so the policy can actually fit it. This is the clean
"can diffu-GRPO move reward AT ALL" test. Run #1 (full set, 1500 steps) saw each prompt once -> flat
reward; here we cap the train set and revisit it.

Identical to d1's diffu_grpo_train.main() (model 4-bit + LoRA, DiffuGRPOTrainer, countdown reward) EXCEPT
the train_set is capped to MAX_TRAIN_SAMPLES prompts (env var, default 32). Faithful otherwise — imports
d1's real trainer/reward/data modules.

Invoke (via accelerate launch), e.g.:
  MAX_TRAIN_SAMPLES=32 accelerate launch --num_processes 1 --mixed_precision bf16 \
    exp/rungA_train.py --config <d1>/slurm_scripts/train.yaml --dataset countdown --model_path ... [overrides]
"""
import os
import sys

import torch

# d1's diffu-grpo modules import each other by bare name -> put that dir on sys.path first.
D1_DIR = os.environ.get("D1_DIR", "/scratch/gupta.yashv/diffusion-rl/d1/diffu-grpo")
sys.path.insert(0, D1_DIR)

from transformers import AutoModel, AutoTokenizer, BitsAndBytesConfig  # noqa: E402
from trl import ModelConfig, TrlParser  # noqa: E402
from peft import LoraConfig  # noqa: E402

from diffu_grpo_trainer import DiffuGRPOTrainer  # noqa: E402
from diffu_grpo_config import DiffuGRPOConfig  # noqa: E402
from reward_func import countdown_reward_func  # noqa: E402
from data_utils import get_countdown_questions, set_random_seed  # noqa: E402


def main(grpo_config, model_config):
    set_random_seed(grpo_config.seed)
    assert grpo_config.dataset == "countdown", "this wrapper is countdown-only"

    n = int(os.environ.get("MAX_TRAIN_SAMPLES", "32"))
    dataset = get_countdown_questions("train").shuffle(seed=grpo_config.seed)
    train_set = dataset.select(range(n))  # SMALL FIXED subset, revisited each epoch
    print(f"[rungA_train] FIXED Countdown subset = {n} prompts (revisited over epochs). "
          f"targets/numbers e.g.: {[ (train_set[i]['target'], train_set[i]['numbers']) for i in range(min(3,n)) ]}")

    reward_functions = [countdown_reward_func]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bnb = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16,
    )
    model = AutoModel.from_pretrained(
        grpo_config.model_path, trust_remote_code=True,
        torch_dtype=torch.bfloat16, quantization_config=bnb,
    ).to(device)
    tok = AutoTokenizer.from_pretrained(grpo_config.model_path, trust_remote_code=True)
    tok.pad_token = tok.eos_token
    model.config.use_cache = False

    peft_config = LoraConfig(
        r=model_config.lora_r, lora_alpha=model_config.lora_alpha,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj", "gate_proj"],
        task_type="CAUSAL_LM", lora_dropout=model_config.lora_dropout,
    )
    trainer = DiffuGRPOTrainer(
        args=grpo_config, model=model, peft_config=peft_config,
        reward_funcs=reward_functions, train_dataset=train_set,
    )
    # resume_from_checkpoint is typed Optional[str] -> "False"/"None" arrive as truthy strings; normalize.
    rc = grpo_config.resume_from_checkpoint
    resume = None if (not rc or str(rc).lower() in ("false", "none")) else rc
    trainer.train(resume_from_checkpoint=resume)


if __name__ == "__main__":
    parser = TrlParser((DiffuGRPOConfig, ModelConfig))
    grpo_config, model_config = parser.parse_args_and_config()
    main(grpo_config, model_config)
