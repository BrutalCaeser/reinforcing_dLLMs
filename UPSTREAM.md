# Upstream pins (NOT vendored — clean clones on the cluster, pinned here)

## d1 / diffu-GRPO (the RL recipe)
- **Repo:** https://github.com/dllm-reasoning/d1
- **Paper:** Zhao, Gupta, Zheng, Grover — "d1: Scaling Reasoning in Diffusion LLMs via RL," arXiv:2504.12216
- **Pinned commit:** `6f5abf5ca8a58c6e08bbf06d412ad260dca6dbd3` (cloned `--depth 1`, 2026-06-03)
- **Cluster clone:** `/scratch/gupta.yashv/diffusion-rl/d1` · read-only ref: `_refs/d1-ro`

## Base model
- **LLaDA-8B-Instruct** (HuggingFace `GSAI-ML/LLaDA-8B-Instruct`) — masked diffusion LM, the d1 base.
  Loaded with `trust_remote_code=True`; LoRA (peft) + 4-bit (bitsandbytes) for RL.
- (Optional SFT base `LLaDA-sft-s1k` — skipped for Rung A; RL-from-Instruct is a valid d1 setting.)

## For Rung C (novel block-diffusion port)
- **bd3lms** @ `1c3e8f4` (see block-pareto/UPSTREAM.md) — our trained block-32 or a block-diffusion base.

## Env
- d1 `env.yml`: torch 2.6.0, transformers 4.49.0, trl@`0f88c179...`, peft 0.15.1, bitsandbytes 0.45.3,
  deepspeed 0.16.4, accelerate 1.4.0, python 3.10. (flash-attn omitted in our build — sdpa first.)
