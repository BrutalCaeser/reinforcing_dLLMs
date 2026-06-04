# Engineering Log — DiffuGRPO (RL for Diffusion LLMs)

Newest at top. Wiki mirror: `~/Documents/wiki/wiki/projects/DiffuGRPO.md`. RL primer: `concepts/rl-for-llms.md`.

---

## 2026-06-03 — Phase 0: recon + compute go/no-go (Gate G-go)

- **Read the d1 stack end-to-end** (grounding rule #1): `diffu_grpo_trainer.py` (the log-prob estimator
  `_get_per_token_logps` + GRPO `compute_loss`), `diffu_grpo_config.py`, `slurm_scripts/{train.yaml,gsm_base.sbatch,accelerate_a100.yaml}`, `env.yml`, `README`. Cloned upstream → `/scratch/gupta.yashv/diffusion-rl/d1` @ `6f5abf5`; read-only ref `_refs/d1-ro`.
- **Algorithm confirmed in code** (matches the rl-for-llms course): log-prob = mask completion fully (+ prompt
  w/ p=0.15) → ONE forward → CE vs true tokens. Loss = −min(ratio·A, clip(ratio,1±ε)·A) + β·KL. The estimator
  is the unit-test target (biased per d1; wd1/AGRPO unbias it).
- **d1 reference:** 8×A100, 72h, DeepSpeed ZeRO-2; LLaDA-8B-Instruct + LoRA r128 + 4-bit + FA2; G=6,
  diffusion_steps=128, block 32, max_completion 256, μ=12, β=0.04, ε=0.5, 10 epochs. (8 GPUs = rollout
  throughput, not capacity — LoRA+4-bit fits one GPU.)
- **Gate G-go DECISION:**
  - Full reproduction ≈ ~576 A100-hrs ≈ ~24 GPU-days `[ESTIMATE]` → **NO-GO** on our 8h-job budget.
  - **Rung A (mechanism demo) = GO:** single H200, RL **from LLaDA-8B-Instruct** (skip SFT), diffusion_steps
    128→64 (NfePareto-informed), G 6→4, GSM8K/countdown subset, ~50–100 opt-steps, checkpoint-resume.
    `[ESTIMATE]` ~0.3–0.7 GPU-day. Refine with a real per-step timing on the first run.
  - Rung B (reduced faithful) ≈ a few GPU-days — decide after A. Rung C (block-diffusion port) — gated on B.
- **Built scaffold:** SPEC.md (recon + G-go), env/build_d1_env.sbatch (sdpa-first; job-local HOME condarc fix),
  repo skeleton. Committed `5fa6ac2`, synced to cluster `repo/`.
- **Rung-A task = Countdown (cd3)** — d1's biggest RL lift (+26.2%); clearest mechanism signal. Verified the
  pipeline in `_refs/d1-ro`: data `{"input":"30,100,93","output":"23"}` (256 test); prompt = R1-style
  `<reasoning>…</reasoning><answer>EXPR</answer>` w/ worked example; reward (`reward_func.compute_score`):
  parse `<answer>`, **all numbers used exactly once** (`validate_equation`), safe-eval, **1.0** correct /
  **0.1** valid-but-wrong / **0** unparseable. Baseline path: `eval/eval.py --dataset countdown --gen_length 128`
  (generate steps=64, block 32 → NFE 64) → `parse_and_get_acc.py`.
- **Jobs running (parallel):** env build 7421414 (gpu-short), LLaDA-8B-Instruct (`GSAI-ML/LLaDA-8B-Instruct`,
  id confirmed) prefetch 7421483 (sharing, into d1 HF_HOME). **Next:** baseline (no-RL) Countdown accuracy when both land = Gate G0-RL.
