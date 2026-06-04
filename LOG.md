# Engineering Log — DiffuGRPO (RL for Diffusion LLMs)

Newest at top. Running engineering/devops log: what ran, where, the result, and the decision.

---

## 2026-06-04 — Phase 1: estimator + reward validation (Gate G1-RL, checks 1&2)

Goal: validate the two no-training components before any RL run — the log-prob estimator
(the part most likely to be subtly wrong) and the reward functions.

- **Check 2 — reward unit tests** (`src/test_rewards.py`): tests d1's REAL `reward_func.py`
  (countdown path), math500 import stubbed (unrelated, sympy-heavy). 19/19 PASS **locally**:
  correct eq → 1.0, valid-but-wrong → 0.1, no/garbage `<answer>` → 0, wrong/reused/missing
  numbers → 0.1; multiset rule + safe-eval + last-`<answer>` parsing confirmed. Noted spec
  quirk: `evaluate_equation` allow-list permits `**`/unary signs (never emitted by countdown).
- **Check 1 — one-step vs brute-force ELBO** (`src/elbo_vs_onestep.py`, GPU): exact mirror of
  `forward_process` + `_get_per_token_logps` (mask all completion, p_mask_prompt∈{0,0.15}, one
  forward, CE on completion slots) vs MDM/LLaDA conditional ELBO (t~U(0,1), iid mask, 1/t
  reweight, MC mean±SE). Measures Pearson(one_step,ELBO), ranking preservation (Spearman, gold
  #1), and seed-variance of the one-step estimate. Mirrors d1 countdown config: mask_id 126336,
  p_mask_prompt 0.15. PASS gates: Pearson≥0.90, gold ranked #1 by both, Spearman≥0.80,
  seed_var_ratio<0.34. Bias (ELBO − one_step) reported, expected >0 (known d1 caveat; wd1/AGRPO unbias).
- Driver: `exp/phase1_gate.sbatch` (gpu-short, 1h, HF_HUB_OFFLINE). Runs both checks, prints verdict.
- **Config grounding:** train.yaml sets `p_mask_prompt: 0.15` (dataclass default is 0.3),
  num_iterations 12, block_length 32, diffusion_steps 128, ε 0.5, β 0.04. Test uses the 0.15 value.
- Next: submit phase1_gate; then check 3 (tiny RL smoke, `exp/phase1_tiny_rl.sbatch`) → close G1-RL.

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
- **Jobs (parallel):** env build 7421414 (gpu-short), LLaDA-8B-Instruct prefetch 7421606 (sharing).

### ✅ Gate G0-RL PASSED (2026-06-04)
- **env built clean** (45 min): torch 2.6.0+cu124, transformers 4.49.0, trl 0.16.0.dev0, peft 0.15.1,
  bitsandbytes 0.45.3, deepspeed 0.16.4. LLaDA-8B-Instruct cached (15G).
- **Public repo:** https://github.com/BrutalCaeser/diffusion-rl (building in the open; git-as-spine).
- **Baseline (job 7426079, V100, 22 min):** LLaDA-8B-Instruct on Countdown cd3 (256 ex, gen_len 128, NFE 64,
  **sdpa — no flash-attn needed**) = **21.48%** (avg 110.0 effective tokens).
- **Validation vs d1's shipped baseline** (`eval/eval_baselines/`, same setting) = **20.70%** → **+0.78% match**
  (sampling noise; d1 averages seeds 1–6). Clean reproduction. (d1 baseline drops w/ length: 20.7%@128 → 19.5%@256 → 16.0%@512.)
- **The number to beat with diffu-GRPO ≈ 21%.**
- **Next (Phase 1):** unit-test d1's log-prob estimator (`_get_per_token_logps`) vs a brute-force ELBO on toy
  sequences (the component most likely to be subtly wrong) → tiny RL loop sanity → Gate G1-RL.
