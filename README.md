# diffusion-rl — RL Post-Training for Diffusion LLMs (diffu-GRPO)

> **Can we reproduce d1's diffu-GRPO** — RL that lifts reasoning in a masked diffusion LM — **and port it
> to block-diffusion?** This is the capability-frontier sequel to [BlockPareto] (block size) and
> [NfePareto] (denoising steps), and one of the **two techniques Inception's Mercury is confirmed to use**.

**Status:** Phase 0 (recon + compute go/no-go) — in progress. Full plan: `SPEC.md`; design playbook + feasibility
ladder: `~/Documents/wiki/wiki/projects/DiffuGRPO.md`; RL fundamentals: `~/Documents/wiki/wiki/concepts/rl-for-llms.md`.

## The recipe (verified from d1 @ `6f5abf5`)
- **diffu-GRPO** = GRPO (critic-free, group-relative advantage, clipped surrogate + KL) with a **one-step
  log-prob estimator** for the non-autoregressive policy: mask the completion, one forward, cross-entropy vs
  the true tokens. Gradients flow because the diffusion loss is cross-entropy (no argmax).
- Base **LLaDA-8B-Instruct** + LoRA + 4-bit. Rewards rule-based (verifiable correctness).

## Why GRPO, not PPO/DPO
Verifiable-reward reasoning + exploration ⇒ not DPO (offline, needs preference pairs); limited compute +
a value-net ill-defined for parallel-denoising diffusion ⇒ not PPO's critic. GRPO fits both. (See `rl-for-llms`.)

## Compute go/no-go (the Phase-0 gate)
Full d1 repro ≈ ~24 GPU-days `[ESTIMATE]` → **NO-GO** here. **Rung A (mechanism demo)** — single H200,
RL-from-Instruct, NfePareto-cut rollout NFE, subset — ≈ ~0.5 GPU-day → **GO**. (Ladder: A → reduced-B → novel-C.)

## Layout
```
SPEC.md        recon-grounded execution spec + the go/no-go decision
LOG.md         engineering log (newest on top)
UPSTREAM.md    pinned d1 commit + LLaDA base (not vendored)
env/           build_d1_env.sbatch (conda env; sdpa-first; job-local HOME condarc fix)
exp/           SLURM jobs (RL runs, baselines) — TBD
src/           our diffu-GRPO glue / unit tests — TBD
eval/          accuracy eval (reuses d1/eval) — TBD
results/       metrics + curves
```

## Infra
Northeastern Explorer (`gpu` H200/A100 8h, `gpu-short` 2h). Upstream d1 + LLaDA on `/scratch`; 8h wall →
checkpoint-resume. Git-as-spine: author locally → push/rsync → cluster runs.
