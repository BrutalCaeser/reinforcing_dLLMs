# reinforcing_dLLMs — RL post-training for diffusion LLMs (diffu-GRPO)

> _Repo `github.com/BrutalCaeser/reinforcing_dLLMs` (renamed from `diffusion-rl`, 2026-06-04)._

An **independent, constrained-compute reproduction and study of `d1`'s diffu-GRPO** — reinforcement
learning that lifts reasoning accuracy in a *masked diffusion* language model — with a planned novel
port to **block-diffusion**. Built in the open; every claim is reproducible from a committed script.

> ⚠️ **Work in progress.** This reproduces an existing method (d1, Zhao et al. 2025); it is not the
> original work and makes no claim about any commercial system's internal recipe. RL post-training of
> diffusion LLMs is an emerging, thinly-reproduced direction — this repo is one careful, honest data point.

**Sister project:** [block-diffusion-pareto](https://github.com/BrutalCaeser/block-diffusion-pareto) — the
*inference* axes (block size × denoising steps / NFE) of the same model family. Findings there (NFE can be
cut without quality loss) directly set the rollout-NFE budget used here.

---

## TL;DR — the idea, and the one hard part

RL for an autoregressive LLM is routine: GRPO/PPO need `log π(completion | prompt)`, which an AR model
gives exactly and for free. A **masked diffusion LM has no left-to-right factorization** — its
sequence log-probability is a *variational bound* (ELBO) that costs many forward passes. d1's key move is
a **one-step estimator**: mask the whole completion, run **one** forward pass, and sum the per-token
cross-entropy against the true tokens. It's biased (it's the hardest, fully-masked slice of the ELBO),
but it's cheap and differentiable, so policy gradients flow. Everything in this repo orbits that estimator
— validating it, then training with it.

**diffu-GRPO** = standard GRPO (critic-free; group-relative advantage `A_i = (r_i − mean) / std`;
clipped surrogate + β·KL) on a diffusion policy, with that one-step log-prob estimator and diffusion
(LLaDA-style block) sampling for rollouts. Rewards are **rule-based and verifiable** (no learned reward
model → no reward hacking). Base model: **LLaDA-8B-Instruct** + LoRA + 4-bit.

**Why GRPO, not PPO/DPO:** verifiable-reward reasoning + on-policy exploration rules out DPO (offline,
needs preference pairs); limited compute + a value function that's ill-defined for parallel denoising
rules out PPO's critic. GRPO needs neither. (Full primer in the project notes.)

## Status

| Gate | What | Result |
|---|---|---|
| **G-go** | compute go/no-go | full repro ≈ ~24 GPU-days `[ESTIMATE]` → **NO-GO**; **Rung A** (mechanism demo, ~0.5 GPU-day) → **GO** |
| **G0-RL** | env + LLaDA loads/generates + **baseline** | ✅ Countdown baseline **21.48%** (256 ex, NFE 64, sdpa) — matches d1's shipped **20.70%**. *The number to beat.* |
| **G1-RL** | log-prob estimator + reward unit tests + tiny RL smoke | 🟡 in progress — rewards **19/19**; estimator vs ELBO + tiny-loop (see `FINDINGS.md`) |
| **G-RL** | Rung-A RL run: baseline→RL **accuracy lift** | ⏳ next |

Numbers and the honest analysis behind each gate live in **`FINDINGS.md`** (updated as results land).

## Feasibility ladder (we never over-commit compute blindly)

- **Rung A — mechanism demo** *(current)*: RL **from LLaDA-8B-Instruct** (skip SFT) on a **Countdown**
  subset, reduced knobs (`diffusion_steps` 128→64 per the sister study, `G` 6→4, ~50–100 opt-steps),
  checkpoint-resume across 8h jobs. Goal: held-out accuracy rises over the 21.48% baseline.
- **Rung B — reduced faithful**: full `diffusion_steps`, more steps — *gated on A working*.
- **Rung C — novel**: port diffu-GRPO to **block-diffusion (bd3lm)** — *gated on B*. The publishable bit.

## Reproduce

All jobs target the Northeastern *Explorer* SLURM cluster (`gpu`/`gpu-short`); paths are in the scripts.
Upstream d1 + LLaDA are **not vendored** — pinned in `UPSTREAM.md` and cloned/downloaded on the cluster.

```bash
# 1. build the conda env on a COMPUTE node (sdpa-first; no flash-attn needed)
sbatch env/build_d1_env.sbatch
# 2. fetch LLaDA-8B-Instruct to $HF_HOME (internet partition)
sbatch env/prefetch_llada.sbatch
# 3. baseline (Gate G0-RL): the no-RL Countdown accuracy = the number to beat
sbatch exp/baseline_countdown.sbatch
# 4. Phase-1 gate (G1-RL): reward unit tests + log-prob estimator vs brute-force ELBO
sbatch exp/phase1_gate.sbatch
# reward tests are pure Python and also run locally:
python src/test_rewards.py
```

## Repo map

```
SPEC.md                  recon-grounded execution spec + the compute go/no-go decision
LOG.md                   engineering log — what ran, where, result, decision (newest on top)
FINDINGS.md              living results & analysis — every gate's numbers + honest interpretation
UPSTREAM.md              pinned d1 commit + LLaDA base (not vendored) + env versions
DOCS.md                  how this repo is documented & kept current (the maintenance discipline)
theory.md                the full theory in plain language — diffusion LMs, RL, diffu-GRPO, the estimator
env/   build_d1_env.sbatch     conda env on a compute node (job-local HOME condarc fix)
       prefetch_llada.sbatch   download LLaDA-8B-Instruct on the internet partition
exp/   baseline_countdown.sbatch   G0-RL: baseline Countdown accuracy (d1's eval/)
       phase1_gate.sbatch          G1-RL checks 1&2: estimator-vs-ELBO + reward unit tests
src/   test_rewards.py             reward unit tests vs d1's real reward_func (countdown path)
       elbo_vs_onestep.py          one-step log-prob estimator vs brute-force MDM/ELBO
results/                 metrics + JSON the scripts emit (large artifacts gitignored; summaries committed)
```

## What "verified" means here

- Every metric is produced by a committed script with the command logged in `LOG.md`.
- The log-prob estimator and reward functions are **unit-tested before any training run** — the estimator
  against a brute-force ELBO, the rewards against d1's actual `reward_func.py`.
- Compute estimates are marked `[ESTIMATE]`; negative or surprising results are reported honestly (see
  `FINDINGS.md` for the Phase-1 finding that two naive estimator gates were *mis-specified*, not failures).

## References

- **d1** — Zhao, Gupta, Zheng, Grover, *"d1: Scaling Reasoning in Diffusion LLMs via RL,"* arXiv:2504.12216 · [code](https://github.com/dllm-reasoning/d1) (pinned `6f5abf5`)
- **LLaDA** — Nie et al., *"Large Language Diffusion Models,"* 2025 (`GSAI-ML/LLaDA-8B-Instruct`)
- **MDLM** — Sahoo et al., *"Simple and Effective Masked Diffusion Language Models,"* 2024 (the ELBO we test against)
- **GRPO** — Shao et al., *"DeepSeekMath,"* 2024

_Author: Yashvardhan Gupta. Reproduction for research/learning; all credit for the method to the d1 authors._
