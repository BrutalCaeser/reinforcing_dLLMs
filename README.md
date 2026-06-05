# reinforcing_dLLMs — a single-GPU reproduction & study of diffu-GRPO (RL for diffusion LLMs)

> _Repo `github.com/BrutalCaeser/reinforcing_dLLMs` (renamed from `diffusion-rl`, 2026-06-04)._

**What this is:** an independent, **compute-constrained (single-GPU) reproduction and study** of `d1`'s
**diffu-GRPO** — GRPO reinforcement learning applied to a *masked diffusion* language model
(LLaDA-8B-Instruct) on a verifiable reasoning task (Countdown). What the project actually digs into is the
method's **hard part — estimating a diffusion LM's sequence log-probability** — plus an honest
characterization of *what makes diffu-GRPO train at all* on a tight budget.

**What this is NOT** (please read before citing):
- **Not original method work.** diffu-GRPO is d1 (Zhao et al., 2025, arXiv:2504.12216). All method credit is theirs.
- **Not a full reproduction.** A faithful d1 run is ≈24 GPU-days; this runs on **one GPU in 8-hour jobs**, so
  it reproduces the *mechanism* at small scale — not d1's headline accuracy numbers.
- **Not a novel "block-diffusion RL" project.** An earlier version of this plan billed a block-diffusion port
  as novel. **It is not:** as of 2026, RL on block / semi-autoregressive diffusion is an active, crowded
  subfield (e.g. TraceRL→TraDo arXiv:2509.06949; MMaDA/UniGRPO arXiv:2505.15809; StableDRL arXiv:2603.06743).
  That direction is **dropped** — see [Direction](#direction).
- **Not a claim about any commercial system's internal recipe.**

Every number here is produced by a committed script with its command in `LOG.md`; negative/surprising
results are reported honestly in `FINDINGS.md`.

---

## TL;DR — the idea, and the one hard part

RL for an autoregressive LLM is routine: GRPO/PPO need `log π(completion | prompt)`, which an AR model gives
exactly and for free (chain rule). A **masked diffusion LM has no left-to-right factorization** — its
sequence log-probability is intractable, only an ELBO bound that costs many forward passes. d1's key move is
a **one-step estimator**: mask the whole completion, run **one** forward pass, sum the per-token
cross-entropy against the true tokens. It's biased (the hardest, fully-masked slice of the ELBO) but cheap
and differentiable, so policy gradients flow. **Everything here orbits that estimator — validating it, then
training with it.** (Full derivations: `mathematics.md`.)

**diffu-GRPO** = standard GRPO on a diffusion policy: critic-free; **group-relative advantage that is
_mean-centered only_, `Â_i = r_i − mean(r)` ("Dr.GRPO"-style, no `/std` — verified in d1's source, not the
paper prose)**; clipped surrogate + β·KL; with the one-step log-prob estimator and diffusion (LLaDA-style)
sampling for rollouts. Rewards are **rule-based and verifiable** (no learned reward model → no reward
hacking). Base model: **LLaDA-8B-Instruct + LoRA + 4-bit**.

**Why GRPO (not PPO/DPO):** verifiable-reward + on-policy exploration rules out DPO (offline, needs
preference pairs); a value function that's ill-defined for parallel denoising + limited compute rules out
PPO's critic. GRPO needs neither.

## Status & results (all measured)

| Gate | What | Result |
|---|---|---|
| **G-go** | compute go/no-go | full repro ≈ ~24 GPU-days `[ESTIMATE]` → **NO-GO**; single-GPU mechanism demo → **GO** |
| **G0-RL** | env + LLaDA loads/generates + **baseline** | ✅ Countdown baseline **21.48%** (256 ex, NFE 64, sdpa) ≈ d1's shipped **20.70%**. *The number to beat.* |
| **G1-RL** | estimator + reward tests + tiny smoke | ✅ estimator **ranking-faithful vs a brute-force ELBO** (matched-seed noise cancels, corr 1.0000); rewards **19/19**; tiny RL loop runs (no OOM, finite grads) |
| **G-RL** | does RL lift **accuracy**? | 🟡 **mechanism reproduced, modestly.** A 32-prompt RL run lifts **held-out** Countdown to **25.78% (+4.3 pp)** — but that's only **~1.6 SE on n=256, so suggestive, not yet significant; needs a multi-seed replication.** A full-set single-pass run stayed flat. |

**Key empirical findings** (full analysis in `FINDINGS.md`):
- **Reward moves via _repetition_, not single-pass volume.** Full training set seen once → flat ~0.30. The
  *same* model on **32 fixed prompts revisited ~12×** → train-reward 0.25→~0.40, and the adapter generalized
  **+4.3 pp held-out** (needs seeds). This mirrors d1's actual regime (≈240k prompts × **10 epochs**).
- **The zero-advantage trap is the practical wall.** If all `G` rollouts for a prompt score equally, the
  advantage is 0 → zero gradient. Observed on ~half of rounds for hard prompts; it caps small-set runs (and
  `Pr[trap] = pᴳ + (1−p)ᴳ`, so larger `G` helps — see `mathematics.md`).
- **Why a biased one-step estimator still trains:** the masking noise is **common-mode and cancels** in the
  matched-seed importance ratio `π_θ/π_old` (measured correlation **1.0000**).

## Direction

- **Rung A — mechanism demo** — *done; a modest, trap-capped lift (above).*
- **Rung B — faithful reproduction** — compute-bound (~24 GPU-days) → **out of this budget.**
- ~~**Rung C — block-diffusion port**~~ — **dropped: not novel** (active 2026 subfield; see above).
- **Where this goes now (honest, feasible on one GPU):** (a) an **apples-to-apples comparison of the
  competing log-prob estimators** — the field has 6+ (one-step / ELBO / AGRPO / wd1 / SPG / GDPO / ESPO)
  with no consensus; and (b) **minimal-compute** RL — how cheaply can a dLLM reasoning lift be obtained.
  The estimator frontier is laid out in `mathematics.md`.

## Reproduce

Jobs target the Northeastern *Explorer* SLURM cluster (`gpu` / `gpu-short`); paths are in the scripts.
Upstream d1 + LLaDA are **not vendored** — pinned in `UPSTREAM.md`, cloned/downloaded on the cluster.

```bash
sbatch env/build_d1_env.sbatch          # 1. conda env on a COMPUTE node (sdpa-first; no flash-attn)
sbatch env/prefetch_llada.sbatch        # 2. fetch LLaDA-8B-Instruct (+ env/prefetch_countdown.sbatch for data)
sbatch exp/baseline_countdown.sbatch    # 3. G0-RL: no-RL Countdown accuracy = the number to beat (21.48%)
sbatch exp/phase1_gate.sbatch           # 4. G1-RL: estimator-vs-ELBO + reward unit tests
python src/test_rewards.py              #    (reward tests are pure Python; also run locally)
sbatch exp/phase1_tiny_rl.sbatch        # 5. tiny end-to-end RL smoke (d1 trainer, shrunk)
sbatch --gres=gpu:h200:1 exp/rungA_subset.sbatch   # 6. the RL run (fixed-subset mechanism demo)
# 7. eval a trained adapter on held-out Countdown vs the 21.48% baseline:
sbatch --gres=gpu:v100-sxm2:1 --export=ALL,CKPT=<adapter_dir> exp/eval_adapter_countdown.sbatch
```

## Repo map

```
SPEC.md          recon-grounded execution spec + the compute go/no-go decision
LOG.md           engineering log — what ran, where, result, decision (newest on top)
FINDINGS.md      living results & analysis — every gate's numbers + honest interpretation
mathematics.md   complete math: ELBO, RL→PPO→GRPO, the estimator + frontier, worked examples
theory.md        the same in plain language — diffusion LMs, RL, diffu-GRPO, the estimator
UPSTREAM.md      pinned d1 commit + LLaDA base (not vendored) + env versions
DOCS.md          how this repo is documented & kept current (the maintenance discipline)
env/   build_d1_env.sbatch · prefetch_llada.sbatch · prefetch_countdown.sbatch
exp/   baseline_countdown.sbatch   G0-RL baseline (d1's eval/)
       phase1_gate.sbatch          G1-RL: estimator-vs-ELBO + reward tests
       phase1_tiny_rl.sbatch       tiny end-to-end RL smoke
       rungA.sbatch                RL on the full Countdown set (single-pass)
       rungA_subset.sbatch         RL on a small FIXED subset, revisited (mechanism demo)
       rungA_train.py              thin wrapper over d1's trainer (caps train set + explicit resume)
       eval_adapter_countdown.sbatch   eval a LoRA adapter on held-out Countdown vs baseline
src/   test_rewards.py             reward unit tests vs d1's real reward_func (countdown path)
       elbo_vs_onestep.py          one-step log-prob estimator vs brute-force MDM/ELBO
analysis/ reward_trend.py          reward trajectory from a checkpoint's trainer_state.json
results/                           metrics/JSON the scripts emit (large artifacts gitignored)
```

## What "verified" means here

- Every metric comes from a committed script with the command logged in `LOG.md`.
- The log-prob estimator and reward functions are **unit-tested before any training run** — the estimator
  against a brute-force ELBO, the rewards against d1's actual `reward_func.py`.
- Compute is marked `[ESTIMATE]`; statistical caveats are explicit (e.g. the +4.3 pp held-out lift is only
  ~1.6 SE on 256 examples and is labeled *needs seed-confirmation*, not asserted as significant).
- An honest correction trail is kept on purpose (e.g. `FINDINGS.md` records that two early estimator "gates"
  were *mis-specified yardsticks*, not estimator failures, and a mid-run "negative" that later turned positive).

## References

- **d1** — Zhao, Gupta, Zheng, Grover, *"d1: Scaling Reasoning in Diffusion LLMs via RL,"* arXiv:2504.12216 · [code](https://github.com/dllm-reasoning/d1) (pinned in `UPSTREAM.md`)
- **LLaDA** — Nie et al., *"Large Language Diffusion Models,"* 2025 (`GSAI-ML/LLaDA-8B-Instruct`)
- **MDLM** — Sahoo et al., *"Simple and Effective Masked Diffusion Language Models,"* 2024 (the ELBO we test against)
- **GRPO** — Shao et al., *"DeepSeekMath,"* 2024
- The broader RL-for-diffusion-LLM landscape (estimators, block-diffusion RL) is surveyed with arXiv IDs in `mathematics.md`.

_Author: Yashvardhan Gupta. A reproduction for research/learning — all method credit to the d1 authors._
