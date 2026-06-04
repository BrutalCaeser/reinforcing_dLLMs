# SPEC — DiffuGRPO: RL Post-Training for Diffusion LLMs

**Owner:** Yashvardhan Gupta · **Started:** 2026-06-03 · **Cluster:** Northeastern Explorer
**Predecessors:** [NfePareto] (NFE axis — COMPLETE; fixes our rollout-NFE budget), [BlockPareto] (block axis).
**Scope:** independent, constrained-compute reproduction of d1's diffu-GRPO + a planned novel block-diffusion port.
The feasibility ladder (cheap-mechanism → faithful → novel) and decision gates are in §2–3 below.

> This file is the **recon-grounded execution spec** (Phase 0 output): the *verified* d1 recipe,
> the **compute go/no-go decision**, and the reduced **Rung-A** plan. The full 11-layer system design
> and phase/gate structure live in the wiki playbook; this doesn't duplicate it — it makes it runnable.

## 0. Thesis
Masked diffusion LMs improve with RL via **diffu-GRPO** (d1): GRPO with a **one-step log-prob estimator**.
We (Rung A) reproduce the *mechanism* — RL lifts reasoning accuracy on a masked dLLM — then (gated) attempt
the **novel** port to block-diffusion. The enabler: the diffusion loss is cross-entropy (no argmax) ⇒ log-prob
is differentiable ⇒ policy gradients flow.

## 1. Upstream recon — VERIFIED (read 2026-06-03, grounding rule #1)
d1 repo `github.com/dllm-reasoning/d1` @ `6f5abf5`, cloned at `/scratch/gupta.yashv/diffusion-rl/d1` (read-only ref: `_refs/d1-ro`).

**The algorithm, confirmed in `diffu-grpo/diffu_grpo_trainer.py`:**
- **Log-prob estimator** (`_get_per_token_logps` → `forward_process`): mask the input — prompt masked w/
  prob `p_mask_prompt=0.15`, **completion fully masked** — run **ONE** forward, take cross-entropy vs the
  true completion tokens = per-token log-prob. A different `mask_seed` per inner-iteration gives the
  perturbed views. **← this is the component we unit-test (vs a brute-force ELBO) before any training.**
- **GRPO loss** (`compute_loss`): `coef_1=exp(logp−old_logp)`, `coef_2=clamp(coef_1,1−ε,1+ε)`,
  `loss=−min(coef_1·A, coef_2·A) + β·KL` — standard clipped surrogate + KL (k3 estimator). Confirmed
  identical to the textbook GRPO update (see rl-for-llms worked example).
- **Rollout** (`generate`): LLaDA-style block diffusion sampling — `block_length`, `diffusion_steps`,
  `remasking=low_confidence`. **This is where the NFE budget lives** (= NfePareto bridge).

**d1's GSM8K reference config** (`slurm_scripts/train.yaml` + `accelerate_a100.yaml`):
| knob | value | | knob | value |
|---|---|---|---|---|
| GPUs / time | **8×A100, 72h**, DeepSpeed ZeRO-2 | | `num_generations` G | 6 |
| base model | LLaDA-8B-Instruct (+opt. s1k SFT) | | `diffusion_steps` (rollout NFE) | **128** |
| adapters | **LoRA r128 α64, 4-bit, FA2, bf16** | | `block_length` | 32 |
| `num_iterations` μ | 12 | | `max_completion_length` | 256 |
| β (KL) / ε (clip) | 0.04 / 0.5 | | epochs | 10 (~7.5k prompts) |
| env | torch 2.6.0, transformers 4.49.0, trl@0f88c17, peft 0.15.1, bitsandbytes 0.45.3, deepspeed 0.16.4 | | LR | 3e-6 |

Note: **8 GPUs are for rollout throughput, not capacity** — LoRA+4-bit LLaDA-8B fits one GPU.

## 2. Compute go/no-go — THE Phase-0 gate (G-go)
- **Full d1 reproduction:** 8×A100 × 72h ≈ **~576 A100-hours ≈ ~24 GPU-days**. Our cap is 8h jobs on `gpu`.
  `[ESTIMATE]` ⇒ **NO-GO** (>> the playbook's 3–4 GPU-day threshold; ~72 resubmissions).
- **Rung A (mechanism demo) — GO.** Single H200, **RL directly from LLaDA-8B-Instruct** (skip SFT), reduced:
  - `diffusion_steps` 128→**64** (NfePareto: quality saturates; correctness reward tolerates fewer steps — *the concrete payoff of doing NFE first*),
  - G 6→**4**, GSM8K (or countdown) **subset ~256 prompts, ~50–100 optimizer steps** (enough to see reward rise), `num_iterations` 12 (cheap-reuse kept).
  - `[ESTIMATE]` ~2–3 min/opt-step on one H200 (generation-dominated) → **~0.3–0.7 GPU-day** → 1–2× 8h jobs + checkpoint-resume. **Feasible.** Refine with a real per-step timing on the first run (like BlockPareto's G1 timing).
- **Rung B (reduced faithful):** 1 task, fewer epochs, full `diffusion_steps` → `[ESTIMATE]` a few GPU-days — **decide only after Rung A works.**
- **Rung C (novel block-diffusion port):** gated on B.

## 3. Phase 0 — remaining steps
- [x] Read d1 trainer/config/env/hardware (done). Clone on cluster (done).
- [ ] Build conda env `d1` on a **compute node** (login node kills conda; job-local `$HOME` for the condarc-corruption fix — see env/build_d1_env.sbatch). Flash-attn is the risk → fall back to sdpa if it won't build.
- [ ] Fetch **LLaDA-8B-Instruct** to `$HF_HOME`; confirm it loads (trust_remote_code) + a 1-prompt generation works.
- [ ] **Baseline accuracy** (no RL) on the chosen task = *the number to beat* (use d1's `eval/`).
- **Gate G0-RL:** env imports + LLaDA generates + baseline reproduced within reason → proceed to Phase 1 (log-prob estimator unit-test + tiny RL loop).

## 4. Reward (Rung A): rule-based, unit-tested
Reuse d1's `diffu-grpo/reward_func.py` (parse final answer, exact-match correctness + format). Pure functions →
unit test before training (no learned reward → no reward hacking).

## 5. Risks (Rung-A-relevant; full table in playbook §8)
| Risk | Mitigation |
|---|---|
| flash-attn won't build (torch 2.6/CUDA) | sdpa fallback; LLaDA supports it |
| env drift (trl pinned commit, torch 2.6 vs cluster CUDA 12.x) | pin exactly per env.yml; build on compute node; lock with `pip freeze` |
| 8h wall kills RL | checkpoint + resume (`resume_from_checkpoint`); proven SIGUSR1 toolkit |
| log-prob estimator wrong/high-variance | **unit-test vs brute-force ELBO on toy seqs before training** (biased per d1; wd1/AGRPO are the unbiased successors) |
| compute blowout | G-go gate above; Rung ladder; NfePareto-cut rollout NFE; LoRA+4-bit |

## 6. Grounding rules (inherited, non-negotiable)
1. No code vs an unread API (d1 + LLaDA read first). 2. Every number reproducible from a committed script + logged command.
3. `[ESTIMATE]` on guesses (esp. compute). 4. Negative results reported honestly. 5. `LOG.md` every session; small Conventional Commits.
6. Log-prob estimator + reward fns **unit-tested** before any training run.

## 7. Definition of done (Rung A)
Validated diffu-GRPO loop + a reproduced **baseline→RL accuracy lift on one task** (Gate G-RL), honestly reported.
Then: rollout-NFE ablation (RQ2, the NfePareto bridge) → stretch: block-diffusion port (RQ3).
