# FINDINGS — reinforcing_dLLMs

Living results & analysis. Every gate's **numbers** and an **honest interpretation**. Updated whenever a
result lands (even partial or negative). The "why" behind the methods is in `theory.md`; the plan is in
`SPEC.md`; the chronological trail is in `LOG.md`.

**Status:** Phase 2 Rung-A — **run #2 (32 fixed prompts) shows a late reward UPTICK (0.25→0.48, ~16%→42% correct) → PRELIMINARY POSITIVE: diffu-GRPO moves reward via _repetition_.** Now **extending run #2** (job 7438917) toward saturation — the mechanism proof. Run #3 leg 1 (job 7437645) left running as a single-pass d1-faithful datapoint; its 5-leg chain **deferred** (single-pass volume = wrong lever, per run #2). See Gate G-RL below.

| Gate | Question | Verdict |
|---|---|---|
| G-go | Is a full reproduction feasible on our budget? | **NO** (~24 GPU-days); **Rung A** (~0.5 GPU-day) **GO** |
| G0-RL | Does the env work + what's the no-RL baseline? | ✅ baseline **21.48%** ≈ d1's **20.70%** |
| G1-RL (1) | Does d1's one-step log-prob estimator hold up vs the ELBO? | ✅ **ALL PASS** (ranking 1.0, bias +0.10, noise cancels) |
| G1-RL (2) | Are the reward functions correct? | ✅ **19/19** |
| G1-RL (3) | Does the full RL loop run (no OOM, finite loss, grad flows)? | ✅ **PASS** (V100, rc=0; loss≈0 w/ grad_norm 1.47; reward var real) |
| G-RL #1 | Did 1500-step run (full set, single-pass) lift reward? | ❌ **NO** — flat ~0.30 (250 one-shot prompts) |
| G-RL #2 | Does a 32-prompt fixed subset (revisited) lift reward? | ✅ **YES (late)** — 0.25→0.48 over 11 epochs (mechanism) |
| G-RL #2-ext | Does it keep climbing toward ~1.0 (clean proof)? | ⏳ **running** (job 7438917) |
| G-RL #3 | Does a d1-faithful **single-pass** full-set run lift? | ⏳ leg 1 running (job 7437645); chain deferred |

---

## Gate G-go — compute go/no-go

Full faithful `d1` = 8×A100 × 72h ≈ **~576 A100-hours ≈ ~24 GPU-days** `[ESTIMATE]` → **NO-GO** on our
8h single-GPU budget. **Rung A** (RL from `LLaDA-8B-Instruct`, `diffusion_steps` 64, `G` 4, ~256-prompt
Countdown subset, ~50–100 steps, checkpoint-resume) ≈ **~0.5 GPU-day** `[ESTIMATE]` → **GO**. Ladder:
A → reduced-faithful B → ~~novel block-diffusion C~~ **(C dropped 2026-06-04 — RL on block diffusion is an active subfield, not novel; see README)**, each gated on the previous.

---

## Gate G0-RL — baseline (the number to beat)

**LLaDA-8B-Instruct, no RL, Countdown cd3** (256 examples, `gen_length` 128 → `diffusion_steps` 64,
`block_length` 32, **sdpa** — no flash-attn needed), job 7426079:

| Metric | Value |
|---|---|
| **Countdown accuracy** | **21.48%** |
| avg effective completion length | 110.0 tokens |
| d1's own shipped baseline (same setting) | **20.70%** |
| gap | **+0.78%** (sampling noise; d1 averages seeds 1–6) → clean reproduction |

For reference, d1's baseline degrades with length (20.7% @128 → 19.5% @256 → 16.0% @512), consistent with
ours. **The number diffu-GRPO must beat ≈ 21%.**

---

## Gate G1-RL, check 2 — reward functions (19/19)

`src/test_rewards.py` against d1's real `reward_func.py` (math500 import stubbed; countdown path untouched).
Confirms the exact reward semantics we'll train on:

- correct equation (uses each number once, evaluates to target) → **1.0**
- valid expression, wrong result → **0.1**; wrong/reused/missing numbers → **0.1**
- no parseable `<answer>` → **0**
- multiset rule, safe-eval guard, and "last `<answer>` wins" parsing all verified.
- noted spec quirk (not a failure): `evaluate_equation`'s allow-list permits `**`/unary signs, which the
  Countdown task never emits.

---

## Gate G1-RL, check 1 — one-step estimator vs brute-force ELBO

`src/elbo_vs_onestep.py` on the real LLaDA-8B-Instruct (job 7426699). We compare d1's one-step estimator
(mirror of `forward_process` + `_get_per_token_logps`) to a Monte-Carlo MDM/ELBO, and measure the
properties **GRPO actually depends on** (not absolute likelihood accuracy).

### Diagnostics (reported, not gated)
- **Bias** = mean(ELBO − one-step@p0) = **+0.099** — tiny and positive: the one-step estimate slightly
  *underestimates* the ELBO, exactly as predicted (it's the hardest, fully-masked `t=1` slice).
- **Per-t curve** (per-token logp of masked slots vs mask ratio `t`): `-0.06` at t≤0.5 rising to `-0.21`
  at t=1.0 — visualizes the bias mechanism: predictions get harder as more of the answer is hidden, and
  the one-step estimator lives at the hardest end (t=1).
- **Cross-prompt Pearson(one-step, ELBO)** = 0.849. **Not gated** — GRPO never compares across prompts;
  this is over 6 near-identical easy 2-token completions (tiny dynamic range), so low correlation here is
  expected and irrelevant.

### Gated checks — the GRPO-relevant properties

**C1a — within-group ranking** (a group of completions for one prompt; advantage = ranking):

| completion | one-step | ELBO |
|---|---|---|
| gold `" Paris."` | −0.41 | −0.29 |
| wrong capital `" Berlin."` | −8.97 | −12.16 |
| off-topic `" pizza."` | −14.88 | −21.79 |
| gibberish `" qwx zzf."` | −75.69 | −45.33 |

→ **gold ranked #1 by both; Spearman = 1.000.** ✅

**C1b — corruption ladders** (gold → 25% → 50% → 100% random tokens; quality monotonically worse):

| ladder | one-step | ELBO | monotone & agree |
|---|---|---|---|
| paris | [−6.1, −42.4, −79.3, −127.1] | [−0.9, −55.4, −95.0, −115.0] | ✅ Spearman 1.0 |
| water | [−4.6, −44.4, −78.4, −149.5] | [−1.7, −45.0, −83.6, −119.3] | ✅ Spearman 1.0 |
| sun   | [−31.2, −63.4, −91.9, −154.9] | [−12.4, −55.1, −86.3, −137.6] | ✅ Spearman 1.0 |

→ both estimators decrease monotonically with corruption and agree perfectly. ✅

**C3 — common-mode cancellation** (why the high-variance estimator still trains):

| quantity | value | meaning |
|---|---|---|
| std(logp_A) across 24 mask seeds | **4.037** | the raw one-step estimate **is** high-variance |
| corr(logp_A, logp_B), *different* completions, matched seed | 0.760 | conservative: partial noise sharing |
| **corr(logp_A^old, logp_A^new), same completion, small policy step, matched seed** | **1.0000** | **GRPO's actual ratio** |
| log-ratio std / absolute std | **0.005** | the matched-seed log-ratio is ~200× quieter than the absolute estimate |

→ For the quantity GRPO actually uses — the **ratio of the same answer under π_new vs π_old at a matched
mask** — the large masking noise is **fully common-mode and cancels** (correlation 1.0). ✅ **This is the
mechanism that makes a biased, high-variance estimator safe to train with.**

### Verdict: **ALL PASS** (job 7426699, `results/phase1_estimator.json`).

### The honest correction trail (kept on purpose)
Our **first** run of this test "failed" two checks — cross-prompt Pearson (0.85) and absolute cross-seed
variance (ratio 11×). On analysis, **both were the wrong yardsticks**, not estimator defects:
- GRPO compares completions **within a group**, never across prompts → cross-prompt Pearson is irrelevant.
- GRPO uses **matched mask seeds** for π_new/π_old → the absolute cross-seed variance cancels in the ratio.

We **rewrote the gates to measure the GRPO-relevant quantities** (within-group ranking, corruption-ladder
monotonicity, same-completion matched-seed correlation) — and then everything passed, *confirming* the
estimator is fit for purpose. Lesson: **fix the yardstick, not the threshold.** We kept the failed numbers
in `LOG.md` — a reproduction that hides its missteps isn't one.

**Bottom line:** d1's one-step estimator is biased and high-variance in absolute terms, but it is
**ranking-perfect**, **low-bias**, and its variance is **neutralized by matched seeding** — exactly the
properties GRPO needs. It is validated for the Rung-A run.

---

## Gate G1-RL, check 3 — tiny end-to-end RL smoke (PASS)

`exp/phase1_tiny_rl.sbatch` ran d1's actual trainer (G=4, max_completion 64, diffusion_steps 32,
num_iterations 2, max_steps 2) on Countdown, V100-SXM2-32GB, rc=0, 3:32 wall. Took 3 tries — each caught a
real env gap the isolated tests couldn't (the RL code itself never errored):

| try | cleared | failed at | fix |
|---|---|---|---|
| v1 | — | `import wandb` → `pkg_resources` (setuptools 82 removed it) | pin `setuptools<81` (+ build script) |
| v2 | wandb | trl→`import deepspeed` → `CUDA_HOME` unset | `module load cuda/12.8.0` (major 12 == torch cu124) |
| **v3** | both | — **PASS** | — |

v3 output: `loss 0.0, grad_norm 1.47→0.52, reward 0.325, reward_std 0.45, zero_std_ratio 0, kl 0,
train_loss 2.98e-8`.
- **`loss≈0` with `grad_norm>0` is correct GRPO** — advantages are mean-centered (ΣA=0 ⇒ loss value ~0),
  but the gradient `ΣAᵢ·∇log π(oᵢ)` is non-zero and finite. The loop learns-ready, no OOM.
- **Reward variance is real** (std 0.45, zero_std_ratio 0): one of 4 completions scored ~1.0 (a correct
  answer from the *un-trained* model) → a meaningful advantage. Reward + KL wiring confirmed live.
- **Timing (V100):** ~66.5s / 2 steps, generation-dominated. A100/H200 (Rung A) ~2–4× faster; the whole
  stack fit in 32GB → ample headroom for larger batches on bigger GPUs.

**→ Gate G1-RL CLOSED.**

## Gate G-RL — Rung-A run #1 (job 7428265): NEGATIVE (no lift) — under-trained

LLaDA-8B-Instruct + LoRA r128/4-bit, diffusion_steps 64, G=4, max_completion 128, μ=6, **max_steps 1500**,
h200, rc=0, ~42 min. Loop healthy (grad finite, kl ~0.03 rising off ref, checkpoints every 24). **But the
reward did NOT rise** (`trainer_state.json`, 250 generation rounds):

| window | mean reward |
|---|---|
| first 25 rounds | 0.313 |
| rounds 100–150 | 0.296 |
| **last 25 rounds** | **0.321** |
| overall | 0.299 (max 1.0; fully-correct rounds 1/125 early vs 2/125 late) |

**Flat at ~0.30 ≈ the base model's level** (consistent with the 21% baseline). Honest read: **no learning yet.**

**Diagnosis — sizing, not (yet) method:**
1. 1500 steps = **0.1% of one epoch** → each of ~250 prompts seen **once** (×6 reuse) → ~250 one-shot
   updates on an 8B model. d1 used thousands of prompts × 10 epochs × 8 GPUs. Far too little signal.
2. Many rounds have `reward_std=0` (all G=4 completions equal) → **advantage 0 → zero gradient** → effective
   signal even sparser.

**Next — Rung-A run #2 (mechanism demo, properly sized):** train on a **small FIXED prompt subset** (≈64
prompts) for **many** rounds (revisit them, several epochs) so the policy can actually fit them — the
clean "can diffu-GRPO move reward at all" test. Needs a thin wrapper (`exp/rungA_train.py`) to cap the
train subset (d1's script hardcodes the full set). Likely also raise G→6 (better advantage) and LR→1e-5.
If reward rises on the fixed set → eval that checkpoint vs 21.48% (the real Gate G-RL). If it *still* won't
move, escalate the investigation (LR, estimator variance, KL β). Run #1 kept on record — honest negatives stay.

## Gate G-RL — Rung-A run #2 (job 7431403): PRELIMINARY POSITIVE — late reward uptick on the fixed subset

Mechanism-demo design: train a **small FIXED subset (32 Countdown prompts)** over many epochs (G=6, μ=6,
LR **1e-5**, diffusion_steps 64, max_completion 128, h200) so the policy could fit them — the cleanest
"can diffu-GRPO move reward *at all*" test. Wrapper `exp/rungA_train.py` caps the train set. **Completed
2000/2000 steps (~11 passes over the 32 prompts). Reward was flat for ~9 epochs, then rose:**

| epoch (32 rounds) | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | **10** | **11** |
|---|---|---|---|---|---|---|---|---|---|---|---|
| mean reward | .246 | .285 | .289 | .246 | .296 | .274 | .30 | .292 | .259 | **.334** | **.465** |

- **first-10 rounds 0.245 → last-10 rounds 0.478** (≈2×). Decoding (reward ∈ {0.1 valid-but-wrong, 1.0
  correct}): fraction-correct rose from **~16% → ~42%** of completions — the policy is **starting to solve
  the fixed set.**
- `frac ≥0.9` still ≈ 0: no round is ~all-6 correct yet — the gain is more rounds landing 2–3/6.

**Verdict: PRELIMINARY POSITIVE — diffu-GRPO *is* moving reward, just slowly (lift emerged only after ~10
passes).** Caveats before over-reading: (1) it's on **32 memorized prompts** → proves the *mechanism*, not
generalization; (2) small sample at the tail (epoch 11 ≈ 14 rounds).

**Honest correction (kept on record):** my mid-run read (through step 1344 / epoch 7) called this a "soft
negative" — that was **premature**; the signal showed up late. It also walks back the earlier "low-fidelity
rollouts starve the signal" diagnosis (`reward_std=0` → zero advantage): the signal *was* there, it just
needed more **passes**. (Note: run #3's live log later shows `zero_std_ratio 0.0` even at 64→128 fidelity,
so zero-advantage starvation was over-stated.)

**The decisive lesson — repetition, not single-pass volume:** run #1 (full set, each prompt once) = flat;
run #2 (32 prompts ×11) = rising. Reward moves by **revisiting** a fixed set — **exactly what d1 did**:
verified in `countdown_base.sbatch` — it inherits `train.yaml`'s **`num_train_epochs: 10`** with **no
`max_steps` override**, over a **240,632-prompt** countdown set (`filter len(nums)==3`) minus 500 eval. So d1
trains ~240k prompts **×10 epochs** (each prompt seen ~10×) on top of **μ=8** inner reuse — unambiguously a
*repetition* regime (the 72h wall may cap realized epochs <10 `[UNVERIFIED]`). → **Next: extend run #2**
(ride the win), not the fresh-prompt chain.

### Run #2 EXTENSION (job 7439596): resume → ~+21 epochs toward saturation
Resume **`checkpoint-1920`**, continue the same 32 prompts to `max_steps 6000` (LR 1e-5, save every epoch).
**Question:** does reward keep climbing toward ~1.0 (clean mechanism proof) or plateau (the model's ceiling
on those 32)? Either is a definitive result + a strong "I reproduced diffu-GRPO" outreach artifact.

**Resume gotcha (cost one job — kept on record):** the first attempt (job 7438917) resumed from
`checkpoint-2000` and **crashed** — `compute_loss` got `inputs=None` (`'NoneType' object is not
subscriptable`). trl GRPO gates "generate fresh rollouts vs. reuse buffer" on `global_step % num_iterations`;
the restored `global_step=2000` has `2000 % 6 = 2 ≠ 0` → it took the *reuse-buffer* branch, but the buffer is
empty in a fresh process → `None`. **A diffu-GRPO checkpoint is only resumable when `global_step % μ == 0`**
(d1's `train.py` warns exactly this). `checkpoint-2000` was the **forced final-save at `max_steps`**, not a
periodic save, so it's μ-misaligned; **`checkpoint-1920`** (1920 % 6 = 0) resumes cleanly. Lesson: when
resuming GRPO, pick a checkpoint at a multiple of `num_iterations`.

## Gate G-RL — Rung-A run #3 (job 7437645, leg 1 RUNNING): isolate rollout fidelity (d1-faithful slice)

Run #3 **raises the rollout to d1's exact fidelity** and runs it on the **full** Countdown set (single pass,
no revisiting) — the largest *faithful* slice one 8h H200 allows.

**Leg 1 status (live):** healthy on h200 — **no OOM at max_comp 256**; training with `reward_std ≈ 0.46,
zero_std_ratio 0.0, completion_length ≈ 227, kl ≈ 0.003, lr 3e-6 (constant)`. The d1-fidelity rollout
produces **non-zero advantages from the start** (contrast run #1's many zero-advantage rounds) — mild early
support for the fidelity angle. But it's **single-pass**, and run #2 just showed reward moves via
*repetition*, so **the planned 5-leg resume-chain is DEFERRED** and the GPU budget redirected to extending
run #2. Leg 1 is left running as a single-pass datapoint (does higher fidelity alone lift, vs run #1's flat?).

| knob | d1 (target) | run #2 (flat) | **run #3** |
|---|---|---|---|
| data | ~250k prompts ×10 ep | 32 fixed | **full set, fresh** |
| diffusion_steps (NFE) | **128** | 64 | **128** ✓ |
| max_completion | **256** | 128 | **256** ✓ |
| num_generations G | 6 | 6 | **6** ✓ |
| μ (num_iterations) | 8† | 6 | **8** ✓ |
| LR | 3e-6 | 1e-5 | **3e-6** ✓ |
| β / ε / block | 0.04 / 0.5 / 32 | same | same ✓ |
| prompts / optim-step | 16 (8 GPU × ga2) | 1 | **1** (1 GPU, ga1) |
| GPUs × wall | 8×A100 × 72h | 1×H200 × ~1.4h | **1×H200 × 8h** |
| unique prompts (8h) | ~250k | 32 | **~575 [EST]** |
| **% of d1's prompt-rollouts** | 100% | 0.013% | **~0.22% (≈1/450)** |

†d1's countdown override = μ 8 (its gsm config uses μ 12). **Run #3 matches every *per-rollout* knob to d1**;
it differs only in GPU count (1 vs 8), total volume (~1/450), and grad-accum (1 vs 2 → 1 prompt/step vs 16).
Submitted 17:33, est start ≤19:19 (likely ~18:01 when run #2 frees its h200). `RUN=rungA_cd_run3`,
save_steps 400, max_steps 6000 (the 8h wall cuts it ~5000). Uses `exp/rungA.sbatch` (d1's full-set trainer).

**Honest expectation:** ~1/450 of d1's data is still tiny. Run #3 either shows a **partial lift** (rollout
fidelity *was* the bottleneck) or **confirms the lift needs scale beyond our single-GPU budget** (the G-go
NO-GO). Both are clean, honest data points.
