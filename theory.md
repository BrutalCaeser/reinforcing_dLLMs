# Theory — reinforcing_dLLMs

The complete theory of this project, in plain language, with no steps skipped. If you read this
top to bottom you will understand **what a diffusion language model is, why reinforcement learning
is hard for one, exactly how `d1`'s diffu-GRPO solves it, and what we are reproducing and measuring** —
without needing any prior background beyond "a neural network maps inputs to outputs."

Every equation is written in plain ASCII (no rendering needed). Every term is defined on first use and
again in the **Glossary (§14)**. Anticipated questions are answered in the **FAQ (§15)**. Where this
document states a fact about `d1`, it reflects the *actual source code* we read (not the paper's idealized
description) — discrepancies are flagged explicitly.

> **⚠️ Correction (2026-06-04):** §13 ("the novel direction: RL on block diffusion") is **outdated**. A 2026
> literature sweep found RL on block / semi-autoregressive diffusion is an **active subfield**, not
> unexplored (TraceRL→TraDo arXiv:2509.06949; MMaDA/UniGRPO; StableDRL). **It is not novel; that direction is
> dropped.** The genuinely open problem is the **log-prob estimator** (`mathematics.md`). Treat
> "novel"/"unexplored" anywhere below as historical.

**Contents**
1. The big picture in one page
2. What a language model actually is
3. Two ways to build one: autoregressive vs. diffusion
4. How a masked diffusion LM is trained (the ELBO, gently)
5. Generation = iterative denoising; what "NFE" means; block diffusion
6. Why reinforcement learning for language models
7. RL from scratch: reward → policy gradient → PPO → GRPO → DPO
8. The crux: getting `log π(answer | question)` for a diffusion model
9. diffu-GRPO, fully assembled (with the real loss from the code)
10. The reward: Countdown (rule-based, verifiable)
11. What we measured, and why the first test was wrong
12. The compute story and the Rung ladder
13. The novel direction: RL on block diffusion
14. Glossary
15. FAQ
16. References

---

## 1. The big picture in one page

A **language model (LM)** is a machine that assigns probabilities to text. **Reinforcement learning (RL)
post-training** nudges an LM to produce text that scores well on some goal — e.g. "get the math answer
right" — by rewarding good outputs and discouraging bad ones.

The standard, easy case is an **autoregressive (AR)** LM (like GPT): it writes one token at a time,
left to right, and the probability it assigns to a sentence is just the product of its
one-token-at-a-time probabilities. RL methods need exactly that number — `log π(answer | question)` —
and an AR model hands it over for free.

A **diffusion LM** (like **LLaDA**, the model we use) does *not* write left to right. It starts from a
fully blanked-out answer and **fills tokens in over several rounds**, in no fixed order. This is faster
and parallel, but it breaks the easy formula: there is no clean left-to-right product, so the
probability of a sentence is only defined through a **variational bound (an ELBO)** that costs *many*
forward passes to estimate. You cannot afford that inside an RL loop that runs thousands of times.

**`d1`'s trick** (the method this repo reproduces) is a cheap **one-step estimator** of that probability:
blank the whole answer, run the network *once*, and read off how confident it is about each true token.
It is *biased* (it's the hardest possible case — predicting every token with no other answer tokens
visible), but it is cheap and differentiable, so RL gradients flow. The rest of `d1` is ordinary
**GRPO** — a popular, critic-free RL recipe — wrapped around that estimator, with **rule-based rewards**
(a calculator checks the answer, so there's nothing to "hack").

This project: (a) reproduce that the trick works (RL raises a diffusion model's reasoning accuracy), and
(b) attempt the **novel** step of applying it to **block diffusion**. Because we have a tiny compute
budget, we validate every piece in isolation before spending GPU-hours — which is why we spent real
effort *measuring the estimator itself* (§11).

---

## 2. What a language model actually is

Text is cut into **tokens** — chunks roughly the size of a word-piece (e.g. `" Paris"`, `"."`). A fixed
list of all possible tokens is the **vocabulary** (LLaDA's has ~126,000 entries). A sentence is a sequence
of token IDs, e.g. `[The, capital, of, France, is, Paris, .]`.

A language model is a function `p_θ` (θ = its trained weights) that, given some context, outputs a
**probability distribution over the vocabulary** — a list of ~126,000 numbers that sum to 1, saying "how
likely is each possible next/blank token here." The network actually emits **logits** (raw scores, any
real number); `softmax` turns logits into probabilities:

```
softmax(z)_k = exp(z_k) / sum_j exp(z_j)        # probability of token k
log p of the true token t = z_t - log sum_j exp(z_j)   # this is just -cross_entropy(z, t)
```

That last line matters a lot: **the log-probability of a token is the negative cross-entropy of the
logits against that token.** Cross-entropy is differentiable and has no "argmax" (no hard pick) in it —
so we can take gradients through a log-probability. Hold that thought; it is the reason RL on a diffusion
LM is possible at all.

---

## 3. Two ways to build one: autoregressive vs. diffusion

### 3.1 Autoregressive (AR) — the familiar kind

An AR model factorizes a sentence's probability with the **chain rule of probability**:

```
p(x_1, x_2, ..., x_L) = p(x_1) · p(x_2 | x_1) · p(x_3 | x_1,x_2) · ... · p(x_L | x_1..x_{L-1})
log p(sentence) = sum_i log p(x_i | x_1..x_{i-1})
```

It predicts each token from everything to its left. So the log-probability of any given sentence is just
a sum of per-step log-probs the model already computes. **Exact, and one forward pass.** This is why RL
for GPT-style models is straightforward.

### 3.2 Masked diffusion — what LLaDA is

A **masked diffusion model (MDM)** throws away the left-to-right rule. Picture the answer as a row of
blanks that get filled in over several rounds, in *arbitrary* order. Two processes define it:

- **Forward (noising) process** — *destroy* information. Take a clean sentence `x_0`. Pick a number
  `t` between 0 and 1 (the "noise level" / "mask ratio"). Replace **each token independently with a
  special `[MASK]` token with probability `t`**. At `t=0` nothing is masked (clean); at `t=1` everything
  is masked (a blank page). Call the result `x_t`. (This is "absorbing-state discrete diffusion": tokens
  fall into the absorbing `[MASK]` state and the only question the model is trained on is how to recover
  them.)

- **Reverse (denoising) process** — *restore* information. A neural network `p_θ(x_0 | x_t)` looks at the
  partially-masked sentence and predicts, **for every masked slot at once**, what the original token was.
  Generation runs this repeatedly, revealing a few tokens each round (§5).

The key structural difference from AR: a diffusion model predicts a masked token using **both left and
right context** that happen to be visible, and it predicts **many slots in parallel**. There is no
single ordering, so there is **no chain-rule product** — and therefore no cheap exact `log p(sentence)`.
That missing number is the entire difficulty this project is about.

---

## 4. How a masked diffusion LM is trained (the ELBO, gently)

We cannot write `log p(sentence)` exactly, but we can **lower-bound** it with a quantity we *can* compute,
called the **ELBO** (Evidence Lower BOund — "evidence" is an old name for the data's probability). Training
maximizes this bound; the better the bound, the better the model fits the data. For a masked diffusion LM
the bound has a clean, intuitive form (LLaDA, Nie et al. 2025; MDLM, Sahoo et al. 2024):

```
log p(x_0)  >=  E over t in (0,1],  E over random maskings x_t :
                  (1/t) * sum over masked positions i  of  log p_θ( x_0^i | x_t )
```

Read it in words: **"Mask the sentence at a random level `t`. Ask the model to fill the blanks. Reward it
for the log-probability it puts on the true tokens. Average this over many noise levels and many random
maskings."** Two details:

- The **sum is only over the masked positions** — the model is graded only on blanks it had to guess.
- The **`(1/t)` weight** corrects for the fact that low-`t` maskings hide few tokens (so few terms appear)
  while high-`t` maskings hide many. It makes the estimator an *unbiased* estimate of the bound.

And here is the crucial practical fact: **each term `log p_θ(x_0^i | x_t)` is just `-cross_entropy`** of
the model's logits at slot `i` against the true token. So the whole training objective is **a weighted
cross-entropy with no `argmax` anywhere.** That is what makes the model's outputs *differentiable
functions of a log-probability*, which is what lets RL gradients flow later. (Compare: a model that
*sampled* a hard token would block gradients.)

> **One-line summary of §4:** the true `log p(sentence)` is an average of "fill-in-the-blanks" cross-entropy
> scores over *all* masking levels `t`. Remember this picture — `d1`'s shortcut keeps only the `t=1` slice.

### Block diffusion (BD3-LM) — the bridge between the two worlds

Pure AR is fully sequential (slow, exact). Pure diffusion is fully parallel (fast, no exact log-prob).
**Block diffusion** sits in between: split the sequence into **blocks** of, say, 32 tokens; generate the
blocks **left-to-right like AR**, but generate **all tokens inside a block in parallel like diffusion**.
A `block_length` knob dials between the two extremes (block = 1 → AR; block = whole sequence → pure
diffusion). Our sister project [block-diffusion-pareto] found throughput peaks around block 32, and that
the denoising-step count can be cut without quality loss — both feed directly into our compute budget here.

---

## 5. Generation = iterative denoising; "NFE"; and the cost knob

To **generate** an answer, a masked diffusion model:

1. starts with the answer region **fully masked** (a blank page after the prompt),
2. runs the network → gets a predicted distribution for every blank,
3. **commits the most confident few** tokens (LLaDA/`d1` use *low-confidence remasking*: keep the
   network's most-confident guesses, leave the rest masked),
4. repeats from step 2 until no blanks remain.

Each pass through the network in step 2 is one **NFE — Number of Function Evaluations** (one "forward
pass"). If you take `T` denoising steps, you spend ~`T` NFEs. **More steps = higher quality but slower.**
This is the "speed dial" our [NfePareto] study mapped. In RL it matters twice over: generating each of the
`G` candidate answers per prompt costs NFEs, so the rollout NFE budget is the dominant compute cost
(§12). The payoff of doing NfePareto first: we know we can run rollouts at **64 steps instead of 128** with
negligible quality loss — halving the most expensive part of training.

---

## 6. Why reinforcement learning for language models

Models are usually built in stages:

- **Pretraining:** learn language by predicting masked/next tokens on huge text. Produces a fluent model
  with no particular goal.
- **Supervised fine-tuning (SFT):** show it example (question → good answer) pairs; it imitates them.
- **Reinforcement learning (RL):** instead of imitating fixed answers, **let the model produce its own
  answers, score them, and shift probability toward the ones that scored well.** This can exceed SFT
  because the model explores and improves on its *own* outputs, and because some goals (be correct, be
  preferred) are easier to *score* than to *demonstrate*.

Two flavors of "score":
- **RLHF** (RL from Human Feedback): a learned *reward model* predicts which answers humans prefer.
  Powerful but the reward model can be **gamed** ("reward hacking").
- **RLVR** (RL from Verifiable Rewards): the reward is a **rule or checker** — e.g. "does this equal the
  target number?" There is nothing to hack; the checker is ground truth. **This is what `d1` and we use.**
  It's ideal for reasoning tasks (math, code, puzzles) where correctness is mechanically checkable.

---

## 7. RL from scratch: reward → policy gradient → PPO → GRPO → DPO

We build up the exact machinery, each step motivated by a problem in the previous one.

### 7.1 The goal
The model is a **policy** `π_θ` — given a prompt `q`, it produces an answer `o` with probability
`π_θ(o | q)`. A **reward** `r(o)` scores the answer. We want weights θ that maximize the **expected
reward**: `J(θ) = E over o ~ π_θ [ r(o) ]`. We can't try all answers, so we estimate by sampling and
nudge θ in the direction that increases reward.

### 7.2 Policy gradient (REINFORCE) — where `log π` first appears
A classic identity (the "log-derivative trick") gives the gradient of expected reward:

```
grad_θ J = E over o ~ π_θ [ r(o) · grad_θ log π_θ(o | q) ]
```

In words: **sample an answer; if it got a high reward, increase its log-probability; if low, decrease it.**
This is the seed of everything. Notice `log π_θ(o|q)` sitting right in the middle — *this is the quantity
a diffusion model can't give cheaply* (§8). For an AR model it's free, so AR RL is easy.

### 7.3 Baselines and advantage — taming the noise
Raw rewards make the estimate jumpy (high variance). Subtract a **baseline** `b` (a reference score)
without changing the average direction:

```
grad_θ J = E [ (r(o) - b) · grad_θ log π_θ(o|q) ]
```

`A = r - b` is the **advantage**: "how much better than expected was this answer?" Positive → push up;
negative → push down. A good baseline cuts variance enormously. **The whole PPO/GRPO family is about how
to compute `A` and how to take safe steps.**

### 7.4 PPO — actor-critic with a safety clip
**PPO (Proximal Policy Optimization)** is the long-time default. It uses:
- a **critic** (a second network) that learns to predict the baseline/value → gives a low-variance `A`;
- a **clipped surrogate** so a single update can't move the policy too far. With the probability ratio
  `ρ = π_new(o)/π_old(o)`:

```
L_PPO = - min( ρ · A , clip(ρ, 1-ε, 1+ε) · A )
```

The `clip` caps how much `ρ` can grow/shrink, so even a big advantage can't yank the policy off a cliff in
one step (that's the "proximal"). PPO works well but needs **four models in memory** (policy, critic,
reference, reward) — heavy — and the critic is **awkward for diffusion**, where a partially-denoised
sequence isn't a clean "state" with a well-defined value.

### 7.5 GRPO — drop the critic, let the group be the baseline
**GRPO (Group Relative Policy Optimization)** is PPO's trick minus the critic. For each prompt, sample a
**group of `G` answers**. Use the group's **own average reward as the baseline**:

```
For prompt q, sample o_1..o_G ;  reward each r_1..r_G
A_i = r_i - mean(r_1..r_G)          # "is this answer better than its siblings?"
```

No value network needed → only two models (policy + reference), far lighter. It keeps PPO's clipped
surrogate and adds a **KL penalty** to a frozen reference policy so the model doesn't drift into gibberish:

```
per-token loss = - min( ρ · A , clip(ρ, 1-ε, 1+ε) · A )  +  β · KL(π_θ || π_ref)
```

> **Accuracy note (from the real `d1` code, not the paper):** the original GRPO paper divides the
> advantage by the group's standard deviation (`A_i = (r_i - mean)/std`). **`d1`'s implementation does
> NOT** — it uses `advantages = rewards - mean_grouped_rewards` (mean-centering only; the std is computed
> but used just for a "all answers identical" diagnostic). Skipping the `/std` follows the "Dr. GRPO"
> finding that std-normalization injects a length/difficulty bias. We reproduce `d1` as written. This is
> exactly the kind of detail that only reading the source — not the paper — reveals.

### 7.6 DPO — the offline alternative we deliberately don't use
**DPO (Direct Preference Optimization)** skips sampling and rewards entirely: given fixed pairs of
(preferred answer, rejected answer), a closed-form loss directly raises the preferred one's relative
probability. Cheap and stable, but **offline** — no exploration, and it needs **preference pairs**, not a
correctness checker.

### 7.7 Why GRPO here (the decision)

| Method | Needs a critic? | On-policy (explores)? | Needs | Fit for us |
|---|---|---|---|---|
| PPO | **yes** (value net) | yes | reward signal | critic ill-defined for parallel denoising; 4 models too heavy |
| **GRPO** | **no** (group baseline) | **yes** | reward signal | **✓ verifiable reward + exploration, light (2 models)** |
| DPO | no | **no** (offline) | preference pairs | no exploration; we have a *checker*, not preferences |

Verifiable-reward reasoning + exploration rules out DPO; limited compute + the diffusion-critic problem
rules out PPO. **GRPO is the natural fit — and it's what `d1` chose.**

---

## 8. The crux: getting `log π(answer | question)` for a diffusion model

Every RL method above needs `log π_θ(o | q)` (§7.2). For AR it's the chain-rule sum — free and exact.
For a diffusion model it's the **ELBO of §4** — an average over *all* masking levels, costing many forward
passes. Putting that inside a loop that runs thousands of times is hopeless. So `d1` approximates it.

### 8.1 `d1`'s one-step estimator — exactly what the code does
(From `diffu_grpo_trainer.py`: `forward_process` + `_get_per_token_logps`.)

1. Concatenate `[prompt | answer]` into one sequence.
2. **Mask every answer token** to `[MASK]`. Also mask each *prompt* token independently with a small
   probability `p_mask_prompt = 0.15` (a regularization noise — *not* part of the likelihood).
3. Run the network **once**.
4. At each answer slot, read `log p_θ(true token | this masked input) = -cross_entropy(logits, true token)`.
5. **Sum** these per-token log-probs → the estimate of `log π(answer | question)`.

That's it: **one forward pass, all answer tokens scored in parallel.**

### 8.2 Why it's biased
Compare to the ELBO (§4): the true bound averages over *all* mask levels `t`, including easy ones where
some answer tokens are visible and help predict their neighbors. The one-step estimator keeps **only the
`t=1` slice** — every answer token predicted with **no other answer token visible** (the hardest case).
It also treats the answer tokens as **conditionally independent** ("mean-field") given the blank answer,
ignoring how they constrain each other. So it **systematically underestimates** the true log-probability.
We *measured* this: a tiny `+0.099` average gap on toy sentences (§11). `d1`'s own paper acknowledges the
bias; later methods (`wd1`, `AGRPO`) remove it with importance weighting.

### 8.3 Why it still works (this is the subtle, important part)
GRPO never needs the *absolute* probability. It needs two things, and a biased estimator can deliver both:

- **Ranking within a group.** The advantage `A_i = r_i - mean` is set by *rewards*, and the gradient
  push is `A_i · grad log π(o_i)`. What matters is that the estimator assigns sensible *relative*
  log-probs across the group's answers. We measured this: on a group of (good / wrong / off-topic /
  gibberish) answers, the one-step estimate ranks them **identically to the full ELBO** (Spearman = 1.0),
  even though each absolute value is biased.

- **A stable ratio `ρ = π_new/π_old`.** Both sides are scored the **same biased way with the same random
  mask** (`d1` fixes one mask seed per inner-iteration and reuses it for new *and* old). So the bias — and
  even the large random-masking noise — is **common-mode and cancels** in the ratio. We measured this too:
  the raw estimate swings a lot across mask seeds (std ≈ 4), but for the *same* answer under a small policy
  step at a *matched* seed, new and old move together with correlation **1.0000** (the log-ratio's noise is
  0.005× the absolute noise). **That cancellation is precisely why a high-variance, biased estimator can
  still train a stable policy** — and why `d1` fixes the seed per iteration.

> §11 tells the story of how our *first* version of this test used the wrong yardsticks and "failed,"
> and how fixing the yardstick (not the threshold) confirmed all of the above.

---

## 9. diffu-GRPO, fully assembled (with the real loss from the code)

Putting §5–§8 together, one **diffu-GRPO** training step is:

```
for each prompt q in the batch:
    # 1. ROLLOUT (diffusion sampling, §5): generate G candidate answers o_1..o_G
    #    cost = G × (diffusion_steps NFEs). This is the dominant compute.
    # 2. REWARD (§10): score each with the rule-based checker -> r_1..r_G
    # 3. ADVANTAGE (§7.5): A_i = r_i - mean(r_1..r_G)          [d1: no /std]
    # 4. for mu = 1..num_iterations (reuse the same G answers, μ=12 in d1):
    #        pick this iteration's fixed mask seed
    #        new_logp_i = one_step_estimator(o_i, q, seed)     [§8.1, ONE forward]
    #        ratio_i    = exp(new_logp_i - old_logp_i)         [matched seed -> noise cancels]
    #        loss_i     = - min(ratio_i · A_i, clip(ratio_i, 1-ε, 1+ε) · A_i) + β · KL(π||π_ref)
    #        backprop loss through the LoRA adapters; AdamW step
```

The exact per-token loss in the code (`compute_loss`), for completeness:

```
coef_1        = exp(per_token_logps - old_per_token_logps)         # the ratio ρ
coef_2        = clamp(coef_1, 1-ε, 1+ε)                             # clipped ratio
per_token_loss = -min(coef_1 · A, coef_2 · A) + β · per_token_kl
loss          = sum(per_token_loss · completion_mask) / sum(completion_mask)   # mean over real answer tokens
# KL is the unbiased "k3" estimator:
per_token_kl  = exp(ref_logp - logp) - (ref_logp - logp) - 1
```

Hyperparameters `d1` uses for Countdown/GSM8K (`slurm_scripts/train.yaml`): `G = 6`, `diffusion_steps =
128`, `block_length = 32`, `max_completion_length = 256`, `num_iterations (μ) = 12`, `β = 0.04`,
`ε = 0.5`, `p_mask_prompt = 0.15`, LR `3e-6`, LoRA rank 128 + 4-bit, on `LLaDA-8B-Instruct`. **Why μ=12:**
generating answers is expensive, so `d1` reuses each batch of answers for 12 cheap gradient updates
(re-scoring with fresh mask seeds each time) — this is where the per-iteration matched-seed scheme lives.

Our **Rung A** shrinks the expensive knobs (`diffusion_steps` 128→64, `G` 6→4, a ~256-prompt subset,
~50–100 steps) to fit an 8-hour single-GPU job — enough to see whether accuracy rises (§12).

---

## 10. The reward: Countdown (rule-based, verifiable)

**Countdown** is a numbers puzzle: given a few numbers and a target, write an arithmetic expression that
hits the target **using each number exactly once** (operators `+ - * /`). Example: numbers `[3,4,5]`,
target `23` → `3 + 4 * 5`. It is perfect for RLVR because correctness is **mechanically checkable** and the
reward is dense enough to learn from. `d1`'s reward (`reward_func.compute_score`, which we unit-tested
19/19):

```
parse the model's <answer>...</answer>           ; if none -> reward 0
check it uses exactly the available numbers       ; if not  -> reward 0.1   (valid format, wrong content)
safe-evaluate the expression
    == target (within 1e-5) -> reward 1.0          (correct)
    != target               -> reward 0.1          (valid but wrong)
```

So: **1.0 correct, 0.1 valid-but-wrong, 0 unparseable.** The small `0.1` gives a gradient toward
"at least produce a valid expression" before the model learns to be correct. No learned reward model →
**no reward hacking** → the accuracy lift we measure is real, not gamed. We chose Countdown for Rung A
because it was `d1`'s largest RL gain (+26.2%) — the clearest signal that the mechanism works.

---

## 11. What we measured, and why the first test was wrong (the honest part)

Before any training, we validated the two no-training pieces (rewards, and the log-prob estimator). The
rewards passed 19/19. The estimator test taught us something about *testing itself*:

- **First attempt — "2 of 4 checks failed."** We gated on (a) the estimator's correlation with the ELBO
  *across different prompts*, and (b) its absolute variance across random mask seeds. Both looked bad
  (Pearson 0.85; variance 11× the between-sentence spread).
- **The realization:** those are **the wrong quantities for GRPO.** GRPO never compares answers *across
  prompts* — only *within one prompt's group*. And it never uses the *absolute* log-prob — only the
  *ratio* at a *matched* mask seed, where the noise cancels. We were grading the estimator on an exam it
  doesn't take.
- **Fixed test — all pass.** Measuring the GRPO-relevant quantities: within-group ranking is **perfect**
  (gold answer ranked #1 by both estimator and ELBO; corruption ladders monotone; Spearman = 1.0); the
  bias is tiny (`+0.099`); and the matched-seed same-answer correlation is **1.0000**, confirming the
  noise is common-mode (§8.3).

The lesson — *fix the yardstick, not the threshold* — is recorded in full in `FINDINGS.md`. We kept the
"failed" numbers in the log on purpose: a reproduction that hides its missteps isn't a reproduction.

---

## 12. The compute story and the Rung ladder

A faithful full reproduction of `d1` is **8×A100 for 72h ≈ ~24 GPU-days** — far beyond an 8-hour
single-GPU budget. So we do **not** start there. We use a **feasibility ladder**, climbing only if the
rung below works:

- **Rung A — mechanism demo (now):** RL *directly* from `LLaDA-8B-Instruct` (skip SFT), reduced knobs
  (`diffusion_steps` 64, `G` 4, ~256 prompts, ~50–100 steps), checkpoint-resume across 8h jobs. Goal:
  held-out Countdown accuracy rises above the measured baseline of **21.48%** (which matches `d1`'s own
  20.70% — a clean reproduction of the *starting point*). ≈ 0.5 GPU-day `[ESTIMATE]`.
- **Rung B — reduced faithful:** full `diffusion_steps`, more steps. A few GPU-days. *Only if A works.*
- **Rung C — novel:** port diffu-GRPO to **block diffusion**. *Only if B works.* (§13)

The discipline: **never start an expensive run without a logged compute estimate and a cheap validation
of every component it depends on.** That's why §11 happened before any training.

---

## 13. RL on block diffusion *(SUPERSEDED 2026-06-04 — not novel; active subfield, see top note)*

`d1` trains on **LLaDA**, a *fully* masked diffusion model. Our sister projects studied **block diffusion**
(BD3-LM, §4) on the *inference* side. The open question (Rung C): **does diffu-GRPO transfer to block
diffusion, and does the block structure change RL dynamics?** Intuition: blocks give a cleaner notion of
"what was generated when" (AR across blocks), which could mean cleaner *credit assignment* (knowing which
part of the answer deserves the reward) than fully-parallel denoising. ~~This is genuinely unexplored and is the part with publication potential~~ — **[SUPERSEDED 2026-06-04: RL on
block diffusion is an active subfield (TraceRL/TraDo arXiv:2509.06949, MMaDA/UniGRPO, StableDRL); it is **not**
novel. The open problem is the log-prob estimator — see `mathematics.md`.]** The original intuition below
(block structure → cleaner per-block credit assignment) is still a reasonable research *question*, but it is
now being actively studied by others, not first-explored here.

---

## 14. Glossary

- **Token / vocabulary:** atomic text chunk / the full set of them (~126k for LLaDA).
- **Logits / softmax / cross-entropy:** raw network scores / their conversion to probabilities /
  `-log p(true token)`; the differentiable bridge between a network and a log-probability.
- **Autoregressive (AR):** generates left-to-right; exact `log p` via the chain rule.
- **Masked diffusion model (MDM):** generates by iteratively un-masking; no chain rule; `log p` only via an ELBO.
- **`[MASK]` / mask ratio `t`:** the "blank" token / the fraction of tokens blanked in the forward process.
- **ELBO:** a computable *lower bound* on `log p(sentence)`; the average fill-in-the-blanks score over all `t`.
- **Denoising step / NFE:** one round of predict-and-reveal / one forward pass; more = better but slower.
- **Block diffusion (BD3-LM):** AR across blocks, diffusion within; `block_length` dials AR↔diffusion.
- **Policy `π_θ`:** the model viewed as "given prompt, produce answer with some probability."
- **Reward `r` / RLVR:** a score for an answer / RL where the reward is a verifiable rule (a checker).
- **Advantage `A`:** how much better than a baseline an answer was; sets the gradient's sign and size.
- **PPO / clipped surrogate:** actor-critic RL / a capped update so one step can't move the policy too far.
- **GRPO:** PPO without a critic; baseline = the group's average reward. **What we use.**
- **DPO:** offline preference-pair RL; no rollouts, no reward model. (Not used here.)
- **Ratio `ρ = π_new/π_old`:** how much the policy changed on this answer; clipped in the loss.
- **KL penalty `β`:** keeps the policy near a frozen reference so it doesn't degenerate.
- **`num_iterations` (μ):** how many gradient updates reuse one batch of generated answers (`d1`: 12).
- **One-step estimator:** `d1`'s cheap, biased `log π` — blank the answer, one forward, sum per-token log-probs.
- **Common-mode cancellation:** because new/old are scored with the *same* random mask, shared noise cancels in `ρ`.
- **LoRA / 4-bit:** train only small adapter weights / store the model in 4-bit — fits 8B on one GPU.
- **Countdown:** the verifiable arithmetic task we train on.
- **Rung A/B/C:** our cheap→faithful→novel feasibility ladder.

---

## 15. FAQ

**Q: Why not just compute the exact `log p` for the diffusion model?**
It has no closed form — only the ELBO, which needs many forward passes per sequence. Inside an RL loop
(thousands of sequences × many updates) that's prohibitive. Hence the one-step shortcut.

**Q: If the estimator is biased, isn't the training wrong?**
Bias in the *absolute* value is fine because GRPO only uses *relative* quantities: within-group ranking
(set by rewards) and the matched-seed ratio (where bias/noise cancel). We measured both and they're sound.
A biased-but-monotone, matched-seed-stable estimator is enough. (Unbiased successors `wd1`/`AGRPO` exist if
we later want them.)

**Q: Why mask the *prompt* at 0.15 if the prompt is given?**
It's a regularization noise that decorrelates the μ repeated updates (it perturbs the input slightly each
inner iteration). It is *not* part of the likelihood; our ELBO comparison keeps the prompt fully visible.

**Q: Why does the same random mask get reused for new and old policy?**
So the ratio `ρ = π_new/π_old` compares apples to apples — identical masked input, only the weights differ.
That makes the (large) masking noise common-mode, so it cancels and `ρ` reflects only the policy change.

**Q: Why GRPO and not PPO or DPO?**
PPO's critic is ill-defined for parallel denoising and needs 4 models (too heavy); DPO is offline and needs
preference pairs, but we have a correctness checker and want exploration. GRPO needs neither a critic nor
preferences. See §7.7.

**Q: Why Countdown and not just GSM8K?**
It was `d1`'s biggest RL gain (clearest signal), and its reward is a clean, dense, ungameble rule.

**Q: Where does NfePareto fit?**
It told us we can run rollouts at 64 denoising steps instead of 128 with negligible quality loss — halving
the dominant training cost. That's the concrete payoff of doing the inference study first.

**Q: What would make this *novel* rather than a reproduction?**
Rung C: applying diffu-GRPO to **block diffusion**, which `d1` did not do — and asking whether the block
structure improves RL credit assignment.

**Q: What's the honest current status?**
Baseline reproduced (21.48% ≈ `d1`'s 20.70%); rewards validated (19/19); estimator validated against the
ELBO (ranking perfect, bias tiny, matched-seed noise cancels). Next: a tiny end-to-end RL smoke test, then
the Rung-A run to see if accuracy actually rises. See `FINDINGS.md` for live numbers.

---

## 16. References

- **`d1`** — Zhao, Gupta, Zheng, Grover, *"d1: Scaling Reasoning in Diffusion LLMs via RL,"* arXiv:2504.12216 · [code](https://github.com/dllm-reasoning/d1)
- **LLaDA** — Nie et al., *"Large Language Diffusion Models,"* 2025 — the masked diffusion base model + its ELBO.
- **MDLM** — Sahoo et al., *"Simple and Effective Masked Diffusion Language Models,"* 2024 — the ELBO we test against.
- **Block diffusion (BD3-LM)** — Arriola et al., 2025 — AR↔diffusion interpolation.
- **GRPO** — Shao et al., *"DeepSeekMath,"* 2024 — group-relative policy optimization.
- **PPO** — Schulman et al., 2017. **DPO** — Rafailov et al., 2023. **Dr. GRPO** (no-std-normalization) — Liu et al., 2025.

_See `theory` ↔ `SPEC.md` (the plan) ↔ `FINDINGS.md` (the measured results) ↔ `LOG.md` (the trail). This
file is the "why"; those are the "what" and "when."_
