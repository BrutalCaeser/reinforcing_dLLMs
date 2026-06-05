# mathematics.md — The Complete Mathematics of GRPO for Diffusion LLMs

> **Purpose.** A from-scratch, self-contained mathematical treatment of reinforcement learning (GRPO)
> applied to masked **diffusion** language models — the *d1 / diffu-GRPO* recipe and the 2025–26 estimator
> frontier around it. Every symbol is defined; every equation is followed by a plain-English reading and,
> where useful, a **worked numeric example**.
>
> **Notation style:** all math is **plain ASCII inside code blocks** (operators spelled out: `sum`, `prod`,
> `E[...]`, `1[...]`; Greek letters as Unicode: θ π β ε μ α). **No LaTeX, nothing to render** — it reads the
> same in an editor, a terminal (`cat`/`less`), or a Markdown preview. (Matches `theory.md`.)
>
> **Companion docs:** `theory.md` (same ideas, more words, less math), `FINDINGS.md` (measured results),
> `SPEC.md` (plan), wiki `concepts/rl-for-diffusion-llms.md` (field landscape). Equations are reconciled with
> d1's actual code (`diffu_grpo_trainer.py`, `reward_func.py`); unproven claims are marked `[UNVERIFIED]`.

---

## Table of contents
- Part 0  — Notation (read first)
- Part I  — What a language model is (probability over sequences)
- Part II — Masked diffusion LMs: forward process, reverse model, the ELBO
- Part III— The log-probability problem (why diffusion breaks RL)
- Part IV — RL foundations: objective -> policy gradient -> PPO -> GRPO
- Part V  — diffu-GRPO: GRPO on a diffusion policy (the full objective)
- Part VI — The estimator frontier (the real open problem)
- Part VII— Diffusion-as-MDP (the DDPO view)
- Part VIII—Block diffusion: the two-level factorization
- Part IX — The reward + a full worked Countdown example
- Part X  — Evaluation mathematics (and statistical honesty)
- Part XI — Inference / sampling mathematics (NFE, blocks, remasking)
- Part XII— Glossary of every symbol
- Part XIII—FAQ / common confusions
- References

---

## Part 0 — Notation (read first)

| symbol | meaning |
|---|---|
| `V`, `|V|` | vocabulary and its size (~126,000 for LLaDA) |
| `[MASK]` | the special mask token (id 126336 for LLaDA) |
| `q = (q_1..q_n)` | the **prompt** (question), n tokens |
| `o = (o_1..o_L)` | a **completion** (answer), L = |o| tokens |
| `π_θ(o|q)` | the **policy**: prob the model (weights θ) assigns to completion o given q |
| `π_old` | the policy **at the moment the rollouts were generated** (frozen during inner updates) |
| `π_ref` | a frozen **reference** policy (the starting model), for the KL leash |
| `t` in [0,1] | diffusion **time** / mask ratio: t=0 clean, t=1 fully masked |
| `x_t` | a partially-masked version of a sequence at time t |
| `G` | **group size**: completions sampled per prompt (d1 countdown: 6) |
| `r_i` | scalar **reward** of completion o_i |
| `A_i` | **advantage** of completion i (how much better than its group) |
| `ρ_{i,t}` | **importance ratio** π_θ/π_old for token t of completion i |
| `β` | KL penalty weight (d1: 0.04) |
| `ε` | PPO clip half-width (d1: 0.5) |
| `μ` | num_iterations: inner gradient updates reusing one rollout batch (d1 countdown: 8) |
| `1[...]` | indicator: 1 if true else 0 |
| `E[...]` | expectation (average over the named randomness) |
| `~` | "distributed as" (e.g. `o ~ π_θ`) |

"Forward pass" = one evaluation of the network f_θ. **NFE** = number of function evaluations (forward
passes), the natural unit of diffusion-generation cost.

---

## Part I — What a language model is

A language model defines a probability over token sequences, `p_θ(x)`. An **autoregressive (AR)** model
factorizes left-to-right by the chain rule:

```
p_θ(x)      = prod_{k=1..|x|} p_θ(x_k | x_<k)
log p_θ(x)  = sum_{k=1..|x|}  log p_θ(x_k | x_<k)
```

**Plain reading.** "Probability of the whole sentence = product of each token's probability given everything
before it." Each factor is a single softmax the model already computes, so the **per-sequence log-prob is
exact and cheap** for AR models. *Hold onto this — it is exactly what diffusion LMs lack, and every
difficulty below flows from that one fact.* For RL we need the log-prob of a *completion* given a *prompt*:

```
log π_θ(o|q) = sum_{k=1..L} log p_θ(o_k | q, o_<k)      # AR case -- exact
```

Aside (used everywhere): the model emits **logits** z (raw scores); softmax turns them into probabilities,
and a log-prob is just a negative cross-entropy — differentiable, no argmax:

```
softmax(z)_k        = exp(z_k) / sum_j exp(z_j)
log p(true token c) = z_c - log sum_j exp(z_j)  = -CrossEntropy(z, c)
```

---

## Part II — Masked diffusion LMs

### II.1 Forward (noising) process
A masked diffusion model (MDM; LLaDA, MDLM arXiv:2406.07524) corrupts a clean sequence x_0 by replacing
tokens with `[MASK]`. At time t each position is independently masked with probability t:

```
q(x_t^i | x_0^i) =  1 - t   ->  x_t^i = x_0^i      (kept)
                    t       ->  x_t^i = [MASK]      (masked)
```

t=0 nothing masked (clean); t=1 everything masked. **Example** (L=4, t=0.5): clean `[2, *, 4, 3]` -> a
sample x_t might be `[2, MASK, 4, MASK]` (each token a coin-flip at rate t).

### II.2 Reverse (denoising) model
The network takes the partially-masked x_t and outputs, for **every masked position**, a categorical
distribution over V predicting the original token:

```
p_θ(x_0^i | x_t) = softmax( f_θ(x_t)_i )      # a distribution over the vocabulary
```

There is **no argmax and no sampling inside the loss** — just a softmax. That keeps everything
**differentiable** (gradients flow), the precondition for policy-gradient RL. (Ermon confirmed in person:
gradients DO backpropagate through discrete diffusion.)

### II.3 Training loss = a (weighted) cross-entropy = the negative ELBO
MDMs are trained by a bound on the negative log-likelihood (the NELBO). For the linear (absorbing) schedule
`α_t = 1 - t`:

```
-log p_θ(x_0)  <=  L(θ)
L(θ) = E_{t ~ U(0,1)}  E_{x_t ~ q(.|x_0,t)} [  (1/t) * sum_{i=1..|x_0|} 1[x_t^i = MASK] * ( -log p_θ(x_0^i | x_t) )  ]
```

**Reading, piece by piece:**
- `1[x_t^i = MASK]`   — only **masked** positions contribute (predict what was hidden).
- `-log p_θ(x_0^i|x_t)`— ordinary **cross-entropy**: "how surprised is the model by the true token."
- `(1/t)`              — a **reweighting**; makes the Monte-Carlo estimate an unbiased bound on `-log p`.
- `E_t E_{x_t}`        — average over a random mask ratio t and a random masking pattern.

**The key structural fact:** the diffusion loss is *just a reweighted cross-entropy with no argmax*. The
marginal likelihood `log p_θ(x_0)` itself is **not** computed — only this bound. That gap is Part III.

---

## Part III — The log-probability problem (why diffusion breaks RL)

RL (Part IV) needs `log π_θ(o|q)`. AR models get it exactly (Part I). A diffusion LM does not:

```
log π_θ(o|q) = log ( sum over all unmasking orders / denoising trajectories of ... )   # INTRACTABLE
```

There is **no closed form**, because tokens are emitted **in parallel, in arbitrary order, over many
denoising steps**. We only have either:
1. the **ELBO** (Part II.3): a *bound*, estimated by Monte-Carlo over (t, mask) — unbiased-ish but
   **high variance**; or
2. a cheap **one-step / mean-field** approximation (Part V.2) — **biased** but a single forward pass.

> The entire field of RL-for-diffusion-LLMs is the search for the right way to estimate this one quantity.
> Every method in Part VI is a different bias / variance / compute trade-off for `log π_θ(o|q)`.

---

## Part IV — RL foundations (objective -> policy gradient -> PPO -> GRPO)

### IV.1 The objective
With a reward R(o) (e.g. "1 if the Countdown answer is correct"), we want a high-reward policy:

```
J(θ) = E_{o ~ π_θ(.|q)} [ R(o) ]
```

### IV.2 Policy gradient (REINFORCE) — derived
Differentiate and use the log-derivative trick `grad π = π * grad log π`:

```
grad_θ J = sum_o ( grad_θ π_θ(o) ) R(o)
         = sum_o π_θ(o) ( grad_θ log π_θ(o) ) R(o)
         = E_{o ~ π_θ} [ R(o) * grad_θ log π_θ(o|q) ]
```

**Reading.** "Push up the log-probability of completions, weighted by reward." Note it needs
`grad_θ log π_θ` — the log-prob again.

### IV.3 Baselines (variance reduction)
Because `E[ grad_θ log π_θ ] = 0`, we may subtract any baseline b without bias:

```
grad_θ J = E[ ( R(o) - b ) * grad_θ log π_θ(o|q) ]
```

A good b (e.g. the average reward) cuts variance: better-than-average samples go up, worse go down. The
quantity `A = R - b` is the **advantage**.

### IV.4 PPO — take several steps per rollout, safely
Rollouts are expensive, so we want many gradient steps per batch. Stepping too far from the policy that
generated the data makes the gradient invalid. PPO uses an **importance ratio** `ρ = π_θ/π_old` and a
**clip**:

```
J_PPO = E[ min( ρ*A , clip(ρ, 1-ε, 1+ε)*A ) ]
```

**Reading.** Improve while the new policy stays within `1 ± ε` of the old; the min+clip removes the
incentive to move further once at the edge. d1 uses a loose `ε = 0.5`.

### IV.5 GRPO — drop the critic, compare within a group
Standard PPO needs a learned value network (critic). **GRPO** removes it: sample a **group** of G
completions for the *same* prompt and use the group as its own baseline.

```
standard GRPO advantage:   A_i = ( r_i - mean(r_1..r_G) ) / std(r_1..r_G)

d1 / "Dr.GRPO" (what our code ACTUALLY does):
   A_i = r_i - mean(r_1..r_G)        # mean-centered ONLY, NO division by std
```

> **Verified from d1's source**, not the paper prose: `advantages = rewards - mean_grouped_rewards`. The std
> is computed only to *log* a `zero_std_ratio` diagnostic. This matters — std-normalization changes the
> gradient scale. Confirmed in `diffu_grpo_trainer.py`.

Full GRPO objective, per token, with PPO clipping and a KL leash to π_ref:

```
J_GRPO(θ) = E[ (1/G) * sum_{i=1..G} (1/|o_i|) * sum_{t=1..|o_i|}
                 min( ρ_{i,t} * A_i ,  clip(ρ_{i,t}, 1-ε, 1+ε) * A_i ) ]
            - β * KL( π_θ || π_ref )

ρ_{i,t} = π_θ(o_{i,t} | q, o_{i,<t}) / π_old(o_{i,t} | q, o_{i,<t})
```

**The KL term** uses Schulman's low-variance, always-nonnegative **k3 estimator** (let `s = log π_ref - log π_θ`):

```
KL( π_θ || π_ref ) ≈ E[ exp(s) - s - 1 ]   >= 0
```

### IV.6 A subtlety we measured: loss ≈ 0 but gradient ≠ 0
Advantages are mean-centered, so `sum_i A_i = 0`. On the **first** inner step `ρ_{i,t} = 1` (since
π_θ = π_old), so the surrogate's *value* is `sum_i A_i * 1 = 0` -> the **reported loss is ~0**. But the
*gradient* is `sum_i A_i * grad_θ log π_θ(o_i) != 0` — it still pushes high-A completions up, low ones down.
Our tiny-RL smoke showed exactly this: `loss 0.0, grad_norm 1.47`. **A zero loss here is correct, not a bug.**

---

## Part V — diffu-GRPO: GRPO on a diffusion policy

### V.1 The assembly
diffu-GRPO (d1, arXiv:2504.12216) is *exactly* the GRPO objective of Part IV.5, with one change: every
`log π(.)` — needed for `ρ_{i,t}` and the KL — comes from a **diffusion log-prob estimator** (Part III says
we can't compute it exactly).

### V.2 d1's one-step (mean-field) estimator — the exact recipe
To score a completion o given prompt q in a **single forward pass**:

```
1. Mask the ENTIRE completion: all L tokens -> [MASK]            (call it o_masked)
2. (optional) perturb the prompt: mask each prompt token w.p. p_mask_prompt = 0.15   (-> q')
3. ONE forward pass f_θ([ q' ; o_masked ]) gives a categorical at each completion position
4. Read off the log-prob of the TRUE token at each position and sum:

   logπ_hat_θ(o|q) = sum_{i=1..L} log p_θ( o_i | q', o_masked )
```

**Reading.** "Hide the whole answer, ask the model to guess every token at once, sum its confidence in the
true tokens." It is the `t ≈ 1` (fully-masked) slice of the ELBO, and it treats answer tokens as
**conditionally independent** given the masked context (a *mean-field* assumption).

**Why biased:** ignores intra-answer dependencies; uses only the hardest (fully-masked) slice. **We measured
the bias** (`elbo_vs_onestep.py`): `mean(ELBO - one_step) = +0.099` (one-step slightly *under*-estimates),
and the per-token curve falls from -0.06 at t<=0.5 to -0.21 at t=1 — exactly the predicted mechanism.

### V.3 Why a biased, high-variance estimator still trains (our key finding)
GRPO never uses the *absolute* log-prob — only the **ratio** `ρ = π_θ/π_old`, computed with **the same
random mask seed** for numerator and denominator:

```
log ρ_{i,t} = logπ_hat_θ(o|q ; ξ) - logπ_hat_old(o|q ; ξ)        # ξ = shared mask pattern
```

The large masking noise ξ is **common-mode** and cancels in the difference. We verified it directly:

```
corr( logπ_hat^old_A , logπ_hat^new_A )  =  1.0000     # same completion, matched seed
std(log-ratio) / std(absolute estimate)  ≈  0.005      # ~200x quieter
```

**This is the mathematical reason diffu-GRPO works despite a crude estimator.** It also tells you what NOT to
do: comparing log-probs across *different* seeds or *different* prompts re-introduces the noise.

### V.4 The reuse loop (μ) and hyperparameters
One expensive generation of G completions is reused for μ inner gradient steps (PPO epochs); on step 1
`ρ = 1`, then it drifts and the clip engages. d1 (Countdown) settings, all confirmed in config:

```
G = 6 ,  μ = 8 ,  β = 0.04 ,  ε = 0.5 ,  p_mask_prompt = 0.15 ,  lr = 3e-6
```

### V.5 The zero-advantage trap (the practical wall — derived)
If **all G completions get the same reward**, then `A_i = 0` for all i -> **zero gradient** -> that prompt
teaches nothing. If each completion succeeds independently with probability p:

```
Pr[zero advantage] = p^G + (1 - p)^G
```

For a weak model on hard prompts (p -> 0): `(1-p)^G -> 1` -> almost always trapped. **Numbers** (G=6):

```
p = 0.05 :  (0.95)^6  = 0.735  trapped
            (0.95)^16 = 0.440          # raising G to 16
            (0.95)^32 = 0.194          # raising G to 32
```

So **bigger G rescues low-p prompts** — a concrete, math-driven lever. We saw this trap live:
`zero_std_ratio = 1.0` on ~half the rounds of our 32-prompt run, which capped reward near 0.35.

---

## Part VI — The estimator frontier (the real open problem)

As of mid-2026 there is **no consensus** on how to estimate `log π_θ(o|q)` for GRPO; >=6 proposals in
~6 months. This is the live frontier (and where Ermon is publishing — ESPO). Each method's trade-off:

| method | arXiv | idea (one line) | trade-off |
|---|---|---|---|
| one-step / mean-field (d1) | 2504.12216 | mask all, 1 forward, sum token CE | cheap, **biased** |
| coupled-GRPO (DiffuCoder) | 2506.20639 | complementary mask pairs -> full coverage | variance down |
| ELBO + VR (VRPO) | 2505.19223 | ELBO + antithetic + optimal MC budget | unbiased-ish, var down |
| GDPO | 2510.08554 | sequence ELBO + semi-deterministic MC | provably lower var |
| AGRPO | 2510.04019 | cache unmask order -> **exact** per-step categorical | unbiased, costly |
| wd1 | 2507.08838 | **ratio-free** weighted log-likelihood | kills ratio variance |
| SPG | 2510.09541 | **sandwich** upper+lower bounds | bias down |
| d2 | 2509.21474 | trajectory likelihood (exact 1-pass any-order) | SOTA (claimed) |
| ESPO (Ermon) | 2512.03759 | **sequence-level** action + ELBO proxy | — |

The bias <-> variance <-> compute triangle is unresolved. Our `elbo_vs_onestep` analysis sits exactly here;
a clean **single-GPU, apples-to-apples comparison** of these estimators is the most honest contribution open
to us.

---

## Part VII — Diffusion-as-MDP (the DDPO view)

A complementary frame (DDPO, Black et al. 2023, arXiv:2305.13301) treats the **denoising trajectory** as a
Markov Decision Process — useful for per-step credit assignment and for the block case (Part VIII):

```
state    s_t = (q, t, x_t)                  # prompt, step index, current partially-denoised sequence
action   a_t = x_{t-1}                       # the next, less-noisy sequence
policy   π(a_t | s_t) = p_θ(x_{t-1} | x_t, q)   # ONE denoising step = ONE decision
reward   r(x_0)                              # sparse: only at the final clean sample
trajectory likelihood factorizes over steps:
         p_θ(x_{0:T} | q) = prod_t p_θ(x_{t-1} | x_t, q)
```

**The trick that makes it tractable:** never compute the marginal `p_θ(x_0)`; use **per-step** log-probs
`log p_θ(x_{t-1}|x_t)`. Continuous diffusion -> closed-form Gaussian per step; **discrete/text** diffusion ->
**categorical** over V per position. Policy gradient = `sum_t grad_θ log p_θ(x_{t-1}|x_t) * A`, with per-step
importance ratios + PPO clip (DDPO_IS). AGRPO (Part VI) is essentially this MDP view done exactly for masked
text.

---

## Part VIII — Block diffusion: the two-level factorization

**Block diffusion** (BD3-LM arXiv:2503.09573; SDAR; Inception's block-32) splits a sequence into B blocks
`x^(1)..x^(B)` and is **autoregressive across blocks, diffusion within a block**:

```
p_θ(x)      = prod_{b=1..B}  p_θ( x^(b) | x^(<b) )       # each factor = masked diffusion within block b
log p_θ(x)  = sum_{b=1..B}   log p_θ( x^(b) | x^(<b) )
```

**Why interesting for RL:** the **outer** product over blocks is an ordinary AR factorization (the tractable
part), while each **inner** block term is a diffusion ELBO (the hard part — but over a *short* block, so
cheaper to estimate well). This gives a real design choice — at what granularity does the GRPO
advantage/ratio live?

```
block-level :  sum_b  ρ^(b) * A
token-level :  sum_b sum_{j in block b}  ρ_{b,j} * A
step-level  :  sum_b sum_τ  ρ_{b,τ} * A          # τ over denoising steps within the block
```

StableDRL (arXiv:2603.06743) introduces **"staircase attention"** for *leakage-free* per-block log-prob
estimation (so block b's probability doesn't peek at future blocks). **[UNVERIFIED whether the
block-vs-token-vs-step choice is settled — Agent C found no paper that cleanly resolves it for the semi-AR
case; it's the narrowest plausibly-open question, but pursuing it means racing active 2026 work.]**

---

## Part IX — The reward, and a full worked Countdown example

### IX.1 The Countdown reward (RLVR — rule-based, no learned reward model)
Given numbers (a multiset) N and target T, the model must output `<answer>EXPR</answer>`. From
`reward_func.countdown_reward_func` (unit-tested 19/19):

```
R(o) = 1.0  if EXPR uses each number in N exactly once (multiset) AND eval(EXPR) == T
       0.1  if EXPR parses to a valid expression but is wrong / misuses numbers
       0.0  if there is no parseable <answer>  (garbage)
```

Parsing: the **last** `<answer>` wins; evaluation is a safe-eval over { + - * / }.

### IX.2 End-to-end, with numbers
**Prompt q:** "Using 30, 100, 93 make an expression equal to 23." G=6 rollouts; suppose rewards:

```
r        = [ 1.0 , 1.0 , 0.1 , 0.1 , 0.1 , 0.0 ]
mean(r)  = 2.3 / 6 = 0.3833

A_i = r_i - mean(r)   (d1 mean-centered):
A        = [ +0.617 , +0.617 , -0.283 , -0.283 , -0.283 , -0.383 ]
sum_i A_i = 0    (check: 2*0.617 - 3*0.283 - 0.383 = 1.234 - 0.849 - 0.383 ≈ 0)  OK
```

**Reading.** The two correct completions (A>0) get their token log-probs pushed **up**; the three
valid-but-wrong and the garbage one get pushed **down**, hardest on the garbage (A=-0.383). On inner step 1,
ρ=1 so reported loss ≈ 0; the gradient is nonzero and updates θ; then μ-1 more clipped steps refine on the
same rollouts.

**Trap example.** If instead all six were valid-but-wrong, `r = [0.1,...,0.1]`, then `mean = 0.1`,
`A = [0,...,0]` -> **no gradient** (Part V.5). This is what capped our 32-prompt run.

---

## Part X — Evaluation mathematics (and statistical honesty)

**Accuracy** on a held-out set of N prompts:

```
acc = (1/N) * sum_{n=1..N} 1[ model's answer to prompt n is correct ]
```

**Train-reward != held-out accuracy.** Mean *training reward* (e.g. our ~0.4) includes the 0.1
partial-credit floor and is on *seen* prompts; convert to an approximate correct-fraction via
`frac_correct ≈ (mean_reward - 0.1) / 0.9`. **Held-out accuracy** is the real generalization metric.

**Is a measured gap real?** A proportion has standard error:

```
SE = sqrt( p_hat * (1 - p_hat) / N )
```

**Our case:** baseline 21.48% vs adapter 25.78% on N=256.

```
SE  ≈ sqrt( 0.23 * 0.77 / 256 ) ≈ 0.026 = 2.6%
gap = +4.3%  ≈  1.6 * SE     ->  SUGGESTIVE, NOT significant
```

So the lift needs a 2–3 seed replication before we trust it. (This is why it is marked `[needs seeds]`
everywhere, never asserted as significant.)

---

## Part XI — Inference / sampling mathematics

Generation runs the reverse process from fully-masked to clean over `diffusion_steps` iterations:

```
1. start x_1 = [MASK, MASK, ..., MASK]   (length = gen_length)
2. for each step:
     - forward pass -> categorical at each masked position
     - fill in predictions, then REMASK the least-confident fraction
       (remasking = "low_confidence") so they get reconsidered next step
3. after diffusion_steps steps, all positions are committed -> the answer
```

**Cost.** `NFE ≈ diffusion_steps` (one forward per step). For LLaDA the default ties
`diffusion_steps = gen_length / 2`. **Semi-autoregressive** generation does this **per block** of
`block_length` tokens, left to right, caching previous blocks (the block-32 design).

**Quality vs NFE.** More steps = higher quality but slower. Our **NfePareto** project measured this "speed
dial": generative perplexity drops steeply then **plateaus** past a knee — so for RL rollouts pick the
smallest NFE on the plateau (we used 64) to maximize throughput without starving the reward signal.

---

## Part XII — Glossary of every symbol
- `π_θ, π_old, π_ref` — current / rollout-time / frozen-reference policies.
- `q, o, L` — prompt, completion, completion length.
- `t, x_t, [MASK]` — diffusion time (mask ratio), masked sequence, mask token.
- `G` — group size (completions per prompt).
- `r_i, A_i` — reward and (mean-centered) advantage of completion i.
- `ρ_{i,t}` — importance ratio π_θ/π_old for token t of completion i.
- `ε, β, μ` — PPO clip half-width; KL weight; inner-update count per rollout.
- `p_mask_prompt` — prompt-perturbation mask prob (0.15) in the estimator.
- `NFE` — forward passes per generation (≈ diffusion_steps).
- `ELBO` — evidence lower bound (the diffusion training bound on log p).
- `KL` — Kullback–Leibler divergence (k3 estimator), the leash to π_ref.

## Part XIII — FAQ / common confusions
- **"Why not compute log π exactly?"** Diffusion LMs have no tractable marginal sequence likelihood
  (Part III). That single fact creates the whole estimator frontier (Part VI).
- **"Loss is ~0 — is training broken?"** No (Part IV.6): mean-centered advantages make the surrogate *value*
  ~0 while the *gradient* is nonzero.
- **"Reward rose on training prompts — are we done?"** No (Part X): that's train-reward on *seen* prompts.
  Only held-out accuracy (vs the 21.48% baseline) counts, and it needs seed-confirmation.
- **"Is GRPO-on-block-diffusion novel?"** No — already done at scale (TraceRL/TraDo, etc.; see the wiki
  landscape). The open part is the **estimator** and **minimal-compute**.
- **"Why does the crude one-step estimator work?"** Matched-seed common-mode cancellation in the ratio
  (Part V.3), verified at correlation 1.0000.

## References (arXiv)
- d1 / diffu-GRPO — 2504.12216 ; MDLM — 2406.07524 ; DDPO — 2305.13301
- AGRPO — 2510.04019 ; wd1 — 2507.08838 ; SPG — 2510.09541 ; GDPO — 2510.08554 ; d2 — 2509.21474
- ESPO (Ermon) — 2512.03759 ; VRPO/LLaDA-1.5 — 2505.19223 ; DiffuCoder — 2506.20639
- Block: BD3-LM — 2503.09573 ; TraceRL/TraDo — 2509.06949 ; StableDRL — 2603.06743 ; MMaDA/UniGRPO — 2505.15809
- Full landscape + verification status: wiki `concepts/rl-for-diffusion-llms.md`. Measured numbers: `FINDINGS.md`.

---

_Maintained alongside the project. If an equation here ever disagrees with the code, the code wins — fix the
doc and note it in `LOG.md`._
