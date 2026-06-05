# mathematics.md — The Complete Mathematics of GRPO for Diffusion LLMs

> **Purpose.** A from-scratch, self-contained mathematical treatment of reinforcement learning (GRPO)
> applied to masked **diffusion** language models — the *d1 / diffu-GRPO* recipe and the 2025–26 estimator
> frontier around it. Every symbol is defined; every equation is followed by a plain-English reading and,
> where useful, a **worked numeric example**. The goal: that you can read this once and (a) understand the
> method to its foundations and (b) train and evaluate it yourself.
>
> **Companion docs:** `theory.md` (intuition, no heavy math), `FINDINGS.md` (our measured results),
> `SPEC.md` (the plan), and the wiki page `concepts/rl-for-diffusion-llms.md` (the field landscape).
> **Grounding:** equations are reconciled with d1's actual code (`diffu_grpo_trainer.py`,
> `reward_func.py`) and the literature (arXiv IDs inline). Where a claim is unproven I mark it `[UNVERIFIED]`.

---

## Table of contents
- Part 0 — Notation (read this first)
- Part I — What a language model is (probability over sequences)
- Part II — Masked diffusion LMs: forward process, reverse model, the ELBO
- Part III — The log-probability problem (why diffusion breaks RL)
- Part IV — RL foundations: objective → policy gradient → PPO → GRPO
- Part V — diffu-GRPO: GRPO on a diffusion policy (the full objective)
- Part VI — The estimator frontier (the real open problem)
- Part VII — Diffusion-as-MDP (the DDPO view)
- Part VIII — Block diffusion: the two-level factorization
- Part IX — The reward and a full worked Countdown example
- Part X — Evaluation mathematics (and statistical honesty)
- Part XI — Inference / sampling mathematics (NFE, blocks, remasking)
- Part XII — Glossary of every symbol
- Part XIII — FAQ / common confusions
- References

---

## Part 0 — Notation (read this first)

| symbol | meaning |
|---|---|
| $V$ | vocabulary (set of tokens); $|V|$ its size |
| $\mathbf{m}$ | the special `[MASK]` token (id `126336` for LLaDA) |
| $q = (q_1,\dots,q_n)$ | the **prompt** (question), $n$ tokens |
| $o = (o_1,\dots,o_L)$ | a **completion** (the model's answer), $L=|o|$ tokens |
| $x = (x_1,\dots)$ | a generic token sequence |
| $\pi_\theta(o\mid q)$ | the **policy**: probability the model (params $\theta$) assigns to completion $o$ given $q$ |
| $\pi_{\text{old}}$ | the policy **at the moment rollouts were generated** (frozen during inner updates) |
| $\pi_{\text{ref}}$ | a frozen **reference** policy (the starting model), for the KL leash |
| $t\in[0,1]$ | diffusion **time** / mask ratio: $t{=}0$ clean, $t{=}1$ fully masked |
| $x_t$ | a partially-masked version of $x$ at time $t$ |
| $G$ | **group size**: number of completions sampled per prompt (d1 countdown: $6$) |
| $r_i$ | scalar **reward** of completion $o_i$ |
| $\hat A_i$ | **advantage** of completion $i$ (how much better than its group) |
| $\rho_{i,t}$ | **importance ratio** $\pi_\theta/\pi_{\text{old}}$ for token $t$ of completion $i$ |
| $\beta$ | KL penalty weight (d1: $0.04$) |
| $\varepsilon$ | PPO clip half-width (d1: $0.5$) |
| $\mu$ | `num_iterations`: inner gradient updates reusing one rollout batch (d1 countdown: $8$) |
| $\mathbb{1}[\cdot]$ | indicator: $1$ if true else $0$ |
| $\mathbb{E}$ | expectation (average over the randomness named in the subscript) |

A **forward pass** = one evaluation of the network $f_\theta$. "NFE" = *number of function evaluations*
(forward passes), the natural unit of diffusion-generation cost.

---

## Part I — What a language model is

A language model defines a probability distribution over token sequences, $p_\theta(x)$. For **autoregressive
(AR)** models (GPT-style) it factorizes left-to-right by the chain rule of probability:

$$ p_\theta(x) = \prod_{k=1}^{|x|} p_\theta(x_k \mid x_{<k}), \qquad \log p_\theta(x) = \sum_{k=1}^{|x|} \log p_\theta(x_k \mid x_{<k}). $$

**Plain reading.** "The probability of the whole sentence = product of the probability of each token given
everything before it." Because each factor $p_\theta(x_k\mid x_{<k})$ is a single softmax the model already
computes, the **per-sequence log-probability is exact and cheap** for AR models. *Hold onto this — it is
exactly what diffusion LMs lack, and the entire difficulty of RL-on-diffusion flows from that one fact.*

For RL we care about $\log \pi_\theta(o\mid q)$ — the log-prob of a *completion* $o$ given a *prompt* $q$:

$$ \log \pi_\theta(o\mid q) = \sum_{k=1}^{L}\log p_\theta\big(o_k \mid q, o_{<k}\big). \quad\text{(AR case — exact)} $$

---

## Part II — Masked diffusion LMs

### II.1 The forward (noising) process
A masked diffusion model (MDM; e.g. **LLaDA**, **MDLM** arXiv:2406.07524) corrupts a clean sequence $x_0$ by
**replacing tokens with `[MASK]`**. At time $t\in[0,1]$ each position is independently masked with
probability $t$:

$$ q(x_t^i \mid x_0^i) = \begin{cases} 1-t, & x_t^i = x_0^i \ \text{(kept)}\\[2pt] t, & x_t^i = \mathbf{m}\ \text{(masked)} \end{cases} $$

At $t{=}0$ nothing is masked (clean); at $t{=}1$ everything is masked. **Example** ($L{=}4$, $t{=}0.5$): a
clean answer `[2,*,4,-,3]`→ a sample $x_t$ might be `[2, MASK, 4, MASK, 3]` (each token a coin-flip).

### II.2 The reverse (denoising) model
The network $f_\theta$ takes the partially-masked $x_t$ and outputs, **for every masked position**, a
categorical distribution over $V$ predicting the original token:

$$ p_\theta(x_0^i \mid x_t) = \mathrm{softmax}\big(f_\theta(x_t)_i\big) \in \Delta^{|V|-1}. $$

Note there is **no argmax and no sampling inside the loss** — it is a plain categorical (softmax)
distribution. That is what keeps everything **differentiable** (gradients flow), which is the precondition
for policy-gradient RL. (Ermon confirmed this in person: *gradients do backpropagate through discrete
diffusion.*)

### II.3 The training loss = a (weighted) cross-entropy = the negative ELBO
MDMs are trained by a bound on the negative log-likelihood (the NELBO). For the linear (absorbing) schedule
$\alpha_t = 1-t$, the continuous-time bound is

$$ -\log p_\theta(x_0)\ \le\ \mathcal L(\theta) = \mathbb{E}_{t\sim U(0,1)}\ \mathbb{E}_{x_t\sim q(\cdot\mid x_0,t)}\!\left[\ \frac{1}{t}\sum_{i=1}^{|x_0|}\mathbb{1}[x_t^i=\mathbf m]\ \big(-\log p_\theta(x_0^i\mid x_t)\big)\right]. $$

**Plain reading, piece by piece.**
- $\mathbb{1}[x_t^i=\mathbf m]$: only **masked** positions contribute (predict what was hidden).
- $-\log p_\theta(x_0^i\mid x_t)$: ordinary **cross-entropy** — "how surprised is the model by the true token."
- $\tfrac1t$: a **reweighting**. Small $t$ (few masks) → easy → upweighted so it isn't ignored; this factor
  makes the Monte-Carlo estimate an unbiased bound on $-\log p$.
- $\mathbb{E}_{t}\,\mathbb{E}_{x_t}$: average over a random mask ratio $t$ and a random masking pattern.

**This is the single most important structural fact:** the diffusion loss is *just a reweighted
cross-entropy with no argmax*. The marginal likelihood $\log p_\theta(x_0)$ itself is **not** computed —
only the bound. That gap is Part III.

---

## Part III — The log-probability problem (why diffusion breaks RL)

RL (Part IV) needs $\log\pi_\theta(o\mid q)$. For AR models it's the exact chain-rule sum (Part I). For a
diffusion LM:

$$ \log\pi_\theta(o\mid q) = \log \sum_{\text{all unmasking orders / trajectories}} (\dots) \quad\text{— intractable.} $$

There is **no closed form** for the marginal probability of a specific sequence, because the model generates
tokens **in parallel, in arbitrary order, over many denoising steps**. We only have:
1. the **ELBO** (Part II.3): a *bound*, computed by Monte-Carlo over $(t, \text{mask})$ — unbiased-ish but
   **high variance**; or
2. a cheap **one-step / mean-field** approximation (Part V.2) — **biased** but a single forward pass.

> **The whole field of RL-for-diffusion-LLMs is the search for the right way to estimate this one quantity.**
> Every method named in Part VI is a different bias/variance/compute trade-off for $\log\pi_\theta(o\mid q)$.

---

## Part IV — RL foundations (objective → policy gradient → PPO → GRPO)

### IV.1 The objective
We have a reward $R(o)$ (e.g. "1 if the Countdown answer is correct"). We want a policy that produces
high-reward completions:

$$ J(\theta) = \mathbb{E}_{o\sim\pi_\theta(\cdot\mid q)}\big[R(o)\big]. $$

### IV.2 The policy gradient (REINFORCE) — derived
Differentiate and use the **log-derivative trick** $\nabla_\theta \pi_\theta = \pi_\theta \nabla_\theta\log\pi_\theta$:

$$ \nabla_\theta J = \sum_o \big(\nabla_\theta\pi_\theta(o)\big) R(o) = \sum_o \pi_\theta(o)\,\big(\nabla_\theta\log\pi_\theta(o)\big)R(o) = \mathbb{E}_{o\sim\pi_\theta}\!\big[R(o)\,\nabla_\theta\log\pi_\theta(o\mid q)\big]. $$

**Plain reading.** "Push up the log-probability of completions, weighted by their reward." High-reward
samples get their probability increased; this is the foundation of every method here. **Note it needs
$\nabla_\theta\log\pi_\theta$** — the log-prob again.

### IV.3 Baselines (variance reduction)
Because $\mathbb{E}[\nabla_\theta\log\pi_\theta]=0$, we may subtract any baseline $b$ without bias:

$$ \nabla_\theta J = \mathbb{E}\big[(R(o)-b)\,\nabla_\theta\log\pi_\theta(o\mid q)\big]. $$

A good $b$ (e.g. the average reward) drastically cuts variance: completions *better than average* get pushed
up, *worse than average* get pushed down. The quantity $A = R-b$ is the **advantage**.

### IV.4 PPO — take several steps per rollout, safely
Rollouts (generation) are expensive, so we want **many gradient steps per batch** of samples. But stepping
far from the policy that generated the data makes the gradient invalid. PPO fixes this with an **importance
ratio** $\rho = \pi_\theta/\pi_{\text{old}}$ and a **clip**:

$$ J_{\text{PPO}} = \mathbb{E}\Big[\min\big(\rho\,A,\ \operatorname{clip}(\rho,1-\varepsilon,1+\varepsilon)\,A\big)\Big]. $$

**Plain reading.** Improve as long as the new policy isn't too far from the old (ratio within $1\pm\varepsilon$);
the $\min$+clip removes the incentive to move further once you're at the edge. $\varepsilon$ is the trust
region half-width (d1 uses a loose $\varepsilon=0.5$).

### IV.5 GRPO — drop the critic, compare within a group
Standard PPO needs a learned **value network** (critic) to compute $A$. **GRPO** (Group Relative Policy
Optimization, Shao et al. 2024) removes it: sample a **group** of $G$ completions for the *same* prompt and
use the group as its own baseline.

$$ \textbf{advantage (standard GRPO):}\quad \hat A_i = \frac{r_i - \operatorname{mean}(r_{1{:}G})}{\operatorname{std}(r_{1{:}G})}. $$

$$ \boxed{\textbf{advantage (d1 / "Dr.GRPO" — what our code actually does):}\quad \hat A_i = r_i - \operatorname{mean}(r_{1{:}G})\ \ \text{(mean-centered only, NO }\div\operatorname{std}).} $$

> **Verified from d1's source**, not the paper prose: `advantages = rewards - mean_grouped_rewards`. The
> std is computed only to *log* a `zero_std_ratio` diagnostic. This matters — std-normalization changes the
> gradient scale. We confirmed this reading in `diffu_grpo_trainer.py`.

The full GRPO objective, per token, with PPO clipping and a KL leash to $\pi_{\text{ref}}$:

$$ \mathcal J_{\text{GRPO}}(\theta) = \mathbb{E}\!\left[\frac{1}{G}\sum_{i=1}^{G}\frac{1}{|o_i|}\sum_{t=1}^{|o_i|} \min\!\big(\rho_{i,t}\hat A_i,\ \operatorname{clip}(\rho_{i,t},1{-}\varepsilon,1{+}\varepsilon)\hat A_i\big)\right]\ -\ \beta\, \mathbb{D}_{\text{KL}}\!\big(\pi_\theta\,\|\,\pi_{\text{ref}}\big), $$

$$ \rho_{i,t} = \frac{\pi_\theta(o_{i,t}\mid q,o_{i,<t})}{\pi_{\text{old}}(o_{i,t}\mid q,o_{i,<t})}. $$

**The KL term** uses Schulman's low-variance, always-positive **k3 estimator** (let $s=\log\pi_{\text{ref}}-\log\pi_\theta$):

$$ \mathbb{D}_{\text{KL}} \approx \mathbb{E}\big[e^{s} - s - 1\big] \ \ge 0. $$

### IV.6 A subtlety we *measured*: loss $\approx 0$ but gradient $\neq 0$
Because advantages are mean-centered, $\sum_i \hat A_i = 0$. On the **first** inner step $\rho_{i,t}=1$
(since $\pi_\theta=\pi_{\text{old}}$), so the surrogate's *value* is $\sum_i \hat A_i \cdot 1 = 0$ → the
**reported loss is ~0**. But the *gradient* is $\sum_i \hat A_i\,\nabla_\theta\log\pi_\theta(o_i)\neq 0$ — it
still pushes high-$\hat A$ completions up and low ones down. In our tiny-RL smoke test we observed exactly
this: `loss 0.0, grad_norm 1.47`. **A zero loss here is correct, not a bug.**

---

## Part V — diffu-GRPO: GRPO on a diffusion policy

### V.1 The assembly
diffu-GRPO (d1, arXiv:2504.12216) is *exactly* the GRPO objective of Part IV.5, with one change: every
$\log\pi(\cdot)$ — needed for $\rho_{i,t}$ and the KL — is supplied by a **diffusion log-prob estimator**,
because Part III says we can't compute it exactly.

### V.2 d1's one-step (mean-field) estimator — the exact recipe
To score a completion $o$ given prompt $q$ in a **single forward pass**:
1. **Mask the entire completion**: replace all $L$ completion tokens with `[MASK]` → $o_{\text{mask}}$.
2. **(Optionally) perturb the prompt**: mask each prompt token independently w.p. $p_{\text{mask\_prompt}}=0.15$ → $q'$.
3. **One forward pass** $f_\theta([\,q';\,o_{\text{mask}}\,])$ gives a categorical $p_\theta(\cdot\mid\cdot)$ at each completion position.
4. **Read off** the log-prob of the *true* token at each position and sum:

$$ \widehat{\log\pi}_\theta(o\mid q) \;=\; \sum_{i=1}^{L}\log p_\theta\big(o_i \,\big|\, q',\,o_{\text{mask}}\big). $$

**Plain reading.** "Hide the whole answer, ask the model to guess every token at once, and sum how confident
it is about the true tokens." It is the $t\approx1$ (fully-masked) slice of the ELBO, and it treats the
answer tokens as **conditionally independent** given the masked context (a *mean-field* assumption).

**Why it's biased:** it ignores intra-answer dependencies and uses only the hardest (fully-masked) slice.
**We measured the bias** (`elbo_vs_onestep.py`): mean(ELBO $-$ one-step) $=+0.099$ (one-step slightly
*under*-estimates), and the per-token curve falls from $-0.06$ at $t{\le}0.5$ to $-0.21$ at $t{=}1$ —
exactly the predicted mechanism.

### V.3 Why a biased, high-variance estimator still trains (our key finding)
GRPO never uses the *absolute* log-prob — only the **ratio** $\rho_{i,t}=\pi_\theta/\pi_{\text{old}}$, with
**the same random mask seed** for numerator and denominator. So:

$$ \log\rho_{i,t} = \widehat{\log\pi}_\theta(o\mid q;\,\xi) - \widehat{\log\pi}_{\text{old}}(o\mid q;\,\xi), \qquad \xi=\text{shared mask pattern}. $$

The large masking noise $\xi$ is **common-mode** and cancels in the difference. We verified this directly:
$\mathrm{corr}\big(\widehat{\log\pi}^{\text{old}}_A,\widehat{\log\pi}^{\text{new}}_A\big)=1.0000$ at matched
seed, and the matched-seed log-ratio is $\sim 200\times$ quieter than the absolute estimate. **This is the
mathematical reason diffu-GRPO works despite a crude estimator.** (It also tells you what *not* to do:
comparing log-probs across *different* seeds or *different* prompts re-introduces the noise.)

### V.4 The reuse loop ($\mu$) and the hyperparameters
One expensive generation of $G$ completions is reused for $\mu$ inner gradient steps (PPO epochs); on step 1
$\rho=1$, then it drifts and the clip engages. d1 (Countdown) settings, all confirmed in config:

$$ G=6,\quad \mu=8,\quad \beta=0.04,\quad \varepsilon=0.5,\quad p_{\text{mask\_prompt}}=0.15,\quad \text{lr}=3\times10^{-6}. $$

### V.5 The zero-advantage trap (the practical wall — derived)
If **all $G$ completions get the same reward**, then $\hat A_i = 0\ \forall i$ → **zero gradient** → that
prompt teaches nothing. For a prompt where each completion succeeds independently with probability $p$:

$$ \Pr[\text{zero advantage}] \;=\; p^{G} + (1-p)^{G}. $$

For a weak base model on hard prompts ($p\to 0$): $(1-p)^G\to 1$ → almost always trapped. **Numbers**
($G{=}6$): $p{=}0.05\Rightarrow (0.95)^6\!=\!0.74$ trapped; raising to $G{=}16\Rightarrow (0.95)^{16}\!=\!0.44$;
$G{=}32\Rightarrow 0.19$. So **bigger $G$ rescues low-$p$ prompts** — a concrete, math-driven lever. We
observed this trap live: `zero_std_ratio=1.0` on ~half the rounds of our 32-prompt run, which capped reward
at ~0.35.

---

## Part VI — The estimator frontier (the real open problem)

As of mid-2026 there is **no consensus** on how to estimate $\log\pi_\theta(o\mid q)$ for GRPO; ≥6 proposals
in ~6 months. This is the live research frontier (and where Ermon is publishing — ESPO). Summary of the
trade-off each makes:

| method | arXiv | idea (one line) | trade-off |
|---|---|---|---|
| one-step / mean-field (d1) | 2504.12216 | mask all, 1 forward, sum token CE | cheap, **biased** |
| coupled-GRPO (DiffuCoder) | 2506.20639 | complementary mask pairs → full coverage | variance ↓ |
| ELBO + VR (VRPO) | 2505.19223 | ELBO + antithetic + optimal MC budget | unbiased-ish, var ↓ |
| GDPO | 2510.08554 | sequence ELBO + semi-deterministic MC | provably lower var |
| AGRPO | 2510.04019 | cache unmask order → **exact** per-step categorical | unbiased, costly |
| wd1 | 2507.08838 | **ratio-free** weighted log-likelihood | kills ratio variance |
| SPG | 2510.09541 | **sandwich** upper+lower bounds | bias ↓ |
| d2 | 2509.21474 | trajectory likelihood (exact 1-pass any-order) | SOTA (claimed) |
| ESPO (Ermon) | 2512.03759 | **sequence-level** action + ELBO proxy | — |

The bias↔variance↔compute triangle is unresolved. *Our* `elbo_vs_onestep` analysis sits exactly here, and a
clean **single-GPU, apples-to-apples comparison** of these estimators is the most honest contribution open
to us.

---

## Part VII — Diffusion-as-MDP (the DDPO view)

A complementary frame (DDPO, Black et al. 2023, arXiv:2305.13301) treats the **denoising trajectory itself**
as a Markov Decision Process — useful for per-step credit assignment and for the block case (Part VIII).

- **state** $s_t=(q,\,t,\,x_t)$ — prompt, step index, current partially-denoised sequence.
- **action** $a_t = x_{t-1}$ — the next (less-noisy) sequence.
- **policy** $\pi(a_t\mid s_t)=p_\theta(x_{t-1}\mid x_t,q)$ — one denoising step *is* one decision.
- **reward** $r(x_0)$ — sparse, only at the final clean sample.
- **trajectory likelihood factorizes over steps:** $\;p_\theta(x_{0:T}\mid q)=\prod_{t} p_\theta(x_{t-1}\mid x_t,q).$

**The trick that makes it tractable:** you never compute the marginal $p_\theta(x_0)$; you use **per-step**
log-probs $\log p_\theta(x_{t-1}\mid x_t)$. For continuous diffusion these are closed-form Gaussians; for
**discrete/text** diffusion each step is a **categorical** over $V$ per position. The policy gradient is
$\sum_t \nabla_\theta\log p_\theta(x_{t-1}\mid x_t)\,A$, with per-step importance ratios + PPO clip
(DDPO$_{\text{IS}}$). AGRPO (Part VI) is essentially this MDP view done exactly for masked text.

---

## Part VIII — Block diffusion: the two-level factorization

**Block diffusion** (BD3-LM, arXiv:2503.09573; SDAR; Inception's block-32) splits the sequence into $B$
blocks $x^{(1)},\dots,x^{(B)}$ and is **autoregressive across blocks, diffusion within a block**:

$$ p_\theta(x) = \prod_{b=1}^{B} \underbrace{p_\theta\big(x^{(b)}\mid x^{(<b)}\big)}_{\text{masked diffusion within block }b}, \qquad \log p_\theta(x)=\sum_{b=1}^{B}\log p_\theta\big(x^{(b)}\mid x^{(<b)}\big). $$

**Why this is interesting for RL:** the **outer** product over blocks is an ordinary AR factorization (the
tractable part), while each **inner** block term is a diffusion ELBO (the hard part, but over a *short* block
→ cheaper to estimate well). This gives a genuine design choice — **at what granularity does the GRPO
advantage/ratio live?**

$$ \text{block-level: } \sum_b \rho^{(b)}\hat A \quad\big|\quad \text{token-level: } \sum_b\sum_{j\in b}\rho_{b,j}\hat A \quad\big|\quad \text{step-level: } \sum_b\sum_\tau \rho_{b,\tau}\hat A. $$

StableDRL (arXiv:2603.06743) introduces **"staircase attention"** for *leakage-free* per-block log-prob
estimation (so block $b$'s probability doesn't peek at future blocks). **[UNVERIFIED whether the block-vs-token-vs-step
choice is settled in the literature — Agent C found no paper that cleanly resolves it for the semi-AR case;
this is the narrowest plausibly-open question, but pursuing it means racing active 2026 work.]**

---

## Part IX — The reward, and a full worked Countdown example

### IX.1 The Countdown reward (RLVR — rule-based, no learned reward model)
Given numbers (a multiset) $\mathcal N$ and target $T$, the model must output `<answer>EXPR</answer>`. From
`reward_func.countdown_reward_func` (we unit-tested this, 19/19):

$$ R(o) = \begin{cases} 1.0, & \texttt{EXPR}\text{ uses each number in }\mathcal N\text{ exactly once (multiset) and }\operatorname{eval}(\texttt{EXPR})=T\\ 0.1, & \texttt{EXPR}\text{ parses to a valid expression but is wrong / misuses numbers}\\ 0.0, & \text{no parseable }\texttt{<answer>}\ \text{(garbage)} \end{cases} $$

(Parsing: the **last** `<answer>` wins; evaluation is safe-`eval` over $\{+,-,\times,\div\}$.)

### IX.2 End-to-end, with numbers
**Prompt** $q$: "Using 30, 100, 93 create an expression equal to 23." $G=6$ rollouts, suppose rewards:

$$ r = [\,1.0,\ 1.0,\ 0.1,\ 0.1,\ 0.1,\ 0.0\,], \qquad \operatorname{mean}(r)=0.3833. $$

**Advantages** (d1 mean-centered): $\hat A = r-\operatorname{mean}(r)$:

$$ \hat A = [\,+0.617,\ +0.617,\ -0.283,\ -0.283,\ -0.283,\ -0.383\,], \qquad \textstyle\sum_i\hat A_i = 0\ \checkmark. $$

**Reading.** The two correct completions ($\hat A>0$) get their token log-probs pushed **up**; the three
"valid-but-wrong" and the garbage one get pushed **down**, hardest on the garbage ($\hat A=-0.383$). On inner
step 1, $\rho=1$ so reported loss $\approx0$; the gradient is nonzero and updates $\theta$; then $\mu{-}1$
more clipped steps refine on the same rollouts.

**Trap example.** If instead all six were valid-but-wrong, $r=[0.1,\dots,0.1]$, then $\operatorname{mean}=0.1$,
$\hat A=[0,\dots,0]$ → **no gradient** (Part V.5). This is what capped our 32-prompt run.

---

## Part X — Evaluation mathematics (and statistical honesty)

**Accuracy** on a held-out set of $N$ prompts:

$$ \text{acc} = \frac{1}{N}\sum_{n=1}^{N}\mathbb{1}\big[\text{model's answer to prompt }n\text{ is correct}\big]. $$

**Train-reward $\neq$ held-out accuracy.** Mean *training reward* (e.g. our $0.4$) includes the $0.1$
partial-credit floor and is measured on *seen* prompts; convert to an approximate correct-fraction via
$\text{frac\_correct}\approx (\bar r - 0.1)/0.9$. **Held-out accuracy** is the real generalization metric.

**Is a measured gap real?** A proportion has standard error

$$ \mathrm{SE} = \sqrt{\frac{\hat p(1-\hat p)}{N}}. $$

**Our case:** baseline $21.48\%$ vs adapter $25.78\%$ on $N=256$. $\mathrm{SE}\approx\sqrt{0.23\cdot0.77/256}\approx 0.026 = 2.6\%$.
The gap $+4.3\%$ is $\approx 1.6\,\mathrm{SE}$ → **suggestive, not significant**; it needs a 2–3 seed
replication before we trust it. (This is why we mark it `[needs seeds]` everywhere.)

---

## Part XI — Inference / sampling mathematics

Generation runs the reverse process from fully-masked to clean over `diffusion_steps` iterations:

1. Start $x_1 = [\mathbf m,\dots,\mathbf m]$ (length = `gen_length`).
2. For each step: forward pass → categorical per masked position → **fill in** predictions, then **remask**
   the least-confident fraction (`remasking="low_confidence"`) so they get reconsidered next step.
3. After `diffusion_steps` steps, all positions are committed → the answer.

**Cost.** $\textbf{NFE} \approx \texttt{diffusion\_steps}$ (one forward per step). For LLaDA the default ties
$\texttt{diffusion\_steps}=\texttt{gen\_length}/2$. **Semi-AR** generation does this **per block** of
`block_length` tokens, left to right, caching previous blocks (that's the block-32 design).

**Quality↔NFE.** More steps = higher sample quality but slower. Our **NfePareto** project measured this
"speed dial": generative perplexity drops steeply then **plateaus** past a knee — so for RL rollouts you pick
the smallest NFE on the plateau (we used $64$) to maximize throughput without hurting reward signal.

---

## Part XII — Glossary of every symbol
- $\pi_\theta,\pi_{\text{old}},\pi_{\text{ref}}$ — current / rollout-time / frozen-reference policies.
- $q,o,L$ — prompt, completion, completion length.
- $t,x_t,\mathbf m$ — diffusion time (mask ratio), masked sequence, mask token.
- $G$ — group size (completions per prompt).
- $r_i,\hat A_i$ — reward and (mean-centered) advantage of completion $i$.
- $\rho_{i,t}$ — importance ratio $\pi_\theta/\pi_{\text{old}}$ for token $t$ of completion $i$.
- $\varepsilon,\beta,\mu$ — PPO clip half-width; KL weight; inner-update count per rollout.
- $p_{\text{mask\_prompt}}$ — prompt-perturbation mask prob (0.15) in the estimator.
- NFE — forward passes per generation ($\approx$ `diffusion_steps`).
- ELBO — evidence lower bound (the diffusion training bound on $\log p$).
- $\mathbb D_{\text{KL}}$ — KL divergence (k3 estimator), the leash to $\pi_{\text{ref}}$.

## Part XIII — FAQ / common confusions
- **"Why not just compute $\log\pi$ exactly?"** Diffusion LMs have no tractable marginal sequence likelihood
  (Part III). That single fact creates the entire estimator frontier (Part VI).
- **"Loss is ~0 — is training broken?"** No (Part IV.6): mean-centered advantages make the surrogate *value*
  ~0 while the *gradient* is nonzero.
- **"Reward rose on training prompts — are we done?"** No (Part X): that's train-reward on *seen* prompts.
  Only held-out accuracy (vs the 21.48% baseline) counts, and it needs seed-confirmation.
- **"Is GRPO-on-block-diffusion novel?"** No — done at scale (TraceRL/TraDo, etc.; see the wiki landscape).
  The open part is the **estimator** and **minimal-compute**.
- **"Why does the crude one-step estimator work?"** Matched-seed common-mode cancellation in the ratio
  (Part V.3), which we verified at correlation $1.0000$.

## References (arXiv)
- d1 / diffu-GRPO — 2504.12216 · MDLM — 2406.07524 · DDPO — 2305.13301
- AGRPO — 2510.04019 · wd1 — 2507.08838 · SPG — 2510.09541 · GDPO — 2510.08554 · d2 — 2509.21474
- ESPO (Ermon) — 2512.03759 · VRPO/LLaDA-1.5 — 2505.19223 · DiffuCoder — 2506.20639
- Block: BD3-LM — 2503.09573 · TraceRL/TraDo — 2509.06949 · StableDRL — 2603.06743 · MMaDA/UniGRPO — 2505.15809
- Verification status + the full landscape: wiki `concepts/rl-for-diffusion-llms.md`. Our measured numbers: `FINDINGS.md`.

---

_Maintained alongside the project. If an equation here ever disagrees with the code, the code wins — fix the
doc and note it in `LOG.md`._
