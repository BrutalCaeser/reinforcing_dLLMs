#!/usr/bin/env python3
"""
Phase 1 / Gate G1-RL — Check 1: d1's one-step log-prob estimator vs a brute-force ELBO.

WHY: diffu-GRPO's whole gradient rides on log p(completion | prompt). For a masked
diffusion LM there is no exact autoregressive factorization; the honest quantity is a
variational bound (ELBO) costing many forward passes. d1 replaces it with a ONE-step
shortcut: mask the entire completion, run one forward, sum per-token log p. That is the
t=1 (everything-masked) slice of the ELBO -> biased (each token predicted from zero
completion-context). This script measures, on the REAL LLaDA-8B-Instruct, how far the
shortcut drifts from the multi-t ELBO and -- the property GRPO actually needs -- whether
it preserves RANKING and is LOW-VARIANCE across mask seeds.

Two estimators of log p(c | prompt), both against the real model:

  one_step(p_mask_prompt):   mirror of diffu_grpo_trainer.forward_process + _get_per_token_logps
        mask ALL completion tokens -> [MASK]; mask each prompt token iid w.p. p_mask_prompt;
        ONE forward; logp_i = log softmax(logits_i)[true c_i] at completion slots; return sum.
        (p_mask_prompt=0.0 isolates the completion-masking approximation = the clean
         conditional; p_mask_prompt=0.15 reproduces d1's countdown training noise.)

  elbo(N):    MDM/LLaDA bound (Nie et al. 2025; Sahoo et al. 2024):
        log p(c|prompt) >= E_{t~U(0,1)} (1/t) E_{c_t} Sum_{i masked} log p_theta(c_i | prompt, c_t)
        prompt always visible; each completion token masked iid w.p. t; score ONLY masked
        positions; reweight by 1/t; Monte-Carlo over N draws of (t, mask). Reports mean +/- SE.

Also emits a per-t diagnostic curve (fixed-count masking, low variance) that visualizes the
bias mechanism: predicted per-token logp rises as t->0 (more context visible); the one-step
estimate is exactly the t=1 endpoint.

PASS (Gate G1-RL, estimator part):
  - Pearson(one_step@p0, elbo) over the base pairs >= 0.90   (tracks the real likelihood)
  - gold ranked #1 by BOTH estimators on the ranking set, Spearman >= 0.80
  - seed-variance of one_step << between-sequence spread     (per-iteration ratio is stable)
  Bias (elbo - one_step) is reported, expected > 0 (one-step underestimates); this is the
  known d1 caveat that wd1/AGRPO later unbias -- documented, not a failure.

Usage (on a GPU node, d1 env):  python src/elbo_vs_onestep.py [--elbo-samples 512]
"""
import argparse
import json
import os
import sys

import torch
import torch.nn.functional as F

MASK_ID = 126336  # LLaDA mask token (diffu_grpo_config default)


# ----------------------------------------------------------------------------- utils
def pearson(x, y):
    x = torch.tensor(x, dtype=torch.float64)
    y = torch.tensor(y, dtype=torch.float64)
    x = x - x.mean()
    y = y - y.mean()
    denom = (x.norm() * y.norm()).item()
    return (x @ y).item() / denom if denom > 0 else float("nan")


def spearman(x, y):
    def rank(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0] * len(v)
        for pos, idx in enumerate(order):
            r[idx] = pos
        return r
    return pearson(rank(x), rank(y))


# ------------------------------------------------------------------- the estimators
@torch.no_grad()
def one_step_logp(model, full_ids, prompt_len, p_mask_prompt, seed):
    """Exact mirror of d1's forward_process + per-token-logp on completion slots.
    Returns scalar sum_i log p(c_i | masked context)."""
    device = full_ids.device
    L = full_ids.size(1)
    g = torch.Generator(device="cpu").manual_seed(int(seed))
    prompt_index = torch.arange(L, device=device) < prompt_len  # [L] bool
    rand = torch.rand(L, generator=g).to(device)
    is_mask_prompt = prompt_index & (rand < p_mask_prompt)
    is_mask_completion = ~prompt_index                          # all completion masked
    is_mask = is_mask_prompt | is_mask_completion
    noisy = torch.where(is_mask.unsqueeze(0), torch.tensor(MASK_ID, device=device), full_ids)
    logits = model(noisy).logits[0]                             # [L, V]
    comp_logits = logits[prompt_len:].float()                  # [n_comp, V]
    comp_targets = full_ids[0, prompt_len:]                    # [n_comp]
    logp = -F.cross_entropy(comp_logits, comp_targets, reduction="none")  # [n_comp]
    return logp.sum().item()


@torch.no_grad()
def elbo_logp(model, full_ids, prompt_len, n_samples, batch=64, seed=0):
    """Brute-force MDM conditional ELBO of log p(completion | prompt).
    t ~ U(0,1); mask completion tokens iid w.p. t; score masked slots; weight 1/t; MC mean.
    Returns (mean, standard_error)."""
    device = full_ids.device
    L = full_ids.size(1)
    n_comp = L - prompt_len
    comp_targets = full_ids[0, prompt_len:]                    # [n_comp]
    g = torch.Generator(device="cpu").manual_seed(int(seed))
    per_sample = []
    done = 0
    while done < n_samples:
        b = min(batch, n_samples - done)
        t = torch.rand(b, generator=g).clamp_min(1e-3)         # [b] in (0,1]
        # iid Bernoulli mask over completion positions, per sample
        u = torch.rand(b, n_comp, generator=g)
        m = u < t.unsqueeze(1)                                  # [b, n_comp] bool
        seqs = full_ids.repeat(b, 1).clone()                   # [b, L]
        comp_block = seqs[:, prompt_len:]
        comp_block[m] = MASK_ID
        seqs[:, prompt_len:] = comp_block
        logits = model(seqs.to(device)).logits[:, prompt_len:, :].float()  # [b, n_comp, V]
        logp = -F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            comp_targets.repeat(b),
            reduction="none",
        ).view(b, n_comp)                                      # [b, n_comp]
        masked_sum = (logp * m.to(device)).sum(dim=1)          # [b] sum over masked slots
        contrib = masked_sum / t.to(device)                    # 1/t reweight
        per_sample.extend(contrib.tolist())
        done += b
    v = torch.tensor(per_sample, dtype=torch.float64)
    mean = v.mean().item()
    se = (v.std(unbiased=True) / (len(v) ** 0.5)).item()
    return mean, se


@torch.no_grad()
def per_t_curve(model, full_ids, prompt_len, t_grid, k=64, seed=0):
    """Diagnostic: mean per-token logp of masked slots vs mask ratio t (fixed count, low var).
    t=1 endpoint == one_step@p0 per-token. Shows the bias mechanism."""
    device = full_ids.device
    L = full_ids.size(1)
    n_comp = L - prompt_len
    comp_targets = full_ids[0, prompt_len:]
    g = torch.Generator(device="cpu").manual_seed(int(seed))
    curve = []
    for t in t_grid:
        m_count = max(1, round(t * n_comp))
        seqs = full_ids.repeat(k, 1).clone()
        masks = torch.zeros(k, n_comp, dtype=torch.bool)
        for j in range(k):
            idx = torch.randperm(n_comp, generator=g)[:m_count]
            masks[j, idx] = True
        comp_block = seqs[:, prompt_len:]
        comp_block[masks] = MASK_ID
        seqs[:, prompt_len:] = comp_block
        logits = model(seqs.to(device)).logits[:, prompt_len:, :].float()
        logp = -F.cross_entropy(
            logits.reshape(-1, logits.size(-1)), comp_targets.repeat(k), reduction="none"
        ).view(k, n_comp)
        per_tok = (logp * masks.to(device)).sum() / masks.to(device).sum()
        curve.append((round(float(t), 3), round(per_tok.item(), 4)))
    return curve


# --------------------------------------------------------------------------- toy data
BASE_PAIRS = [
    ("The capital of France is", " Paris."),
    ("Two plus two equals", " four."),
    ("The opposite of hot is", " cold."),
    ("Water is made of hydrogen and", " oxygen."),
    ("The sun rises in the", " east."),
    ("The first president of the United States was George", " Washington."),
]
RANK_PROMPT = "The capital of France is"
RANK_COMPLETIONS = [
    ("gold_paris", " Paris."),
    ("wrong_capital_berlin", " Berlin."),
    ("offtopic_pizza", " pizza."),
    ("gibberish", " qwx zzf."),
]


def tokenize(tok, prompt, completion):
    p = tok(prompt, add_special_tokens=True).input_ids
    c = tok(completion, add_special_tokens=False).input_ids
    full = torch.tensor([p + c], dtype=torch.long)
    return full, len(p), len(c)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", default=os.environ.get("MODEL", "GSAI-ML/LLaDA-8B-Instruct"))
    ap.add_argument("--elbo_samples", type=int, default=512)
    ap.add_argument("--seed_reps", type=int, default=16)
    ap.add_argument("--out", default="results/phase1_estimator.json")
    args = ap.parse_args()

    from transformers import AutoModel, AutoTokenizer

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={dev} model={args.model_path} elbo_samples={args.elbo_samples}")
    tok = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        args.model_path, trust_remote_code=True, torch_dtype=torch.bfloat16
    ).to(dev).eval()

    results = {"model": args.model_path, "elbo_samples": args.elbo_samples, "pairs": []}

    # ---- base pairs: one_step (p=0, p=0.15) vs elbo + per-t curve ------------------
    os_p0, os_p15, elbos = [], [], []
    print("\n=== base pairs: one_step vs ELBO (sum log p(completion|prompt)) ===")
    print(f"{'pair':34s} {'1step@p0':>10s} {'1step@.15':>10s} {'ELBO':>14s} {'comp_tok':>9s}")
    for prompt, comp in BASE_PAIRS:
        full, plen, clen = tokenize(tok, prompt, comp)
        full = full.to(dev)
        a = one_step_logp(model, full, plen, p_mask_prompt=0.0, seed=42)
        b = one_step_logp(model, full, plen, p_mask_prompt=0.15, seed=42)
        e, se = elbo_logp(model, full, plen, n_samples=args.elbo_samples)
        os_p0.append(a); os_p15.append(b); elbos.append(e)
        print(f"{prompt[:34]:34s} {a:10.3f} {b:10.3f} {e:8.3f}+-{se:4.2f} {clen:9d}")
        results["pairs"].append(
            {"prompt": prompt, "completion": comp, "n_comp": clen,
             "one_step_p0": a, "one_step_p15": b, "elbo": e, "elbo_se": se}
        )

    # diagnostic per-t curve on the first pair
    full, plen, _ = tokenize(tok, *BASE_PAIRS[0]); full = full.to(dev)
    curve = per_t_curve(model, full, plen, t_grid=[0.1, 0.25, 0.5, 0.75, 0.9, 1.0])
    print("\nper-t diagnostic (pair 0)  [per-token logp of masked slots vs t]:")
    print("  " + "  ".join(f"t={t}:{v}" for t, v in curve))
    results["per_t_curve_pair0"] = curve

    r_pearson = pearson(os_p0, elbos)
    bias = sum(e - a for e, a in zip(elbos, os_p0)) / len(elbos)
    print(f"\nPearson(one_step@p0, ELBO) = {r_pearson:.4f}")
    print(f"mean bias (ELBO - one_step@p0) = {bias:+.3f}  (>0 => one-step underestimates, expected)")

    # ---- ranking test --------------------------------------------------------------
    print("\n=== ranking test (prompt fixed; do both estimators agree on order?) ===")
    rank_os, rank_elbo, names = [], [], []
    for name, comp in RANK_COMPLETIONS:
        full, plen, clen = tokenize(tok, RANK_PROMPT, comp); full = full.to(dev)
        a = one_step_logp(model, full, plen, p_mask_prompt=0.0, seed=42)
        e, _ = elbo_logp(model, full, plen, n_samples=args.elbo_samples)
        rank_os.append(a); rank_elbo.append(e); names.append(name)
        print(f"  {name:22s} one_step={a:9.3f}  elbo={e:9.3f}")
    gold_os_top = names[max(range(len(rank_os)), key=lambda i: rank_os[i])] == "gold_paris"
    gold_elbo_top = names[max(range(len(rank_elbo)), key=lambda i: rank_elbo[i])] == "gold_paris"
    rank_sp = spearman(rank_os, rank_elbo)
    print(f"  gold ranked #1 by one_step={gold_os_top}  by elbo={gold_elbo_top}  Spearman={rank_sp:.3f}")
    results["ranking"] = {"names": names, "one_step": rank_os, "elbo": rank_elbo,
                          "gold_top_one_step": gold_os_top, "gold_top_elbo": gold_elbo_top,
                          "spearman": rank_sp}

    # ---- variance across mask seeds (stability the GRPO ratio relies on) -----------
    print("\n=== seed-variance of one_step@p0.15 (per-iteration mask noise) ===")
    full, plen, _ = tokenize(tok, *BASE_PAIRS[0]); full = full.to(dev)
    seed_vals = [one_step_logp(model, full, plen, p_mask_prompt=0.15, seed=s)
                 for s in range(args.seed_reps)]
    seed_std = torch.tensor(seed_vals).std(unbiased=True).item()
    between_std = torch.tensor(os_p0).std(unbiased=True).item()
    ratio = seed_std / between_std if between_std > 0 else float("inf")
    print(f"  seed-std={seed_std:.3f}  between-pair-std={between_std:.3f}  ratio={ratio:.3f}")
    results["variance"] = {"seed_std": seed_std, "between_pair_std": between_std, "ratio": ratio}

    # ---- verdict -------------------------------------------------------------------
    checks = {
        "pearson>=0.90": r_pearson >= 0.90,
        "gold_top_both": gold_os_top and gold_elbo_top,
        "rank_spearman>=0.80": rank_sp >= 0.80,
        "seed_var_ratio<0.34": ratio < 0.34,
    }
    results["checks"] = checks
    results["bias_elbo_minus_onestep_p0"] = bias
    results["pearson"] = r_pearson
    print("\n=== VERDICT (Gate G1-RL, estimator) ===")
    for k, v in checks.items():
        print(f"  [{'PASS' if v else 'FAIL'}] {k}")
    all_pass = all(checks.values())
    print(f"\nESTIMATOR TEST: {'ALL PASS' if all_pass else 'SOME FAILED'}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"wrote {args.out}")
    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
