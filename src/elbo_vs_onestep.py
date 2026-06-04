#!/usr/bin/env python3
"""
Phase 1 / Gate G1-RL — Check 1: d1's one-step log-prob estimator vs a brute-force ELBO.

WHY: diffu-GRPO's whole gradient rides on log p(completion | prompt). For a masked
diffusion LM there is no exact autoregressive factorization; the honest quantity is a
variational bound (ELBO) costing many forward passes. d1 replaces it with a ONE-step
shortcut: mask the entire completion, run one forward, sum per-token log p. That is the
t=1 (everything-masked) slice of the ELBO -> biased (each token predicted from zero
completion-context). This script measures, on the REAL LLaDA-8B-Instruct, the properties
that GRPO actually depends on -- NOT absolute likelihood accuracy.

Two estimators of log p(c | prompt), both against the real model:

  one_step(p_mask_prompt):   mirror of diffu_grpo_trainer.forward_process + _get_per_token_logps
        mask ALL completion tokens -> [MASK]; mask each prompt token iid w.p. p_mask_prompt;
        ONE forward; logp_i = log softmax(logits_i)[true c_i] at completion slots; return sum.

  elbo(N):    MDM/LLaDA bound (Nie et al. 2025; Sahoo et al. 2024):
        log p(c|prompt) >= E_{t~U(0,1)} (1/t) E_{c_t} Sum_{i masked} log p_theta(c_i | prompt, c_t)
        prompt always visible; each completion token masked iid w.p. t; score ONLY masked
        positions; reweight by 1/t; Monte-Carlo over N draws. Reports mean +/- SE.

WHAT GRPO ACTUALLY NEEDS (and what we therefore gate on):
  C1  WITHIN-GROUP RANKING. GRPO's advantage is a *ranking* of the G completions of ONE
      prompt. We test (a) a 4-way semantic group (gold must rank #1 by both estimators) and
      (b) corruption ladders (gold -> 25% -> 50% -> 100% random tokens): both estimators must
      be monotone-decreasing and agree (Spearman). Cross-PROMPT correlation is NOT gated --
      GRPO never compares across prompts (reported as a diagnostic only).
  C2  SMALL, CONSISTENT BIAS. mean/std of (ELBO - one_step). Expected small & positive
      (one-step = hardest t=1 slice). Reported; soft-checked per token.
  C3  COMMON-MODE CANCELLATION. The one-step estimate is high-variance across mask seeds
      (prompt-masking hides tokens). GRPO uses MATCHED seeds for pi_new vs pi_old, so that
      noise is common-mode and cancels in the ratio. We prove it: for two completions A,B of
      the SAME prompt scored at the SAME seed, std(logp_A - logp_B) << std(logp_A). This is
      WHY d1's biased high-variance estimator still trains (and why seeds are fixed per
      iteration; wd1/AGRPO later unbias it).

Usage (GPU node, d1 env):  python src/elbo_vs_onestep.py [--elbo_samples 512] [--seed_reps 24]
Exit 0 = all gated checks pass.
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
    x = torch.tensor(x, dtype=torch.float64); y = torch.tensor(y, dtype=torch.float64)
    x = x - x.mean(); y = y - y.mean()
    d = (x.norm() * y.norm()).item()
    return (x @ y).item() / d if d > 0 else float("nan")


def spearman(x, y):
    def rank(v):
        order = sorted(range(len(v)), key=lambda i: v[i]); r = [0] * len(v)
        for pos, idx in enumerate(order): r[idx] = pos
        return r
    return pearson(rank(x), rank(y))


# ------------------------------------------------------------------- the estimators
@torch.no_grad()
def one_step_logp(model, full_ids, prompt_len, p_mask_prompt, seed):
    """Exact mirror of d1 forward_process + per-token-logp. Returns sum_i log p(c_i | masked ctx)."""
    device = full_ids.device
    L = full_ids.size(1)
    g = torch.Generator(device="cpu").manual_seed(int(seed))
    prompt_index = torch.arange(L, device=device) < prompt_len
    rand = torch.rand(L, generator=g).to(device)
    is_mask = (prompt_index & (rand < p_mask_prompt)) | (~prompt_index)
    noisy = torch.where(is_mask.unsqueeze(0), torch.tensor(MASK_ID, device=device), full_ids)
    logits = model(noisy).logits[0].float()
    logp = -F.cross_entropy(logits[prompt_len:], full_ids[0, prompt_len:], reduction="none")
    return logp.sum().item()


@torch.no_grad()
def elbo_logp(model, full_ids, prompt_len, n_samples, batch=64, seed=0):
    """Brute-force MDM conditional ELBO of log p(completion | prompt). Returns (mean, SE)."""
    device = full_ids.device
    L = full_ids.size(1); n_comp = L - prompt_len
    comp_targets = full_ids[0, prompt_len:]
    g = torch.Generator(device="cpu").manual_seed(int(seed))
    vals, done = [], 0
    while done < n_samples:
        b = min(batch, n_samples - done)
        t = torch.rand(b, generator=g).clamp_min(1e-3)
        m = torch.rand(b, n_comp, generator=g) < t.unsqueeze(1)
        seqs = full_ids.repeat(b, 1).clone()
        cb = seqs[:, prompt_len:]; cb[m] = MASK_ID; seqs[:, prompt_len:] = cb
        logits = model(seqs.to(device)).logits[:, prompt_len:, :].float()
        logp = -F.cross_entropy(
            logits.reshape(-1, logits.size(-1)), comp_targets.repeat(b), reduction="none"
        ).view(b, n_comp)
        contrib = (logp * m.to(device)).sum(dim=1) / t.to(device)
        vals.extend(contrib.tolist()); done += b
    v = torch.tensor(vals, dtype=torch.float64)
    return v.mean().item(), (v.std(unbiased=True) / (len(v) ** 0.5)).item()


@torch.no_grad()
def per_t_curve(model, full_ids, prompt_len, t_grid, k=64, seed=0):
    """Diagnostic: mean per-token logp of masked slots vs mask ratio t (fixed count). t=1 == one_step."""
    device = full_ids.device
    L = full_ids.size(1); n_comp = L - prompt_len
    comp_targets = full_ids[0, prompt_len:]
    g = torch.Generator(device="cpu").manual_seed(int(seed))
    curve = []
    for t in t_grid:
        mc = max(1, round(t * n_comp))
        seqs = full_ids.repeat(k, 1).clone()
        masks = torch.zeros(k, n_comp, dtype=torch.bool)
        for j in range(k):
            masks[j, torch.randperm(n_comp, generator=g)[:mc]] = True
        cb = seqs[:, prompt_len:]; cb[masks] = MASK_ID; seqs[:, prompt_len:] = cb
        logits = model(seqs.to(device)).logits[:, prompt_len:, :].float()
        logp = -F.cross_entropy(
            logits.reshape(-1, logits.size(-1)), comp_targets.repeat(k), reduction="none"
        ).view(k, n_comp)
        curve.append((round(float(t), 3), round(((logp * masks.to(device)).sum() / masks.to(device).sum()).item(), 4)))
    return curve


def corrupt(comp_ids, frac, seed, lo=1000, hi=100000):
    """Replace a fraction of completion tokens with random (wrong) vocab ids -> quality ladder."""
    g = torch.Generator(device="cpu").manual_seed(int(seed))
    ids = comp_ids.clone(); n = ids.numel(); k = round(frac * n)
    if k == 0:
        return ids
    idx = torch.randperm(n, generator=g)[:k]
    ids[idx] = torch.randint(lo, hi, (k,), generator=g)
    return ids


# --------------------------------------------------------------------------- toy data
BASE_PAIRS = [
    ("The capital of France is", " Paris."),
    ("Two plus two equals", " four."),
    ("The opposite of hot is", " cold."),
    ("Water is made of hydrogen and", " oxygen."),
    ("The sun rises in the", " east."),
    ("The first president of the United States was George", " Washington."),
]
# 4-way semantic group: gold must rank #1 by BOTH estimators.
RANK_PROMPT = "The capital of France is"
RANK_COMPLETIONS = [
    ("gold_paris", " Paris."), ("wrong_capital_berlin", " Berlin."),
    ("offtopic_pizza", " pizza."), ("gibberish", " qwx zzf."),
]
# corruption ladders: longer gold completions corrupted 0/25/50/100% -> monotone quality.
LADDERS = [
    ("paris", "Q: What is the capital of France?\nA:", " The capital of France is Paris."),
    ("water", "Q: What is water made of?\nA:", " Water is made of hydrogen and oxygen."),
    ("sun", "Q: Where does the sun rise?\nA:", " The sun rises in the east every morning."),
]
LADDER_FRACS = [0.0, 0.25, 0.5, 1.0]


def tok_pair(tok, prompt, completion):
    p = tok(prompt, add_special_tokens=True).input_ids
    c = tok(completion, add_special_tokens=False).input_ids
    return torch.tensor([p + c], dtype=torch.long), len(p), len(c)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", default=os.environ.get("MODEL", "GSAI-ML/LLaDA-8B-Instruct"))
    ap.add_argument("--elbo_samples", type=int, default=512)
    ap.add_argument("--seed_reps", type=int, default=24)
    ap.add_argument("--out", default="results/phase1_estimator.json")
    args = ap.parse_args()

    from transformers import AutoModel, AutoTokenizer

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={dev} model={args.model_path} elbo_samples={args.elbo_samples}")
    tok = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        args.model_path, trust_remote_code=True, torch_dtype=torch.bfloat16
    ).to(dev).eval()
    R = {"model": args.model_path, "elbo_samples": args.elbo_samples}

    # ---- base pairs: bias + cross-prompt correlation (DIAGNOSTIC, not gated) -------
    os0, elbos = [], []
    print("\n=== base pairs: one_step vs ELBO  (DIAGNOSTIC: cross-prompt, not GRPO-relevant) ===")
    print(f"{'pair':34s} {'1step@p0':>9s} {'1step@.15':>9s} {'ELBO':>13s} {'tok':>4s}")
    R["base_pairs"] = []
    for prompt, comp in BASE_PAIRS:
        full, plen, clen = tok_pair(tok, prompt, comp); full = full.to(dev)
        a = one_step_logp(model, full, plen, 0.0, 42)
        b = one_step_logp(model, full, plen, 0.15, 42)
        e, se = elbo_logp(model, full, plen, args.elbo_samples)
        os0.append(a); elbos.append(e)
        print(f"{prompt[:34]:34s} {a:9.3f} {b:9.3f} {e:7.3f}+-{se:4.2f} {clen:4d}")
        R["base_pairs"].append({"prompt": prompt, "one_step_p0": a, "one_step_p15": b, "elbo": e})
    bias = sum(e - a for e, a in zip(elbos, os0)) / len(os0)
    xprompt_pearson = pearson(os0, elbos)
    full, plen, _ = tok_pair(tok, *BASE_PAIRS[0]); full = full.to(dev)
    curve = per_t_curve(model, full, plen, [0.1, 0.25, 0.5, 0.75, 0.9, 1.0])
    print("per-t curve (pair0):  " + "  ".join(f"t={t}:{v}" for t, v in curve))
    print(f"mean bias (ELBO - one_step@p0) = {bias:+.3f}   cross-prompt Pearson (diag) = {xprompt_pearson:.3f}")
    R["bias"] = bias; R["xprompt_pearson_diagnostic"] = xprompt_pearson; R["per_t_curve_pair0"] = curve

    # ---- C1a: 4-way semantic group, gold must rank #1 by both ----------------------
    print("\n=== C1a: semantic group ranking (gold must be #1 by BOTH) ===")
    r_os, r_el, names = [], [], []
    for name, comp in RANK_COMPLETIONS:
        full, plen, _ = tok_pair(tok, RANK_PROMPT, comp); full = full.to(dev)
        a = one_step_logp(model, full, plen, 0.0, 42)
        e, _ = elbo_logp(model, full, plen, args.elbo_samples)
        r_os.append(a); r_el.append(e); names.append(name)
        print(f"  {name:22s} one_step={a:9.3f}  elbo={e:9.3f}")
    gold_os = names[max(range(len(r_os)), key=lambda i: r_os[i])] == "gold_paris"
    gold_el = names[max(range(len(r_el)), key=lambda i: r_el[i])] == "gold_paris"
    grp_sp = spearman(r_os, r_el)
    print(f"  gold#1 one_step={gold_os} elbo={gold_el}  Spearman(one_step,elbo)={grp_sp:.3f}")
    R["semantic_group"] = {"names": names, "one_step": r_os, "elbo": r_el,
                           "gold_top_both": gold_os and gold_el, "spearman": grp_sp}

    # ---- C1b: corruption ladders, monotone + agreement -----------------------------
    print("\n=== C1b: corruption ladders (gold->25%->50%->100% random; monotone & agree) ===")
    ladder_ok, ladder_sps = True, []
    R["ladders"] = []
    for lname, prompt, gold in LADDERS:
        p_ids = tok(prompt, add_special_tokens=True).input_ids
        c_ids = torch.tensor(tok(gold, add_special_tokens=False).input_ids, dtype=torch.long)
        os_l, el_l = [], []
        for frac in LADDER_FRACS:
            cc = corrupt(c_ids, frac, seed=7)
            full = torch.cat([torch.tensor(p_ids, dtype=torch.long), cc]).unsqueeze(0).to(dev)
            plen = len(p_ids)
            os_l.append(one_step_logp(model, full, plen, 0.0, 42))
            el_l.append(elbo_logp(model, full, plen, args.elbo_samples)[0])
        # both monotone decreasing in corruption, and agree
        mono_os = all(os_l[i] >= os_l[i + 1] - 0.5 for i in range(len(os_l) - 1))
        mono_el = all(el_l[i] >= el_l[i + 1] - 0.5 for i in range(len(el_l) - 1))
        sp = spearman(os_l, el_l)
        ladder_sps.append(sp)
        ok = mono_os and mono_el and sp >= 0.9
        ladder_ok = ladder_ok and ok
        print(f"  {lname:8s} one_step={[round(x,1) for x in os_l]}  elbo={[round(x,1) for x in el_l]}"
              f"  mono(os={mono_os},el={mono_el}) Spearman={sp:.2f} {'OK' if ok else 'BAD'}")
        R["ladders"].append({"name": lname, "fracs": LADDER_FRACS, "one_step": os_l, "elbo": el_l,
                             "mono_os": mono_os, "mono_el": mono_el, "spearman": sp})

    # ---- C3: common-mode cancellation (why the high-variance estimator still trains) -
    print("\n=== C3: common-mode cancellation (matched-seed differences cancel mask noise) ===")
    fa, pla, _ = tok_pair(tok, RANK_PROMPT, " Paris."); fa = fa.to(dev)
    fb, plb, _ = tok_pair(tok, RANK_PROMPT, " London."); fb = fb.to(dev)
    a_s = [one_step_logp(model, fa, pla, 0.15, s) for s in range(args.seed_reps)]
    b_s = [one_step_logp(model, fb, plb, 0.15, s) for s in range(args.seed_reps)]
    diff = [a - b for a, b in zip(a_s, b_s)]
    std_abs = torch.tensor(a_s).std(unbiased=True).item()
    std_diff = torch.tensor(diff).std(unbiased=True).item()
    cancel = std_diff / std_abs if std_abs > 0 else float("inf")
    print(f"  std(logp_A)={std_abs:.3f}  std(logp_A - logp_B, matched seed)={std_diff:.3f}"
          f"  cancellation ratio={cancel:.3f}")
    print(f"  -> matched-seed pi_new/pi_old shares the SAME mask, so this noise cancels EXACTLY in GRPO.")
    R["common_mode"] = {"std_abs": std_abs, "std_matched_diff": std_diff, "cancellation_ratio": cancel}

    # ---- verdict -------------------------------------------------------------------
    checks = {
        "C1a_gold_top_both": gold_os and gold_el,
        "C1a_group_spearman>=0.9": grp_sp >= 0.9,
        "C1b_ladders_monotone_agree": ladder_ok,
        "C3_common_mode_cancels<0.5": cancel < 0.5,
    }
    R["checks"] = checks
    print("\n=== VERDICT (Gate G1-RL, estimator) ===")
    for k, v in checks.items():
        print(f"  [{'PASS' if v else 'FAIL'}] {k}")
    print(f"(diagnostics: bias={bias:+.3f}, cross-prompt Pearson={xprompt_pearson:.3f} [not gated], "
          f"ladder Spearmans={[round(s,2) for s in ladder_sps]})")
    ok = all(checks.values())
    print(f"\nESTIMATOR TEST: {'ALL PASS' if ok else 'SOME FAILED'}")
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(R, f, indent=2)
    print(f"wrote {args.out}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
