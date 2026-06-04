# How this project is documented (and kept current)

Documentation here is a **first-class artifact, written in parallel with the work — not after it.**
The goal: anyone (a collaborator, a reviewer, or future-me) can open this repo cold and understand
*what* was done, *why*, *what was found*, and *how to rerun it* — without a verbal handoff.

## Reading order for a newcomer
1. **`README.md`** — what the project is, the one hard part (the log-prob estimator), current status.
2. **`SPEC.md`** — the recon-grounded execution plan: the verified d1 recipe, the compute go/no-go, the Rung ladder.
3. **`FINDINGS.md`** — every gate's numbers and the honest interpretation (the actual results).
4. **`LOG.md`** — the chronological engineering trail (what ran where, and why each decision was made).
5. **`UPSTREAM.md`** — exact pins (d1 commit, model, env versions) so the stack is reconstructable.
6. **`src/` docstrings** — each script opens with *why it exists and what it proves*, not just what it does.

## The documentation surfaces and when each is updated

| File | Role | Updated |
|---|---|---|
| `README.md` | Front door: overview, status table, reproduce steps, repo map | when status/gates change, or layout changes |
| `SPEC.md` | The plan + the gates + the decisions | when the plan or a gate decision changes |
| `LOG.md` | Append-only engineering log, **newest on top** | **every work session** — what ran, the result, the decision |
| `FINDINGS.md` | Living results & analysis, per gate | **whenever a result lands** (even partial/negative) |
| `UPSTREAM.md` | Pinned upstreams + env versions | when a pin or version changes |
| `src/*.py` | Module docstring states the *why*; functions documented | with the code, same commit |
| `exp/*.sbatch` | Header comment: what it runs, how to submit | with the script |

## The discipline (non-negotiable, inherited from SPEC §"grounding rules")
1. **No code against an unread API** — read the upstream source first (d1, LLaDA).
2. **Every number is reproducible** from a committed script + a command logged in `LOG.md`.
3. **`[ESTIMATE]`** marks any guessed quantity (especially compute); never dressed up as measured.
4. **Negative & surprising results are reported honestly** — e.g. the Phase-1 finding that two naive
   estimator gates were *mis-specified* is documented in full, not quietly dropped.
5. **A `LOG.md` entry every session; small, conventional commits** (`exp(p1): ...`, `docs: ...`).
6. **Estimator + reward functions are unit-tested before any training run.**

## Public/private split
This repo is **public**. It contains only the technical reproduction: methods, code, numbers, analysis.
No credentials, no private infrastructure paths, no third-party-confidential claims, and no unverified
assertions about any commercial system's internal recipe. Strategic/personal context is kept elsewhere.

## Commit hygiene
- Conventional-commit prefixes: `exp(<phase>)`, `docs`, `chore`, `fix`, `env`.
- Body explains *why*, not just *what* — a commit is a documentation surface too.
- Large artifacts (checkpoints, raw generations) stay on `/scratch` and are gitignored; small result
  summaries (JSON, curves, tables) are committed so the findings are self-contained.
