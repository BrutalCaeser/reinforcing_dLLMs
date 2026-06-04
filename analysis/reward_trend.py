#!/usr/bin/env python3
"""Reward trend from a diffu-GRPO checkpoint's trainer_state.json.
Shows whether mean reward rises across epochs over a fixed prompt subset (Rung-A mechanism demo).
Usage: python analysis/reward_trend.py <path/to/trainer_state.json> [epoch_size=32]
"""
import json
import sys

d = json.load(open(sys.argv[1]))
ep = int(sys.argv[2]) if len(sys.argv) > 2 else 32
xs = [e["reward"] for e in d["log_history"] if "reward" in e]
n = len(xs)
mean = lambda a: sum(a) / len(a) if a else 0.0

print("global_step %s/%s | reward-logged rounds=%d (epoch_size=%d)"
      % (d.get("global_step"), d.get("max_steps"), n, ep))
if n:
    print("first10=%.3f  last10=%.3f  overall=%.3f  max=%.2f"
          % (mean(xs[:10]), mean(xs[-10:]), mean(xs), max(xs)))
    epochs = [round(mean(xs[i:i + ep]), 3) for i in range(0, n, ep)]
    print("per-epoch mean reward:", epochs)
    frac = [round(sum(1 for r in xs[i:i + ep] if r >= 0.9) / len(xs[i:i + ep]), 2)
            for i in range(0, n, ep)]
    print("per-epoch frac reward>=0.9:", frac)
