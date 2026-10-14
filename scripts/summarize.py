"""Speedup and acceptance length from paired runs of scripts/run.py.

Speedup is the mean over requests of TPOT_AR / TPOT_method, where each method is
paired with an AR run of the same requests on the same GPU. The average over
tasks weights tasks equally. 95% confidence intervals use 2,000 bootstrap
resamples of paired requests, drawn within each task.

    python scripts/summarize.py --ar ar.jsonl --method DFlash=dflash.jsonl --method FOLD=fold.jsonl
"""
import argparse
import json
from collections import defaultdict

import numpy as np


def load(path):
    return {r["id"]: r for r in map(json.loads, open(path)) if r.get("tpot")}


def paired(ar, method):
    tasks = defaultdict(lambda: ([], []))
    for rid, r in method.items():
        if rid in ar:
            ratio, tau = tasks[r.get("task") or "all"]
            ratio.append(ar[rid]["tpot"] / r["tpot"])
            tau.append(r["tau"] if r["tau"] is not None else np.nan)
    return {t: (np.array(a), np.array(b)) for t, (a, b) in sorted(tasks.items())}


def bootstrap_avg(groups, n, seed):
    rng = np.random.default_rng(seed)
    draws = np.empty(n)
    for i in range(n):
        draws[i] = np.mean([g[rng.integers(0, len(g), len(g))].mean() for g in groups])
    return np.percentile(draws, [2.5, 97.5])


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ar", required=True)
    p.add_argument("--method", action="append", required=True, metavar="NAME=PATH")
    p.add_argument("--resamples", type=int, default=2000)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    ar = load(args.ar)
    for spec in args.method:
        name, path = spec.split("=", 1)
        tasks = paired(ar, load(path))
        print(f"\n{name}")
        print(f"  {'task':<24}{'n':>6}{'tau':>8}{'speedup':>10}")
        for task, (ratio, tau) in tasks.items():
            print(f"  {task:<24}{len(ratio):>6}{np.nanmean(tau):>8.2f}{ratio.mean():>9.2f}x")
        ratios = [r for r, _ in tasks.values()]
        lo, hi = bootstrap_avg(ratios, args.resamples, args.seed)
        avg_tau = np.mean([np.nanmean(t) for _, t in tasks.values()])
        avg = np.mean([r.mean() for r in ratios])
        print(f"  {'Avg.':<24}{sum(map(len, ratios)):>6}{avg_tau:>8.2f}{avg:>9.2f}x  95% CI [{lo:.2f}, {hi:.2f}]")


if __name__ == "__main__":
    main()
