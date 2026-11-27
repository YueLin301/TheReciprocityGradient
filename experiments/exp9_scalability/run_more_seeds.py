"""Run additional seeds (seed_start..seed_end) on the amortized configs only.

The default 3-seed run (seeds 0,1,2) leaves visible variance at N=10 in the
action-vs-L6 setting and a U-shape across N for the joint/signal anchors.
This driver extends the seed pool so the table values stabilize.

Targets only `amortized` (the strategy the paper-text table reports).
Calls run_expD from train_expD; results are merged into the same JSON file
that train_expD writes, so the existing aggregation pipeline keeps working.
"""
import argparse
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.abspath(os.path.join(HERE, '..')))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, '..', '..')))

from train_expD import run_expD
from checkpoint_utils import atomic_save_json, load_json_if_exists

NS = [5, 10, 15]
POOL_FRACTIONS = [0.2, 0.3, 0.5, 0.7, 0.8]


def build_grid(scope: str):
    grid = []
    n_only = scope in ('n10', 'n10_no_action')
    if scope in ('action', 'all', 'n10'):
        for N in NS if not n_only else [10]:
            grid.append(('action', N, None))
    if scope in ('signal', 'all', 'n10', 'n10_no_action'):
        for N in NS if not n_only else [10]:
            for p in POOL_FRACTIONS:
                grid.append(('signal', N, p))
    if scope in ('joint', 'all', 'n10', 'n10_no_action'):
        for N in NS if not n_only else [10]:
            for p in POOL_FRACTIONS:
                grid.append(('joint', N, p))
    return grid


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--scope', choices=['action', 'signal', 'joint', 'n10', 'n10_no_action', 'all'],
                    default='all')
    ap.add_argument('--seed-start', type=int, default=3)
    ap.add_argument('--seed-end', type=int, default=10)
    args = ap.parse_args()

    json_path = os.path.join(HERE, 'data', 'results_expD.json')
    all_results = load_json_if_exists(json_path) or {}

    grid = build_grid(args.scope)
    seeds = list(range(args.seed_start, args.seed_end))
    todo = []
    for pathway, N, p in grid:
        for seed in seeds:
            if pathway == 'action':
                key = f"action_N{N}_amortized_seed{seed}"
            else:
                key = f"{pathway}_N{N}_p{int(round(p*100)):02d}_amortized_seed{seed}"
            if key not in all_results:
                todo.append((pathway, N, p, seed, key))

    print(f"\n{'='*70}")
    print(f"  scope={args.scope}, seeds {args.seed_start}..{args.seed_end-1}")
    print(f"  {len(todo)} runs to do (amortized only)")
    print(f"{'='*70}\n")

    t_start = time.time()
    for i, (pathway, N, p, seed, key) in enumerate(todo):
        t0 = time.time()
        print(f"[{i+1}/{len(todo)}] {key}", flush=True)
        try:
            hist = run_expD(N=N, est_strategy='amortized', seed=seed,
                            pathway=pathway, pool_fraction=p)
        except Exception as e:
            print(f"  FAIL {key}: {e}", flush=True)
            import traceback; traceback.print_exc()
            continue
        slim = {k: v for k, v in hist.items()
                if isinstance(v, (list, tuple, str, int, float))}
        all_results[key] = slim
        atomic_save_json(all_results, json_path)
        elapsed = time.time() - t0
        total = time.time() - t_start
        print(f"  ok in {elapsed:.1f}s; total {total/60:.1f} min", flush=True)

    print(f"\nDONE — {len(todo)} runs in {(time.time()-t_start)/60:.1f} min")


if __name__ == '__main__':
    main()
