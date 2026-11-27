"""
Exp (D): Scalability sweep for the new matrix-table (D) design.

Reuses the action / signal training infrastructure of train_exp9_scalability.py
but with two opponent-population builders that match the (B) anchors:

  • Action pathway against L6: N-1 frozen L6 leading-eight opponents.
  • Signal pathway against ProudCoop+AllDefector pool: round(p * (N-1))
    ProudCoopL3 + (N-1 - round(p * (N-1))) AllDefectL3, with at least one of each.

Sweep:
  • N ∈ {5, 10, 15}             (N=3 anchor lives in (B) itself)
  • Action: L6 only              (no ratio sweep)
  • Signal: p ∈ {0.2, 0.3, 0.5, 0.7, 0.8}
  • Strategy: per_opp / amortized
  • Seeds: 3
"""
import os
import sys
import json
import torch
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(HERE, '../..')))
sys.path.append(os.path.abspath(os.path.join(HERE, '..')))

from agent import Agent
from exp2_action_optimization.leading_eight_agents import LeadingEightAgent
from exp6_signal_opponent_modeling.train_exp6_signal_only import ProudCoopL3, AllDefectL3
sys.path.append(os.path.abspath(os.path.join(HERE, '..', 'exp4_joint_gt')))
from hybrid_opponents import HybridCoopL3
from checkpoint_utils import atomic_save_json, load_json_if_exists

# Reuse the run loop and helper utilities from the original exp9 script.
from train_exp9_scalability import (
    run_exp9 as _run_exp9_orig,
    BETA, B_VALUE,
)
import train_exp9_scalability as _exp9


# ══════════════════════════════════════════════════════════════════════════════
# New opponent-population builders matching (D) design
# ══════════════════════════════════════════════════════════════════════════════

def build_pop_L6_only(N: int, device):
    """N-1 frozen L6 leading-eight opponents (action-pathway sweep)."""
    n_opps = N - 1
    opps = []
    for i in range(n_opps):
        opp = LeadingEightAgent(name=f'OppL6_{i+1}', norm='L6', continuous=True,
                                 decay=0.5, beta=BETA, device=device)
        for p in opp.parameters():
            p.requires_grad_(False)
        opps.append(opp)
    return opps


def build_pop_PC_AD(N: int, pc_fraction: float, device):
    """ProudCoop + AllDefector pool. Counts: round(p*(N-1)) PC, rest AD; ≥1 each."""
    n_opps = N - 1
    n_pc = max(1, min(n_opps - 1, int(round(pc_fraction * n_opps))))
    n_ad = n_opps - n_pc
    opps = []
    for i in range(n_pc):
        opp = ProudCoopL3(name=f'PC_{i+1}', device=device)
        for p in opp.parameters():
            p.requires_grad_(False)
        opps.append(opp)
    for i in range(n_ad):
        opp = AllDefectL3(name=f'AD_{i+1}', device=device)
        for p in opp.parameters():
            p.requires_grad_(False)
        opps.append(opp)
    return opps, n_pc, n_ad


def build_pop_Hybrid_AD(N: int, hybrid_fraction: float, device):
    """HybridCoop + AllDefector pool. Same shape as PC+AD but with HybridCoopL3."""
    n_opps = N - 1
    n_hc = max(1, min(n_opps - 1, int(round(hybrid_fraction * n_opps))))
    n_ad = n_opps - n_hc
    opps = []
    for i in range(n_hc):
        opp = HybridCoopL3(name=f'HC_{i+1}', device=device)
        for p in opp.parameters():
            p.requires_grad_(False)
        opps.append(opp)
    for i in range(n_ad):
        opp = AllDefectL3(name=f'AD_{i+1}', device=device)
        for p in opp.parameters():
            p.requires_grad_(False)
        opps.append(opp)
    return opps, n_hc, n_ad


# ══════════════════════════════════════════════════════════════════════════════
# Patch the imported run_exp9 to use these new builders, controlled by opp_mode.
# ══════════════════════════════════════════════════════════════════════════════

_original_build_opp_pop = _exp9.build_opp_pop


def _build_opp_pop_dispatcher(N, device, mode='alternating', pool_fraction=0.5):
    if mode == 'L6_only':
        return build_pop_L6_only(N, device)
    if mode == 'PC_AD':
        opps, _, _ = build_pop_PC_AD(N, pool_fraction, device)
        return opps
    if mode == 'Hybrid_AD':
        opps, _, _ = build_pop_Hybrid_AD(N, pool_fraction, device)
        return opps
    return _original_build_opp_pop(N, device)


def _patch_per_agent_init(N, mode):
    """Patch the per-agent init list inside run_exp9 indirectly.

    For L6_only: all opps use L6 stationary distribution.
    For PC_AD: PC opps and AD opps both use uniform [0,1] init.
    For alternating (default): keep original alternating L3/L6.

    The original run_exp9 builds per_agent_init internally; we have to write a
    thin wrapper that overrides it. The cleanest path is to monkey-patch the
    builder at module level for the duration of the call.
    """
    # done in run_expD wrapper below
    pass


def run_expD(N: int, est_strategy: str, seed: int, pathway: str,
              pool_fraction: float = None) -> dict:
    """Drive the original exp9 training loop with a new opp-pop builder.

    pathway: 'action'  → L6-only opps
             'signal'  → ProudCoop+AllDefector pool, fraction = ProudCoop share
             'joint'   → HybridCoop+AllDefector pool, fraction = HybridCoop share
    """
    if pathway == 'action':
        mode = 'L6_only'; pool_fraction = None; c_val = 1.0
    elif pathway == 'signal':
        mode = 'PC_AD'; c_val = 5.0          # match Section 4.3.2 (sec:exp6)
        if pool_fraction is None:
            raise ValueError("pool_fraction required for signal pathway")
    elif pathway == 'joint':
        mode = 'Hybrid_AD'; c_val = 1.0       # match Section 4.3.3 (sec:exp7)
        if pool_fraction is None:
            raise ValueError("pool_fraction required for joint pathway")
    else:
        raise ValueError(f"Unsupported pathway: {pathway}")

    _exp9.build_opp_pop = lambda N_, device: _build_opp_pop_dispatcher(
        N_, device, mode=mode, pool_fraction=pool_fraction or 0.5)
    saved_c = _exp9.C_VALUE
    _exp9.C_VALUE = c_val

    try:
        history = _run_exp9_orig(N=N, est_strategy=est_strategy,
                                  seed=seed, pathway=pathway)
    finally:
        _exp9.build_opp_pop = _original_build_opp_pop
        _exp9.C_VALUE = saved_c

    history['opp_mode'] = mode
    history['c_value'] = c_val
    if pool_fraction is not None:
        history['pool_fraction'] = pool_fraction
        n_opps = N - 1
        n_main = max(1, min(n_opps - 1, int(round(pool_fraction * n_opps))))
        history['n_main'] = n_main
        history['n_ad'] = n_opps - n_main
    return history


# ══════════════════════════════════════════════════════════════════════════════
# Sweep config + main
# ══════════════════════════════════════════════════════════════════════════════

NS = [5, 10, 15]
POOL_FRACTIONS = [0.2, 0.3, 0.5, 0.7, 0.8]
NUM_SEEDS = 3
STRATEGIES = ['per_opp', 'amortized']


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--smoke', action='store_true',
                        help='quick smoke test: N=5, 1 seed, 1 ratio per pathway')
    parser.add_argument('--pathway', choices=['action', 'signal', 'joint', 'all'],
                        default='all')
    parser.add_argument('--seed-start', type=int, default=0)
    parser.add_argument('--seed-end', type=int, default=NUM_SEEDS)
    parser.add_argument('--json-suffix', default='')
    args = parser.parse_args()

    out_dir = os.path.join(HERE, 'data')
    os.makedirs(out_dir, exist_ok=True)
    json_path = os.path.join(out_dir,
                              f'results_expD{args.json_suffix}.json')

    if args.smoke:
        config_grid = []
        if args.pathway in ('action', 'all'):
            config_grid.append(('action', 5, 'per_opp', None))
        if args.pathway in ('signal', 'all'):
            config_grid.append(('signal', 5, 'per_opp', 0.5))
        if args.pathway in ('joint', 'all'):
            config_grid.append(('joint', 5, 'per_opp', 0.5))
        seed_range = [0]
    else:
        config_grid = []
        if args.pathway in ('action', 'all'):
            for N in NS:
                for strat in STRATEGIES:
                    config_grid.append(('action', N, strat, None))
        if args.pathway in ('signal', 'all'):
            for N in NS:
                for p in POOL_FRACTIONS:
                    for strat in STRATEGIES:
                        config_grid.append(('signal', N, strat, p))
        if args.pathway in ('joint', 'all'):
            for N in NS:
                for p in POOL_FRACTIONS:
                    for strat in STRATEGIES:
                        config_grid.append(('joint', N, strat, p))
        seed_range = list(range(args.seed_start, args.seed_end))

    n_total = len(config_grid) * len(seed_range)
    print(f"\n{'='*70}\n  Exp (D): Scalability sweep — {len(config_grid)} configs × "
          f"{len(seed_range)} seeds = {n_total} runs\n{'='*70}\n")

    all_results = load_json_if_exists(json_path) or {}

    for cfg_idx, (pathway, N, strat, p) in enumerate(config_grid):
        for seed in seed_range:
            if pathway == 'action':
                key = f"action_N{N}_{strat}_seed{seed}"
            else:
                key = f"{pathway}_N{N}_p{int(round(p*100)):02d}_{strat}_seed{seed}"

            if key in all_results:
                print(f"  ⏭  skip {key} (already done)")
                continue

            print(f"\n  [{cfg_idx+1}/{len(config_grid)}] {key}")
            try:
                hist = run_expD(N=N, est_strategy=strat, seed=seed,
                                 pathway=pathway, pool_fraction=p)
            except Exception as e:
                print(f"  ✗ failed {key}: {e}")
                import traceback; traceback.print_exc()
                continue

            slim = {k: v for k, v in hist.items()
                    if isinstance(v, (list, tuple, str, int, float))}
            all_results[key] = slim
            atomic_save_json(all_results, json_path)
            print(f"  ✓ saved → {json_path}")

    print(f"\n{'='*70}\n  All done — {len(all_results)} entries in {json_path}\n{'='*70}")


if __name__ == '__main__':
    main()
