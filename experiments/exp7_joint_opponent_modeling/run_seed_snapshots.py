"""
Re-run exp7 joint flagship for seeds 0 and 2, saving full pi/phi snapshots
to JSON for the policy-evolution panel of the main-text summary figure.
"""
import os, sys, json
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import train_exp7_joint
# Use T_OUTER=125 for x-axis alignment across exp5/6/7 snapshot figures.
train_exp7_joint.T_OUTER = 125
from train_exp7_joint import run_exp7

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data',
                    'results_exp7_snapshots.json')

from train_exp7_joint import _strip_for_json
# Force re-run with new T_OUTER (don't reuse previous T_OUTER=100 results).
results = {}
for seed in [0, 2]:
    print(f'\n=== seed {seed} ===', flush=True)
    hist = run_exp7(opp_set='HybridCoopAllD', init_mode='all_uniform', seed=seed)
    results[str(seed)] = _strip_for_json(hist)
    json.dump(results, open(OUT, 'w'))
    print(f'  ✓ saved → {OUT}', flush=True)

print(f'\nDone. {len(results)} seeds saved to {OUT}')
