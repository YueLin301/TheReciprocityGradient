"""
Run exp5e for seed 4 (best disc seed) to obtain pi snapshots.
Saves to a separate JSON to avoid disturbing the main results file.
"""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import train_exp5e
# Match exp7's π config exactly: T_OUTER=125, LR=3e-5, N_TRAIN=50.
train_exp5e.T_OUTER = 125
train_exp5e.LR_POLICY = 3e-5
train_exp5e.N_TRAIN = 50
from train_exp5e import run_exp5e, _strip

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                    'data', 'results_5e_snapshot.json')
os.makedirs(os.path.dirname(OUT), exist_ok=True)

# Seed 4 — best disc per existing 10-seed sweep
result = run_exp5e(seed=4)
data = {'4': _strip(result)}
json.dump(data, open(OUT, 'w'))
print(f'\n✓ saved → {OUT}')
print(f'  per-int = {result["final_perint"]:.3f}')
print(f'  std[pi] = {result["final_pi_var"]:.3f}')
print(f'  n_snapshots = {len(result["pi_snapshots"])}')
