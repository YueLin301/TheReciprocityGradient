"""
Multi-seed (0..4) version of the exp5e snapshot collector for the RG B.1 panel.
T_OUTER=125, LR=3e-5, N_TRAIN=50 (matches exp7 π config). Resumable.
"""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import train_exp5e
train_exp5e.T_OUTER = 125
train_exp5e.LR_POLICY = 3e-5
train_exp5e.N_TRAIN = 50
from train_exp5e import run_exp5e, _strip

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                    'data', 'results_5e_snapshot_multi.json')
os.makedirs(os.path.dirname(OUT), exist_ok=True)

results = json.load(open(OUT)) if os.path.exists(OUT) else {}

for seed in range(5):
    if str(seed) in results and len(results[str(seed)].get('pi_snapshots', [])) > 0:
        print(f'[skip] seed {seed} already done', flush=True)
        continue
    print(f'\n=== seed {seed} ===', flush=True)
    result = run_exp5e(seed=seed)
    results[str(seed)] = _strip(result)
    json.dump(results, open(OUT, 'w'))
    print(f'  ✓ saved → {OUT}', flush=True)
    print(f'  seed {seed}: per-int={result["final_perint"]:.3f} std[pi]={result["final_pi_var"]:.3f}', flush=True)

print(f'\nDone. {len(results)} seeds saved.')
