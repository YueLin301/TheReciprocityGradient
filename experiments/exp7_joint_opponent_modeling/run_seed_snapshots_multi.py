"""
Multi-seed (0..4) version of the exp7 snapshot collector for the RG B.3 panel.
T_OUTER=125, default asymmetric LRs (π=3e-5, φ=3e-3). Resumable.
"""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import train_exp7_joint
train_exp7_joint.T_OUTER = 125
from train_exp7_joint import run_exp7, _strip_for_json

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                    'data', 'results_exp7_snapshots_multi.json')
os.makedirs(os.path.dirname(OUT), exist_ok=True)

results = json.load(open(OUT)) if os.path.exists(OUT) else {}

for seed in range(5):
    if str(seed) in results and len(results[str(seed)].get('pi_snapshots', [])) > 0:
        print(f'[skip] seed {seed} already done', flush=True)
        continue
    print(f'\n=== seed {seed} ===', flush=True)
    hist = run_exp7(opp_set='HybridCoopAllD', init_mode='all_uniform', seed=seed)
    results[str(seed)] = _strip_for_json(hist)
    json.dump(results, open(OUT, 'w'))
    print(f'  ✓ saved → {OUT}', flush=True)
    print(f'  seed {seed}: per-int={hist["final_perint"]:.3f}  std[pi]={hist["final_pi_var"]:.3f}  std[phi]={hist["final_phi_var"]:.3f}', flush=True)

print(f'\nDone. {len(results)} seeds saved.')
