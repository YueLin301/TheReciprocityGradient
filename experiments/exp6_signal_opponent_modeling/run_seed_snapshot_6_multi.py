"""
Multi-seed (0..4) version of the exp6 snapshot collector for the RG B.2 panel.
T_OUTER=125, LR=8e-3 (decelerated for visual coherence with B.1/B.3). Resumable.
"""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import train_exp6_signal_only
train_exp6_signal_only.T_OUTER = 125
train_exp6_signal_only.LR_POLICY = 8e-3
from train_exp6_signal_only import run_exp6

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                    'data', 'results_6_snapshot_multi.json')
os.makedirs(os.path.dirname(OUT), exist_ok=True)

results = json.load(open(OUT)) if os.path.exists(OUT) else {}
results.setdefault('MLP_all_uniform', {})

for seed in range(5):
    if str(seed) in results['MLP_all_uniform'] and \
       len(results['MLP_all_uniform'][str(seed)].get('signal_snapshots', [])) > 0:
        print(f'[skip] seed {seed} already done', flush=True)
        continue
    print(f'\n=== seed {seed} ===', flush=True)
    result = run_exp6(arch='MLP', init_mode='all_uniform', seed=seed)
    keep = ['real_perint', 'surrogate_perint', 'real_perint_per_ep', 'surrogate_perint_per_ep',
            'act_est_loss_1', 'act_est_loss_2', 'sig_est_loss_1', 'sig_est_loss_2',
            'spread_log', 'rep_log', 'signal_snapshots',
            'final_perint', 'final_signal_profile', 'final_signal_spread', 'final_disc_type']
    results['MLP_all_uniform'][str(seed)] = {k: result[k] for k in keep if k in result}
    json.dump(results, open(OUT, 'w'))
    print(f'  ✓ saved → {OUT}', flush=True)
    print(f'  seed {seed}: per-int={result["final_perint"]:.3f} spread={result["final_signal_spread"]:.3f}', flush=True)

print(f'\nDone. {len(results["MLP_all_uniform"])} seeds saved.')
