"""
Run exp6 (B.2 signal-only learned models) for MLP × all_uniform × seed 4
to obtain signal_snapshots. This is the best disc seed
(per-int 1.252 = 100% of 1.25 ceiling, spread 0.375).
"""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import train_exp6_signal_only
# Override T_OUTER=125 and slow LR (5e-2 -> 3e-3, ~17x slower) so φ discrimination
# rises gradually over the full 125 iters instead of saturating in <10 iters.
train_exp6_signal_only.T_OUTER = 125
train_exp6_signal_only.LR_POLICY = 8e-3
from train_exp6_signal_only import run_exp6

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                    'data', 'results_6_snapshot.json')
os.makedirs(os.path.dirname(OUT), exist_ok=True)

result = run_exp6(arch='MLP', init_mode='all_uniform', seed=4)

# Keep snapshots + everything needed for plotting
keep = ['real_perint', 'surrogate_perint', 'real_perint_per_ep', 'surrogate_perint_per_ep',
        'act_est_loss_1', 'act_est_loss_2', 'sig_est_loss_1', 'sig_est_loss_2',
        'spread_log', 'rep_log', 'signal_snapshots',
        'final_perint', 'final_signal_profile', 'final_signal_spread',
        'final_disc_type']
data = {'MLP_all_uniform': {'4': {k: result[k] for k in keep if k in result}}}
json.dump(data, open(OUT, 'w'))
print(f'\n✓ saved → {OUT}')
print(f'  per-int = {result["final_perint"]:.3f}')
print(f'  signal_spread = {result["final_signal_spread"]:.3f} ({result["final_disc_type"]})')
print(f'  n_snapshots = {len(result["signal_snapshots"])}')
