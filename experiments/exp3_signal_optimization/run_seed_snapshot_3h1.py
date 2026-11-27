"""
Run exp3h1 const05 seed 0 to obtain signal snapshots (10/10 disc cell).
Saves to a separate JSON to avoid disturbing the main results file.
"""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_vs_mixed_exp3h1 import train_signal_3h1, _strip_for_json

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                    'data', 'results_3h1_snapshot.json')
os.makedirs(os.path.dirname(OUT), exist_ok=True)

result = train_signal_3h1(seed=0, init_mode='const05')
data = {'const05': {'0': _strip_for_json(result)}}
json.dump(data, open(OUT, 'w'))
print(f'\n✓ saved → {OUT}')
print(f'  per_int = {result["last50_per_int"]:.3f}, signal_spread = {result["signal_spread"]:.3f} ({result["disc_type"]})')
print(f'  n_snapshots = {len(result["signal_snapshots"])}')
