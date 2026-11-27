"""
Exp 2b variant: Action optimization against leading-eight opponents,
where the learning agent's signal uses the SAME norm as the opponents
(tanh L3/L6) instead of identity mapping.

Uses warmup_init for opponents, uniform for agent 0.
"""
import os
import sys
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from train_vs_leading_eight import (
    train_action_vs_norm, load_steady_state,
    NUM_SEEDS, FULL_COOP_BENCHMARK, NORM_NAMES,
    SEED_COLORS, BG_COLOR
)

C_COOP = '#6BAED6'
C_DEF  = '#E07A5F'
cmap_cr = LinearSegmentedColormap.from_list('coop_def', [C_DEF, '#F5F5F5', C_COOP])


def signal_heatmap(norm, beta=5.0, n=200):
    """Compute 2D signal map σ(action, own_rep) for a norm."""
    a = np.linspace(0, 1, n)
    r = np.linspace(0, 1, n)
    A, R = np.meshgrid(a, r)
    sx = np.tanh(beta * (A - 0.5))
    sy = np.tanh(beta * (R - 0.5))
    if norm == 'L3':
        return 1.0 - 0.25 * (1 - sx) * (1 + sy)
    elif norm == 'L6':
        return 0.5 * (1 + sx * sy)
    return A  # identity fallback


def save_normsignal_figure(norm_name, seed_results, save_dir, fig_name=None, suffix=''):
    """Figure with payoff, action profile, and 2D signal heatmap."""
    x = np.linspace(0, 1, 100)
    full_name = NORM_NAMES[norm_name]

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5), facecolor=BG_COLOR)
    for ax in axes[:2]:
        ax.set_facecolor(BG_COLOR)

    # (1) Per-interaction payoff
    ax = axes[0]
    for s in range(NUM_SEEDS):
        r = [e['per_int_reward'] for e in seed_results[str(s)]['episode_log']]
        ax.plot(r, alpha=0.15, color=SEED_COLORS[s])
        ema = [r[0]]
        for v in r[1:]:
            ema.append(0.05 * v + 0.95 * ema[-1])
        ax.plot(ema, linewidth=1.8, color=SEED_COLORS[s], label=f"seed {s}")
    ax.axhline(y=FULL_COOP_BENCHMARK, color='#888', linestyle='--', alpha=0.5,
               label=f'benchmark={FULL_COOP_BENCHMARK}')
    ax.set_title("Per-Interaction Payoff", fontsize=10, fontweight='bold')
    ax.set_xlabel("Episode")
    ax.set_ylabel("R⁰ / n₀")
    ax.legend(fontsize=6, framealpha=0.8)
    ax.grid(True, alpha=0.25)

    # (2) Action profile
    ax = axes[1]
    for s in range(NUM_SEEDS):
        ax.plot(x, seed_results[str(s)]['action_profile'], linewidth=1.6,
                color=SEED_COLORS[s], alpha=0.7)
    ax.plot(x, x, ':', color='#888', alpha=0.3, label='identity')
    ax.set_title("Learned Action π(opp_rep)", fontsize=10, fontweight='bold')
    ax.set_xlabel("Opponent Reputation")
    ax.set_ylabel("Cooperation Level")
    ax.set_ylim([-0.05, 1.05])
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.25)

    # (3) 2D signal heatmap (hardcoded norm — same for all seeds)
    ax = axes[2]
    Z = signal_heatmap(norm_name)
    im = ax.imshow(Z, extent=[0, 1, 0, 1], origin='lower',
                   aspect='auto', cmap=cmap_cr, vmin=0, vmax=1)
    cs = ax.contour(np.linspace(0, 1, 200), np.linspace(0, 1, 200), Z,
                     levels=[0.25, 0.5, 0.75], colors='black', linewidths=0.7, alpha=0.5)
    ax.clabel(cs, fmt='%.2f', fontsize=6)
    ax.set_title(f"Signal φ(a, own_rep)  [hardcode {norm_name}]", fontsize=10, fontweight='bold')
    ax.set_xlabel("Donor Action")
    ax.set_ylabel("Own Reputation")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    avg = np.mean([seed_results[str(s)]['last50_per_int'] for s in range(NUM_SEEDS)])
    fig.suptitle(f"Exp 2b: Action vs {norm_name} ({full_name}), signal={norm_name}   "
                 f"(avg={avg:.3f}, benchmark={FULL_COOP_BENCHMARK})",
                 fontsize=11, fontweight='bold', y=1.02)
    plt.tight_layout()
    if fig_name is None:
        fig_name = f"fig_action_vs_{norm_name}{suffix}.png"
    fig_path = os.path.join(save_dir, "figures", fig_name)
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  -> {fig_path}")


def main():
    save_dir = os.path.dirname(os.path.abspath(__file__))
    os.makedirs(os.path.join(save_dir, "figures"), exist_ok=True)
    os.makedirs(os.path.join(save_dir, "data"), exist_ok=True)
    norms = ['L3', 'L6']
    all_results = {}

    for init_mode in ['const05', 'all_uniform', 'warmup_init']:
        for opp_mode in ['mean', 'ema']:
            for norm in norms:
                key = f"{norm}_{opp_mode}_{init_mode}_normsignal"
                label = f"{norm} ({NORM_NAMES[norm]}, {opp_mode}, {init_mode}, signal={norm})"
                print(f"\n{'='*60}")
                print(f"Exp 2b: {label}")
                print(f"{'='*60}")
                results = {}
                for seed in range(NUM_SEEDS):
                    print(f"  seed {seed}...", end="", flush=True)
                    r = train_action_vs_norm(
                        norm, seed, opp_mode=opp_mode,
                        init_mode=init_mode, signal_mode='norm')
                    results[str(seed)] = r
                    print(f" per_int={r['last50_per_int']:.3f}", flush=True)

                save_normsignal_figure(norm, results, save_dir,
                                       fig_name=f"fig_exp2b_{norm}_{opp_mode}_{init_mode}_normsignal.png")
                all_results[key] = results

    # Summary
    print(f"\n{'='*60}")
    print(f"{'Setting':<55} {'Per-Int':>10}")
    print(f"{'-'*67}")
    for init_mode in ['const05', 'all_uniform', 'warmup_init']:
        for opp_mode in ['mean', 'ema']:
            for norm in norms:
                key = f"{norm}_{opp_mode}_{init_mode}_normsignal"
                res = all_results[key]
                pi_avg = np.mean([res[str(s)]['last50_per_int'] for s in range(NUM_SEEDS)])
                print(f"  {norm} {opp_mode} {init_mode} signal={norm}{'':<10} {pi_avg:10.3f}")
    print(f"\n  Benchmark: {FULL_COOP_BENCHMARK}")

    json_path = os.path.join(save_dir, "data", "results_action_vs_leading_eight_normsignal.json")
    with open(json_path, 'w') as f:
        json.dump(all_results, f)
    print(f"Results saved: {json_path}")


if __name__ == "__main__":
    main()
