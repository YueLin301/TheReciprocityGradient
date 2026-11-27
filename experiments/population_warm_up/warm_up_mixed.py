"""
Warm-up burn-in for MIXED populations (heterogeneous agent types).

Two population compositions:
  1. mixed-hard: ProudCooperator (sigmoid T=10) + GuiltyDefector (sigmoid T=10)
  2. mixed-soft: SoftProudCooperator (linear) + SoftGuiltyDefector (linear)

Each type has its own action and signal rule. 50/50 split.
All use honest signal (σ = a). Action depends on own_rep.

Continuous only (no discrete version for mixed).
"""
import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt

SAVE_DIR = os.path.dirname(os.path.abspath(__file__))

N_AGENTS = 100  # 50 ProudCoop + 50 GuiltyDef
N_ROUNDS = 5000
EXEC_NOISE_STD = 0.05
ASSESS_NOISE_STD = 0.05
EMA_ALPHA = 0.3
SEED = 42
T_SIGMOID = 10.0  # temperature for hard version


# ═══════════════════════════════════════
# Agent type definitions (scalar functions)
# ═══════════════════════════════════════

def proud_action_hard(own_rep):
    """ProudCooperator (hard): a = sigmoid(T*(own_rep - 0.5))"""
    return 1.0 / (1.0 + np.exp(-T_SIGMOID * (own_rep - 0.5)))

def guilty_action_hard(own_rep):
    """GuiltyDefector (hard): a = sigmoid(T*(0.5 - own_rep))"""
    return 1.0 / (1.0 + np.exp(-T_SIGMOID * (0.5 - own_rep)))

def proud_action_soft(own_rep):
    """SoftProudCooperator: a = own_rep"""
    return own_rep

def guilty_action_soft(own_rep):
    """SoftGuiltyDefector: a = 1 - own_rep"""
    return 1.0 - own_rep

def signal_identity(a):
    """All agents: honest signal σ = a"""
    return a


def run_mixed_warmup(variant='hard', n_agents=N_AGENTS, n_rounds=N_ROUNDS,
                      exec_noise=EXEC_NOISE_STD, assess_noise=ASSESS_NOISE_STD,
                      ema_alpha=EMA_ALPHA, seed=SEED):
    """
    Run burn-in for a mixed population.
    First half = ProudCooperator, second half = GuiltyDefector.
    """
    np.random.seed(seed)

    if variant == 'hard':
        action_fns = [proud_action_hard] * (n_agents // 2) + \
                      [guilty_action_hard] * (n_agents // 2)
    else:  # soft
        action_fns = [proud_action_soft] * (n_agents // 2) + \
                      [guilty_action_soft] * (n_agents // 2)

    agent_types = ['proud'] * (n_agents // 2) + ['guilty'] * (n_agents // 2)

    # Init: all at 1.0 (standard)
    reps = np.ones(n_agents)

    all_pairs = [(i, j) for i in range(n_agents) for j in range(n_agents) if i != j]

    mean_history = []
    proud_mean_history = []
    guilty_mean_history = []
    std_history = []

    for rnd in range(n_rounds):
        np.random.shuffle(all_pairs)
        for d, r in all_pairs:
            # Donor acts based on own rep
            intended_action = action_fns[d](reps[d])
            action = np.clip(intended_action + np.random.normal(0, exec_noise), 0, 1)

            # Recipient signals (honest) + assess noise
            signal = np.clip(signal_identity(action) + np.random.normal(0, assess_noise), 0, 1)

            # Update donor rep
            reps[d] = (1 - ema_alpha) * reps[d] + ema_alpha * signal

        mean_history.append(np.mean(reps))
        std_history.append(np.std(reps))
        proud_mean_history.append(np.mean(reps[:n_agents//2]))
        guilty_mean_history.append(np.mean(reps[n_agents//2:]))

        if (rnd + 1) % 1000 == 0:
            print(f"    Round {rnd+1}: mean={mean_history[-1]:.4f} "
                  f"proud={proud_mean_history[-1]:.4f} guilty={guilty_mean_history[-1]:.4f}")

    return {
        'final_reps': reps.copy(),
        'agent_types': agent_types,
        'mean_history': mean_history,
        'std_history': std_history,
        'proud_mean_history': proud_mean_history,
        'guilty_mean_history': guilty_mean_history,
        'steady_mean': np.mean(mean_history[-500:]),
        'steady_proud': np.mean(proud_mean_history[-500:]),
        'steady_guilty': np.mean(guilty_mean_history[-500:]),
    }


def plot_all(results_dict):
    variants = list(results_dict.keys())
    fig, axes = plt.subplots(2, len(variants), figsize=(6*len(variants), 8), facecolor='white')
    if len(variants) == 1:
        axes = axes.reshape(2, 1)

    colors = {'mixed-hard': '#E07A5F', 'mixed-soft': '#81B29A'}
    c_proud = '#3498DB'
    c_guilty = '#E74C3C'

    for col, var in enumerate(variants):
        r = results_dict[var]
        color = colors.get(var, '#888')
        w = 50

        # Top: convergence by type
        ax = axes[0, col]
        h_p = r['proud_mean_history']
        h_g = r['guilty_mean_history']
        h_all = r['mean_history']

        if len(h_all) > w:
            sm_p = np.convolve(h_p, np.ones(w)/w, mode='valid')
            sm_g = np.convolve(h_g, np.ones(w)/w, mode='valid')
            sm_all = np.convolve(h_all, np.ones(w)/w, mode='valid')
            x_sm = range(w-1, len(h_all))
            ax.plot(x_sm, sm_p, color=c_proud, linewidth=1.5, label=f'proud ({r["steady_proud"]:.3f})')
            ax.plot(x_sm, sm_g, color=c_guilty, linewidth=1.5, label=f'guilty ({r["steady_guilty"]:.3f})')
            ax.plot(x_sm, sm_all, color='black', linewidth=1.5, linestyle='--',
                    label=f'overall ({r["steady_mean"]:.3f})')

        ax.set_title(f'{var}', fontsize=12, fontweight='bold')
        ax.set_xlabel('Round')
        ax.set_ylabel('Mean Reputation')
        ax.set_ylim([0, 1])
        ax.legend(fontsize=8, frameon=False)
        ax.grid(True, alpha=0.3)

        # Bottom: final distribution by type
        ax = axes[1, col]
        reps_p = r['final_reps'][:len(r['final_reps'])//2]
        reps_g = r['final_reps'][len(r['final_reps'])//2:]
        ax.hist(reps_p, bins=30, range=(0,1), color=c_proud, alpha=0.6, label='proud', edgecolor='white')
        ax.hist(reps_g, bins=30, range=(0,1), color=c_guilty, alpha=0.6, label='guilty', edgecolor='white')
        ax.set_xlabel('Reputation')
        ax.set_ylabel('Count')
        ax.set_title('Final Distribution', fontsize=10)
        ax.legend(fontsize=8, frameon=False)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig_path = os.path.join(SAVE_DIR, 'fig_warm_up_mixed_convergence.png')
    plt.savefig(fig_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f'Figure saved: {fig_path}')


def main():
    print("=" * 60)
    print("Mixed Population Warm-Up")
    print(f"  N={N_AGENTS} (50 proud + 50 guilty), rounds={N_ROUNDS}")
    print(f"  exec_noise={EXEC_NOISE_STD}, assess_noise={ASSESS_NOISE_STD}")
    print("=" * 60)

    all_results = {}

    for variant in ['mixed-hard', 'mixed-soft']:
        v = 'hard' if variant == 'mixed-hard' else 'soft'
        print(f"\n{variant}:")
        result = run_mixed_warmup(variant=v)
        all_results[variant] = result

        # Save
        import torch
        torch.save({
            'continuous_final_reps': torch.tensor(result['final_reps'], dtype=torch.float32),
            'steady_mean': result['steady_mean'],
            'steady_proud': result['steady_proud'],
            'steady_guilty': result['steady_guilty'],
        }, os.path.join(SAVE_DIR, f'steady_state_{variant.replace("-","_")}.pt'))

        json_data = {
            'variant': variant,
            'steady_mean': result['steady_mean'],
            'steady_proud': result['steady_proud'],
            'steady_guilty': result['steady_guilty'],
            'final_reps': result['final_reps'].tolist(),
        }
        with open(os.path.join(SAVE_DIR, f'steady_state_{variant.replace("-","_")}.json'), 'w') as f:
            json.dump(json_data, f, indent=2)

        print(f"  Steady: mean={result['steady_mean']:.4f} "
              f"proud={result['steady_proud']:.4f} guilty={result['steady_guilty']:.4f}")

    plot_all(all_results)

    print(f"\n{'='*60}")
    print(f"  {'Variant':<15} {'Overall':>8} {'Proud':>8} {'Guilty':>8}")
    for var in all_results:
        r = all_results[var]
        print(f"  {var:<15} {r['steady_mean']:8.4f} {r['steady_proud']:8.4f} {r['steady_guilty']:8.4f}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
