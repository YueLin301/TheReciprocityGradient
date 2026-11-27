"""
Stationary Distribution Burn-in for reputation initialization.

Two versions:
  1. DISCRETE: binary reps {0, 1}, exec error flips action, assess error flips signal
  2. CONTINUOUS: scalar reps [0, 1], exec error → random action, assess error → flip signal

Both guarantee ergodic Markov chain → unique stationary distribution.
Both start from all-Good (rep=1 or rep=1.0) per standard convention.

Saves steady-state parameters for use as initial conditions in experiments.
"""
import os
import sys
import json
import torch
import numpy as np
import matplotlib.pyplot as plt

SAVE_DIR = os.path.dirname(os.path.abspath(__file__))

# ── Config ──
N_AGENTS = 100
N_ROUNDS = 5000
EXEC_ERROR = 0.02       # α (discrete: flip prob; continuous: not used)
ASSESS_ERROR = 0.02     # ε (discrete: flip prob; continuous: not used)
EXEC_NOISE_STD = 0.05   # σ₁: continuous execution noise (truncated Gaussian)
ASSESS_NOISE_STD = 0.05 # σ₂: continuous assessment noise (truncated Gaussian)
EMA_ALPHA = 0.3         # continuous reputation update rate
SEED = 42


# ═══════════════════════════════════════════════
# Norm definitions
# ═══════════════════════════════════════════════

# Discrete assessment: (action_coop, recip_good) → donor Good?
DISCRETE_ASSESSMENT = {
    'identity': {(True,True):True, (True,False):True, (False,True):False, (False,False):False},
    'L3': {(True,True):True, (True,False):True, (False,True):False, (False,False):True},
    'L6': {(True,True):True, (True,False):False, (False,True):False, (False,False):True},
}

# Discrete action: cooperate iff opp Good (same for all norms)
def discrete_action(own_good, opp_good):
    return opp_good

# Continuous action: a = opp_rep
def continuous_action(own_rep, opp_rep):
    return opp_rep

# Continuous signal functions (tanh spin-space, β=5)
BETA = 5.0

def continuous_signal_identity(a, r_recip, r_donor):
    return a  # honest reporting

def continuous_signal_L3(a, r_recip, r_donor):
    sx = np.tanh(BETA * (a - 0.5))
    sy = np.tanh(BETA * (r_recip - 0.5))
    return 1.0 - 0.25 * (1 - sx) * (1 + sy)

def continuous_signal_L6(a, r_recip, r_donor):
    sx = np.tanh(BETA * (a - 0.5))
    sy = np.tanh(BETA * (r_recip - 0.5))
    return 0.5 * (1 + sx * sy)

CONTINUOUS_SIGNAL = {
    'identity': continuous_signal_identity,
    'L3': continuous_signal_L3,
    'L6': continuous_signal_L6,
}


# ═══════════════════════════════════════════════
# Discrete burn-in
# ═══════════════════════════════════════════════

def run_discrete(norm_type, n_agents=N_AGENTS, n_rounds=N_ROUNDS,
                 exec_error=EXEC_ERROR, assess_error=ASSESS_ERROR, seed=SEED):
    np.random.seed(seed)
    assessment = DISCRETE_ASSESSMENT[norm_type]
    reps = np.ones(n_agents, dtype=int)  # all-Good init

    all_pairs = [(i, j) for i in range(n_agents) for j in range(n_agents) if i != j]
    frac_good_history = []

    for rnd in range(n_rounds):
        np.random.shuffle(all_pairs)
        for d, r in all_pairs:
            intended = discrete_action(bool(reps[d]), bool(reps[r]))
            actual = (not intended) if np.random.random() < exec_error else intended
            intended_good = assessment[(actual, bool(reps[r]))]
            if np.random.random() < assess_error:
                new_rep = 0 if intended_good else 1
            else:
                new_rep = 1 if intended_good else 0
            reps[d] = new_rep

        frac_good_history.append(np.mean(reps))
        if (rnd + 1) % 1000 == 0:
            recent = frac_good_history[-200:]
            print(f"    Round {rnd+1}: frac_good={frac_good_history[-1]:.3f} "
                  f"(avg200={np.mean(recent):.4f})")

    p_star = np.mean(frac_good_history[-500:])
    return {
        'final_reps': reps.copy(),
        'frac_good_history': frac_good_history,
        'p_star': p_star,
        'p_std': np.std(frac_good_history[-500:]),
    }


# ═══════════════════════════════════════════════
# Continuous burn-in
# ═══════════════════════════════════════════════

def run_continuous(norm_type, n_agents=N_AGENTS, n_rounds=N_ROUNDS,
                   exec_noise_std=EXEC_NOISE_STD, assess_noise_std=ASSESS_NOISE_STD,
                   ema_alpha=EMA_ALPHA, seed=SEED):
    """
    Continuous burn-in with additive Gaussian noise (Doeblin condition).

    Execution noise:  A_actual = clip(A_intended + ξ₁, 0, 1),  ξ₁ ~ N(0, σ₁²)
    Assessment noise:  R_new = clip(f(A, R_recip) + ξ₂, 0, 1),  ξ₂ ~ N(0, σ₂²)

    Both noises have density > 0 on [0,1] (truncated Gaussian),
    guaranteeing ψ-irreducibility → unique stationary distribution.
    """
    np.random.seed(seed)
    signal_fn = CONTINUOUS_SIGNAL[norm_type]
    reps = np.ones(n_agents)  # all-Good init (1.0)

    all_pairs = [(i, j) for i in range(n_agents) for j in range(n_agents) if i != j]
    mean_rep_history = []
    std_rep_history = []

    for rnd in range(n_rounds):
        np.random.shuffle(all_pairs)
        for d, r in all_pairs:
            # Action + execution noise ξ₁
            intended = continuous_action(reps[d], reps[r])
            xi1 = np.random.normal(0, exec_noise_std)
            action = np.clip(intended + xi1, 0, 1)

            # Signal + assessment noise ξ₂
            signal_clean = signal_fn(action, reps[r], reps[d])
            xi2 = np.random.normal(0, assess_noise_std)
            signal = np.clip(signal_clean + xi2, 0, 1)

            # EMA update
            reps[d] = (1 - ema_alpha) * reps[d] + ema_alpha * signal

        mean_rep_history.append(np.mean(reps))
        std_rep_history.append(np.std(reps))
        if (rnd + 1) % 1000 == 0:
            recent_mean = np.mean(mean_rep_history[-200:])
            recent_std = np.mean(std_rep_history[-200:])
            print(f"    Round {rnd+1}: mean={mean_rep_history[-1]:.4f} "
                  f"(avg200={recent_mean:.4f}, std={recent_std:.4f})")

    steady_mean = np.mean(mean_rep_history[-500:])
    steady_std = np.mean(std_rep_history[-500:])
    return {
        'final_reps': reps.copy(),
        'mean_history': mean_rep_history,
        'std_history': std_rep_history,
        'steady_mean': steady_mean,
        'steady_std': steady_std,
    }


# ═══════════════════════════════════════════════
# Plotting
# ═══════════════════════════════════════════════

def plot_all(discrete_results, continuous_results, norms):
    colors = {'identity': '#5B8DBE', 'L3': '#81B29A', 'L6': '#E07A5F'}

    fig, axes = plt.subplots(2, len(norms), figsize=(5*len(norms), 8), facecolor='white')

    for col, norm in enumerate(norms):
        color = colors[norm]

        # Top: discrete
        ax = axes[0, col]
        h = discrete_results[norm]['frac_good_history']
        ax.plot(h, color=color, linewidth=0.3, alpha=0.3)
        w = 50
        if len(h) > w:
            sm = np.convolve(h, np.ones(w)/w, mode='valid')
            ax.plot(range(w-1, len(h)), sm, color=color, linewidth=2)
        p = discrete_results[norm]['p_star']
        ax.axhline(y=p, color='black', linestyle='--', linewidth=1,
                    label=f'p* = {p:.3f}')
        ax.set_title(f'{norm} (discrete)', fontsize=11, fontweight='bold')
        ax.set_ylabel('Fraction Good')
        ax.set_ylim([0, 1.05])
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

        # Bottom: continuous
        ax = axes[1, col]
        h = continuous_results[norm]['mean_history']
        ax.plot(h, color=color, linewidth=0.5, alpha=0.4)
        if len(h) > w:
            sm = np.convolve(h, np.ones(w)/w, mode='valid')
            ax.plot(range(w-1, len(h)), sm, color=color, linewidth=2)
        m = continuous_results[norm]['steady_mean']
        ax.axhline(y=m, color='black', linestyle='--', linewidth=1,
                    label=f'mean = {m:.3f}')
        ax.set_title(f'{norm} (continuous)', fontsize=11, fontweight='bold')
        ax.set_xlabel('Round')
        ax.set_ylabel('Mean Reputation')
        ax.set_ylim([0, 1.05])
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig_path = os.path.join(SAVE_DIR,
        f'fig_warm_up_discrete_and_continuous_tanh_beta{int(BETA)}_N{N_AGENTS}_R{N_ROUNDS}.png')
    plt.savefig(fig_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Figure saved: {fig_path}")


# ═══════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════

def main():
    norms = ['identity', 'L3', 'L6']

    print("=" * 60)
    print("Population Warm-Up: Stationary Distribution Burn-in")
    print(f"  N={N_AGENTS}, rounds={N_ROUNDS}")
    print(f"  exec_error α={EXEC_ERROR}, assess_error ε={ASSESS_ERROR}")
    print(f"  continuous EMA α={EMA_ALPHA}")
    print(f"  Init: all-Good")
    print("=" * 60)

    discrete_results = {}
    continuous_results = {}

    for norm in norms:
        print(f"\n{'─'*40}")
        print(f"{norm}:")

        print(f"  [discrete]")
        discrete_results[norm] = run_discrete(norm)
        p = discrete_results[norm]['p_star']
        print(f"    → p* = {p:.4f}")

        print(f"  [continuous]")
        continuous_results[norm] = run_continuous(norm)
        m = continuous_results[norm]['steady_mean']
        s = continuous_results[norm]['steady_std']
        print(f"    → mean = {m:.4f}, std = {s:.4f}")

    # Save
    for norm in norms:
        d = discrete_results[norm]
        c = continuous_results[norm]

        save_data = {
            'norm': norm,
            'config': {
                'n_agents': N_AGENTS, 'n_rounds': N_ROUNDS,
                'exec_error': EXEC_ERROR, 'assess_error': ASSESS_ERROR,
                'ema_alpha': EMA_ALPHA, 'seed': SEED,
            },
            'discrete': {
                'p_star': float(d['p_star']),
                'p_std': float(d['p_std']),
                'final_reps': d['final_reps'].tolist(),
            },
            'continuous': {
                'steady_mean': float(c['steady_mean']),
                'steady_std': float(c['steady_std']),
                'final_reps': c['final_reps'].tolist(),
            },
        }
        json_path = os.path.join(SAVE_DIR, f"steady_state_{norm}.json")
        with open(json_path, 'w') as f:
            json.dump(save_data, f, indent=2)

        torch.save({
            'discrete_p_star': d['p_star'],
            'discrete_final_reps': torch.tensor(d['final_reps'], dtype=torch.float32),
            'continuous_steady_mean': c['steady_mean'],
            'continuous_final_reps': torch.tensor(c['final_reps'], dtype=torch.float32),
        }, os.path.join(SAVE_DIR, f"steady_state_{norm}.pt"))

    plot_all(discrete_results, continuous_results, norms)

    # Summary
    print(f"\n{'='*60}")
    print(f"{'Norm':>10} │ {'Discrete p*':>12} │ {'Continuous mean':>16} │ {'Cont. std':>10}")
    print(f"{'─'*10}─┼─{'─'*12}─┼─{'─'*16}─┼─{'─'*10}")
    for norm in norms:
        p = discrete_results[norm]['p_star']
        m = continuous_results[norm]['steady_mean']
        s = continuous_results[norm]['steady_std']
        print(f"{norm:>10} │ {p:12.4f} │ {m:16.4f} │ {s:10.4f}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
