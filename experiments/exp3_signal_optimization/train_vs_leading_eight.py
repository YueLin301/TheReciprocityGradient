"""
Exp 3c: Signal optimization against leading-eight opponents (L3/L6).

Agent 0: trainable signal_net, action hardcoded as identity (a = opp_rep).
Opponents: LeadingEightAgent with tanh β=5 continuous norms.

The learning agent's hardcoded action matches the opponent's action rule
(both are a = opp_rep in continuous form). Only the signal is trained.
"""
import os
import sys
import json
import torch
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../exp2_action_optimization')))
from agent import Agent
from leading_eight_agents import LeadingEightAgent, NORM_NAMES
from env_batched import BatchedDonationGame, compute_per_interaction_payoff

BATCH_SIZE = 128
NUM_SEEDS = 5
TRAIN_EPISODES = 200
B_VALUE = 10.0
DELTA = 0.98
FULL_COOP_BENCHMARK = (B_VALUE - 1) / 2

SEED_COLORS = ['#5B8DBE', '#E07A5F', '#81B29A', '#F2CC8F', '#9A8FB4']
BG_COLOR = '#FAFAFA'
C_COOP = '#6BAED6'
C_DEF  = '#E07A5F'
cmap_cr = LinearSegmentedColormap.from_list('coop_def', [C_DEF, '#F5F5F5', C_COOP])


def load_steady_state(norm_name):
    warm_up_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                '..', 'population_warm_up')
    pt_path = os.path.join(warm_up_dir, f'steady_state_{norm_name}.pt')
    data = torch.load(pt_path, map_location='cpu', weights_only=False)
    return data['continuous_final_reps']


def train_signal_vs_norm(norm_name, seed, opp_mode='mean', init_mode='warmup_init'):
    torch.manual_seed(seed)
    np.random.seed(seed)

    decay = None if opp_mode == 'mean' else 0.5

    agent0 = Agent(name="A_0", signal_dim=1, second_order=True, is_demo=False)
    opp1 = LeadingEightAgent(name="Opp_1", norm=norm_name, continuous=True, decay=decay)
    opp2 = LeadingEightAgent(name="Opp_2", norm=norm_name, continuous=True, decay=decay)

    # Action: hardcode as identity (same as opponent's continuous action rule)
    agent0.set_hardcode_action(decay=decay)

    # Trainable: signal network
    signal_params = agent0.get_signal_params()
    optimizer = optim.Adam(signal_params, lr=5e-2)

    if init_mode == 'const05':
        per_agent_init = [
            {"dist": "constant", "params": {"value": 0.5}},
            {"dist": "constant", "params": {"value": 0.5}},
            {"dist": "constant", "params": {"value": 0.5}},
        ]
    elif init_mode == 'all_uniform':
        per_agent_init = [
            {"dist": "uniform", "params": {"low": 0.0, "high": 1.0}},
            {"dist": "uniform", "params": {"low": 0.0, "high": 1.0}},
            {"dist": "uniform", "params": {"low": 0.0, "high": 1.0}},
        ]
    else:  # warmup_init
        steady_reps = load_steady_state(norm_name)
        per_agent_init = [
            {"dist": "uniform", "params": {"low": 0.0, "high": 1.0}},
            {"dist": "empirical", "params": {"values": steady_reps}},
            {"dist": "empirical", "params": {"values": steady_reps}},
        ]

    agents = [agent0, opp1, opp2]
    env = BatchedDonationGame(
        agents=agents, b=B_VALUE, continuation_prob=DELTA, batch_size=BATCH_SIZE,
        balanced_matching=True, min_rounds=3,
        per_agent_init=per_agent_init)

    episode_log = []

    for ep in range(TRAIN_EPISODES):
        optimizer.zero_grad()

        cum_rewards, matching_seq, rep_histories = env.play_batch()
        per_int_reward, n_plays = compute_per_interaction_payoff(cum_rewards, matching_seq, 0)
        loss = -per_int_reward.mean()

        if loss.requires_grad:
            grads = torch.autograd.grad(loss, signal_params, allow_unused=True)
            for p, g in zip(signal_params, grads):
                if g is not None and not (torch.isnan(g).any() or torch.isinf(g).any()):
                    p.grad = g
            torch.nn.utils.clip_grad_norm_(signal_params, max_norm=1.0)

        optimizer.step()

        episode_log.append({
            'episode': ep,
            'per_int_reward': per_int_reward.mean().item(),
            'cum_reward': cum_rewards[0].mean().item(),
            'n_plays': n_plays,
            'T': len(matching_seq),
        })

    # Policy profiles
    x_vals = torch.linspace(0, 1, 100)
    if init_mode == 'warmup_init':
        ss = load_steady_state(norm_name)
        own_rep_val = float(ss.mean())
    else:
        own_rep_val = 0.5
    with torch.no_grad():
        action_profile = [agent0.take_action(x.view(1, 1)).item() for x in x_vals]
        dummy_own_rep = torch.tensor([[own_rep_val]])
        signal_profile = [agent0.send_signal(
            x.view(1), own_reputation=dummy_own_rep).item() for x in x_vals]

    # Save weights
    exp_dir = os.path.dirname(os.path.abspath(__file__))
    weights_dir = os.path.join(exp_dir, "data", "weights")
    os.makedirs(weights_dir, exist_ok=True)
    torch.save(agent0.state_dict(),
               os.path.join(weights_dir, f"agent0_signal_{norm_name}_{opp_mode}_{init_mode}_seed{seed}.pt"))

    last50 = float(np.mean([e['per_int_reward'] for e in episode_log[-50:]]))

    return {
        'episode_log': episode_log,
        'action_profile': action_profile,
        'signal_profile': signal_profile,
        'last50_per_int': last50,
        'last50_cum': float(np.mean([e['cum_reward'] for e in episode_log[-50:]])),
    }


def save_figure(norm_name, seed_results, save_dir, suffix=''):
    x = np.linspace(0, 1, 100)
    full_name = NORM_NAMES[norm_name]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), facecolor=BG_COLOR)
    for ax in axes:
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

    # (2) Action profile (hardcode — identity)
    ax = axes[1]
    for s in range(NUM_SEEDS):
        ax.plot(x, seed_results[str(s)]['action_profile'], linewidth=1.6,
                color=SEED_COLORS[s], alpha=0.7)
    ax.plot(x, x, '--', color='#888', alpha=0.4, label='identity')
    ax.set_title("Action π(opp_rep)  [hardcode]", fontsize=10, fontweight='bold')
    ax.set_xlabel("Opponent Reputation")
    ax.set_ylabel("Cooperation Level")
    ax.set_ylim([-0.05, 1.05])
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.25)

    # (3) Learned signal profile
    ax = axes[2]
    for s in range(NUM_SEEDS):
        ax.plot(x, seed_results[str(s)]['signal_profile'], linewidth=1.6,
                color=SEED_COLORS[s], alpha=0.7)
    ax.plot(x, x, '--', color='#888', alpha=0.4, label='honest (identity)')
    ax.set_title("Learned Signal φ(a)", fontsize=10, fontweight='bold')
    ax.set_xlabel("Donor Action")
    ax.set_ylabel("Signal")
    ax.set_ylim([-0.05, 1.05])
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.25)

    avg = np.mean([seed_results[str(s)]['last50_per_int'] for s in range(NUM_SEEDS)])
    fig.suptitle(f"Exp 3c: Signal vs {norm_name} ({full_name})   "
                 f"(avg={avg:.3f}, benchmark={FULL_COOP_BENCHMARK})",
                 fontsize=11, fontweight='bold', y=1.02)
    plt.tight_layout()
    fig_path = os.path.join(save_dir, "figures", f"fig_signal_vs_{norm_name}{suffix}.png")
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  -> {fig_path}")


def main():
    save_dir = os.path.dirname(os.path.abspath(__file__))
    os.makedirs(os.path.join(save_dir, "figures"), exist_ok=True)
    os.makedirs(os.path.join(save_dir, "data"), exist_ok=True)
    norms = ['L3', 'L6']
    all_results = {}

    for init_mode in ['const05', 'warmup_init', 'all_uniform']:
        for opp_mode in ['mean', 'ema']:
            for norm in norms:
                key = f"{norm}_{opp_mode}_{init_mode}"
                label = f"{norm} ({NORM_NAMES[norm]}, {opp_mode}, {init_mode})"
                print(f"\n{'='*60}")
                print(f"Exp 3c: Signal vs {label}")
                print(f"{'='*60}")
                results = {}
                for seed in range(NUM_SEEDS):
                    print(f"  seed {seed}...", end="", flush=True)
                    r = train_signal_vs_norm(norm, seed, opp_mode=opp_mode, init_mode=init_mode)
                    results[str(seed)] = r
                    print(f" per_int={r['last50_per_int']:.3f}", flush=True)

                save_figure(norm, results, save_dir,
                            suffix=f"_{opp_mode}_{init_mode}")
                all_results[key] = results

    # Summary
    print(f"\n{'='*60}")
    print(f"{'Setting':<45} {'Per-Int':>10}")
    print(f"{'-'*57}")
    for init_mode in ['const05', 'warmup_init', 'all_uniform']:
        for opp_mode in ['mean', 'ema']:
            for norm in norms:
                key = f"{norm}_{opp_mode}_{init_mode}"
                res = all_results[key]
                avg = np.mean([res[str(s)]['last50_per_int'] for s in range(NUM_SEEDS)])
                label = f"{norm} ({NORM_NAMES[norm]}, {opp_mode}, {init_mode})"
                print(f"{label:<45} {avg:10.3f}")
    print(f"\nBenchmark: {FULL_COOP_BENCHMARK}")

    json_path = os.path.join(save_dir, "data", "results_signal_vs_leading_eight.json")
    with open(json_path, 'w') as f:
        json.dump(all_results, f)
    print(f"Results saved: {json_path}")


if __name__ == "__main__":
    main()
