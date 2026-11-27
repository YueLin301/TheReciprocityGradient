"""
Exp 2a (memory window variant): Action optimization against identity opponents
with bounded memory.

Opponents only look at the last k signals of the recipient's reputation history.
Tests k = 1, 3, full (None).

Agent 0 still uses full Transformer (sees full history).
EMA mode (decay=0.5) for all variants.
"""
import os
import sys
import json
import torch
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from agent import Agent
from identity_agent import IdentityAgent
from env_batched import BatchedDonationGame, compute_per_interaction_payoff

BATCH_SIZE = 128
NUM_SEEDS = 5
TRAIN_EPISODES = 200
B_VALUE = 10.0
DELTA = 0.98
FULL_COOP_BENCHMARK = (B_VALUE - 1) / 2

SEED_COLORS = ['#5B8DBE', '#E07A5F', '#81B29A', '#F2CC8F', '#9A8FB4']
BG_COLOR = '#FAFAFA'


def train_action_only(seed, memory_window=None):
    torch.manual_seed(seed)
    np.random.seed(seed)

    agent0 = Agent(name="Agent_0", signal_dim=1, second_order=False, is_demo=False)
    agent0.set_hardcode_signal()

    opp1 = IdentityAgent(name="Opp_1", decay=0.5, memory_window=memory_window)
    opp2 = IdentityAgent(name="Opp_2", decay=0.5, memory_window=memory_window)

    action_params = agent0.get_action_params()
    optimizer = optim.Adam(action_params, lr=5e-3)

    agents = [agent0, opp1, opp2]
    env = BatchedDonationGame(
        agents=agents, b=B_VALUE, continuation_prob=DELTA, batch_size=BATCH_SIZE,
        balanced_matching=True, min_rounds=3,
        per_agent_init=[
            {"dist": "uniform", "params": {"low": 0.0, "high": 1.0}},
            {"dist": "constant", "params": {"value": 0.5}},
            {"dist": "constant", "params": {"value": 0.5}},
        ])

    episode_log = []

    for ep in range(TRAIN_EPISODES):
        optimizer.zero_grad()

        cum_rewards, matching_seq, rep_histories = env.play_batch()
        per_int_reward, n_plays = compute_per_interaction_payoff(cum_rewards, matching_seq, 0)
        loss = -per_int_reward.mean()

        if loss.requires_grad:
            grads = torch.autograd.grad(loss, action_params, allow_unused=True)
            for p, g in zip(action_params, grads):
                if g is not None:
                    p.grad = g
            torch.nn.utils.clip_grad_norm_(action_params, max_norm=1.0)

        optimizer.step()

        episode_log.append({
            'episode': ep,
            'per_int_reward': per_int_reward.mean().item(),
            'cum_reward': cum_rewards[0].mean().item(),
            'n_plays': n_plays,
            'T': len(matching_seq),
        })

    x_vals = torch.linspace(0, 1, 100)
    with torch.no_grad():
        action_profile = [agent0.take_action(x.view(1, 1)).item() for x in x_vals]
        signal_profile = [agent0.send_signal(x.view(1)).item() for x in x_vals]

    last50 = float(np.mean([e['per_int_reward'] for e in episode_log[-50:]]))
    mw_str = str(memory_window) if memory_window else 'full'
    print(f"  Seed {seed} done. per_int last50={last50:.3f} (window={mw_str})")

    return {
        'episode_log': episode_log,
        'action_profile': action_profile,
        'signal_profile': signal_profile,
        'last50_per_int': last50,
    }


def save_figure(results_by_window, save_dir):
    """One figure with subplots for each memory window."""
    windows = list(results_by_window.keys())
    x = np.linspace(0, 1, 100)

    fig, axes = plt.subplots(len(windows), 3, figsize=(15, 4.5 * len(windows)),
                              facecolor=BG_COLOR)
    if len(windows) == 1:
        axes = axes.reshape(1, 3)

    for row, mw in enumerate(windows):
        results = results_by_window[mw]
        mw_str = str(mw) if mw else 'full'

        ax = axes[row, 0]
        ax.set_facecolor(BG_COLOR)
        for s in range(NUM_SEEDS):
            r = [e['per_int_reward'] for e in results[s]['episode_log']]
            ax.plot(r, alpha=0.15, color=SEED_COLORS[s])
            ema = [r[0]]
            for v in r[1:]:
                ema.append(0.05 * v + 0.95 * ema[-1])
            ax.plot(ema, linewidth=1.8, color=SEED_COLORS[s], label=f"seed {s}")
        ax.axhline(y=FULL_COOP_BENCHMARK, color='#888', linestyle='--', alpha=0.5,
                    label=f'benchmark={FULL_COOP_BENCHMARK}')
        ax.set_title(f"Per-Int Payoff (window={mw_str})", fontsize=10, fontweight='bold')
        ax.set_xlabel("Episode")
        ax.set_ylabel("R⁰ / n₀")
        ax.legend(fontsize=6, framealpha=0.8)
        ax.grid(True, alpha=0.25)

        ax = axes[row, 1]
        ax.set_facecolor(BG_COLOR)
        for s in range(NUM_SEEDS):
            ax.plot(x, results[s]['action_profile'], linewidth=1.6,
                    color=SEED_COLORS[s], alpha=0.7)
        ax.plot(x, x, '--', color='#888', alpha=0.4, label='identity')
        ax.set_title(f"Action π(opp_rep) (window={mw_str})", fontsize=10, fontweight='bold')
        ax.set_xlabel("Opponent Reputation")
        ax.set_ylabel("Cooperation Level")
        ax.set_ylim([-0.05, 1.05])
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.25)

        ax = axes[row, 2]
        ax.set_facecolor(BG_COLOR)
        for s in range(NUM_SEEDS):
            ax.plot(x, results[s]['signal_profile'], linewidth=1.6,
                    color=SEED_COLORS[s], alpha=0.7)
        ax.plot(x, x, '--', color='#888', alpha=0.4, label='identity')
        ax.set_title(f"Signal φ(a) [hardcode] (window={mw_str})", fontsize=10, fontweight='bold')
        ax.set_xlabel("Donor Action")
        ax.set_ylabel("Signal")
        ax.set_ylim([-0.05, 1.05])
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.25)

    plt.tight_layout()
    fig_path = os.path.join(save_dir, "figures",
                             "fig_action_vs_identity_memory_window_ema.png")
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Figure saved: {fig_path}")


def main():
    save_dir = os.path.dirname(os.path.abspath(__file__))
    os.makedirs(os.path.join(save_dir, "figures"), exist_ok=True)
    os.makedirs(os.path.join(save_dir, "data"), exist_ok=True)

    windows = [1, 3, None]  # 1-step, 3-step, full history
    results_by_window = {}

    for mw in windows:
        mw_str = str(mw) if mw else 'full'
        print(f"\n{'='*50}")
        print(f"Memory window = {mw_str} (EMA decay=0.5)")
        print(f"{'='*50}")
        results = {}
        for seed in range(NUM_SEEDS):
            results[seed] = train_action_only(seed, memory_window=mw)
        results_by_window[mw] = results

        avg = np.mean([results[s]['last50_per_int'] for s in range(NUM_SEEDS)])
        print(f"  Average: {avg:.3f} / {FULL_COOP_BENCHMARK}")

    save_figure(results_by_window, save_dir)

    # Summary
    print(f"\n{'='*40}")
    for mw in windows:
        mw_str = str(mw) if mw else 'full'
        avg = np.mean([results_by_window[mw][s]['last50_per_int'] for s in range(NUM_SEEDS)])
        print(f"  window={mw_str:>5}: {avg:.3f} / {FULL_COOP_BENCHMARK}")
    print(f"{'='*40}")

    json_path = os.path.join(save_dir, "data", "results_action_vs_identity_memory_window.json")
    serializable = {}
    for mw in windows:
        key = str(mw) if mw else 'full'
        serializable[key] = {str(s): results_by_window[mw][s] for s in range(NUM_SEEDS)}
    with open(json_path, 'w') as f:
        json.dump(serializable, f)
    print(f"Results saved: {json_path}")


if __name__ == "__main__":
    main()
