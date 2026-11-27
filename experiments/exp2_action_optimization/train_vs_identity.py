"""
Exp 2a: Action optimization against hard-coded identity opponents.

Two variants:
  2a1: Opponent uses mean(full history) — cold start problem with init_token
  2a2: Opponent uses EMA (exponential decay=0.5) — recent signals dominate

Setup:
- Agent 0: learnable action_net (Transformer + MLP, random init)
  Signal is hard-coded exact identity (no NN).
- Opponents: hard-coded IdentityAgent (part of the environment)

Full cooperation benchmark: (b-1)/2 = 4.5  (b=10)
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
TRAIN_EPISODES = 300
B_VALUE = 10.0
DELTA = 0.98
FULL_COOP_BENCHMARK = (B_VALUE - 1) / 2

SEED_COLORS = ['#5B8DBE', '#E07A5F', '#81B29A', '#F2CC8F', '#9A8FB4']
BG_COLOR = '#FAFAFA'


def train_action_only(seed, opp_mode='mean', init_mode='const05'):
    """
    Train Agent 0's action_net against identity opponents.
    opp_mode: 'mean' or 'ema'
    init_mode: 'const05' (all 0.5), 'all_uniform' (all U[0,1]), 'warmup_init' (agent0=U[0,1], opps=0.5)
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    agent0 = Agent(name="Agent_0", signal_dim=1, second_order=False, is_demo=False)
    agent0.set_hardcode_signal()

    if opp_mode == 'mean':
        opp1 = IdentityAgent(name="Opp_1", decay=None)
        opp2 = IdentityAgent(name="Opp_2", decay=None)
    else:
        opp1 = IdentityAgent(name="Opp_1", decay=0.5)
        opp2 = IdentityAgent(name="Opp_2", decay=0.5)

    action_params = agent0.get_action_params()
    optimizer = optim.Adam(action_params, lr=5e-3)

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
    else:  # warmup_init — for identity opponents, steady-state ≈ 0.5
        per_agent_init = [
            {"dist": "uniform", "params": {"low": 0.0, "high": 1.0}},
            {"dist": "constant", "params": {"value": 0.5}},
            {"dist": "constant", "params": {"value": 0.5}},
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

        if (ep + 1) % 50 == 0:
            e = episode_log[-1]
            print(f"    ep {ep+1:3d}/{TRAIN_EPISODES}  per_int={e['per_int_reward']:.3f}  "
                  f"cum={e['cum_reward']:.1f}  n_plays={e['n_plays']}  T={e['T']}", flush=True)

    # Policy profiles
    x_vals = torch.linspace(0, 1, 100)
    with torch.no_grad():
        action_profile = [agent0.take_action(x.view(1, 1)).item() for x in x_vals]
        signal_profile = [agent0.send_signal(x.view(1)).item() for x in x_vals]

    last50 = float(np.mean([e['per_int_reward'] for e in episode_log[-50:]]))
    print(f"  Seed {seed} done. per_int last50={last50:.3f} (benchmark={FULL_COOP_BENCHMARK})")

    # Save weights
    exp_dir = os.path.dirname(os.path.abspath(__file__))
    weights_dir = os.path.join(exp_dir, "data", "weights")
    os.makedirs(weights_dir, exist_ok=True)
    torch.save(agent0.state_dict(),
               os.path.join(weights_dir, f"agent0_identity_{opp_mode}_seed{seed}.pt"))

    return {
        'config': {
            'experiment': f'exp2a_action_vs_identity_{opp_mode}',
            'seed': seed,
            'opp_mode': opp_mode,
            'opp_decay': None if opp_mode == 'mean' else 0.5,
            'agent': {
                'class': 'Agent',
                'signal_dim': 1, 'emb_dim': 16,
                'action_params': sum(p.numel() for p in agent0.get_action_params()),
                'signal': 'hardcode_identity',
            },
            'opponents': {
                'class': 'IdentityAgent',
                'mode': opp_mode,
                'decay': None if opp_mode == 'mean' else 0.5,
            },
            'environment': {
                'N': 3, 'b': B_VALUE, 'delta': DELTA,
                'batch_size': BATCH_SIZE,
                'balanced_matching': True,
            },
            'training': {
                'episodes': TRAIN_EPISODES,
                'action_lr': 5e-3,
                'grad_clip': 1.0,
            },
            'benchmark': FULL_COOP_BENCHMARK,
        },
        'episode_log': episode_log,
        'action_profile': action_profile,
        'signal_profile': signal_profile,
        'last50_per_int': last50,
        'last50_cum': float(np.mean([e['cum_reward'] for e in episode_log[-50:]])),
    }


def save_figure(results, save_dir, variant_name, variant_label):
    x = np.linspace(0, 1, 100)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), facecolor=BG_COLOR)
    for ax in axes:
        ax.set_facecolor(BG_COLOR)

    ax = axes[0]
    for s in range(NUM_SEEDS):
        r = [e['per_int_reward'] for e in results[s]['episode_log']]
        ax.plot(r, alpha=0.15, color=SEED_COLORS[s])
        ema = [r[0]]
        for v in r[1:]:
            ema.append(0.05 * v + 0.95 * ema[-1])
        ax.plot(ema, linewidth=1.8, color=SEED_COLORS[s], label=f"seed {s}")
    ax.axhline(y=FULL_COOP_BENCHMARK, color='#888888', linestyle='--', alpha=0.5,
               label=f'full coop = {FULL_COOP_BENCHMARK}')
    ax.axhline(y=0, color='#CCCCCC', linestyle='-', alpha=0.3)
    ax.set_title("Per-Interaction Payoff", fontsize=10, fontweight='bold')
    ax.set_xlabel("Episode")
    ax.set_ylabel("R⁰ / n₀")
    ax.legend(fontsize=7, framealpha=0.8)
    ax.grid(True, alpha=0.25)

    ax = axes[1]
    for s in range(NUM_SEEDS):
        ax.plot(x, results[s]['action_profile'], linewidth=1.6,
                color=SEED_COLORS[s], alpha=0.7)
    ax.plot(x, x, '--', color='#888888', alpha=0.4, label='identity')
    ax.set_title("Learned Action Policy  π(opp_rep)", fontsize=10, fontweight='bold')
    ax.set_xlabel("Opponent Reputation")
    ax.set_ylabel("Cooperation Level")
    ax.set_ylim([-0.05, 1.05])
    ax.legend(fontsize=8, framealpha=0.8)
    ax.grid(True, alpha=0.25)

    ax = axes[2]
    for s in range(NUM_SEEDS):
        ax.plot(x, results[s]['signal_profile'], linewidth=1.6,
                color=SEED_COLORS[s], alpha=0.7)
    ax.plot(x, x, '--', color='#888888', alpha=0.4, label='identity')
    ax.set_title("Signal Policy  φ(a)  [hardcode]", fontsize=10, fontweight='bold')
    ax.set_xlabel("Donor Action")
    ax.set_ylabel("Signal")
    ax.set_ylim([-0.05, 1.05])
    ax.legend(fontsize=8, framealpha=0.8)
    ax.grid(True, alpha=0.25)

    avg = np.mean([results[s]['last50_per_int'] for s in range(NUM_SEEDS)])
    fig.suptitle(f"Exp {variant_label}   (avg per-int = {avg:.3f}, benchmark = {FULL_COOP_BENCHMARK})",
                 fontsize=12, fontweight='bold', y=1.02)
    plt.tight_layout()
    fig_path = os.path.join(save_dir, "figures", f"fig_action_vs_identity_{variant_name}.png")
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Figure saved to {fig_path}")


def main():
    save_dir = os.path.dirname(os.path.abspath(__file__))
    os.makedirs(os.path.join(save_dir, "figures"), exist_ok=True)
    os.makedirs(os.path.join(save_dir, "data"), exist_ok=True)

    all_results = {}

    for init_mode in ['const05', 'all_uniform', 'warmup_init']:
        for opp_mode in ['mean', 'ema']:
            key = f"{opp_mode}_{init_mode}"
            label = f"{opp_mode}, init={init_mode}"
            print(f"\n{'='*60}")
            print(f"Exp 2a: Action vs Identity ({label})")
            print(f"{'='*60}")
            results = {}
            for seed in range(NUM_SEEDS):
                print(f"  Seed {seed}:", flush=True)
                results[seed] = train_action_only(seed, opp_mode=opp_mode, init_mode=init_mode)
            save_figure(results, save_dir,
                        f'{opp_mode}_{init_mode}',
                        f'2a: Action vs Identity ({label})')
            all_results[key] = {str(k): v for k, v in results.items()}

    # ── Summary ──
    print(f"\n{'='*60}")
    print(f"{'Setting':<30} {'Per-Int':>10}")
    print(f"{'-'*42}")
    for key in all_results:
        avg = np.mean([all_results[key][str(s)]['last50_per_int'] for s in range(NUM_SEEDS)])
        print(f"  {key:<28} {avg:10.3f}")
    print(f"  Benchmark: {FULL_COOP_BENCHMARK}")
    print(f"{'='*60}")

    json_path = os.path.join(save_dir, "data", "results_action_vs_identity.json")
    with open(json_path, 'w') as f:
        json.dump(all_results, f)
    print(f"Results saved to {json_path}")


if __name__ == "__main__":
    main()
