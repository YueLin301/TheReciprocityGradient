"""
Exp 2b: Action optimization against hard-coded leading-eight norm opponents.

Setup:
- Agent 0: learnable action_net (Transformer + MLP, random init)
  Signal_net pre-trained as honest reporter, then frozen.
- Opponents (Agent 1 & 2): hard-coded LeadingEightAgent
  Fixed social norm strategies (L1-L8), implemented as differentiable
  lookup tables with soft sigmoid thresholds. No neural networks.
  These are part of the environment.

Training objective:
- Maximize Agent 0's per-own-interaction payoff via reciprocity gradient
- Tests whether action gradient can discover optimal strategy per norm

Saves one result figure per norm.
"""
import os
import sys
import json
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from agent import Agent
from env_batched import BatchedDonationGame, compute_per_interaction_payoff
from leading_eight_agents import LeadingEightAgent, NORM_NAMES

BATCH_SIZE = 128
NUM_SEEDS = 5
TRAIN_EPISODES = 200
B_VALUE = 10.0
DELTA = 0.98
FULL_COOP_BENCHMARK = (B_VALUE - 1) / 2  # 4.5

# ── Style ──
SEED_COLORS = ['#5B8DBE', '#E07A5F', '#81B29A', '#F2CC8F', '#9A8FB4']
OPP_COLOR = '#AAAAAA'
BG_COLOR = '#FAFAFA'


def pretrain_honest_signal(agent, epochs=500):
    """Pre-train signal network as honest reporter via send_signal."""
    opt = optim.Adam(agent.get_signal_params(), lr=1e-3)
    criterion = nn.MSELoss()
    for _ in range(epochs):
        opt.zero_grad()
        loss = torch.tensor(0.0)
        for _ in range(16):
            a = torch.rand(1)
            own_rep = torch.rand(torch.randint(1, 8, (1,)).item(), 1)
            pred = agent.send_signal(a, own_reputation=own_rep)
            loss = loss + criterion(pred, a.view_as(pred))
        (loss / 16).backward()
        opt.step()


def get_opponent_action_profile(norm_name):
    """Opponent's action as function of recipient rep, assuming own_rep=0.8."""
    opp = LeadingEightAgent(name="tmp", norm=norm_name, temperature=10.0)
    x_vals = torch.linspace(0, 1, 100)
    profile = []
    with torch.no_grad():
        for x in x_vals:
            opp.reputation_history = torch.tensor([[0.8]])
            profile.append(opp.take_action(x.view(1, 1)).item())
    return profile


def load_steady_state(norm_name):
    """Load warm-up steady-state reputation distribution."""
    warm_up_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                '..', 'population_warm_up')
    pt_path = os.path.join(warm_up_dir, f'steady_state_{norm_name}.pt')
    data = torch.load(pt_path, map_location='cpu', weights_only=False)
    return data['continuous_final_reps']  # [N_agents] tensor of steady-state reps


def train_action_vs_norm(norm_name, seed, opp_mode='mean', init_mode='warmup_init',
                         signal_mode='identity'):
    """
    signal_mode: 'identity' (σ=a) or 'norm' (σ=L3/L6 tanh, same as opponents)
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    decay = None if opp_mode == 'mean' else 0.5

    agent0 = Agent(name="A_0", signal_dim=1, second_order=True, is_demo=False)
    opp1 = LeadingEightAgent(name="Opp_1", norm=norm_name, continuous=True, decay=decay)
    opp2 = LeadingEightAgent(name="Opp_2", norm=norm_name, continuous=True, decay=decay)

    # Signal: hardcode
    if signal_mode == 'norm':
        agent0.set_hardcode_signal(norm=norm_name, beta=5.0)
    else:
        agent0.set_hardcode_signal()

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

    action_params = agent0.get_action_params()
    optimizer = optim.Adam(action_params, lr=5e-3)

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

    # Policy profile — use steady-state own_rep for signal evaluation
    x_vals = torch.linspace(0, 1, 100)
    if init_mode == 'warmup_init':
        ss_reps = load_steady_state(norm_name)
        own_rep_val = float(ss_reps.mean())
    else:
        own_rep_val = 0.5
    with torch.no_grad():
        action_profile = [agent0.take_action(x.view(1, 1)).item() for x in x_vals]
        dummy_own_rep = torch.tensor([[own_rep_val]])
        signal_profile = [agent0.send_signal(
            x.view(1), own_reputation=dummy_own_rep).item() for x in x_vals]

    # Save model weights
    exp_dir = os.path.dirname(os.path.abspath(__file__))
    weights_path = f"weights/agent0_{norm_name}_{opp_mode}_{init_mode}_{signal_mode}_seed{seed}.pt"
    checkpoint_path = os.path.join(exp_dir, "data", weights_path)
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    torch.save(agent0.state_dict(), checkpoint_path)

    last50_pi = float(np.mean([e['per_int_reward'] for e in episode_log[-50:]]))

    return {
        'config': {
            'experiment': f'exp2b_action_vs_{norm_name}',
            'seed': seed,
            'agent': {
                'class': 'Agent',
                'signal_dim': 1,
                'second_order': True,
                'emb_dim': 16,
                'action_hidden': [16],
                'signal_hidden': [16],
                'action_params': sum(p.numel() for p in agent0.get_action_params()),
                'signal_params': sum(p.numel() for p in agent0.get_signal_params()),
                'architecture': str(agent0),
            },
            'opponents': {
                'class': 'LeadingEightAgent',
                'norm': norm_name,
                'norm_name': NORM_NAMES[norm_name],
                'temperature': 10.0,
                'count': 2,
            },
            'environment': {
                'N': 3, 'b': B_VALUE, 'delta': DELTA,
                'batch_size': BATCH_SIZE,
                'balanced_matching': True,
                'init_reputation': 0.5,
            },
            'training': {
                'train_episodes': TRAIN_EPISODES,
                'signal_pretrain_epochs': 500,
                'action_lr': 5e-3,
                'grad_clip': 1.0,
                'trained_network': 'action',
                'frozen_network': 'signal',
            },
            'benchmark': FULL_COOP_BENCHMARK,
        },
        'episode_log': episode_log,
        'action_profile': action_profile,
        'signal_profile': signal_profile,
        'last50_per_int': last50_pi,
        'last50_cum': float(np.mean([e['cum_reward'] for e in episode_log[-50:]])),
        'pi_02': action_profile[20],
        'pi_08': action_profile[80],
        'weights_path': weights_path,
    }


def save_norm_figure(norm_name, seed_results, save_dir, suffix=''):
    x = np.linspace(0, 1, 100)
    full_name = NORM_NAMES[norm_name]
    display = f"{norm_name}" if full_name == norm_name else f"{norm_name} ({full_name})"

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), facecolor=BG_COLOR)
    for ax in axes:
        ax.set_facecolor(BG_COLOR)

    # (1) Per-interaction reward curves
    ax = axes[0]
    for s in range(NUM_SEEDS):
        r = [e['per_int_reward'] for e in seed_results[str(s)]['episode_log']]
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

    # (2) Learned action policy
    ax = axes[1]
    for s in range(NUM_SEEDS):
        ax.plot(x, seed_results[str(s)]['action_profile'], linewidth=1.6,
                color=SEED_COLORS[s], alpha=0.7)
    ax.plot(x, x, ':', color='#888888', alpha=0.3, label='identity')
    ax.set_title("Learned Action Policy  π(opp_rep)", fontsize=10, fontweight='bold')
    ax.set_xlabel("Opponent Reputation")
    ax.set_ylabel("Cooperation Level")
    ax.set_ylim([-0.05, 1.05])
    ax.legend(fontsize=7, framealpha=0.8)
    ax.grid(True, alpha=0.25)

    # (3) Signal policy (frozen)
    ax = axes[2]
    for s in range(NUM_SEEDS):
        ax.plot(x, seed_results[str(s)]['signal_profile'], linewidth=1.6,
                color=SEED_COLORS[s], alpha=0.7)
    ax.plot(x, x, '--', color='#888888', alpha=0.4, label='honest (identity)')
    ax.set_title("Signal Policy  φ(a)  [frozen]", fontsize=10, fontweight='bold')
    ax.set_xlabel("Donor Action")
    ax.set_ylabel("Signal")
    ax.set_ylim([-0.05, 1.05])
    ax.legend(fontsize=8, framealpha=0.8)
    ax.grid(True, alpha=0.25)

    avg = np.mean([seed_results[str(s)]['last50_per_int'] for s in range(NUM_SEEDS)])
    fig.suptitle(f"Exp 2b: Action vs {display}   "
                 f"(avg per-int = {avg:.3f}, benchmark = {FULL_COOP_BENCHMARK})",
                 fontsize=12, fontweight='bold', y=1.02)
    plt.tight_layout()
    fig_path = os.path.join(save_dir, "figures", f"fig_action_vs_{norm_name}{suffix}.png")
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
            print("=" * 60)
            print(f"Exp 2b: Action vs leading-eight ({opp_mode}, init={init_mode})")
            print("=" * 60)

            for norm in norms:
                key = f"{norm}_{opp_mode}_{init_mode}"
                print(f"\n{norm} ({NORM_NAMES[norm]}, {opp_mode}, {init_mode}):", flush=True)
                results = {}
                for seed in range(NUM_SEEDS):
                    print(f"  seed {seed}...", end="", flush=True)
                    r = train_action_vs_norm(norm, seed, opp_mode=opp_mode, init_mode=init_mode)
                    results[str(seed)] = r
                    print(f" per_int={r['last50_per_int']:.3f}", flush=True)

                save_norm_figure(norm, results, save_dir,
                                 suffix=f"_{opp_mode}_{init_mode}")
                all_results[key] = results

    # Summary table
    print(f"\n{'='*60}")
    print(f"{'Setting':<45} {'Per-Int':>10} {'π(.2)':>8} {'π(.8)':>8}")
    print(f"{'-'*73}")
    for init_mode in ['const05', 'warmup_init', 'all_uniform']:
        for opp_mode in ['mean', 'ema']:
            for norm in norms:
                key = f"{norm}_{opp_mode}_{init_mode}"
                res = all_results[key]
                pi_avg = np.mean([res[str(s)]['last50_per_int'] for s in range(NUM_SEEDS)])
                pi02 = np.mean([res[str(s)]['pi_02'] for s in range(NUM_SEEDS)])
                pi08 = np.mean([res[str(s)]['pi_08'] for s in range(NUM_SEEDS)])
                name = NORM_NAMES[norm]
                label = f"{norm} ({name}, {opp_mode}, {init_mode})"
                print(f"{label:<45} {pi_avg:10.3f} {pi02:8.3f} {pi08:8.3f}")
    print(f"\nBenchmark (full coop): {FULL_COOP_BENCHMARK}")

    json_path = os.path.join(save_dir, "data", "results_action_vs_leading_eight.json")
    with open(json_path, 'w') as f:
        json.dump(all_results, f)
    print(f"\nAll results saved to {json_path}")


if __name__ == "__main__":
    main()
