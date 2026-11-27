"""
Exp 3h: Signal optimization against ProudCoop(L3 signal) + AllD(L3 signal).

Opponent 1 — ProudCooperator with L3 signal:
    action = sigmoid(10*(own_rep[-1] - 0.5))  (cooperates when own rep is high)
    signal = tanh L3: 1 - 0.25*(1-tanh(β(a-0.5)))*(1+tanh(β(r-0.5)))

Opponent 2 — AllDefector with L3 signal:
    action = 0  (never cooperates)
    signal = tanh L3  (same assessment norm)

Agent 0 (learner):
    action = mean(opp_rep) [frozen identity]
    signal = trainable NN

150 episodes, 10 seeds, const05 init.
"""
import os
import sys
import json
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from typing import Optional

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from agent import Agent
from env_batched import BatchedDonationGame, compute_per_interaction_payoff

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from checkpoint_utils import atomic_save_json, load_json_if_exists

mpl.rcParams.update({
    'font.family': 'sans-serif', 'font.size': 9,
    'axes.spines.top': False, 'axes.spines.right': False,
    'axes.linewidth': 0.6,
})

BATCH_SIZE = 128
NUM_SEEDS = 5
TRAIN_EPISODES = 100
B_VALUE = 10.0
DELTA = 0.98
# With AllD opponent: best case = cooperate only with ProudCoop
# per_int = (-1 + 0 + b + 0) / 4 = (b-1)/4
FULL_COOP_BENCHMARK = (B_VALUE - 1) / 4  # 2.25
SNAPSHOT_EVERY = 15
BETA = 5.0
T_ACTION = 10.0


def _tanh_L3_signal(action, recip_rep, beta=BETA):
    """Tanh L3 (Simple Standing): Good unless defect against Good."""
    sx = torch.tanh(beta * (action - 0.5))
    sy = torch.tanh(beta * (recip_rep - 0.5))
    return (1.0 - 0.25 * (1 - sx) * (1 + sy)).clamp(0, 1)


class ProudCoopL3(nn.Module):
    """ProudCoop action (sigmoid T=10 on own_rep) + L3 tanh signal (memory-1)."""

    def __init__(self, name, signal_dim=1, device=None):
        super().__init__()
        self.name = name
        self.signal_dim = signal_dim
        self.second_order = True
        self.device = device if device is not None else torch.device("cpu")
        self.register_buffer("init_token", torch.full((1, signal_dim), 0.5))
        self.register_buffer("reputation_history", torch.full((1, signal_dim), 0.5))
        self.to(self.device)

    def reset_reputation(self, initial_reputation=None):
        if initial_reputation is not None:
            self.reputation_history = initial_reputation.view(1, self.signal_dim).to(
                dtype=torch.float32, device=self.device)
        else:
            self.reputation_history = self.init_token.clone()

    def append_reputation(self, signal_t):
        sig = signal_t.to(self.device).view(1, self.signal_dim)
        self.reputation_history = torch.cat([self.reputation_history, sig], dim=0)

    def get_observation(self):
        return self.reputation_history

    def take_action(self, recipient_reputation):
        own_rep = self.reputation_history[-1].mean()
        return torch.sigmoid(T_ACTION * (own_rep - 0.5)).view(1)

    def send_signal(self, donor_action, own_reputation=None, donor_reputation=None):
        a = donor_action.to(self.device).view(-1)
        if own_reputation is not None:
            recip_rep = own_reputation[-1].mean()
        else:
            recip_rep = self.reputation_history[-1].mean()
        return _tanh_L3_signal(a, recip_rep).view(self.signal_dim)


class AllDefectL3(nn.Module):
    """AllD action + L3 tanh signal (memory-1)."""

    def __init__(self, name, signal_dim=1, device=None):
        super().__init__()
        self.name = name
        self.signal_dim = signal_dim
        self.second_order = True
        self.device = device if device is not None else torch.device("cpu")
        self.register_buffer("init_token", torch.full((1, signal_dim), 0.5))
        self.register_buffer("reputation_history", torch.full((1, signal_dim), 0.5))
        self.to(self.device)

    def reset_reputation(self, initial_reputation=None):
        if initial_reputation is not None:
            self.reputation_history = initial_reputation.view(1, self.signal_dim).to(
                dtype=torch.float32, device=self.device)
        else:
            self.reputation_history = self.init_token.clone()

    def append_reputation(self, signal_t):
        sig = signal_t.to(self.device).view(1, self.signal_dim)
        self.reputation_history = torch.cat([self.reputation_history, sig], dim=0)

    def get_observation(self):
        return self.reputation_history

    def take_action(self, recipient_reputation):
        return torch.zeros(1, device=self.device)

    def send_signal(self, donor_action, own_reputation=None, donor_reputation=None):
        a = donor_action.to(self.device).view(-1)
        if own_reputation is not None:
            recip_rep = own_reputation[-1].mean()
        else:
            recip_rep = self.reputation_history[-1].mean()
        return _tanh_L3_signal(a, recip_rep).view(self.signal_dim)


def train_signal_3h(seed, init_mode='const05'):
    torch.manual_seed(seed)
    np.random.seed(seed)

    agent0 = Agent(name="A_0", signal_dim=1, second_order=False, is_demo=False)
    opp_proud = ProudCoopL3(name="ProudL3")
    opp_alld = AllDefectL3(name="AllD_L3")

    agent0.set_hardcode_action(decay=None)  # frozen identity mean

    signal_params = agent0.get_signal_params()
    optimizer = optim.Adam(signal_params, lr=5e-2)

    if init_mode == 'const05':
        per_agent_init = [
            {"dist": "constant", "params": {"value": 0.5}},
            {"dist": "constant", "params": {"value": 0.5}},
            {"dist": "constant", "params": {"value": 0.5}},
        ]
    else:  # all_uniform
        per_agent_init = [
            {"dist": "uniform", "params": {"low": 0.0, "high": 1.0}},
            {"dist": "uniform", "params": {"low": 0.0, "high": 1.0}},
            {"dist": "uniform", "params": {"low": 0.0, "high": 1.0}},
        ]

    agents = [agent0, opp_proud, opp_alld]
    env = BatchedDonationGame(
        agents=agents, b=B_VALUE, continuation_prob=DELTA, batch_size=BATCH_SIZE,
        balanced_matching=True, min_rounds=3,
        per_agent_init=per_agent_init)

    episode_log = []
    rep_log = []
    signal_snapshots = []

    # Snapshot TRUE initialization (before any training)
    agent0.eval()
    with torch.no_grad():
        x_snap = torch.linspace(0, 1, 100)
        dummy_own_rep = torch.tensor([[0.5]])
        profile = [agent0.send_signal(x.view(1), own_reputation=dummy_own_rep).item()
                   for x in x_snap]
        signal_snapshots.append({'episode': -1, 'profile': profile})
    agent0.train()

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

        # Track reputation — extract then free graph
        with torch.no_grad():
            agent_reps = [rep_histories[i][:, -1, 0].mean().item() for i in range(3)]
            rep_log.append(agent_reps)
        del rep_histories

        episode_log.append({
            'episode': ep,
            'per_int_reward': per_int_reward.mean().item(),
        })

        # Snapshot: eval mode to avoid LayerNorm corruption
        if ep % SNAPSHOT_EVERY == 0 or ep == TRAIN_EPISODES - 1:
            agent0.eval()
            with torch.no_grad():
                x_snap = torch.linspace(0, 1, 100)
                dummy_own_rep = torch.tensor([[0.5]])
                profile = [agent0.send_signal(x.view(1), own_reputation=dummy_own_rep).item()
                           for x in x_snap]
                signal_snapshots.append({'episode': ep, 'profile': profile})
            agent0.train()

    # Final profiles
    x_vals = torch.linspace(0, 1, 100)
    agent0.eval()
    with torch.no_grad():
        action_profile = [agent0.take_action(x.view(1, 1)).item() for x in x_vals]
        dummy_own_rep = torch.tensor([[0.5]])
        signal_profile = [agent0.send_signal(x.view(1), own_reputation=dummy_own_rep).item()
                          for x in x_vals]
    agent0.train()

    last50 = float(np.mean([e['per_int_reward'] for e in episode_log[-50:]]))
    sig = np.array(signal_profile)
    spread = np.mean(sig[80:]) - np.mean(sig[:20])
    disc_type = 'disc' if spread > 0.3 else ('part' if spread > 0.1 else 'flat')
    print(f"  Seed {seed}: per_int={last50:.3f}, spread={spread:.3f} ({disc_type})")

    return {
        'episode_log': episode_log,
        'action_profile': action_profile,
        'signal_profile': signal_profile,
        'signal_snapshots': signal_snapshots,
        'rep_log': rep_log,
        'last50_per_int': last50,
        'signal_spread': float(spread),
        'disc_type': disc_type,
    }


def plot_seed_details(seed_idx, result, save_dir, suffix=''):
    x = np.linspace(0, 1, 100)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    # (a) Learning curve
    ax = axes[0, 0]
    rewards = [e['per_int_reward'] for e in result['episode_log']]
    ax.plot(rewards, alpha=0.2, color='#3498DB', linewidth=0.5, label='raw')
    sm = [rewards[0]]
    for v in rewards[1:]:
        sm.append(0.05 * v + 0.95 * sm[-1])
    ax.plot(sm, color='#3498DB', linewidth=1.5, label='smoothed')
    ax.axhline(y=FULL_COOP_BENCHMARK, color='#AAA', linestyle='--', linewidth=0.8,
               label=f'benchmark ({FULL_COOP_BENCHMARK})')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Per-interaction payoff')
    ax.set_title('(a) Learning curve', fontweight='bold', loc='left')
    ax.legend(loc='lower right', frameon=False, fontsize=7)
    ax.grid(True, alpha=0.2)

    # (b) Reputation evolution
    ax = axes[0, 1]
    rep_log = np.array(result['rep_log'])
    ax.plot(rep_log[:, 0], color='#3498DB', linewidth=1.0, alpha=0.8, label='Agent 0')
    ax.plot(rep_log[:, 1], color='#2ECC71', linewidth=1.0, alpha=0.8, label='ProudCoop(L3)')
    ax.plot(rep_log[:, 2], color='#E74C3C', linewidth=1.0, alpha=0.8, label='AllD(L3)')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Mean reputation (last signal)')
    ax.set_title('(b) Reputation evolution', fontweight='bold', loc='left')
    ax.set_ylim([-0.05, 1.05])
    ax.legend(loc='upper right', frameon=True, fancybox=False, edgecolor='#CCC',
              framealpha=0.95, fontsize=7)
    ax.grid(True, alpha=0.2)

    # (c) Signal policy evolution
    ax = axes[1, 0]
    snapshots = result['signal_snapshots']
    n_snap = len(snapshots)
    cmap = plt.cm.plasma
    for i, snap in enumerate(snapshots):
        color = cmap(i / max(n_snap - 1, 1))
        ax.plot(x, snap['profile'], color=color, linewidth=1.2, alpha=0.8,
                label=f"ep {snap['episode']}")
    ax.plot(x, x, '--', color='#AAA', linewidth=0.8, label='identity')
    ax.set_xlabel('Donor action')
    ax.set_ylabel('Signal')
    ax.set_title('(c) Signal policy evolution', fontweight='bold', loc='left')
    ax.set_xlim([0, 1]); ax.set_ylim([-0.05, 1.05])
    ax.legend(loc='upper left', frameon=True, fancybox=False, edgecolor='#CCC',
              framealpha=0.95, fontsize=6, ncol=2)
    ax.grid(True, alpha=0.2)

    # (d) Final signal policy
    ax = axes[1, 1]
    ax.plot(x, result['signal_profile'], color='#3498DB', linewidth=2, label='learned $\\varphi$')
    ax.plot(x, x, '--', color='#AAA', linewidth=0.8, label='identity')
    ax.set_xlabel('Donor action')
    ax.set_ylabel('Signal')
    ax.set_title(f'(d) Final signal ({result["disc_type"]}, spread={result["signal_spread"]:.2f})',
                 fontweight='bold', loc='left')
    ax.set_xlim([0, 1]); ax.set_ylim([-0.05, 1.05])
    ax.legend(loc='lower right', frameon=False, fontsize=8)
    ax.grid(True, alpha=0.2)

    plt.suptitle(f'Exp 3h — Seed {seed_idx} (per-int={result["last50_per_int"]:.3f})',
                 fontsize=12, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    path = os.path.join(save_dir, 'figures', f'fig_3h_seed{seed_idx}{suffix}_details.png')
    plt.savefig(path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f'    → {path}')


def _strip_for_json(r):
    return {k: r[k] for k in
            ['episode_log', 'action_profile', 'signal_profile',
             'last50_per_int', 'signal_spread', 'disc_type']}


def main():
    save_dir = os.path.dirname(os.path.abspath(__file__))
    os.makedirs(os.path.join(save_dir, "figures"), exist_ok=True)
    os.makedirs(os.path.join(save_dir, "data"), exist_ok=True)
    json_path = os.path.join(save_dir, "data", "results_signal_vs_mixed_3h.json")

    print(f"Exp 3h: ProudCoop(L3) + AllD(L3), {NUM_SEEDS} seeds, {TRAIN_EPISODES} episodes")

    save_all = load_json_if_exists(json_path)
    if save_all:
        done = sum(len(v) for v in save_all.values())
        print(f"[resume] loaded {done} completed (init, seed) pairs from {json_path}")

    for init_mode in ['const05', 'all_uniform']:
        print(f"\n  Init: {init_mode}")
        save_all.setdefault(init_mode, {})
        for seed in range(NUM_SEEDS):
            if str(seed) in save_all[init_mode]:
                print(f"  Seed {seed}: [skip — already in JSON]", flush=True)
                continue
            print(f"  Seed {seed}:", flush=True)
            result = train_signal_3h(seed, init_mode=init_mode)
            plot_seed_details(seed, result, save_dir, suffix=f'_{init_mode}')
            save_all[init_mode][str(seed)] = _strip_for_json(result)
            atomic_save_json(save_all, json_path)

    print(f"\n{'='*60}")
    for init_mode in save_all:
        res = save_all[init_mode]
        disc_count = sum(1 for s in res if res[s]['disc_type'] == 'disc')
        part_count = sum(1 for s in res if res[s]['disc_type'] == 'part')
        flat_count = sum(1 for s in res if res[s]['disc_type'] == 'flat')
        avg = np.mean([res[s]['last50_per_int'] for s in res])
        print(f"  {init_mode}: disc={disc_count} part={part_count} flat={flat_count} avg={avg:.3f}")
    print(f"{'='*60}")
    print(f"Results saved to {json_path}")


if __name__ == "__main__":
    main()
