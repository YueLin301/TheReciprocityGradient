"""
Exp 4 (new) — Joint π + φ optimization with ground-truth opponent access.

Agent 0 optimizes BOTH action policy π⁰ and gossip policy φ⁰ simultaneously.
Gradients flow through the REAL opponent networks (white-box / GT access).
No opponent modeling. This is the GT ceiling for joint optimization.

Mirrors the GT→Learned pattern:
  Exp 2 (GT action) → Exp 5 (learned action)
  Exp 3 (GT signal) → Exp 6 (learned signal)
  Exp 4 (GT joint)  → Exp 7 (learned joint)   ← this + next

Opponents: ProudCoop(L3) + AllD(L3), L3, L6.

Expected on ProudCoop+AllD:
  - π⁰: discriminative (cooperate w/ ProudCoop, defect w/ AllD)
  - φ⁰: discriminative (high signal for cooperators → boosts ProudCoop's self-image)
"""
import os, sys, json
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from agent import Agent
from env_batched import BatchedDonationGame, compute_per_interaction_payoff

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from exp2_action_optimization.leading_eight_agents import LeadingEightAgent
from checkpoint_utils import atomic_save_json, load_json_if_exists

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from paper_style import *
apply_style()

# Reuse ProudCoop/AllD from exp3h
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),
                                              '..', 'exp3_signal_optimization')))
from train_vs_mixed_exp3h import ProudCoopL3, AllDefectL3, _tanh_L3_signal

BATCH_SIZE = 128
NUM_SEEDS = 5
TRAIN_EPISODES = 150
B_VALUE = 10.0
DELTA = 0.98
BETA = 5.0
SNAPSHOT_EVERY = 20
LR = 5e-2

BENCHMARKS = {
    'L3': (B_VALUE - 1) / 2,           # 4.5
    'L6': (B_VALUE - 1) / 2,           # 4.5
    'ProudCoopAllD': (B_VALUE - 1) / 4, # 2.25
}


def build_opponents(opp_set, device):
    if opp_set == 'L3':
        o1 = LeadingEightAgent('Opp1', norm='L3', continuous=True, decay=0.5, beta=BETA, device=device)
        o2 = LeadingEightAgent('Opp2', norm='L3', continuous=True, decay=0.5, beta=BETA, device=device)
    elif opp_set == 'L6':
        o1 = LeadingEightAgent('Opp1', norm='L6', continuous=True, decay=None, beta=BETA, device=device)
        o2 = LeadingEightAgent('Opp2', norm='L6', continuous=True, decay=None, beta=BETA, device=device)
    elif opp_set == 'ProudCoopAllD':
        o1 = ProudCoopL3('ProudL3', device=device)
        o2 = AllDefectL3('AllD_L3', device=device)
    else:
        raise ValueError(opp_set)
    for p in list(o1.parameters()) + list(o2.parameters()):
        p.requires_grad_(False)
    return o1, o2


def train_joint_gt(seed, opp_set, init_mode):
    torch.manual_seed(seed); np.random.seed(seed)
    device = torch.device("cpu")

    agent0 = Agent(name="A_0", signal_dim=1, second_order=False, is_demo=False, device=device)
    # BOTH action and signal trainable — no hardcode
    opp1, opp2 = build_opponents(opp_set, device)

    all_params = list(agent0.get_action_params()) + list(agent0.get_signal_params())
    optimizer = optim.Adam(all_params, lr=LR)

    if init_mode == 'const05':
        per_agent_init = [{"dist": "constant", "params": {"value": 0.5}}] * 3
    else:
        per_agent_init = [{"dist": "uniform", "params": {"low": 0.0, "high": 1.0}}] * 3

    env = BatchedDonationGame(
        agents=[agent0, opp1, opp2], b=B_VALUE, continuation_prob=DELTA,
        batch_size=BATCH_SIZE, balanced_matching=True, min_rounds=3,
        per_agent_init=per_agent_init)

    benchmark = BENCHMARKS[opp_set]
    episode_log = []
    pi_snapshots, phi_snapshots = [], []
    pi_var_log, phi_var_log = [], []

    def measure_profiles():
        agent0.eval()
        x = torch.linspace(0, 1, 100)
        with torch.no_grad():
            pi_prof = [agent0.take_action(x[i].view(1, 1)).item() for i in range(100)]
            phi_prof = [agent0.send_signal(x[i].view(1),
                        own_reputation=torch.tensor([[0.5]])).item() for i in range(100)]
        agent0.train()
        return pi_prof, phi_prof

    pi_p, phi_p = measure_profiles()
    pi_var_log.append(float(np.std(pi_p)))
    phi_var_log.append(float(np.std(phi_p)))
    pi_snapshots.append({'episode': -1, 'profile': pi_p})
    phi_snapshots.append({'episode': -1, 'profile': phi_p})

    for ep in range(TRAIN_EPISODES):
        optimizer.zero_grad()
        cum_rewards, matching_seq, rep_histories = env.play_batch()
        per_int_reward, _ = compute_per_interaction_payoff(cum_rewards, matching_seq, 0)
        loss = -per_int_reward.mean()

        if loss.requires_grad:
            grads = torch.autograd.grad(loss, all_params, allow_unused=True)
            for p, g in zip(all_params, grads):
                if g is not None and not (torch.isnan(g).any() or torch.isinf(g).any()):
                    p.grad = g
            torch.nn.utils.clip_grad_norm_(all_params, max_norm=1.0)
        optimizer.step()

        episode_log.append({'episode': ep, 'per_int_reward': per_int_reward.mean().item()})
        del rep_histories

        if ep % SNAPSHOT_EVERY == 0 or ep == TRAIN_EPISODES - 1:
            pi_p, phi_p = measure_profiles()
            pi_var_log.append(float(np.std(pi_p)))
            phi_var_log.append(float(np.std(phi_p)))
            pi_snapshots.append({'episode': ep, 'profile': pi_p})
            phi_snapshots.append({'episode': ep, 'profile': phi_p})

    pi_final, phi_final = measure_profiles()
    last50 = np.mean([e['per_int_reward'] for e in episode_log[-50:]])

    result = {
        'episode_log': episode_log,
        'action_profile': pi_final,
        'signal_profile': phi_final,
        'pi_snapshots': pi_snapshots,
        'phi_snapshots': phi_snapshots,
        'pi_var': float(np.std(pi_final)),
        'phi_var': float(np.std(phi_final)),
        'last50_per_int': last50,
        'pi_disc': float(np.std(pi_final)) > 0.25,
        'phi_disc': float(np.std(phi_final)) > 0.25,
        'benchmark': benchmark,
    }

    print(f"    [{opp_set} {init_mode} seed {seed}] per-int={last50:.3f} "
          f"({last50/benchmark*100:.0f}%) std[π]={result['pi_var']:.3f} "
          f"std[φ]={result['phi_var']:.3f}", flush=True)
    return result


def plot_result(result, opp_set, init_mode, seed, save_dir):
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    ep_log = result['episode_log']
    eps = [e['episode'] for e in ep_log]
    rewards = [e['per_int_reward'] for e in ep_log]
    from scipy.ndimage import uniform_filter1d

    ax = axes[0]
    ax.plot(eps, rewards, color=PRIMARY, alpha=0.3, lw=0.6)
    sm = uniform_filter1d(np.array(rewards), size=min(15, len(rewards)//5+1))
    ax.plot(eps, sm, color=PRIMARY, lw=2.5, label='smoothed')
    ax.axhline(result['benchmark'], color=SLATE, ls=':', lw=0.8, label=f'benchmark ({result["benchmark"]})')
    ax.set_xlabel('Episode'); ax.set_ylabel('Per-int payoff')
    ax.set_title('(a) Learning curve', color=PRIMARY_DARK)
    ax.legend(fontsize=8)

    ax = axes[1]
    xg = np.linspace(0, 1, 100)
    ax.plot(xg, result['action_profile'], color=PRIMARY, lw=2.5, label='learned $\\pi^0$')
    ax.plot([0,1],[0,1], color=SLATE, ls='--', lw=0.8, alpha=0.4, label='identity')
    ax.set_xlabel('Recipient reputation $s$'); ax.set_ylabel('$\\pi^0(s)$')
    ax.set_title(f'(b) Action (std={result["pi_var"]:.2f})', color=PRIMARY_DARK)
    ax.set_xlim(0,1); ax.set_ylim(-0.02,1.02); ax.legend(fontsize=8)

    ax = axes[2]
    ax.plot(xg, result['signal_profile'], color=NAVY, lw=2.5, label='learned $\\varphi^0$')
    ax.plot([0,1],[0,1], color=SLATE, ls='--', lw=0.8, alpha=0.4, label='identity')
    ax.set_xlabel('Donor action $a$'); ax.set_ylabel('$\\varphi^0(a)$')
    ax.set_title(f'(c) Signal (std={result["phi_var"]:.2f})', color=PRIMARY_DARK)
    ax.set_xlim(0,1); ax.set_ylim(-0.02,1.02); ax.legend(fontsize=8)

    fig.suptitle(f'Exp 4 GT Joint: {opp_set} / {init_mode} / seed {seed}  '
                 f'(per-int={result["last50_per_int"]:.3f})',
                 fontsize=11, color=PRIMARY_DARK, y=1.01)
    fig.tight_layout()
    fn = os.path.join(save_dir, 'figures',
                      f'fig_exp4gt_{opp_set}_{init_mode}_seed{seed}.png')
    fig.savefig(fn, bbox_inches='tight'); plt.close()


def main():
    save_dir = os.path.dirname(os.path.abspath(__file__))
    json_path = os.path.join(save_dir, 'data', 'results_exp4_joint_gt.json')
    all_results = load_json_if_exists(json_path) or {}

    opp_sets = ['ProudCoopAllD', 'L3', 'L6']
    init_modes = ['const05', 'all_uniform']

    print(f"\n{'='*70}")
    print(f"  Exp 4 — Joint π+φ optimization with GT opponent access")
    print(f"  {len(opp_sets)} opp sets × {len(init_modes)} inits × {NUM_SEEDS} seeds")
    print(f"{'='*70}")

    for opp_set in opp_sets:
        for init_mode in init_modes:
            cfg = f"{opp_set}_{init_mode}"
            all_results.setdefault(cfg, {})
            for seed in range(NUM_SEEDS):
                if str(seed) in all_results[cfg]:
                    print(f"  [{cfg} seed {seed}] skip"); continue
                result = train_joint_gt(seed, opp_set, init_mode)
                all_results[cfg][str(seed)] = result
                plot_result(result, opp_set, init_mode, seed, save_dir)
                atomic_save_json(all_results, json_path)

    # Summary
    print(f"\n{'='*70}\n  SUMMARY\n{'='*70}")
    for opp_set in opp_sets:
        for init_mode in init_modes:
            cfg = f"{opp_set}_{init_mode}"
            seeds = all_results[cfg]
            pi_v = [seeds[s]['pi_var'] for s in seeds]
            phi_v = [seeds[s]['phi_var'] for s in seeds]
            perint = [seeds[s]['last50_per_int'] for s in seeds]
            bench = BENCHMARKS[opp_set]
            pi_disc = sum(1 for v in pi_v if v > 0.25)
            phi_disc = sum(1 for v in phi_v if v > 0.25)
            print(f"  {cfg:30s}  per-int={np.mean(perint):.3f} ({np.mean(perint)/bench*100:.0f}%)  "
                  f"π disc={pi_disc}/{len(pi_v)}  φ disc={phi_disc}/{len(phi_v)}")


if __name__ == "__main__":
    main()
