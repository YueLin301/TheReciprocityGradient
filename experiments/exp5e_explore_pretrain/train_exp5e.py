"""
Exp 5e — Exploration-based pretraining (no oracle access).

Replaces Exp 5d's synthetic-grid pretraining (which queries the true opponent
at arbitrary (a,s) inputs — effectively oracle access) with a random-policy
exploration phase that collects diverse interaction data from actual gameplay.

The key change:
  - Exp 5d: pretrain estimators on a 32x32 uniform grid by calling
            opp.forward(synthetic_input) — oracle access.
  - Exp 5e: pretrain estimators on real interaction data collected by
            playing K_EXPLORE_PRETRAIN episodes with a uniform-random
            policy π(s) ~ Uniform[0,1]. No oracle access — the opponent
            is observed only through natural interactions.

Everything else (FactoredMLP estimator, freeze after pretrain, exploration
noise during training, T_OUTER=25) is identical to Exp 5d.

This is standard MBRL practice: explore before exploit.
"""
import os, sys, json, copy
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from agent import Agent
from env_batched import BatchedDonationGame, compute_per_interaction_payoff

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from exp2_action_optimization.leading_eight_agents import LeadingEightAgent
from opponent_modeling_utils import (
    build_estimators, EstimatedOpponent, DataBuffer, RecordingWrapper,
    train_action_estimator, train_signal_estimator,
)
from checkpoint_utils import atomic_save_json, load_json_if_exists
from paper_style import *
apply_style()

BATCH_SIZE = 128
NUM_SEEDS = 10
# ── PP_v2-style budget match: slow LR + more epochs ─────────────────────────
T_OUTER = 400                     # 40 → 400: 10× more budget to match slowed LR
N_EVAL = 2
N_EXPLORE = 5
K_EST = 100
K_EXPLORE_PRETRAIN = 100
K_EST_PRETRAIN = 800
N_TRAIN = 10
W_BUFFER = 3
B_VALUE = 10.0
DELTA = 0.98
BETA = 5.0
LR_POLICY = 5e-4                  # 5e-3 → 5e-4: 10× slower, helps borderline seeds
LR_EST = 1e-2
LR_EST_PRETRAIN = 1e-2
EXPLORATION_STD = 0.5
FREEZE_OPP_MODELS = True
FULL_COOP_BENCHMARK = (B_VALUE - 1) / 2
# Total π update budget: LR_POLICY · T_OUTER · N_TRAIN = 5e-4 · 400 · 10 = 2.0
# (old was 5e-3 · 40 · 10 = 2.0 — same total, finer-grained Adam steps)


class RandomPolicyAgent(nn.Module):
    """Thin wrapper: replaces action with uniform-random, delegates everything else.
    Preserves _hardcode_signal so the env handles signal correctly."""
    def __init__(self, base_agent):
        super().__init__()
        self.base = base_agent
        self.name = base_agent.name
        self.signal_dim = base_agent.signal_dim
        self.second_order = base_agent.second_order
        self.is_demo = base_agent.is_demo
        self.device = base_agent.device
        # Copy hardcode flags so env handles signal/action the same way
        for attr in ('_hardcode_signal', '_hardcode_signal_norm',
                     '_hardcode_signal_beta', '_hardcode_action',
                     '_hardcode_action_decay'):
            if hasattr(base_agent, attr):
                setattr(self, attr, getattr(base_agent, attr))

    def batched_take_action(self, recipient_rep_histories, own_rep_histories=None):
        batch = recipient_rep_histories.shape[0]
        return torch.rand(batch, 1)

    def reset_reputation(self, *args, **kwargs):
        return self.base.reset_reputation(*args, **kwargs)

    @property
    def reputation_history(self):
        return self.base.reputation_history

    @reputation_history.setter
    def reputation_history(self, v):
        self.base.reputation_history = v

    def append_reputation(self, *args, **kwargs):
        return self.base.append_reputation(*args, **kwargs)

    def get_observation(self):
        return self.base.get_observation()


class ExplorationEnv(BatchedDonationGame):
    def __init__(self, *args, noise_agent_idx=0, noise_std=EXPLORATION_STD, **kwargs):
        super().__init__(*args, **kwargs)
        self.noise_agent_idx = noise_agent_idx
        self.noise_std = noise_std

    def _batched_take_action(self, agent, rep_histories, own_rep_histories=None):
        det = super()._batched_take_action(agent, rep_histories, own_rep_histories)
        if agent is self.agents[self.noise_agent_idx]:
            noise = torch.randn_like(det) * self.noise_std
            return (det + noise).clamp(0, 1)
        return det


def explore_and_pretrain(agent0, opp1, opp2, buf1, buf2,
                         act_est_1, sig_est_1, act_est_2, sig_est_2,
                         per_agent_init, device):
    """Phase 0: Random-policy exploration + estimator pretraining.

    1. Replace agent0 with a uniform-random policy wrapper.
    2. Play K_EXPLORE_PRETRAIN episodes, recording all opponent interactions.
    3. Train estimators on this diverse buffer.
    4. Return final MSE.

    No oracle access: opponents are observed only through natural gameplay.
    """
    random_agent = RandomPolicyAgent(agent0)
    rec_opp1 = RecordingWrapper(opp1, buf1)
    rec_opp2 = RecordingWrapper(opp2, buf2)

    explore_env = BatchedDonationGame(
        agents=[random_agent, rec_opp1, rec_opp2], b=B_VALUE, c=1.0,
        continuation_prob=DELTA, batch_size=BATCH_SIZE,
        balanced_matching=True, min_rounds=3, per_agent_init=per_agent_init)

    # Use a single buffer iteration for ALL exploration episodes so nothing is discarded
    buf1.new_iter(); buf2.new_iter()
    n_transitions = 0
    with torch.no_grad():
        for ep in range(K_EXPLORE_PRETRAIN):
            _, matching_seq, rep_histories = explore_env.play_batch()
            n_transitions += len(matching_seq[0]) * BATCH_SIZE
            del rep_histories

    print(f"    Exploration: {K_EXPLORE_PRETRAIN} episodes, ~{n_transitions} transitions collected")

    # Train estimators on the exploration buffer
    loss_a1 = train_action_estimator(act_est_1, buf1, K_EST_PRETRAIN, LR_EST_PRETRAIN)
    loss_a2 = train_action_estimator(act_est_2, buf2, K_EST_PRETRAIN, LR_EST_PRETRAIN)
    loss_s1 = train_signal_estimator(sig_est_1, buf1, K_EST_PRETRAIN, LR_EST_PRETRAIN)
    loss_s2 = train_signal_estimator(sig_est_2, buf2, K_EST_PRETRAIN, LR_EST_PRETRAIN)
    return loss_a1, loss_s1, loss_a2, loss_s2, n_transitions


def build_setup(seed):
    torch.manual_seed(seed); np.random.seed(seed)
    device = torch.device("cpu")
    agent0 = Agent(name="A_0", signal_dim=1, second_order=False, is_demo=False, device=device)
    agent0.set_hardcode_signal()

    opp1 = LeadingEightAgent(name="Opp1", norm='L6', continuous=True,
                              decay=None, beta=BETA, device=device)
    opp2 = LeadingEightAgent(name="Opp2", norm='L6', continuous=True,
                              decay=None, beta=BETA, device=device)
    for p in list(opp1.parameters()) + list(opp2.parameters()):
        p.requires_grad_(False)
    per_agent_init = [{"dist": "uniform", "params": {"low": 0.0, "high": 1.0}}] * 3
    return agent0, opp1, opp2, per_agent_init


def run_exp5e(seed):
    agent0, opp1, opp2, per_agent_init = build_setup(seed)
    device = torch.device("cpu")

    act_est_1, sig_est_1 = build_estimators('Factored')
    act_est_2, sig_est_2 = build_estimators('Factored')
    for m in (act_est_1, sig_est_1, act_est_2, sig_est_2):
        m.to(device)

    buf1 = DataBuffer(window=W_BUFFER)
    buf2 = DataBuffer(window=W_BUFFER)

    # ── Phase 0: Exploration-based pretraining (no oracle) ──
    print(f"  [seed {seed}] Random-policy exploration + estimator pretraining...", flush=True)
    pre_a1, pre_s1, pre_a2, pre_s2, n_trans = explore_and_pretrain(
        agent0, opp1, opp2, buf1, buf2,
        act_est_1, sig_est_1, act_est_2, sig_est_2,
        per_agent_init, device)
    print(f"    pretrain MSE: pi1={pre_a1:.4f} pi2={pre_a2:.4f} "
          f"phi1={pre_s1:.4f} phi2={pre_s2:.4f}", flush=True)

    # Reset buffers for the training phase
    buf1 = DataBuffer(window=W_BUFFER)
    buf2 = DataBuffer(window=W_BUFFER)
    rec_opp1 = RecordingWrapper(opp1, buf1)
    rec_opp2 = RecordingWrapper(opp2, buf2)

    policy_opt = optim.Adam(agent0.get_action_params(), lr=LR_POLICY)
    benchmark = FULL_COOP_BENCHMARK

    history = {
        'real_perint': [], 'surrogate_perint': [],
        'real_perint_per_ep': [], 'surrogate_perint_per_ep': [],
        'pi_var': [], 'pi_snapshots': [],
        'rep_log': [],
        'est_loss_a1': [], 'est_loss_a2': [],
        'est_loss_s1': [], 'est_loss_s2': [],
        'pretrain_mse': {'a1': pre_a1, 'a2': pre_a2, 's1': pre_s1, 's2': pre_s2},
        'n_explore_transitions': n_trans,
    }

    def _measure_pi_var():
        agent0.eval()
        x = torch.linspace(0, 1, 100)
        with torch.no_grad():
            prof = [agent0.take_action(x[i].view(1, 1)).item() for i in range(100)]
        agent0.train()
        return float(np.std(prof)), prof

    v, p = _measure_pi_var()
    history['pi_var'].append(v)
    history['pi_snapshots'].append({'outer_iter': -1, 'profile': p})

    for t in range(T_OUTER):
        buf1.new_iter(); buf2.new_iter()

        # Phase 1a: deterministic eval
        eval_env = BatchedDonationGame(
            agents=[agent0, rec_opp1, rec_opp2], b=B_VALUE, c=1.0,
            continuation_prob=DELTA, batch_size=BATCH_SIZE,
            balanced_matching=True, min_rounds=3, per_agent_init=per_agent_init)
        agent0.eval()
        real_eps, rep = [], None
        with torch.no_grad():
            for _ in range(N_EVAL):
                cum_rewards, matching_seq, rep_histories = eval_env.play_batch()
                pi_payoff, _ = compute_per_interaction_payoff(cum_rewards, matching_seq, 0)
                real_eps.append(pi_payoff.mean().item())
                rep = [rep_histories[i][:, -1, 0].mean().item() for i in range(3)]
                del rep_histories
        agent0.train()

        # Phase 1b: noisy exploration
        explore_env = ExplorationEnv(
            agents=[agent0, rec_opp1, rec_opp2], b=B_VALUE, c=1.0,
            continuation_prob=DELTA, batch_size=BATCH_SIZE,
            balanced_matching=True, min_rounds=3, per_agent_init=per_agent_init,
            noise_agent_idx=0, noise_std=EXPLORATION_STD)
        agent0.eval()
        with torch.no_grad():
            for _ in range(N_EXPLORE):
                _, _, rep_histories = explore_env.play_batch()
                del rep_histories
        agent0.train()

        history['real_perint_per_ep'].append(real_eps)
        history['real_perint'].append(float(np.mean(real_eps)))
        history['rep_log'].append(rep)

        # Phase 2: est update (skipped if frozen)
        if FREEZE_OPP_MODELS:
            history['est_loss_a1'].append(0.0)
            history['est_loss_a2'].append(0.0)
            history['est_loss_s1'].append(0.0)
            history['est_loss_s2'].append(0.0)
        else:
            history['est_loss_a1'].append(train_action_estimator(act_est_1, buf1, K_EST, LR_EST))
            history['est_loss_a2'].append(train_action_estimator(act_est_2, buf2, K_EST, LR_EST))
            history['est_loss_s1'].append(train_signal_estimator(sig_est_1, buf1, K_EST, LR_EST))
            history['est_loss_s2'].append(train_signal_estimator(sig_est_2, buf2, K_EST, LR_EST))

        # Phase 3: surrogate policy update
        est_opp1 = EstimatedOpponent("EstOpp1", act_est_1, sig_est_1, device=device)
        est_opp2 = EstimatedOpponent("EstOpp2", act_est_2, sig_est_2, device=device)
        for m in (act_est_1, sig_est_1, act_est_2, sig_est_2):
            for pp in m.parameters(): pp.requires_grad_(False)

        surrogate_env = BatchedDonationGame(
            agents=[agent0, est_opp1, est_opp2], b=B_VALUE, c=1.0,
            continuation_prob=DELTA, batch_size=BATCH_SIZE,
            balanced_matching=True, min_rounds=3, per_agent_init=per_agent_init)

        surr_eps = []
        for _ in range(N_TRAIN):
            policy_opt.zero_grad()
            cum_rewards, matching_seq, rep_histories = surrogate_env.play_batch()
            pi_payoff, _ = compute_per_interaction_payoff(cum_rewards, matching_seq, 0)
            loss = -pi_payoff.mean()
            if loss.requires_grad:
                grads = torch.autograd.grad(loss, agent0.get_action_params(), allow_unused=True)
                for pp, g in zip(agent0.get_action_params(), grads):
                    if g is not None and not (torch.isnan(g).any() or torch.isinf(g).any()):
                        pp.grad = g
                torch.nn.utils.clip_grad_norm_(agent0.get_action_params(), max_norm=1.0)
            policy_opt.step()
            surr_eps.append(pi_payoff.mean().item())
            del rep_histories

        history['surrogate_perint_per_ep'].append(surr_eps)
        history['surrogate_perint'].append(float(np.mean(surr_eps)))

        for m in (act_est_1, sig_est_1, act_est_2, sig_est_2):
            for pp in m.parameters(): pp.requires_grad_(True)

        v, pprof = _measure_pi_var()
        history['pi_var'].append(v)
        history['pi_snapshots'].append({'outer_iter': t, 'profile': pprof})

        print(f"    [seed {seed}] t={t:2d} | real={history['real_perint'][-1]:.3f} "
              f"surr={history['surrogate_perint'][-1]:.3f}  std[pi]={v:.3f}  "
              f"MSE pi={history['est_loss_a1'][-1]:.4f}/{history['est_loss_a2'][-1]:.4f}",
              flush=True)

    # Final eval
    agent0.eval()
    final = 0.0
    with torch.no_grad():
        env = BatchedDonationGame(
            agents=[agent0, opp1, opp2], b=B_VALUE, c=1.0,
            continuation_prob=DELTA, batch_size=BATCH_SIZE,
            balanced_matching=True, min_rounds=3, per_agent_init=per_agent_init)
        for _ in range(5):
            cum_rewards, matching_seq, _ = env.play_batch()
            pi_payoff, _ = compute_per_interaction_payoff(cum_rewards, matching_seq, 0)
            final += pi_payoff.mean().item()
    final /= 5
    v_final, prof_final = _measure_pi_var()
    history['final_perint'] = final
    history['final_pi_var'] = v_final
    history['final_pi_profile'] = prof_final
    history['benchmark'] = benchmark

    print(f"  [seed {seed}] FINAL={final:.3f} ({final/benchmark*100:.0f}%)  "
          f"std[pi]={v_final:.3f}", flush=True)
    return history


def plot_one_run(history, seed, save_dir):
    x_grid = np.linspace(0, 1, 100)
    benchmark = history['benchmark']
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    t_axis = np.arange(len(history['real_perint']))

    ax = axes[0, 0]
    ax.plot(t_axis + 0.5, history['real_perint'], '-', color=PRIMARY, lw=2.0, label='real')
    ax.plot(t_axis + 0.5, history['surrogate_perint'], '--', color=C_BASELINE, lw=1.6, label='surrogate')
    ax.axhline(benchmark, color=SLATE, ls=':', label=f'benchmark {benchmark}')
    ax.axhline(3.25, color=ROSE, ls=':', label='Exp 2-40 disc target')
    ax.set_xlabel('Outer iter'); ax.set_ylabel('Per-int payoff')
    ax.set_title('(a) Payoff', fontweight='bold', loc='left', color=PRIMARY_DARK)
    ax.legend(fontsize=7, frameon=False)

    ax = axes[0, 1]
    ax.plot(t_axis, history['est_loss_a1'], color=TEAL, label=r'$\hat\pi^1$')
    ax.plot(t_axis, history['est_loss_a2'], color=TEAL, ls='--', label=r'$\hat\pi^2$')
    ax.plot(t_axis, history['est_loss_s1'], color=C_BASELINE, label=r'$\hat\varphi^1$')
    ax.plot(t_axis, history['est_loss_s2'], color=C_BASELINE, ls='--', label=r'$\hat\varphi^2$')
    ax.set_yscale('log')
    ax.set_xlabel('Outer iter'); ax.set_ylabel('MSE')
    ax.set_title('(b) Est. MSE (log)', fontweight='bold', loc='left', color=PRIMARY_DARK)
    ax.legend(fontsize=7, frameon=False)

    ax = axes[0, 2]
    var_t = np.arange(-1, len(history['pi_var']) - 1)
    ax.plot(var_t, history['pi_var'], 'o-', color=PRIMARY)
    ax.axhline(0.25, color=ROSE, ls='--', label='disc threshold')
    ax.set_xlabel('Outer iter'); ax.set_ylabel(r'std[$\pi^0$]')
    ax.set_title('(c) Policy variance', fontweight='bold', loc='left', color=PRIMARY_DARK)
    ax.legend(fontsize=7, frameon=False)

    ax = axes[1, 0]
    cmap = plt.cm.magma
    snaps = history['pi_snapshots']
    for i, snap in enumerate(snaps):
        col = cmap(i / max(len(snaps) - 1, 1))
        lab = f"t={snap['outer_iter']}" if (i % max(len(snaps) // 5, 1) == 0 or i == len(snaps) - 1) else None
        ax.plot(x_grid, snap['profile'], color=col, lw=1.2, alpha=0.85, label=lab)
    ax.set_xlabel('Recipient rep'); ax.set_ylabel(r'$\pi^0$')
    ax.set_title(r'(d) $\pi^0$ evolution', fontweight='bold', loc='left', color=PRIMARY_DARK)
    ax.set_ylim([-0.05, 1.05])
    ax.legend(fontsize=6, frameon=False, loc='upper left', ncol=2)

    ax = axes[1, 1]
    ax.plot(x_grid, history['final_pi_profile'], color=PRIMARY, lw=2,
            label=f'final (std={history["final_pi_var"]:.2f})')
    ax.plot(x_grid, snaps[0]['profile'], color=SLATE, lw=1.2, ls=':', label='init')
    ax.set_xlabel('Recipient rep'); ax.set_ylabel(r'$\pi^0$')
    ax.set_title('(e) Final policy', fontweight='bold', loc='left', color=PRIMARY_DARK)
    ax.set_ylim([-0.05, 1.05])
    ax.legend(fontsize=7, frameon=False)

    ax = axes[1, 2]
    rep_log = np.array(history['rep_log'])
    ax.plot(t_axis, rep_log[:, 0], color=PRIMARY, lw=1.4, label='Agent 0')
    ax.plot(t_axis, rep_log[:, 1], color=TEAL, lw=1.2, label='Opp1')
    ax.plot(t_axis, rep_log[:, 2], color=C_BASELINE, lw=1.2, label='Opp2')
    ax.set_xlabel('Outer iter'); ax.set_ylabel('Mean rep')
    ax.set_title('(f) Reputation evolution', fontweight='bold', loc='left', color=PRIMARY_DARK)
    ax.set_ylim([-0.05, 1.05]); ax.legend(fontsize=7, frameon=False)

    plt.suptitle(
        f'Exp 5e — L6 FactoredMLP explore-pretrain | seed {seed} | '
        f'final={history["final_perint"]:.3f} std[pi]={history["final_pi_var"]:.2f}',
        fontsize=11, fontweight='bold', color=PRIMARY_DARK)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    path = os.path.join(save_dir, 'figures', f'fig_exp5e_seed{seed}.png')
    plt.savefig(path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f'    -> {path}')


def _strip(h):
    keep = ['real_perint', 'surrogate_perint', 'real_perint_per_ep', 'surrogate_perint_per_ep',
            'est_loss_a1', 'est_loss_a2', 'est_loss_s1', 'est_loss_s2',
            'pi_var', 'rep_log', 'pi_snapshots',
            'final_perint', 'final_pi_profile', 'final_pi_var',
            'benchmark', 'pretrain_mse', 'n_explore_transitions']
    return {k: h[k] for k in keep if k in h}


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--seed-start', type=int, default=0)
    ap.add_argument('--seed-end', type=int, default=None, help='exclusive')
    ap.add_argument('--json-suffix', type=str, default='')
    args = ap.parse_args()

    save_dir = os.path.dirname(os.path.abspath(__file__))
    suffix = args.json_suffix if args.json_suffix else ''
    json_path = os.path.join(save_dir, 'data', f'results_exp5e{suffix}.json')
    all_results = load_json_if_exists(json_path) or {}
    cell = 'L6_Factored_mean_all_uniform_sigidentity'
    all_results.setdefault(cell, {})

    seed_start = args.seed_start
    seed_end = args.seed_end if args.seed_end is not None else NUM_SEEDS

    print(f"\n{'='*70}\n  Exp 5e — Exploration-based pretraining (no oracle)\n"
          f"  Cell: {cell} | seeds: {seed_start}..{seed_end}\n"
          f"  Explore: {K_EXPLORE_PRETRAIN} eps random policy + "
          f"{K_EST_PRETRAIN} est steps | Freeze: {FREEZE_OPP_MODELS}\n{'='*70}")

    for seed in range(seed_start, seed_end):
        if str(seed) in all_results[cell]:
            print(f"  Seed {seed}: [skip]"); continue
        print(f"\n  Seed {seed}:")
        hist = run_exp5e(seed)
        all_results[cell][str(seed)] = _strip(hist)
        plot_one_run(hist, seed, save_dir)
        atomic_save_json(all_results, json_path)

    print(f"\n{'='*70}\n  SUMMARY\n{'='*70}")
    seeds_data = all_results[cell]
    stds = [seeds_data[s]['final_pi_var'] for s in seeds_data]
    perints = [seeds_data[s]['final_perint'] for s in seeds_data]
    n_disc = sum(1 for s in stds if s > 0.25)
    print(f"  std[pi] mean={np.mean(stds):.3f} +- {np.std(stds):.3f}   "
          f"disc seeds: {n_disc}/{len(stds)}")
    print(f"  per-int mean={np.mean(perints):.3f} +- {np.std(perints):.3f}")


if __name__ == "__main__":
    main()
