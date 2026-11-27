"""
Exp 7 — JOINT π + φ optimization with learned opponent models.

Closes RQ1c: in deployment, an agent should learn BOTH action and signal
pathways simultaneously. Exps 5 and 6 froze one pathway each to isolate
causal factors; Exp 7 verifies the method works when both are jointly
trained against the same surrogate-env loss.

Setup:
  • Agent 0: trainable π^0 AND trainable φ^0.
  • Opponents (frozen, real): four configurations:
      - HybridCoop(L3) + AllD(L3) (matches Exp 4 — primary cell)
      - Leading-eight L3 or L6 (matches Exp 5)
      - ProudCoop(L3) + AllD(L3) (matches Exp 6 / Exp 3h)
  • Opponent models: MLP (winner of Exp 5/6) — 4 estimator nets.

Loop (per outer iter t):
  Phase 1 — INTERACT in real env, record (input, output) pairs per opp.
  Phase 2 — train each estimator with K_EST MSE steps on sliding-window data.
  Phase 3 — POLICY UPDATE: surrogate env (real opps replaced by EstimatedOpponent),
            backprop loss into BOTH agent0.action_params AND agent0.signal_params.
            Uses PP_v2 recipe: πLR=3e-5, φLR=3e-3 (100× ratio), budget-matched.

Metrics tracked: per-int payoff (welfare) + std[π] (action variance) +
std[σ] (signal variance). All variance computed on uniform input grids.
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
from exp2_action_optimization.leading_eight_agents import LeadingEightAgent
from opponent_modeling_utils import (
    build_estimators, EstimatedOpponent, DataBuffer, RecordingWrapper,
    train_action_estimator, train_signal_estimator,
)
from checkpoint_utils import atomic_save_json, load_json_if_exists

# Need ProudCoopL3 / AllDefectL3 from exp6 (signal OM)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),
                                              '..', 'exp6_signal_opponent_modeling')))
from train_exp6_signal_only import ProudCoopL3, AllDefectL3, _tanh_L3_signal

# HybridCoop from exp4_joint_gt — gives BOTH π and φ a gradient signal
from pathlib import Path as _Path
sys.path.insert(0, str(_Path(__file__).resolve().parents[1] / "exp4_joint_gt"))
from hybrid_opponents import HybridCoopL3

mpl.rcParams.update({
    'font.family': 'sans-serif', 'font.size': 9,
    'axes.spines.top': False, 'axes.spines.right': False,
    'axes.linewidth': 0.6,
})

# ══════════════════════════════════════════════════════════════════════════════
# Hyperparameters (matched to Exp 4/5)
# ══════════════════════════════════════════════════════════════════════════════
BATCH_SIZE = 128
NUM_SEEDS = 5
T_OUTER = 200               # 15 → 200: PP_v2 recipe — more budget for slow π
N_PLAY = 5
K_EST = 100
N_TRAIN = 50                # 10 → 50: PP_v2 recipe — more inner updates per OM refit
W_BUFFER = 3
B_VALUE = 10.0
DELTA = 0.98
BETA = 5.0
# ── PP_v2 recipe: asymmetric LR, π 100× slower than φ ────────────────────────
LR_POLICY_PI = 3e-5          # was 5e-3 shared — now 167× slower
LR_POLICY_PHI = 3e-3         # was 5e-3 shared — slightly slower
LR_EST = 1e-2
# Total inner updates: 200 × 50 = 10 000 (was 150). π budget 3e-5·10 000 = 0.3,
# roughly 40% of PP_v2's 0.75 — partial budget match, OM drift limits further scaling.

# Each opponent set has its own benchmark
BENCHMARKS = {
    'L3':           (B_VALUE - 1) / 2,    # 4.5
    'L6':           (B_VALUE - 1) / 2,    # 4.5
    'ProudCoopAllD': (B_VALUE - 1) / 4,   # 2.25 (Exp 3h benchmark, AllD ~half interactions)
    'HybridCoopAllD': (B_VALUE - 1) / 4,  # 2.25 (Exp 4 joint setup — primary cell)
}


# ══════════════════════════════════════════════════════════════════════════════
# Opponent-set builders
# ══════════════════════════════════════════════════════════════════════════════

def build_opponents(opp_set: str, device):
    """Return (opp1, opp2). All frozen."""
    if opp_set == 'L3':
        opp1 = LeadingEightAgent(name='Opp1', norm='L3', continuous=True,
                                  decay=0.5, beta=BETA, device=device)
        opp2 = LeadingEightAgent(name='Opp2', norm='L3', continuous=True,
                                  decay=0.5, beta=BETA, device=device)
    elif opp_set == 'L6':
        opp1 = LeadingEightAgent(name='Opp1', norm='L6', continuous=True,
                                  decay=0.5, beta=BETA, device=device)
        opp2 = LeadingEightAgent(name='Opp2', norm='L6', continuous=True,
                                  decay=0.5, beta=BETA, device=device)
    elif opp_set == 'ProudCoopAllD':
        opp1 = ProudCoopL3(name='ProudL3', device=device)
        opp2 = AllDefectL3(name='AllD_L3', device=device)
    elif opp_set == 'HybridCoopAllD':
        opp1 = HybridCoopL3(name='HybridL3', device=device)
        opp2 = AllDefectL3(name='AllD_L3', device=device)
    else:
        raise ValueError(f"Unknown opp_set: {opp_set}")
    for p in list(opp1.parameters()) + list(opp2.parameters()):
        p.requires_grad_(False)
    return opp1, opp2


def per_agent_init_for(opp_set: str, init_mode: str):
    """Build per_agent_init list of length 3 for the chosen setup."""
    if init_mode == 'const05':
        return [{"dist": "constant", "params": {"value": 0.5}}] * 3
    elif init_mode == 'all_uniform':
        return [{"dist": "uniform", "params": {"low": 0.0, "high": 1.0}}] * 3
    else:
        raise ValueError(f"Unknown init_mode: {init_mode}")


# ══════════════════════════════════════════════════════════════════════════════
# Variance metrics on learned policies
# ══════════════════════════════════════════════════════════════════════════════

def _measure_action_variance(agent0):
    """std_s[π^0(s)] over uniform rep grid (own_rep history fixed at 0.5)."""
    was_training = agent0.training
    agent0.eval()
    x = torch.linspace(0, 1, 100)
    with torch.no_grad():
        # take_action expects rep_history of shape [seq, signal_dim]
        prof = [agent0.take_action(x[i].view(1, 1)).item() for i in range(100)]
    if was_training:
        agent0.train()
    return float(np.std(prof)), prof


def _measure_signal_variance(agent0):
    """std_a[φ^0(a, own_rep=0.5)] over uniform action grid."""
    was_training = agent0.training
    agent0.eval()
    x = torch.linspace(0, 1, 100)
    dummy_own_rep = torch.tensor([[0.5]])
    with torch.no_grad():
        prof = [agent0.send_signal(x[i].view(1), own_reputation=dummy_own_rep).item()
                for i in range(100)]
    if was_training:
        agent0.train()
    return float(np.std(prof)), prof


# ══════════════════════════════════════════════════════════════════════════════
# Main per-run training loop
# ══════════════════════════════════════════════════════════════════════════════

def run_exp7(opp_set: str, init_mode: str, seed: int) -> dict:
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device("cpu")
    benchmark = BENCHMARKS[opp_set]

    # ── Agent 0: trainable π AND trainable φ ──
    agent0 = Agent(name="A_0", signal_dim=1, second_order=False, is_demo=False, device=device)
    # No hardcoding — both nets remain trainable.

    # ── Real frozen opponents ──
    opp1, opp2 = build_opponents(opp_set, device)

    # ── Estimators (MLP per Exp 4/5 winner) ──
    act_est_1, sig_est_1 = build_estimators('MLP')
    act_est_2, sig_est_2 = build_estimators('MLP')
    for m in (act_est_1, sig_est_1, act_est_2, sig_est_2):
        m.to(device)

    # ── Data buffers + recording wrappers ──
    buf1 = DataBuffer(window=W_BUFFER)
    buf2 = DataBuffer(window=W_BUFFER)
    rec_opp1 = RecordingWrapper(opp1, buf1)
    rec_opp2 = RecordingWrapper(opp2, buf2)

    # ── JOINT optimizer: PP_v2 recipe — asymmetric LR per parameter group ──
    pi_params = list(agent0.get_action_params())
    phi_params = list(agent0.get_signal_params())
    joint_params = pi_params + phi_params
    policy_opt = optim.Adam([
        {'params': pi_params,  'lr': LR_POLICY_PI},
        {'params': phi_params, 'lr': LR_POLICY_PHI},
    ])

    per_agent_init = per_agent_init_for(opp_set, init_mode)

    history = {
        'real_perint': [], 'surrogate_perint': [],
        'real_perint_per_ep': [], 'surrogate_perint_per_ep': [],
        'act_est_loss_1': [], 'act_est_loss_2': [],
        'sig_est_loss_1': [], 'sig_est_loss_2': [],
        'pi_var': [], 'phi_var': [],
        'pi_snapshots': [], 'phi_snapshots': [],
        'rep_log': [],
    }

    # Initial snapshots
    pi_v, pi_p = _measure_action_variance(agent0)
    phi_v, phi_p = _measure_signal_variance(agent0)
    history['pi_var'].append(pi_v); history['phi_var'].append(phi_v)
    history['pi_snapshots'].append({'outer_iter': -1, 'profile': pi_p})
    history['phi_snapshots'].append({'outer_iter': -1, 'profile': phi_p})

    for t in range(T_OUTER):
        # ── Phase 1: INTERACT (real env, record data) ──
        buf1.new_iter()
        buf2.new_iter()

        real_env = BatchedDonationGame(
            agents=[agent0, rec_opp1, rec_opp2], b=B_VALUE, c=1.0,
            continuation_prob=DELTA, batch_size=BATCH_SIZE,
            balanced_matching=True, min_rounds=3,
            per_agent_init=per_agent_init)

        agent0.eval()
        this_iter_real_ep = []
        this_iter_rep = None
        with torch.no_grad():
            for _ in range(N_PLAY):
                cum_rewards, matching_seq, rep_histories = real_env.play_batch()
                pi_payoff, _ = compute_per_interaction_payoff(cum_rewards, matching_seq, 0)
                this_iter_real_ep.append(pi_payoff.mean().item())
                this_iter_rep = [rep_histories[i][:, -1, 0].mean().item() for i in range(3)]
                del rep_histories
        agent0.train()
        history['real_perint_per_ep'].append(this_iter_real_ep)
        history['real_perint'].append(float(np.mean(this_iter_real_ep)))
        history['rep_log'].append(this_iter_rep)

        # ── Phase 2: ESTIMATOR UPDATE ──
        history['act_est_loss_1'].append(train_action_estimator(act_est_1, buf1, K_EST, LR_EST))
        history['act_est_loss_2'].append(train_action_estimator(act_est_2, buf2, K_EST, LR_EST))
        history['sig_est_loss_1'].append(train_signal_estimator(sig_est_1, buf1, K_EST, LR_EST))
        history['sig_est_loss_2'].append(train_signal_estimator(sig_est_2, buf2, K_EST, LR_EST))

        # ── Phase 3: JOINT POLICY UPDATE (surrogate env) ──
        est_opp1 = EstimatedOpponent("EstOpp1", act_est_1, sig_est_1, device=device)
        est_opp2 = EstimatedOpponent("EstOpp2", act_est_2, sig_est_2, device=device)
        for m in (act_est_1, sig_est_1, act_est_2, sig_est_2):
            for p in m.parameters():
                p.requires_grad_(False)

        surrogate_env = BatchedDonationGame(
            agents=[agent0, est_opp1, est_opp2], b=B_VALUE, c=1.0,
            continuation_prob=DELTA, batch_size=BATCH_SIZE,
            balanced_matching=True, min_rounds=3,
            per_agent_init=per_agent_init)

        this_iter_surr_ep = []
        for _ in range(N_TRAIN):
            policy_opt.zero_grad()
            cum_rewards, matching_seq, rep_histories = surrogate_env.play_batch()
            pi_payoff, _ = compute_per_interaction_payoff(cum_rewards, matching_seq, 0)
            loss = -pi_payoff.mean()
            if loss.requires_grad:
                grads = torch.autograd.grad(loss, joint_params, allow_unused=True)
                for p, g in zip(joint_params, grads):
                    if g is not None and not (torch.isnan(g).any() or torch.isinf(g).any()):
                        p.grad = g
                torch.nn.utils.clip_grad_norm_(joint_params, max_norm=1.0)
            policy_opt.step()
            this_iter_surr_ep.append(pi_payoff.mean().item())
            del rep_histories
        history['surrogate_perint_per_ep'].append(this_iter_surr_ep)
        history['surrogate_perint'].append(float(np.mean(this_iter_surr_ep)))

        # Snapshot policies AFTER joint update
        pi_v, pi_p = _measure_action_variance(agent0)
        phi_v, phi_p = _measure_signal_variance(agent0)
        history['pi_var'].append(pi_v); history['phi_var'].append(phi_v)
        history['pi_snapshots'].append({'outer_iter': t, 'profile': pi_p})
        history['phi_snapshots'].append({'outer_iter': t, 'profile': phi_p})

        # Unfreeze estimators for next outer iter
        for m in (act_est_1, sig_est_1, act_est_2, sig_est_2):
            for p in m.parameters():
                p.requires_grad_(True)

        print(f"    [{opp_set} {init_mode} seed={seed}] t={t:2d} | "
              f"real={history['real_perint'][-1]:.3f}  "
              f"surr={history['surrogate_perint'][-1]:.3f}  "
              f"std[π]={history['pi_var'][-1]:.3f}  "
              f"std[φ]={history['phi_var'][-1]:.3f}  "
              f"lossπ̂={history['act_est_loss_1'][-1]:.4f}/"
              f"{history['act_est_loss_2'][-1]:.4f}  "
              f"lossφ̂={history['sig_est_loss_1'][-1]:.4f}/"
              f"{history['sig_est_loss_2'][-1]:.4f}", flush=True)

    # ── Final eval in real env ──
    agent0.eval()
    final_perint = 0.0
    with torch.no_grad():
        real_env = BatchedDonationGame(
            agents=[agent0, opp1, opp2], b=B_VALUE, c=1.0,
            continuation_prob=DELTA, batch_size=BATCH_SIZE,
            balanced_matching=True, min_rounds=3,
            per_agent_init=per_agent_init)
        for _ in range(5):
            cum_rewards, matching_seq, _ = real_env.play_batch()
            pi_payoff, _ = compute_per_interaction_payoff(cum_rewards, matching_seq, 0)
            final_perint += pi_payoff.mean().item()
    final_perint /= 5

    pi_v_final, pi_p_final = _measure_action_variance(agent0)
    phi_v_final, phi_p_final = _measure_signal_variance(agent0)
    history['final_perint'] = final_perint
    history['final_pi_profile'] = pi_p_final
    history['final_phi_profile'] = phi_p_final
    history['final_pi_var'] = pi_v_final
    history['final_phi_var'] = phi_v_final
    history['benchmark'] = benchmark

    print(f"  [{opp_set} {init_mode} seed={seed}]  FINAL real per-int = {final_perint:.3f} "
          f"({final_perint / benchmark * 100:.0f}% of {benchmark})  "
          f"std[π]={pi_v_final:.3f}  std[φ]={phi_v_final:.3f}")

    return history


# ══════════════════════════════════════════════════════════════════════════════
# Plotting (3×3 layout — track both pathways)
# ══════════════════════════════════════════════════════════════════════════════

def plot_one_run(history: dict, opp_set: str, init_mode: str, seed: int, save_dir: str):
    x_grid = np.linspace(0, 1, 100)
    benchmark = history['benchmark']
    fig, axes = plt.subplots(3, 3, figsize=(16, 12))

    t_axis = np.arange(len(history['real_perint']))

    # (a) Per-int payoff
    ax = axes[0, 0]
    for t, eps in enumerate(history['real_perint_per_ep']):
        xs = [t + (i + 0.5) / len(eps) for i in range(len(eps))]
        ax.scatter(xs, eps, s=8, color='#3498DB', alpha=0.35)
    for t, eps in enumerate(history['surrogate_perint_per_ep']):
        xs = [t + (i + 0.5) / len(eps) for i in range(len(eps))]
        ax.scatter(xs, eps, s=8, color='#E67E22', alpha=0.35, marker='s')
    ax.plot(t_axis + 0.5, history['real_perint'], '-', color='#2C3E50', linewidth=2.0, label='real')
    ax.plot(t_axis + 0.5, history['surrogate_perint'], '--', color='#C0392B', linewidth=1.6, label='surrogate')
    ax.axhline(benchmark, color='#AAA', linestyle=':', linewidth=0.8, label=f'benchmark {benchmark}')
    ax.set_xlabel('Outer iter'); ax.set_ylabel('Per-int payoff')
    ax.set_title('(a) Payoff', fontweight='bold', loc='left')
    ax.legend(fontsize=7, frameon=False, loc='lower right'); ax.grid(True, alpha=0.2)

    # (b) Estimator MSE
    ax = axes[0, 1]
    ax.plot(t_axis, history['act_est_loss_1'], color='#2ECC71', label=r'$\hat\pi^1$')
    ax.plot(t_axis, history['act_est_loss_2'], color='#27AE60', linestyle='--', label=r'$\hat\pi^2$')
    ax.plot(t_axis, history['sig_est_loss_1'], color='#E74C3C', label=r'$\hat\varphi^1$')
    ax.plot(t_axis, history['sig_est_loss_2'], color='#C0392B', linestyle='--', label=r'$\hat\varphi^2$')
    ax.set_xlabel('Outer iter'); ax.set_ylabel('MSE')
    ax.set_title('(b) Estimator MSE (log)', fontweight='bold', loc='left')
    ax.set_yscale('log'); ax.legend(fontsize=7, frameon=False); ax.grid(True, alpha=0.2)

    # (c) Objective value gap
    ax = axes[0, 2]
    gap = np.array(history['real_perint']) - np.array(history['surrogate_perint'])
    ax.plot(t_axis, gap, 'o-', color='#9B59B6')
    ax.axhline(0, color='#AAA', linestyle=':')
    ax.set_xlabel('Outer iter'); ax.set_ylabel(r'$J_{\rm real} - J_{\rm surr}$')
    ax.set_title('(c) Objective gap', fontweight='bold', loc='left'); ax.grid(True, alpha=0.2)

    # (d) Action policy evolution
    ax = axes[1, 0]
    snaps = history['pi_snapshots']
    cmap = plt.cm.plasma
    for i, snap in enumerate(snaps):
        col = cmap(i / max(len(snaps) - 1, 1))
        lab = f"t={snap['outer_iter']}" if (i % max(len(snaps) // 5, 1) == 0 or i == len(snaps) - 1) else None
        ax.plot(x_grid, snap['profile'], color=col, linewidth=1.2, alpha=0.85, label=lab)
    ax.plot(x_grid, x_grid, '--', color='#AAA', linewidth=0.8, label='identity')
    ax.set_xlabel('Recipient rep'); ax.set_ylabel(r'$\pi^0$ action')
    ax.set_title(r'(d) Action $\pi^0$ evolution', fontweight='bold', loc='left')
    ax.set_xlim([0, 1]); ax.set_ylim([-0.05, 1.05])
    ax.legend(fontsize=6, frameon=False, loc='upper left', ncol=2); ax.grid(True, alpha=0.2)

    # (e) Final π
    ax = axes[1, 1]
    ax.plot(x_grid, history['final_pi_profile'], color='#3498DB', linewidth=2,
            label=f'final $\\pi^0$ (std={history["final_pi_var"]:.2f})')
    ax.plot(x_grid, snaps[0]['profile'], color='#888', linewidth=1.2, linestyle=':', label='init $\\pi^0$')
    ax.plot(x_grid, x_grid, '--', color='#AAA', linewidth=0.8, label='identity')
    ax.set_xlabel('Recipient rep'); ax.set_ylabel(r'$\pi^0$ action')
    ax.set_title('(e) Final action policy', fontweight='bold', loc='left')
    ax.set_xlim([0, 1]); ax.set_ylim([-0.05, 1.05])
    ax.legend(fontsize=7, frameon=False, loc='upper left'); ax.grid(True, alpha=0.2)

    # (f) Variance trajectories (both π and φ)
    ax = axes[1, 2]
    var_t = np.arange(-1, len(history['pi_var']) - 1)
    ax.plot(var_t, history['pi_var'], 'o-', color='#3498DB', label=r'std[$\pi^0$]')
    ax.plot(var_t, history['phi_var'], 's-', color='#E74C3C', label=r'std[$\varphi^0$]')
    ax.axhline(0, color='#CCC', linewidth=0.6)
    ax.set_xlabel('Outer iter'); ax.set_ylabel('Policy variance')
    ax.set_title('(f) Variance over time', fontweight='bold', loc='left')
    ax.set_ylim([-0.02, 0.55])
    ax.legend(fontsize=7, frameon=False); ax.grid(True, alpha=0.2)

    # (g) Signal policy evolution
    ax = axes[2, 0]
    snaps = history['phi_snapshots']
    for i, snap in enumerate(snaps):
        col = cmap(i / max(len(snaps) - 1, 1))
        lab = f"t={snap['outer_iter']}" if (i % max(len(snaps) // 5, 1) == 0 or i == len(snaps) - 1) else None
        ax.plot(x_grid, snap['profile'], color=col, linewidth=1.2, alpha=0.85, label=lab)
    ax.plot(x_grid, x_grid, '--', color='#AAA', linewidth=0.8, label='identity')
    ax.set_xlabel('Donor action'); ax.set_ylabel(r'$\varphi^0$ signal')
    ax.set_title(r'(g) Signal $\varphi^0$ evolution', fontweight='bold', loc='left')
    ax.set_xlim([0, 1]); ax.set_ylim([-0.05, 1.05])
    ax.legend(fontsize=6, frameon=False, loc='upper left', ncol=2); ax.grid(True, alpha=0.2)

    # (h) Final φ
    ax = axes[2, 1]
    ax.plot(x_grid, history['final_phi_profile'], color='#E74C3C', linewidth=2,
            label=f'final $\\varphi^0$ (std={history["final_phi_var"]:.2f})')
    ax.plot(x_grid, snaps[0]['profile'], color='#888', linewidth=1.2, linestyle=':', label='init $\\varphi^0$')
    ax.plot(x_grid, x_grid, '--', color='#AAA', linewidth=0.8, label='identity')
    ax.set_xlabel('Donor action'); ax.set_ylabel(r'$\varphi^0$ signal')
    ax.set_title('(h) Final signal policy', fontweight='bold', loc='left')
    ax.set_xlim([0, 1]); ax.set_ylim([-0.05, 1.05])
    ax.legend(fontsize=7, frameon=False, loc='lower right'); ax.grid(True, alpha=0.2)

    # (i) Reputation evolution
    ax = axes[2, 2]
    rep_log = np.array(history['rep_log'])
    ax.plot(t_axis, rep_log[:, 0], color='#3498DB', linewidth=1.4, label='Agent 0')
    ax.plot(t_axis, rep_log[:, 1], color='#2ECC71', linewidth=1.2, label='Opp1')
    ax.plot(t_axis, rep_log[:, 2], color='#E67E22', linewidth=1.2, label='Opp2')
    ax.set_xlabel('Outer iter'); ax.set_ylabel('Mean last-signal rep')
    ax.set_title('(i) Reputation evolution', fontweight='bold', loc='left')
    ax.set_ylim([-0.05, 1.05])
    ax.legend(fontsize=7, frameon=False, loc='lower right'); ax.grid(True, alpha=0.2)

    pct = history['final_perint'] / benchmark * 100
    plt.suptitle(
        f'Exp 7 joint π+φ — {opp_set} × {init_mode} — seed {seed} | '
        f'final per-int = {history["final_perint"]:.3f} ({pct:.0f}% of {benchmark})  '
        f'| std[π]={history["final_pi_var"]:.2f}  std[φ]={history["final_phi_var"]:.2f}',
        fontsize=11, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    fname = f'fig_exp7_joint_{opp_set}_{init_mode}_seed{seed}.png'
    path = os.path.join(save_dir, 'figures', fname)
    plt.savefig(path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f'    → {path}')


# ══════════════════════════════════════════════════════════════════════════════
# Main with checkpoint/resume
# ══════════════════════════════════════════════════════════════════════════════

def _strip_for_json(hist):
    keep = ['real_perint', 'surrogate_perint',
            'real_perint_per_ep', 'surrogate_perint_per_ep',
            'act_est_loss_1', 'act_est_loss_2',
            'sig_est_loss_1', 'sig_est_loss_2',
            'pi_var', 'phi_var', 'rep_log',
            'pi_snapshots', 'phi_snapshots',
            'final_perint', 'final_pi_profile', 'final_phi_profile',
            'final_pi_var', 'final_phi_var', 'benchmark']
    return {k: hist[k] for k in keep if k in hist}


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--opp-set', choices=['HybridCoopAllD', 'L3', 'L6', 'ProudCoopAllD'],
                    default=None, help='If set, only run this opp_set (for parallel sharding).')
    ap.add_argument('--seed-start', type=int, default=0)
    ap.add_argument('--seed-end', type=int, default=None, help='exclusive; defaults to NUM_SEEDS')
    ap.add_argument('--json-suffix', type=str, default='',
                    help='If set, append to JSON filename (e.g. "_HybridCoopAllD") for per-shard results.')
    args = ap.parse_args()

    save_dir = os.path.dirname(os.path.abspath(__file__))
    os.makedirs(os.path.join(save_dir, "figures"), exist_ok=True)
    os.makedirs(os.path.join(save_dir, "data"), exist_ok=True)
    suffix = args.json_suffix if args.json_suffix else ''
    json_path = os.path.join(save_dir, "data", f"results_exp7_joint{suffix}.json")

    # PP_v2 recipe targets HybridCoopAllD primarily; others kept for completeness
    if args.opp_set:
        opp_sets = [args.opp_set]
    else:
        opp_sets = ['HybridCoopAllD', 'L3', 'L6', 'ProudCoopAllD']
    init_modes = ['all_uniform']      # PP_v2 used uniform init; drop const05 to save runtime
    seed_start = args.seed_start
    seed_end = args.seed_end if args.seed_end is not None else NUM_SEEDS

    all_results = load_json_if_exists(json_path)
    if all_results:
        done = sum(len(v) for v in all_results.values())
        print(f"[resume] loaded {done} completed (config, seed) pairs")

    total_configs = len(opp_sets) * len(init_modes)
    print(f"\n{'='*70}\n"
          f"  Exp 7 — JOINT π+φ with learned opponent models (PP_v2 recipe)\n"
          f"  {total_configs} configs × {NUM_SEEDS} seeds = {total_configs * NUM_SEEDS} runs\n"
          f"{'='*70}")

    config_idx = 0
    for opp_set in opp_sets:
        for init_mode in init_modes:
            config_idx += 1
            key = f"{opp_set}_{init_mode}"
            print(f"\n[{config_idx}/{total_configs}] {key}  (benchmark={BENCHMARKS[opp_set]})")
            all_results.setdefault(key, {})
            for seed in range(seed_start, seed_end):
                if str(seed) in all_results[key]:
                    print(f"  Seed {seed}: [skip]", flush=True)
                    continue
                print(f"  Seed {seed}:", flush=True)
                hist = run_exp7(opp_set=opp_set, init_mode=init_mode, seed=seed)
                all_results[key][str(seed)] = _strip_for_json(hist)
                plot_one_run(hist, opp_set, init_mode, seed, save_dir)
                atomic_save_json(all_results, json_path)
                done = sum(len(v) for v in all_results.values())
                print(f"    [checkpoint] {done} pairs saved")

    print(f"\n{'='*70}\n  SUMMARY\n{'='*70}")
    rows = []
    for key in sorted(all_results.keys()):
        seeds = all_results[key]
        avg = float(np.mean([seeds[s]['final_perint'] for s in seeds]))
        bench = next(iter(seeds.values()))['benchmark']
        n = len(seeds)
        rows.append((key, avg, bench, n))
    for key, avg, bench, n in sorted(rows, key=lambda x: -x[1] / x[2]):
        pct = avg / bench * 100
        print(f"  {key:<35}  avg={avg:6.3f} ({pct:3.0f}% of {bench})  n={n}")
    print(f"{'='*70}")

    atomic_save_json(all_results, json_path)
    print(f"Final results saved to {json_path}")


if __name__ == "__main__":
    main()
