"""
Exp 9 — Scalability sweep (RQ3).

Tests how the method scales when the opponent population grows from N=3
to N=15. Two estimator strategies:
  • Per-opponent: maintain N-1 separate (π̂, φ̂) pairs → O(N) parameters.
  • Amortized:    one shared (π̂, φ̂) conditioned on opp-ID embedding → O(1).

Setup:
  • N ∈ {3, 5, 10, 15} opponents (N-1 are non-learner; mix L3 / L6 alternating).
  • Action-only training: agent 0 trains π^0, signal hardcoded identity.
    (Signal-side scalability is similar surface area, deferred to future work.)
  • MLP estimator (winner from Exp 5/6).

Mirror Exp 5's training loop with the generalization that we now have N-1 opps.
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

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from agent import Agent
from env_batched import BatchedDonationGame, compute_per_interaction_payoff

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from exp2_action_optimization.leading_eight_agents import LeadingEightAgent
from opponent_modeling_utils import (
    build_estimators, EstimatedOpponent,
    AmortizedActionEstimator, AmortizedSignalEstimator, AmortizedEstimatedOpponent,
    DataBuffer, RecordingWrapper,
    train_action_estimator, train_signal_estimator,
    train_amortized_action_estimator, train_amortized_signal_estimator,
)
from checkpoint_utils import atomic_save_json, load_json_if_exists

mpl.rcParams.update({
    'font.family': 'sans-serif', 'font.size': 9,
    'axes.spines.top': False, 'axes.spines.right': False,
    'axes.linewidth': 0.6,
})

# ══════════════════════════════════════════════════════════════════════════════
# Hyperparameters (matched to Exp 5)
# ══════════════════════════════════════════════════════════════════════════════
BATCH_SIZE = 128
NUM_SEEDS = 3
T_OUTER = 12              # slightly fewer than Exp 5's 15 to manage compute at large N
N_PLAY = 4                # episodes per outer iter (real env)
K_EST = 100               # MSE steps per estimator
N_TRAIN = 8
W_BUFFER = 3
B_VALUE = 10.0
C_VALUE = 1.0   # may be overridden per-call (e.g., signal pathway uses c=5)
DELTA = 0.98
BETA = 5.0
LR_POLICY = 5e-3
LR_EST = 1e-2
FULL_COOP_BENCHMARK = (B_VALUE - 1) / 2  # 4.5 (full cooperation reference)


# ══════════════════════════════════════════════════════════════════════════════
# Population builder: N-1 opps alternating L3 / L6
# ══════════════════════════════════════════════════════════════════════════════

def build_opp_pop(N: int, device):
    """N-1 frozen opponents, alternating L3 and L6 (continuous tanh, EMA)."""
    n_opps = N - 1
    opps = []
    for i in range(n_opps):
        norm = 'L3' if i % 2 == 0 else 'L6'
        opp = LeadingEightAgent(name=f'Opp{i+1}', norm=norm, continuous=True,
                                 decay=0.5, beta=BETA, device=device)
        for p in opp.parameters():
            p.requires_grad_(False)
        opps.append(opp)
    return opps


# ══════════════════════════════════════════════════════════════════════════════
# Training loop (per N, per estimator strategy, per seed)
# ══════════════════════════════════════════════════════════════════════════════

def run_exp9(N: int, est_strategy: str, seed: int, pathway: str = 'action') -> dict:
    """est_strategy: 'per_opp' or 'amortized'. pathway: 'action', 'signal', 'joint'."""
    torch.manual_seed(seed); np.random.seed(seed)
    device = torch.device("cpu")
    n_opps = N - 1

    # ── Agent 0: configure trainable params based on pathway ──
    agent0 = Agent(name="A_0", signal_dim=1, second_order=False, is_demo=False, device=device)
    if pathway == 'action':
        agent0.set_hardcode_signal()
    elif pathway == 'signal':
        agent0.set_hardcode_action(decay=None)
    # 'joint': both trainable, no hardcoding

    # ── Real frozen opps ──
    opps = build_opp_pop(N, device)

    # ── Estimators ──
    if est_strategy == 'per_opp':
        # N-1 independent (π̂, φ̂) MLPs
        action_ests = []
        signal_ests = []
        for _ in range(n_opps):
            a, s = build_estimators('MLP')
            action_ests.append(a); signal_ests.append(s)
        all_est_params = []
        for m in action_ests + signal_ests:
            m.to(device)
            all_est_params.extend(list(m.parameters()))
        n_est_params = sum(p.numel() for p in all_est_params)
    elif est_strategy == 'amortized':
        action_est = AmortizedActionEstimator(n_opps=n_opps, opp_emb_dim=8, hidden=16).to(device)
        signal_est = AmortizedSignalEstimator(n_opps=n_opps, opp_emb_dim=8, hidden=16).to(device)
        n_est_params = (sum(p.numel() for p in action_est.parameters())
                        + sum(p.numel() for p in signal_est.parameters()))
    else:
        raise ValueError(est_strategy)

    # ── Per-opp data buffers + recording wrappers ──
    buffers = [DataBuffer(window=W_BUFFER) for _ in range(n_opps)]
    rec_opps = [RecordingWrapper(opps[i], buffers[i]) for i in range(n_opps)]

    # ── Policy optimizer for agent 0 ──
    if pathway == 'action':
        train_params = list(agent0.get_action_params())
    elif pathway == 'signal':
        train_params = list(agent0.get_signal_params())
    else:  # joint
        train_params = list(agent0.get_action_params()) + list(agent0.get_signal_params())
    policy_opt = optim.Adam(train_params, lr=LR_POLICY)

    # ── Per-agent init: agent 0 uniform, opps from L3/L6 warmup steady ──
    warm_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             '..', 'population_warm_up')
    steady_l3 = torch.load(os.path.join(warm_dir, 'steady_state_L3.pt'),
                            map_location='cpu', weights_only=False)['continuous_final_reps']
    steady_l6 = torch.load(os.path.join(warm_dir, 'steady_state_L6.pt'),
                            map_location='cpu', weights_only=False)['continuous_final_reps']
    per_agent_init = [{"dist": "uniform", "params": {"low": 0.0, "high": 1.0}}]
    for i in range(n_opps):
        steady = steady_l3 if i % 2 == 0 else steady_l6
        per_agent_init.append({"dist": "empirical", "params": {"values": steady}})

    history = {
        'real_perint': [], 'surrogate_perint': [],
        'real_perint_per_ep': [], 'surrogate_perint_per_ep': [],
        'est_loss_action': [], 'est_loss_signal': [],
        'pi_var': [],
        'rep_log': [],
        'N': N, 'n_opps': n_opps, 'est_strategy': est_strategy,
        'n_est_params': n_est_params,
    }

    # initial pi var
    def _measure_pi_var():
        agent0.eval()
        x = torch.linspace(0, 1, 100)
        with torch.no_grad():
            prof = [agent0.take_action(x[i].view(1, 1)).item() for i in range(100)]
        agent0.train()
        return float(np.std(prof)), prof
    v, p = _measure_pi_var()
    history['pi_var'].append(v)
    history['initial_pi_profile'] = p

    for t in range(T_OUTER):
        # ── Phase 1: INTERACT (real env, record) ──
        for buf in buffers:
            buf.new_iter()
        all_agents = [agent0] + rec_opps
        real_env = BatchedDonationGame(
            agents=all_agents, b=B_VALUE, c=C_VALUE,
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
                this_iter_rep = [rep_histories[i][:, -1, 0].mean().item() for i in range(N)]
                del rep_histories
        agent0.train()
        history['real_perint_per_ep'].append(this_iter_real_ep)
        history['real_perint'].append(float(np.mean(this_iter_real_ep)))
        history['rep_log'].append(this_iter_rep)

        # ── Phase 2: ESTIMATOR UPDATE ──
        if est_strategy == 'per_opp':
            losses_a = [train_action_estimator(action_ests[i], buffers[i], K_EST, LR_EST)
                        for i in range(n_opps)]
            losses_s = [train_signal_estimator(signal_ests[i], buffers[i], K_EST, LR_EST)
                        for i in range(n_opps)]
            history['est_loss_action'].append(float(np.nanmean(losses_a)))
            history['est_loss_signal'].append(float(np.nanmean(losses_s)))
        else:
            la = train_amortized_action_estimator(action_est, buffers, K_EST * n_opps, LR_EST)
            ls = train_amortized_signal_estimator(signal_est, buffers, K_EST * n_opps, LR_EST)
            history['est_loss_action'].append(la)
            history['est_loss_signal'].append(ls)

        # ── Phase 3: POLICY UPDATE (surrogate env) ──
        if est_strategy == 'per_opp':
            est_opps = [EstimatedOpponent(f"EstOpp{i}", action_ests[i], signal_ests[i],
                                           device=device)
                        for i in range(n_opps)]
            for m in action_ests + signal_ests:
                for p_ in m.parameters(): p_.requires_grad_(False)
        else:
            est_opps = [AmortizedEstimatedOpponent(f"AmEstOpp{i}", i, action_est, signal_est,
                                                    device=device)
                        for i in range(n_opps)]
            for p_ in list(action_est.parameters()) + list(signal_est.parameters()):
                p_.requires_grad_(False)

        surrogate_env = BatchedDonationGame(
            agents=[agent0] + est_opps, b=B_VALUE, c=C_VALUE,
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
                grads = torch.autograd.grad(loss, train_params, allow_unused=True)
                for p_, g in zip(train_params, grads):
                    if g is not None and not (torch.isnan(g).any() or torch.isinf(g).any()):
                        p_.grad = g
                torch.nn.utils.clip_grad_norm_(train_params, max_norm=1.0)
            policy_opt.step()
            this_iter_surr_ep.append(pi_payoff.mean().item())
            del rep_histories
        history['surrogate_perint_per_ep'].append(this_iter_surr_ep)
        history['surrogate_perint'].append(float(np.mean(this_iter_surr_ep)))

        # Re-enable estimator gradients for next outer iter
        if est_strategy == 'per_opp':
            for m in action_ests + signal_ests:
                for p_ in m.parameters(): p_.requires_grad_(True)
        else:
            for p_ in list(action_est.parameters()) + list(signal_est.parameters()):
                p_.requires_grad_(True)

        v, _ = _measure_pi_var()
        history['pi_var'].append(v)

        print(f"    [N={N} {est_strategy} seed={seed}] t={t:2d} | "
              f"real={history['real_perint'][-1]:.3f}  "
              f"surr={history['surrogate_perint'][-1]:.3f}  "
              f"std[π]={v:.3f}  "
              f"loss π̂={history['est_loss_action'][-1]:.4f}  "
              f"φ̂={history['est_loss_signal'][-1]:.4f}", flush=True)

    # Final eval in real env
    agent0.eval()
    final_perint = 0.0
    with torch.no_grad():
        env = BatchedDonationGame(
            agents=[agent0] + opps, b=B_VALUE, c=C_VALUE,
            continuation_prob=DELTA, batch_size=BATCH_SIZE,
            balanced_matching=True, min_rounds=3,
            per_agent_init=per_agent_init)
        for _ in range(3):
            cum_rewards, matching_seq, _ = env.play_batch()
            pi_payoff, _ = compute_per_interaction_payoff(cum_rewards, matching_seq, 0)
            final_perint += pi_payoff.mean().item()
    final_perint /= 3

    v_final, prof_final = _measure_pi_var()
    history['final_perint'] = final_perint
    history['final_pi_profile'] = prof_final
    history['final_pi_var'] = v_final
    history['benchmark'] = FULL_COOP_BENCHMARK

    print(f"  [N={N} {est_strategy} seed={seed}] FINAL real per-int = {final_perint:.3f} "
          f"({final_perint / FULL_COOP_BENCHMARK * 100:.0f}% of {FULL_COOP_BENCHMARK})  "
          f"std[π]={v_final:.3f}  est params={n_est_params}")

    return history


# ══════════════════════════════════════════════════════════════════════════════
# Plotting (per-run, 1×3)
# ══════════════════════════════════════════════════════════════════════════════

def plot_one_run(history: dict, N: int, est_strategy: str, seed: int, save_dir: str):
    x_grid = np.linspace(0, 1, 100)
    benchmark = history['benchmark']
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    t_axis = np.arange(len(history['real_perint']))

    ax = axes[0]
    for t, eps in enumerate(history['real_perint_per_ep']):
        xs = [t + (i + 0.5) / len(eps) for i in range(len(eps))]
        ax.scatter(xs, eps, s=8, color='#3498DB', alpha=0.35)
    for t, eps in enumerate(history['surrogate_perint_per_ep']):
        xs = [t + (i + 0.5) / len(eps) for i in range(len(eps))]
        ax.scatter(xs, eps, s=8, color='#E67E22', alpha=0.35, marker='s')
    ax.plot(t_axis + 0.5, history['real_perint'], '-', color='#2C3E50', linewidth=2.0, label='real')
    ax.plot(t_axis + 0.5, history['surrogate_perint'], '--', color='#C0392B', linewidth=1.6, label='surrogate')
    ax.axhline(benchmark, color='#AAA', linestyle=':', label=f'benchmark {benchmark}')
    ax.set_xlabel('Outer iter'); ax.set_ylabel('Per-int payoff')
    ax.set_title(f'(a) Payoff (N={N}, {est_strategy})', fontweight='bold', loc='left')
    ax.legend(fontsize=7, frameon=False, loc='lower right'); ax.grid(True, alpha=0.2)

    ax = axes[1]
    ax.plot(t_axis, history['est_loss_action'], color='#2ECC71', label=r'avg MSE $\hat\pi$')
    ax.plot(t_axis, history['est_loss_signal'], color='#E74C3C', label=r'avg MSE $\hat\varphi$')
    ax.set_xlabel('Outer iter'); ax.set_ylabel('MSE')
    ax.set_yscale('log')
    ax.set_title(f'(b) Estimator MSE ({est_strategy})', fontweight='bold', loc='left')
    ax.legend(fontsize=7, frameon=False); ax.grid(True, alpha=0.2)

    ax = axes[2]
    var_t = np.arange(-1, len(history['pi_var']) - 1)
    ax.plot(var_t, history['pi_var'], 'o-', color='#3498DB')
    ax.axhline(0, color='#CCC', linewidth=0.6)
    ax.set_xlabel('Outer iter'); ax.set_ylabel(r'std[$\pi^0$]')
    ax.set_title('(c) Action policy variance', fontweight='bold', loc='left')
    ax.set_ylim([-0.02, 0.55])
    ax.grid(True, alpha=0.2)

    pct = history['final_perint'] / benchmark * 100
    plt.suptitle(
        f'Exp 9 scalability — N={N}, {est_strategy}, seed {seed} | '
        f'final per-int = {history["final_perint"]:.3f} ({pct:.0f}% of {benchmark})  '
        f'| est params = {history["n_est_params"]}',
        fontsize=11, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    fname = f'fig_exp9_N{N}_{est_strategy}_seed{seed}.png'
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
            'est_loss_action', 'est_loss_signal',
            'pi_var', 'rep_log',
            'final_perint', 'final_pi_profile', 'final_pi_var',
            'initial_pi_profile',
            'benchmark', 'N', 'n_opps', 'est_strategy', 'n_est_params']
    return {k: hist[k] for k in keep if k in hist}


def main():
    save_dir = os.path.dirname(os.path.abspath(__file__))
    os.makedirs(os.path.join(save_dir, "figures"), exist_ok=True)
    os.makedirs(os.path.join(save_dir, "data"), exist_ok=True)
    json_path = os.path.join(save_dir, "data", "results_exp9_scalability.json")

    Ns = [3, 5, 10, 15]
    strategies = ['per_opp', 'amortized']
    pathways = ['action', 'signal', 'joint']

    all_results = load_json_if_exists(json_path)
    if all_results:
        done = sum(len(v) for v in all_results.values())
        print(f"[resume] {done} (config, seed) pairs in JSON")

    n_total = len(Ns) * len(strategies) * len(pathways) * NUM_SEEDS
    print(f"\n{'='*70}\n  Exp 9 — Scalability sweep\n  "
          f"{len(Ns)} N × {len(strategies)} strat × {len(pathways)} pathways × {NUM_SEEDS} seeds = {n_total} runs\n"
          f"{'='*70}")

    for N in Ns:
        for est_strategy in strategies:
            for pathway in pathways:
                key = f"N{N}_{est_strategy}_{pathway}"
                print(f"\n  Config: {key}")
                all_results.setdefault(key, {})
                for seed in range(NUM_SEEDS):
                    if str(seed) in all_results[key]:
                        print(f"  Seed {seed}: [skip]", flush=True)
                        continue
                    print(f"  Seed {seed}:", flush=True)
                    hist = run_exp9(N=N, est_strategy=est_strategy, seed=seed, pathway=pathway)
                    all_results[key][str(seed)] = _strip_for_json(hist)
                    plot_one_run(hist, N, est_strategy, seed, save_dir)
                    atomic_save_json(all_results, json_path)
                done = sum(len(v) for v in all_results.values())
                print(f"    [checkpoint] {done} pairs saved")

    print(f"\n{'='*70}\n  SUMMARY\n{'='*70}")
    for key in sorted(all_results.keys(),
                       key=lambda k: (int(k.split('_')[0][1:]), k.split('_', 1)[1])):
        seeds = all_results[key]
        if not seeds: continue
        avg = float(np.mean([seeds[s]['final_perint'] for s in seeds]))
        bench = next(iter(seeds.values()))['benchmark']
        n_p = next(iter(seeds.values()))['n_est_params']
        n_runs = len(seeds)
        pct = avg / bench * 100
        print(f"  {key:<25}  avg={avg:6.3f} ({pct:3.0f}% of {bench})  "
              f"est_params={n_p:>5}  n={n_runs}")
    print(f"{'='*70}")

    atomic_save_json(all_results, json_path)
    print(f"Final results saved to {json_path}")


if __name__ == "__main__":
    main()
