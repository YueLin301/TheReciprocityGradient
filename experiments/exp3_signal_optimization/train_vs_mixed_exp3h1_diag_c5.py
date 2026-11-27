"""
Exp 3h1-diag-c5: Same as exp3h1-diag but with **c=5** (donor cost) instead
of the default c=1. Purpose: widen the gap between frozen and trained by
raising the cost of indiscriminate cooperation — random-init signal that
coops ~50% with AllD burns substantial reward, while a learned
discriminative signal saves that cost.

Env setting: b=10, c=5. Theoretical:
  • Ceiling (A0 coops with P only, P coops back at ~92%):  (b − c)/4 = 1.25
  • Floor (random-init ~50% coop everywhere): per-int ≈ 0 analytically
    (cooperation near break-even), env dynamics may push it slightly up
  • Expected Δ ≈ ~1.0 (80%+ of benchmark) — a clearly visible climb

Each seed runs TWO conditions back-to-back with the same random seed:
  1. TRAINED: signal net optimized (lr=1e-3, 300 eps)
  2. FROZEN BASELINE: signal net held at random init (no optimizer step)

Panels (2×3 layout):
  (a) Per-int learning curve: trained + frozen baseline, with right y-axis
      showing % of benchmark (normalization)
  (b) Reputation evolution: all 3 agents, trained (solid) vs frozen (dotted)
  (c) Signal policy evolution snapshots
  (d) Final signal profile: trained vs frozen-init overlay
  (e) Signal spread  std_a[σ(a)]  over episodes (std over uniform grid)
  (f) Reputation gap  rep(ProudCoop) − rep(AllD)  over episodes
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
from checkpoint_utils import atomic_save_json, load_json_if_exists

mpl.rcParams.update({
    'font.family': 'sans-serif', 'font.size': 9,
    'axes.spines.top': False, 'axes.spines.right': False,
    'axes.linewidth': 0.6,
})

BATCH_SIZE = 128
NUM_SEEDS = 10
TRAIN_EPISODES = 300
B_VALUE = 10.0
C_VALUE = 5.0                             # donor cost — raised from 1 to 5 to widen the learning gap
DELTA = 0.98
FULL_COOP_BENCHMARK = (B_VALUE - C_VALUE) / 4  # 1.25
SNAPSHOT_EVERY = 40
BETA = 5.0
T_ACTION = 10.0
LEARNING_RATE = 1e-3


def _tanh_L3_signal(action, recip_rep, beta=BETA):
    sx = torch.tanh(beta * (action - 0.5))
    sy = torch.tanh(beta * (recip_rep - 0.5))
    return (1.0 - 0.25 * (1 - sx) * (1 + sy)).clamp(0, 1)


class ProudCoopL3(nn.Module):
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


def _measure_signal_spread(agent0):
    """Signal spread = std of σ over uniform grid of donor actions.

        Sample σ(a, own_rep=0.5) at a ∈ linspace(0, 1, 100);
        spread = std(σ) = sqrt( (1/N) Σ_i (σ(a_i) − \\bar σ)² )

    Geometric interpretation: amplitude of σ's deviation from a constant.
    Reference values on σ ∈ [0, 1]:
        • flat σ = const       → std = 0
        • linear ramp σ = a     → std = 1/√12 ≈ 0.289
        • step (half 0, half 1) → std = 0.5 (theoretical max)

    Robust to noise (averaging), well-defined for monotone or jumpy
    σ alike, and scale-bounded to [0, 0.5].

    Assumes agent0 is already in eval() mode (we keep it there throughout
    both trained and frozen runs to disable Transformer dropout, so forward
    passes are deterministic and the two conditions have identical ep-0
    behavior).
    """
    with torch.no_grad():
        x = torch.linspace(0, 1, 100)
        dummy_own_rep = torch.tensor([[0.5]])
        sig = np.array([agent0.send_signal(x[i].view(1), own_reputation=dummy_own_rep).item()
                        for i in range(100)])
    return float(np.std(sig))


def run_one_condition(seed, init_mode, do_train):
    """Run 200 eps either with training (do_train=True) or frozen (baseline)."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    agent0 = Agent(name="A_0", signal_dim=1, second_order=False, is_demo=False)
    opp_proud = ProudCoopL3(name="ProudL3")
    opp_alld = AllDefectL3(name="AllD_L3")
    agent0.set_hardcode_action(decay=None)

    signal_params = agent0.get_signal_params()
    if do_train:
        optimizer = optim.Adam(signal_params, lr=LEARNING_RATE)
    else:
        optimizer = None
        # Freeze signal params AND put agent0 in eval() permanently so LayerNorm
        # running stats don't drift during forward passes (this was causing
        # the frozen baseline to show a false trend over episodes).
        for p in signal_params:
            p.requires_grad_(False)

    if init_mode == 'const05':
        per_agent_init = [{"dist": "constant", "params": {"value": 0.5}}] * 3
    else:
        per_agent_init = [{"dist": "uniform", "params": {"low": 0.0, "high": 1.0}}] * 3

    agents = [agent0, opp_proud, opp_alld]
    env = BatchedDonationGame(
        agents=agents, b=B_VALUE, c=C_VALUE, continuation_prob=DELTA, batch_size=BATCH_SIZE,
        balanced_matching=True, min_rounds=3,
        per_agent_init=per_agent_init)

    episode_log = []
    rep_log = []
    spread_log = []  # per-episode signal spread
    signal_snapshots = []

    # Keep agent0 in eval() for the ENTIRE loop (both trained and frozen).
    # This disables Transformer dropout so forward passes are deterministic.
    # With the same seed + same weights, trained and frozen runs will have
    # IDENTICAL ep-0 per_int — they only diverge from ep 1 onward because
    # the trained run applies gradient updates.
    # Note: eval() does NOT disable autograd; optimizer.step() works fine.
    agent0.eval()

    # Initial snapshot (ep=-1).
    with torch.no_grad():
        x_snap = torch.linspace(0, 1, 100)
        dummy_own_rep = torch.tensor([[0.5]])
        profile = [agent0.send_signal(x.view(1), own_reputation=dummy_own_rep).item()
                   for x in x_snap]
        signal_snapshots.append({'episode': -1, 'profile': profile})

    for ep in range(TRAIN_EPISODES):
        if do_train:
            optimizer.zero_grad()

        cum_rewards, matching_seq, rep_histories = env.play_batch()
        per_int_reward, n_plays = compute_per_interaction_payoff(cum_rewards, matching_seq, 0)
        loss = -per_int_reward.mean()

        if do_train and loss.requires_grad:
            grads = torch.autograd.grad(loss, signal_params, allow_unused=True)
            for p, g in zip(signal_params, grads):
                if g is not None and not (torch.isnan(g).any() or torch.isinf(g).any()):
                    p.grad = g
            torch.nn.utils.clip_grad_norm_(signal_params, max_norm=1.0)
            optimizer.step()

        with torch.no_grad():
            agent_reps = [rep_histories[i][:, -1, 0].mean().item() for i in range(3)]
            rep_log.append(agent_reps)
        del rep_histories

        episode_log.append({
            'episode': ep,
            'per_int_reward': per_int_reward.mean().item(),
        })
        spread_log.append(_measure_signal_spread(agent0))

        if ep % SNAPSHOT_EVERY == 0 or ep == TRAIN_EPISODES - 1:
            with torch.no_grad():
                x_snap = torch.linspace(0, 1, 100)
                dummy_own_rep = torch.tensor([[0.5]])
                profile = [agent0.send_signal(x.view(1), own_reputation=dummy_own_rep).item()
                           for x in x_snap]
                signal_snapshots.append({'episode': ep, 'profile': profile})

    # Final profiles (agent0 already in eval mode)
    x_vals = torch.linspace(0, 1, 100)
    with torch.no_grad():
        action_profile = [agent0.take_action(x.view(1, 1)).item() for x in x_vals]
        dummy_own_rep = torch.tensor([[0.5]])
        signal_profile = [agent0.send_signal(x.view(1), own_reputation=dummy_own_rep).item()
                          for x in x_vals]

    last50 = float(np.mean([e['per_int_reward'] for e in episode_log[-50:]]))
    sig = np.array(signal_profile)
    spread = float(np.std(sig))   # std over uniform grid a ∈ [0,1]; see _measure_signal_spread
    disc_type = 'disc' if spread > 0.25 else ('part' if spread > 0.10 else 'flat')

    return {
        'episode_log': episode_log,
        'action_profile': action_profile,
        'signal_profile': signal_profile,
        'signal_snapshots': signal_snapshots,
        'rep_log': rep_log,
        'spread_log': spread_log,
        'last50_per_int': last50,
        'signal_spread': float(spread),
        'disc_type': disc_type,
    }


def train_both_3h1_diag(seed, init_mode='const05'):
    print(f"    [trained]:", flush=True)
    trained = run_one_condition(seed, init_mode, do_train=True)
    print(f"    [frozen baseline]:", flush=True)
    frozen = run_one_condition(seed, init_mode, do_train=False)
    print(f"  → trained last50={trained['last50_per_int']:.3f} (spread={trained['signal_spread']:.2f} {trained['disc_type']}) | "
          f"frozen last50={frozen['last50_per_int']:.3f}")
    return {'trained': trained, 'frozen': frozen}


def plot_seed_details(seed_idx, both, save_dir, suffix=''):
    trained = both['trained']
    frozen = both['frozen']
    x = np.linspace(0, 1, 100)
    fig, axes = plt.subplots(2, 3, figsize=(16, 8.5))

    # ── (a) Learning curve, with normalized right axis ──
    ax = axes[0, 0]
    rw_t = np.asarray([e['per_int_reward'] for e in trained['episode_log']])
    rw_f = np.asarray([e['per_int_reward'] for e in frozen['episode_log']])

    # Centered rolling mean with reflected padding: no initialization bias.
    # (Using EMA with seed = vals[0] would falsely pull frozen baseline toward
    # the first episode's noise and create a phantom trend.)
    def smooth(vals, window=15):
        pad = window // 2
        padded = np.concatenate([vals[:pad][::-1], vals, vals[-pad:][::-1]])
        ker = np.ones(window) / window
        out = np.convolve(padded, ker, mode='valid')
        return out[:len(vals)]

    ax.plot(rw_t, alpha=0.15, color='#3498DB', linewidth=0.5)
    ax.plot(rw_f, alpha=0.15, color='#888', linewidth=0.5)
    ax.plot(smooth(rw_t), color='#3498DB', linewidth=1.8, label='trained')
    ax.plot(smooth(rw_f), color='#555', linewidth=1.4, linestyle='--', label='frozen baseline')
    ax.axhline(y=FULL_COOP_BENCHMARK, color='#AAA', linestyle=':', linewidth=0.8,
               label=f'benchmark ({FULL_COOP_BENCHMARK})')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Per-interaction payoff')
    ax.set_title('(a) Learning curve', fontweight='bold', loc='left')
    ax.legend(loc='lower right', frameon=False, fontsize=7)
    ax.grid(True, alpha=0.2)
    # Right axis: % of benchmark
    ax2 = ax.twinx()
    ymin, ymax = ax.get_ylim()
    ax2.set_ylim(ymin/FULL_COOP_BENCHMARK*100, ymax/FULL_COOP_BENCHMARK*100)
    ax2.set_ylabel('% of benchmark', fontsize=8, color='#666')
    ax2.tick_params(axis='y', colors='#666', labelsize=7)
    ax2.spines['right'].set_visible(True)
    ax2.spines['right'].set_color('#666')

    # ── (b) Reputation evolution: trained solid, frozen dotted ──
    ax = axes[0, 1]
    rep_t = np.array(trained['rep_log'])
    rep_f = np.array(frozen['rep_log'])
    colors = ['#3498DB', '#2ECC71', '#E74C3C']
    labels = ['Agent 0', 'ProudCoop(L3)', 'AllD(L3)']
    for i in range(3):
        # faint raw + bold smoothed
        ax.plot(rep_t[:, i], color=colors[i], linewidth=0.4, alpha=0.15)
        ax.plot(rep_f[:, i], color=colors[i], linewidth=0.4, alpha=0.12)
        ax.plot(smooth(rep_t[:, i]), color=colors[i], linewidth=1.4, alpha=0.95,
                label=f'{labels[i]} (trained)')
        ax.plot(smooth(rep_f[:, i]), color=colors[i], linewidth=1.0, linestyle=':', alpha=0.75)
    ax.set_xlabel('Episode')
    ax.set_ylabel('Mean reputation (last signal)')
    ax.set_title('(b) Reputation evolution (dotted=frozen)', fontweight='bold', loc='left')
    ax.set_ylim([-0.05, 1.05])
    ax.legend(loc='center right', frameon=True, fancybox=False, edgecolor='#CCC',
              framealpha=0.95, fontsize=7)
    ax.grid(True, alpha=0.2)

    # ── (c) Signal policy evolution (trained) ──
    ax = axes[0, 2]
    snapshots = trained['signal_snapshots']
    n_snap = len(snapshots)
    cmap = plt.cm.plasma
    for i, snap in enumerate(snapshots):
        color = cmap(i / max(n_snap - 1, 1))
        ax.plot(x, snap['profile'], color=color, linewidth=1.2, alpha=0.85,
                label=f"ep {snap['episode']}")
    ax.plot(x, x, '--', color='#AAA', linewidth=0.8, label='identity')
    ax.set_xlabel('Donor action')
    ax.set_ylabel('Signal')
    ax.set_title('(c) Signal policy evolution (trained)', fontweight='bold', loc='left')
    ax.set_xlim([0, 1]); ax.set_ylim([-0.05, 1.05])
    ax.legend(loc='upper left', frameon=True, fancybox=False, edgecolor='#CCC',
              framealpha=0.95, fontsize=6, ncol=2)
    ax.grid(True, alpha=0.2)

    # ── (d) Final signal: trained vs frozen-init ──
    ax = axes[1, 0]
    ax.plot(x, trained['signal_profile'], color='#3498DB', linewidth=2, label='trained $\\varphi$')
    ax.plot(x, frozen['signal_profile'], color='#555', linewidth=1.5, linestyle='--',
            label='frozen (random init)')
    ax.plot(x, x, ':', color='#AAA', linewidth=0.8, label='identity')
    ax.set_xlabel('Donor action')
    ax.set_ylabel('Signal')
    ax.set_title(f'(d) Final signal (trained spread={trained["signal_spread"]:.2f} {trained["disc_type"]})',
                 fontweight='bold', loc='left')
    ax.set_xlim([0, 1]); ax.set_ylim([-0.05, 1.05])
    ax.legend(loc='lower right', frameon=False, fontsize=7)
    ax.grid(True, alpha=0.2)

    # ── (e) Signal spread over episodes ──
    ax = axes[1, 1]
    sp_t = np.asarray(trained['spread_log'])
    sp_f = np.asarray(frozen['spread_log'])
    ax.plot(sp_t, color='#3498DB', linewidth=0.4, alpha=0.2)
    ax.plot(sp_f, color='#555', linewidth=0.4, alpha=0.2)
    ax.plot(smooth(sp_t), color='#3498DB', linewidth=1.8, label='trained')
    ax.plot(smooth(sp_f), color='#555', linewidth=1.4, linestyle='--', label='frozen')
    ax.axhline(0, color='#CCC', linewidth=0.6)
    ax.set_xlabel('Episode')
    ax.set_ylabel(r'Signal spread  $\mathrm{std}_a[\sigma(a)]$')
    ax.set_title('(e) Signal discrimination over time', fontweight='bold', loc='left')
    ax.set_ylim([-0.02, 0.55])   # std ∈ [0, 0.5] for σ ∈ [0,1]
    ax.legend(loc='lower right', frameon=False, fontsize=7)
    ax.grid(True, alpha=0.2)

    # ── (f) Rep gap rep(P)-rep(D) over episodes ──
    ax = axes[1, 2]
    gap_t = rep_t[:, 1] - rep_t[:, 2]
    gap_f = rep_f[:, 1] - rep_f[:, 2]
    ax.plot(gap_t, color='#3498DB', linewidth=0.4, alpha=0.2)
    ax.plot(gap_f, color='#555', linewidth=0.4, alpha=0.2)
    ax.plot(smooth(gap_t), color='#3498DB', linewidth=1.8, label='trained')
    ax.plot(smooth(gap_f), color='#555', linewidth=1.4, linestyle='--', label='frozen')
    ax.axhline(0, color='#CCC', linewidth=0.6)
    ax.set_xlabel('Episode')
    ax.set_ylabel('Reputation gap  $r_P - r_D$')
    ax.set_title('(f) Reputation gap (P − D) over time', fontweight='bold', loc='left')
    ax.set_ylim([-0.1, 1.05])
    ax.legend(loc='lower right', frameon=False, fontsize=7)
    ax.grid(True, alpha=0.2)

    pct = trained['last50_per_int'] / FULL_COOP_BENCHMARK * 100
    pct_f = frozen['last50_per_int'] / FULL_COOP_BENCHMARK * 100
    plt.suptitle(
        f'Exp 3h1-diag-c5 (b={int(B_VALUE)}, c={int(C_VALUE)}) — Seed {seed_idx} | '
        f'trained {trained["last50_per_int"]:.2f} ({pct:.0f}% of {FULL_COOP_BENCHMARK}) '
        f'vs frozen {frozen["last50_per_int"]:.2f} ({pct_f:.0f}%)',
        fontsize=12, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    path = os.path.join(save_dir, 'figures', f'fig_3h1_diag_c5_seed{seed_idx}{suffix}_details.png')
    plt.savefig(path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f'    → {path}')


def _strip_for_json(r):
    keep = ['episode_log', 'action_profile', 'signal_profile',
            'spread_log', 'last50_per_int', 'signal_spread', 'disc_type']
    return {
        'trained': {k: r['trained'][k] for k in keep},
        'frozen':  {k: r['frozen'][k]  for k in keep},
    }


def main():
    save_dir = os.path.dirname(os.path.abspath(__file__))
    os.makedirs(os.path.join(save_dir, "figures"), exist_ok=True)
    os.makedirs(os.path.join(save_dir, "data"), exist_ok=True)
    json_path = os.path.join(save_dir, "data", "results_signal_vs_mixed_3h1_diag_c5.json")

    print(f"Exp 3h1-diag-c5: ProudCoop(L3) + AllD(L3), {NUM_SEEDS} seeds, {TRAIN_EPISODES} eps")
    print(f"  lr={LEARNING_RATE}, b={B_VALUE}, c={C_VALUE}, benchmark={FULL_COOP_BENCHMARK}")
    print(f"  Each seed runs TRAINED and FROZEN BASELINE back-to-back")

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
            result = train_both_3h1_diag(seed, init_mode=init_mode)
            plot_seed_details(seed, result, save_dir, suffix=f'_{init_mode}')
            save_all[init_mode][str(seed)] = _strip_for_json(result)
            atomic_save_json(save_all, json_path)

    print(f"\n{'='*70}")
    for init_mode in save_all:
        res = save_all[init_mode]
        t_avg = np.mean([res[s]['trained']['last50_per_int'] for s in res])
        f_avg = np.mean([res[s]['frozen']['last50_per_int'] for s in res])
        t_disc = sum(1 for s in res if res[s]['trained']['disc_type'] == 'disc')
        print(f"  {init_mode}: trained avg={t_avg:.3f} ({t_avg/FULL_COOP_BENCHMARK*100:.0f}%)  "
              f"frozen avg={f_avg:.3f} ({f_avg/FULL_COOP_BENCHMARK*100:.0f}%)  "
              f"disc={t_disc}/{len(res)}")
    print(f"{'='*70}")
    print(f"Results saved to {json_path}")


if __name__ == "__main__":
    main()
