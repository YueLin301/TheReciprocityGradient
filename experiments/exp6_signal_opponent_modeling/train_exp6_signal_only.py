"""
Exp 6 (signal-only, opponent modeling) = Exp 3h + opponent modeling.

Mirrors exp 3h opponents:
  - Opp 1: ProudCoop with sigmoid(T=10*(own_rep − 0.5)) action and tanh L3 signal
  - Opp 2: AllDefector (always 0) with tanh L3 signal

Agent 0 (the learner):
  - Trainable signal network  φ^0  (the optimization target)
  - Hardcoded identity action  π^0(rep) = mean(rep)   (frozen)

Agent 0 maintains FOUR estimator networks (not white-box access to opponents):
  π̂^j_0  for j ∈ {1, 2}  (action estimators)
  φ̂^j_0  for j ∈ {1, 2}  (signal estimators)

Two estimator architectures tested:
  • MLP   — 2-layer tanh MLP (Markov: reads only last rep)
  • Transformer — single-layer encoder + tanh MLP head

Variant axes (matching exp 3h):
  • estimator_arch ∈ {MLP, TF}
  • init_mode      ∈ {const05, all_uniform}

Nested training loop (Algorithm 1 of the paper, signal-only instantiation):
  for t in range(T_OUTER):
    1. INTERACT: play N_PLAY real episodes, record (input, output) pairs
    2. ESTIMATOR UPDATE: K_EST MSE gradient steps per estimator, on recent W iters
    3. POLICY UPDATE: N_TRAIN rollouts in surrogate env (real opp → est), gradient
                      flows through estimators into φ^0
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
from collections import deque
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

# ══════════════════════════════════════════════════════════════════════════════
# Hyperparameters
# ══════════════════════════════════════════════════════════════════════════════
BATCH_SIZE = 128
NUM_SEEDS = 5
T_OUTER = 15                  # outer iterations
N_PLAY = 5                    # real episodes per outer iter
K_EST = 100                   # MSE grad steps per estimator per outer iter
N_TRAIN = 10                  # policy grad steps per outer iter
W_BUFFER = 3                  # sliding window: keep last W outer iters of data
B_VALUE = 10.0
C_VALUE = 5.0                 # donor cost = 5 (matches exp3h1_diag_c5 — larger dynamic range)
DELTA = 0.98
BETA = 5.0
T_ACTION = 10.0               # ProudCoop sigmoid sharpness (matches exp 3h)
# Benchmark: with AllD opponent, ideal = cooperate only with ProudCoop
#   per_int = (b − c)/4 (cost c payed once to P, benefit b received once from P)
FULL_COOP_BENCHMARK = (B_VALUE - C_VALUE) / 4   # 1.25 under c=5
LR_POLICY = 5e-2              # signal LR — matches exp 3h
LR_EST = 1e-2
EMB_DIM = 16


# ══════════════════════════════════════════════════════════════════════════════
# True opponents (frozen) — exact copies from train_vs_mixed_exp3h.py
# ══════════════════════════════════════════════════════════════════════════════

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

    # ── Batched fast paths (used by env_batched._batched_*) ──
    def batched_take_action(self, rep_histories, own_rep_histories):
        # own_rep_histories: [B, seq, 1] — last own rep
        own_rep = own_rep_histories[:, -1, :]               # [B, 1]
        return torch.sigmoid(T_ACTION * (own_rep - 0.5))     # [B, 1]

    def batched_send_signal(self, actions, own_reps, donor_reps=None):
        # actions: [B, 1], own_reps: [B, seq, 1]
        recip_rep = own_reps[:, -1, :]                       # [B, 1]
        return _tanh_L3_signal(actions, recip_rep).clamp(0, 1)


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

    def batched_take_action(self, rep_histories, own_rep_histories):
        B = rep_histories.shape[0]
        return torch.zeros(B, 1, device=self.device)

    def batched_send_signal(self, actions, own_reps, donor_reps=None):
        recip_rep = own_reps[:, -1, :]
        return _tanh_L3_signal(actions, recip_rep).clamp(0, 1)


# ══════════════════════════════════════════════════════════════════════════════
# Estimators — tanh MLP and Transformer (same as exp 4)
# ══════════════════════════════════════════════════════════════════════════════

def _sinusoidal_pe(max_len: int, emb_dim: int) -> torch.Tensor:
    pe = torch.zeros(max_len, emb_dim)
    pos = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
    div = torch.exp(torch.arange(0, emb_dim, 2, dtype=torch.float32)
                    * (-torch.log(torch.tensor(10000.0)) / emb_dim))
    pe[:, 0::2] = torch.sin(pos * div)
    if emb_dim > 1:
        pe[:, 1::2] = torch.cos(pos * div[:emb_dim // 2])
    return pe


class MLPActionEstimator(nn.Module):
    """π̂^j (MLP): reads last opp_rep + last own_rep, predicts action in [0,1]."""
    def __init__(self, hidden=16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, 1), nn.Sigmoid()
        )

    def forward(self, opp_rep_hist, own_rep_hist):
        opp_last = opp_rep_hist[:, -1, :]
        own_last = own_rep_hist[:, -1, :] if own_rep_hist is not None \
                   else torch.full_like(opp_last, 0.5)
        return self.net(torch.cat([opp_last, own_last], dim=-1))


class MLPSignalEstimator(nn.Module):
    """φ̂^j (MLP): reads donor action + j's last own_rep, predicts signal in [0,1]."""
    def __init__(self, hidden=16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, 1), nn.Sigmoid()
        )

    def forward(self, action, own_rep_hist, donor_rep_hist=None):
        own_last = own_rep_hist[:, -1, :] if own_rep_hist is not None \
                   else torch.full_like(action, 0.5)
        return self.net(torch.cat([action, own_last], dim=-1))


class TFActionEstimator(nn.Module):
    """π̂^j (Transformer): full rep history → tanh MLP head."""
    def __init__(self, emb_dim=EMB_DIM, hidden=16):
        super().__init__()
        self.opp_proj = nn.Linear(1, emb_dim)
        self.own_proj = nn.Linear(1, emb_dim)
        layer_o = nn.TransformerEncoderLayer(d_model=emb_dim, nhead=1,
                                              dim_feedforward=32, batch_first=True)
        self.opp_enc = nn.TransformerEncoder(layer_o, num_layers=1)
        layer_w = nn.TransformerEncoderLayer(d_model=emb_dim, nhead=1,
                                              dim_feedforward=32, batch_first=True)
        self.own_enc = nn.TransformerEncoder(layer_w, num_layers=1)
        self.head = nn.Sequential(
            nn.Linear(emb_dim * 2, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, 1), nn.Sigmoid()
        )
        self.register_buffer("_pe", _sinusoidal_pe(256, emb_dim))

    def forward(self, opp_rep_hist, own_rep_hist):
        opp_p = self.opp_proj(opp_rep_hist) + self._pe[:opp_rep_hist.shape[1]].unsqueeze(0)
        opp_emb = self.opp_enc(opp_p)[:, -1, :]
        if own_rep_hist is None:
            own_emb = torch.zeros_like(opp_emb)
        else:
            own_p = self.own_proj(own_rep_hist) + self._pe[:own_rep_hist.shape[1]].unsqueeze(0)
            own_emb = self.own_enc(own_p)[:, -1, :]
        return self.head(torch.cat([opp_emb, own_emb], dim=-1))


class TFSignalEstimator(nn.Module):
    """φ̂^j (Transformer): action scalar + own-rep history → signal."""
    def __init__(self, emb_dim=EMB_DIM, hidden=16):
        super().__init__()
        self.own_proj = nn.Linear(1, emb_dim)
        self.action_proj = nn.Linear(1, emb_dim)
        layer = nn.TransformerEncoderLayer(d_model=emb_dim, nhead=1,
                                            dim_feedforward=32, batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=1)
        self.head = nn.Sequential(
            nn.Linear(emb_dim * 2, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, 1), nn.Sigmoid()
        )
        self.register_buffer("_pe", _sinusoidal_pe(256, emb_dim))

    def forward(self, action, own_rep_hist, donor_rep_hist=None):
        act_emb = self.action_proj(action)
        own_p = self.own_proj(own_rep_hist) + self._pe[:own_rep_hist.shape[1]].unsqueeze(0)
        own_emb = self.encoder(own_p)[:, -1, :]
        return self.head(torch.cat([act_emb, own_emb], dim=-1))


def build_estimators(arch: str):
    if arch == 'MLP':
        return MLPActionEstimator(), MLPSignalEstimator()
    elif arch == 'TF':
        return TFActionEstimator(), TFSignalEstimator()
    elif arch == 'Factored':
        # General multiplicative-interaction architecture — see
        # opponent_modeling_utils.FactoredMLP* and exp4b diagnostic for rationale.
        from opponent_modeling_utils import (FactoredMLPActionEstimator,
                                              FactoredMLPSignalEstimator)
        return (FactoredMLPActionEstimator(factor_dim=8, hidden=16),
                FactoredMLPSignalEstimator(factor_dim=8, hidden=16))
    else:
        raise ValueError(f"Unknown estimator arch: {arch}")


# ══════════════════════════════════════════════════════════════════════════════
# EstimatedOpponent: env-compatible wrapper around (π̂, φ̂)
# ══════════════════════════════════════════════════════════════════════════════

class EstimatedOpponent(nn.Module):
    """Env-compatible wrapper: routes batched_take_action / batched_send_signal
    through the learned estimators."""
    def __init__(self, name: str, action_est: nn.Module, signal_est: nn.Module,
                 signal_dim: int = 1, device: Optional[torch.device] = None):
        super().__init__()
        self.name = name
        self.signal_dim = signal_dim
        self.second_order = True
        self.device = device if device is not None else torch.device("cpu")
        self.action_est = action_est
        self.signal_est = signal_est
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

    def batched_take_action(self, rep_histories, own_rep_histories):
        return self.action_est(rep_histories, own_rep_histories).clamp(0, 1)

    def batched_send_signal(self, actions, own_reps, donor_reps=None):
        return self.signal_est(actions, own_reps, donor_reps).clamp(0, 1)


# ══════════════════════════════════════════════════════════════════════════════
# Data buffer + recording wrapper
# ══════════════════════════════════════════════════════════════════════════════

class DataBuffer:
    """Sliding window of (input, observed output) pairs for one opponent."""
    def __init__(self, window: int):
        self.window = window
        self.action_data = deque(maxlen=window)
        self.signal_data = deque(maxlen=window)

    def new_iter(self):
        self.action_data.append([])
        self.signal_data.append([])

    def record_action(self, opp_rep, own_rep, obs_a):
        self.action_data[-1].append(
            (opp_rep.detach().clone(), own_rep.detach().clone(), obs_a.detach().clone())
        )

    def record_signal(self, action, own_rep, donor_rep, obs_s):
        self.signal_data[-1].append(
            (action.detach().clone(), own_rep.detach().clone(),
             donor_rep.detach().clone() if donor_rep is not None else None,
             obs_s.detach().clone())
        )

    def all_action_records(self):
        return [rec for iter_data in self.action_data for rec in iter_data]

    def all_signal_records(self):
        return [rec for iter_data in self.signal_data for rec in iter_data]


class RecordingWrapper(nn.Module):
    """Wraps any opponent (with batched_take_action/send_signal) to record all
    batched calls into a DataBuffer."""
    def __init__(self, inner: nn.Module, buffer: DataBuffer):
        super().__init__()
        self.inner = inner
        self.buffer = buffer
        self.name = inner.name
        self.signal_dim = inner.signal_dim
        self.second_order = inner.second_order

    def reset_reputation(self, initial_reputation=None):
        self.inner.reset_reputation(initial_reputation)

    def append_reputation(self, signal_t):
        self.inner.append_reputation(signal_t)

    def get_observation(self):
        return self.inner.get_observation()

    def batched_take_action(self, rep_histories, own_rep_histories):
        action = self.inner.batched_take_action(rep_histories, own_rep_histories)
        self.buffer.record_action(rep_histories, own_rep_histories, action)
        return action

    def batched_send_signal(self, actions, own_reps, donor_reps=None):
        signal = self.inner.batched_send_signal(actions, own_reps, donor_reps)
        self.buffer.record_signal(actions, own_reps, donor_reps, signal)
        return signal


# ══════════════════════════════════════════════════════════════════════════════
# Estimator training
# ══════════════════════════════════════════════════════════════════════════════

def train_action_estimator(est: nn.Module, buf: DataBuffer,
                           k_steps: int = K_EST, lr: float = LR_EST) -> float:
    records = buf.all_action_records()
    if not records:
        return float('nan')
    opt = optim.Adam(est.parameters(), lr=lr)
    final_loss = 0.0
    for _ in range(k_steps):
        rec = records[np.random.randint(len(records))]
        opp_rep, own_rep, obs_a = rec
        pred = est(opp_rep, own_rep).clamp(0, 1)
        loss = ((pred - obs_a) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        final_loss = loss.item()
    return final_loss


def train_signal_estimator(est: nn.Module, buf: DataBuffer,
                           k_steps: int = K_EST, lr: float = LR_EST) -> float:
    records = buf.all_signal_records()
    if not records:
        return float('nan')
    opt = optim.Adam(est.parameters(), lr=lr)
    final_loss = 0.0
    for _ in range(k_steps):
        rec = records[np.random.randint(len(records))]
        action, own_rep, donor_rep, obs_s = rec
        pred = est(action, own_rep, donor_rep).clamp(0, 1)
        loss = ((pred - obs_s) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        final_loss = loss.item()
    return final_loss


# ══════════════════════════════════════════════════════════════════════════════
# Main training per (estimator arch, init mode, seed)
# ══════════════════════════════════════════════════════════════════════════════

def _measure_signal_spread(agent0):
    """std_a[σ(a, own_rep=0.5)] over uniform grid (matches exp3h definition)."""
    was_training = agent0.training
    agent0.eval()
    x = torch.linspace(0, 1, 100)
    dummy_own_rep = torch.tensor([[0.5]])
    with torch.no_grad():
        sig = np.array([agent0.send_signal(x[i].view(1), own_reputation=dummy_own_rep).item()
                        for i in range(100)])
    if was_training:
        agent0.train()
    return float(np.std(sig))


def _snapshot_signal_policy(agent0):
    was_training = agent0.training
    agent0.eval()
    x = torch.linspace(0, 1, 100)
    dummy_own_rep = torch.tensor([[0.5]])
    with torch.no_grad():
        profile = [agent0.send_signal(x[i].view(1), own_reputation=dummy_own_rep).item()
                   for i in range(100)]
    if was_training:
        agent0.train()
    return profile


def run_exp6(arch: str, init_mode: str, seed: int) -> dict:
    """
    arch     : 'MLP' or 'TF'
    init_mode: 'const05' or 'all_uniform'
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    device = torch.device("cpu")

    # ── Agent 0: trainable signal, hardcoded identity action (mean) ──
    agent0 = Agent(name="A_0", signal_dim=1, second_order=False, is_demo=False, device=device)
    agent0.set_hardcode_action(decay=None)              # identity mean action (frozen)

    # ── Real opponents (frozen) ──
    opp1 = ProudCoopL3(name="ProudL3", device=device)
    opp2 = AllDefectL3(name="AllD_L3", device=device)
    for p in list(opp1.parameters()) + list(opp2.parameters()):
        p.requires_grad_(False)

    # ── Estimators (4 networks) ──
    act_est_1, sig_est_1 = build_estimators(arch)
    act_est_2, sig_est_2 = build_estimators(arch)
    for m in (act_est_1, sig_est_1, act_est_2, sig_est_2):
        m.to(device)

    # ── Data buffers + recording wrappers ──
    buf1 = DataBuffer(window=W_BUFFER)
    buf2 = DataBuffer(window=W_BUFFER)
    rec_opp1 = RecordingWrapper(opp1, buf1)
    rec_opp2 = RecordingWrapper(opp2, buf2)

    # ── Policy optimizer for agent0's SIGNAL net ──
    signal_params = agent0.get_signal_params()
    policy_opt = optim.Adam(signal_params, lr=LR_POLICY)

    # ── Per-agent reputation init (matches exp3h modes) ──
    if init_mode == 'const05':
        per_agent_init = [{"dist": "constant", "params": {"value": 0.5}}] * 3
    else:                                                 # all_uniform
        per_agent_init = [{"dist": "uniform", "params": {"low": 0.0, "high": 1.0}}] * 3

    # ── History ──
    history = {
        'real_perint': [], 'surrogate_perint': [],
        'real_perint_per_ep': [], 'surrogate_perint_per_ep': [],
        'act_est_loss_1': [], 'act_est_loss_2': [],
        'sig_est_loss_1': [], 'sig_est_loss_2': [],
        'spread_log': [],            # std_a[σ(a)] per outer iter
        'signal_snapshots': [],      # σ^0(a) at outer iter end
        'rep_log': [],               # [A0, Opp1, Opp2] last-signal mean
    }

    # Initial snapshot (before any training)
    history['signal_snapshots'].append({'outer_iter': -1, 'profile': _snapshot_signal_policy(agent0)})
    history['spread_log'].append(_measure_signal_spread(agent0))

    for t in range(T_OUTER):
        # ── Phase 1: INTERACT (real env, record data) ──
        buf1.new_iter()
        buf2.new_iter()

        real_env = BatchedDonationGame(
            agents=[agent0, rec_opp1, rec_opp2], b=B_VALUE, c=C_VALUE,
            continuation_prob=DELTA, batch_size=BATCH_SIZE,
            balanced_matching=True, min_rounds=3,
            per_agent_init=per_agent_init)

        agent0.eval()
        this_iter_real_ep = []
        this_iter_rep = None
        with torch.no_grad():
            for _ in range(N_PLAY):
                cum_rewards, matching_seq, rep_histories = real_env.play_batch()
                pi, _ = compute_per_interaction_payoff(cum_rewards, matching_seq, 0)
                this_iter_real_ep.append(pi.mean().item())
                this_iter_rep = [rep_histories[i][:, -1, 0].mean().item() for i in range(3)]
                del rep_histories
        agent0.train()
        history['real_perint_per_ep'].append(this_iter_real_ep)
        history['real_perint'].append(float(np.mean(this_iter_real_ep)))
        history['rep_log'].append(this_iter_rep)

        # ── Phase 2: ESTIMATOR UPDATE ──
        history['act_est_loss_1'].append(train_action_estimator(act_est_1, buf1))
        history['act_est_loss_2'].append(train_action_estimator(act_est_2, buf2))
        history['sig_est_loss_1'].append(train_signal_estimator(sig_est_1, buf1))
        history['sig_est_loss_2'].append(train_signal_estimator(sig_est_2, buf2))

        # ── Phase 3: POLICY UPDATE (surrogate env, gradient flows through estimators) ──
        est_opp1 = EstimatedOpponent("EstOpp1", act_est_1, sig_est_1, device=device)
        est_opp2 = EstimatedOpponent("EstOpp2", act_est_2, sig_est_2, device=device)
        for m in (act_est_1, sig_est_1, act_est_2, sig_est_2):
            for p in m.parameters():
                p.requires_grad_(False)

        surrogate_env = BatchedDonationGame(
            agents=[agent0, est_opp1, est_opp2], b=B_VALUE, c=C_VALUE,
            continuation_prob=DELTA, batch_size=BATCH_SIZE,
            balanced_matching=True, min_rounds=3,
            per_agent_init=per_agent_init)

        this_iter_surr_ep = []
        for _ in range(N_TRAIN):
            policy_opt.zero_grad()
            cum_rewards, matching_seq, rep_histories = surrogate_env.play_batch()
            pi, _ = compute_per_interaction_payoff(cum_rewards, matching_seq, 0)
            loss = -pi.mean()
            if loss.requires_grad:
                grads = torch.autograd.grad(loss, signal_params, allow_unused=True)
                for p, g in zip(signal_params, grads):
                    if g is not None and not (torch.isnan(g).any() or torch.isinf(g).any()):
                        p.grad = g
                torch.nn.utils.clip_grad_norm_(signal_params, max_norm=1.0)
            policy_opt.step()
            this_iter_surr_ep.append(pi.mean().item())
            del rep_histories
        history['surrogate_perint_per_ep'].append(this_iter_surr_ep)
        history['surrogate_perint'].append(float(np.mean(this_iter_surr_ep)))

        # Snapshot AFTER policy update
        history['signal_snapshots'].append({'outer_iter': t, 'profile': _snapshot_signal_policy(agent0)})
        history['spread_log'].append(_measure_signal_spread(agent0))

        # Unfreeze estimators for next outer iter
        for m in (act_est_1, sig_est_1, act_est_2, sig_est_2):
            for p in m.parameters():
                p.requires_grad_(True)

        print(f"    [{arch} {init_mode} seed={seed}] t={t:2d} | "
              f"real={history['real_perint'][-1]:.3f}  "
              f"surr={history['surrogate_perint'][-1]:.3f}  "
              f"spread={history['spread_log'][-1]:.3f}  "
              f"loss π̂={history['act_est_loss_1'][-1]:.4f}/"
              f"{history['act_est_loss_2'][-1]:.4f}  "
              f"loss φ̂={history['sig_est_loss_1'][-1]:.4f}/"
              f"{history['sig_est_loss_2'][-1]:.4f}", flush=True)

    # ── Final evaluation in real env ──
    agent0.eval()
    final_perint = 0.0
    with torch.no_grad():
        real_env = BatchedDonationGame(
            agents=[agent0, opp1, opp2], b=B_VALUE, c=C_VALUE,
            continuation_prob=DELTA, batch_size=BATCH_SIZE,
            balanced_matching=True, min_rounds=3,
            per_agent_init=per_agent_init)
        for _ in range(5):
            cum_rewards, matching_seq, _ = real_env.play_batch()
            pi, _ = compute_per_interaction_payoff(cum_rewards, matching_seq, 0)
            final_perint += pi.mean().item()
    final_perint /= 5

    # Final signal profile + classification
    final_profile = _snapshot_signal_policy(agent0)
    final_spread = float(np.std(np.array(final_profile)))
    if final_spread > 0.25:
        disc_type = 'disc'
    elif final_spread > 0.10:
        disc_type = 'part'
    else:
        disc_type = 'flat'

    history['final_perint'] = final_perint
    history['final_signal_profile'] = final_profile
    history['final_signal_spread'] = final_spread
    history['final_disc_type'] = disc_type

    print(f"  [{arch} {init_mode} seed={seed}]  FINAL real per-int = {final_perint:.3f} "
          f"({final_perint / FULL_COOP_BENCHMARK * 100:.0f}% of {FULL_COOP_BENCHMARK})  "
          f"spread={final_spread:.3f} ({disc_type})")

    return history


# ══════════════════════════════════════════════════════════════════════════════
# Plotting (same 2×3 layout as exp4, but for SIGNAL policy)
# ══════════════════════════════════════════════════════════════════════════════

def plot_one_run(history: dict, arch: str, init_mode: str, seed: int, save_dir: str):
    x_grid = np.linspace(0, 1, 100)
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))

    # ── (a) Per-int payoff ──
    ax = axes[0, 0]
    for t, eps in enumerate(history['real_perint_per_ep']):
        xs = [t + (i + 0.5) / len(eps) for i in range(len(eps))]
        ax.scatter(xs, eps, s=8, color='#3498DB', alpha=0.35)
    for t, eps in enumerate(history['surrogate_perint_per_ep']):
        xs = [t + (i + 0.5) / len(eps) for i in range(len(eps))]
        ax.scatter(xs, eps, s=8, color='#E67E22', alpha=0.35, marker='s')
    t_axis = np.arange(len(history['real_perint']))
    ax.plot(t_axis + 0.5, history['real_perint'], '-', color='#2C3E50', linewidth=2.0,
            label='real env (mean)')
    ax.plot(t_axis + 0.5, history['surrogate_perint'], '--', color='#C0392B', linewidth=1.6,
            label='surrogate env (mean)')
    ax.axhline(FULL_COOP_BENCHMARK, color='#AAA', linestyle=':', linewidth=0.8,
               label=f'benchmark {FULL_COOP_BENCHMARK}')
    ax.set_xlabel('Outer iter')
    ax.set_ylabel('Per-interaction payoff')
    ax.set_title('(a) Payoff (dots=per-episode, lines=per-iter mean)',
                 fontweight='bold', loc='left')
    ax.legend(fontsize=7, frameon=False, loc='lower right')
    ax.grid(True, alpha=0.2)

    # ── (b) Estimator MSE ──
    ax = axes[0, 1]
    ax.plot(t_axis, history['act_est_loss_1'], color='#2ECC71', label=r'$\hat\pi^1$ (Proud)')
    ax.plot(t_axis, history['act_est_loss_2'], color='#27AE60', linestyle='--', label=r'$\hat\pi^2$ (AllD)')
    ax.plot(t_axis, history['sig_est_loss_1'], color='#E74C3C', label=r'$\hat\varphi^1$')
    ax.plot(t_axis, history['sig_est_loss_2'], color='#C0392B', linestyle='--', label=r'$\hat\varphi^2$')
    ax.set_xlabel('Outer iter')
    ax.set_ylabel('MSE loss (last step)')
    ax.set_title('(b) Estimator MSE (log scale)', fontweight='bold', loc='left')
    ax.set_yscale('log')
    ax.legend(fontsize=7, frameon=False)
    ax.grid(True, alpha=0.2)

    # ── (c) Objective value gap ──
    ax = axes[0, 2]
    gap = np.array(history['real_perint']) - np.array(history['surrogate_perint'])
    ax.plot(t_axis, gap, 'o-', color='#9B59B6')
    ax.axhline(0, color='#AAA', linestyle=':')
    ax.set_xlabel('Outer iter')
    ax.set_ylabel(r'$J_{\mathrm{real}} - J_{\mathrm{surr}}$')
    ax.set_title('(c) Objective value gap (proxy for gradient bias)', fontweight='bold', loc='left')
    ax.grid(True, alpha=0.2)

    # ── (d) Signal policy evolution ──
    ax = axes[1, 0]
    snaps = history['signal_snapshots']
    n = len(snaps)
    cmap = plt.cm.plasma
    for i, snap in enumerate(snaps):
        color = cmap(i / max(n - 1, 1))
        label = f"t={snap['outer_iter']}" if (i % max(n // 5, 1) == 0 or i == n - 1) else None
        ax.plot(x_grid, snap['profile'], color=color, linewidth=1.2, alpha=0.85, label=label)
    ax.plot(x_grid, x_grid, '--', color='#AAA', linewidth=0.8, label='identity')
    ax.set_xlabel('Donor action')
    ax.set_ylabel('Signal σ')
    ax.set_title(r'(d) Signal policy $\varphi^0$ evolution', fontweight='bold', loc='left')
    ax.set_xlim([0, 1]); ax.set_ylim([-0.05, 1.05])
    ax.legend(fontsize=6, frameon=False, loc='upper left', ncol=2)
    ax.grid(True, alpha=0.2)

    # ── (e) Final signal policy ──
    ax = axes[1, 1]
    final_profile = snaps[-1]['profile']
    init_profile = snaps[0]['profile']
    ax.plot(x_grid, final_profile, color='#3498DB', linewidth=2,
            label=f'final $\\varphi^0$ ({history["final_disc_type"]}, std={history["final_signal_spread"]:.2f})')
    ax.plot(x_grid, init_profile, color='#888', linewidth=1.2, linestyle=':', label='init $\\varphi^0$')
    ax.plot(x_grid, x_grid, '--', color='#AAA', linewidth=0.8, label='identity')
    ax.set_xlabel('Donor action')
    ax.set_ylabel('Signal σ')
    ax.set_title('(e) Final signal policy', fontweight='bold', loc='left')
    ax.set_xlim([0, 1]); ax.set_ylim([-0.05, 1.05])
    ax.legend(fontsize=7, frameon=False, loc='lower right')
    ax.grid(True, alpha=0.2)

    # ── (f) Reputation evolution per agent ──
    ax = axes[1, 2]
    rep_log = np.array(history['rep_log'])
    ax.plot(t_axis, rep_log[:, 0], color='#3498DB', linewidth=1.4, label='Agent 0')
    ax.plot(t_axis, rep_log[:, 1], color='#2ECC71', linewidth=1.2, label='Opp1 (ProudCoop)')
    ax.plot(t_axis, rep_log[:, 2], color='#E67E22', linewidth=1.2, label='Opp2 (AllD)')
    ax.set_xlabel('Outer iter')
    ax.set_ylabel('Mean last-signal reputation')
    ax.set_title('(f) Reputation evolution (real env)', fontweight='bold', loc='left')
    ax.set_ylim([-0.05, 1.05])
    ax.legend(fontsize=7, frameon=False, loc='lower right')
    ax.grid(True, alpha=0.2)

    plt.suptitle(
        f'Exp 6 signal-only — {arch} × {init_mode} — seed {seed} '
        f'| final real per-int = {history["final_perint"]:.3f} '
        f'({history["final_perint"] / FULL_COOP_BENCHMARK * 100:.0f}% of {FULL_COOP_BENCHMARK})  '
        f'| signal {history["final_disc_type"]} (std={history["final_signal_spread"]:.2f})',
        fontsize=11, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    fname = f'fig_exp6_signal_{arch}_{init_mode}_c{int(C_VALUE)}_seed{seed}.png'
    path = os.path.join(save_dir, 'figures', fname)
    plt.savefig(path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f'    → {path}')


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def _strip_for_json(hist):
    """Drop heavy fields not needed downstream (matches exp4 pattern)."""
    keep = ['real_perint', 'surrogate_perint',
            'real_perint_per_ep', 'surrogate_perint_per_ep',
            'act_est_loss_1', 'act_est_loss_2',
            'sig_est_loss_1', 'sig_est_loss_2',
            'spread_log', 'rep_log',
            'final_perint', 'final_signal_profile',
            'final_signal_spread', 'final_disc_type']
    return {k: hist[k] for k in keep if k in hist}


def main():
    save_dir = os.path.dirname(os.path.abspath(__file__))
    os.makedirs(os.path.join(save_dir, "figures"), exist_ok=True)
    os.makedirs(os.path.join(save_dir, "data"), exist_ok=True)
    json_path = os.path.join(save_dir, "data", f"results_exp6_signal_only_c{int(C_VALUE)}.json")

    archs = ['MLP', 'TF', 'Factored']   # MLP and TF act as baseline / ablation vs Factored
    init_modes = ['const05', 'all_uniform']

    all_results = load_json_if_exists(json_path)
    if all_results:
        done = sum(len(v) for v in all_results.values())
        print(f"[resume] loaded {done} completed (config, seed) pairs from {json_path}")

    total_configs = len(archs) * len(init_modes)
    print(f"\n{'='*70}\n"
          f"  Exp 6 signal-only (= Exp 3h + opponent modeling)\n"
          f"  Variant grid: {total_configs} configs × {NUM_SEEDS} seeds\n"
          f"  Opponents: ProudCoop(L3) + AllD(L3)\n"
          f"  benchmark = (b−1)/4 = {FULL_COOP_BENCHMARK}\n"
          f"{'='*70}")

    config_idx = 0
    for arch in archs:
        for init_mode in init_modes:
            config_idx += 1
            key = f"{arch}_{init_mode}"
            print(f"\n[{config_idx}/{total_configs}] {key}")
            all_results.setdefault(key, {})
            for seed in range(NUM_SEEDS):
                if str(seed) in all_results[key]:
                    print(f"  Seed {seed}: [skip]", flush=True)
                    continue
                print(f"  Seed {seed}:", flush=True)
                hist = run_exp6(arch=arch, init_mode=init_mode, seed=seed)
                all_results[key][str(seed)] = _strip_for_json(hist)
                # Plot needs the full snapshots — use full hist before strip
                plot_one_run(hist, arch, init_mode, seed, save_dir)
                atomic_save_json(all_results, json_path)
                done = sum(len(v) for v in all_results.values())
                print(f"    [checkpoint] {done} pairs saved")

    # ── Summary ──
    print(f"\n{'='*70}\n  SUMMARY\n{'='*70}")
    rows = []
    for key in sorted(all_results.keys()):
        seeds = all_results[key]
        avg = float(np.mean([seeds[s]['final_perint'] for s in seeds]))
        disc = sum(1 for s in seeds if seeds[s]['final_disc_type'] == 'disc')
        n = len(seeds)
        rows.append((key, avg, disc, n))
    for key, avg, disc, n in sorted(rows, key=lambda x: -x[1]):
        pct = avg / FULL_COOP_BENCHMARK * 100
        print(f"  {key:<30}  avg={avg:6.3f} ({pct:3.0f}%)  disc={disc}/{n}")
    print(f"{'='*70}")

    atomic_save_json(all_results, json_path)
    print(f"Final results saved to {json_path}")


if __name__ == "__main__":
    main()
