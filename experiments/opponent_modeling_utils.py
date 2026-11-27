"""
Shared opponent-modeling scaffolding for Exp 4, 5, 6, 7, 8.

Provides:
  • Estimator architectures (MLP / Transformer, with tanh activations).
  • EstimatedOpponent — env-compatible wrapper around (action_est, signal_est).
  • DataBuffer + RecordingWrapper — collect (input, output) pairs from real opps.
  • train_action_estimator / train_signal_estimator — supervised MSE updates.

Design contract (env_batched.py expects these methods on any agent placed into
BatchedDonationGame):
  - reset_reputation(initial_reputation=None)
  - append_reputation(signal_t)
  - get_observation()
  - batched_take_action(rep_histories, own_rep_histories) -> [B, 1]
  - batched_send_signal(actions, own_reps, donor_reps=None) -> [B, signal_dim]
  - attributes: name, signal_dim, second_order
"""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
from typing import Optional


# ══════════════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════════════

def sinusoidal_pe(max_len: int, emb_dim: int) -> torch.Tensor:
    """Vaswani-style positional encoding."""
    pe = torch.zeros(max_len, emb_dim)
    pos = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
    div = torch.exp(torch.arange(0, emb_dim, 2, dtype=torch.float32)
                    * (-torch.log(torch.tensor(10000.0)) / emb_dim))
    pe[:, 0::2] = torch.sin(pos * div)
    if emb_dim > 1:
        pe[:, 1::2] = torch.cos(pos * div[:emb_dim // 2])
    return pe


# ══════════════════════════════════════════════════════════════════════════════
# Estimators — tanh activation (NOT ReLU) to match smooth sigmoid shapes.
# ══════════════════════════════════════════════════════════════════════════════

class MLPActionEstimator(nn.Module):
    """π̂^j (MLP, Markov-1): reads last opp_rep + last own_rep, predicts action ∈ [0,1]."""
    def __init__(self, hidden=16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, 1), nn.Sigmoid()
        )

    def forward(self, opp_rep_hist, own_rep_hist):
        # [B, seq, 1] → [B, 1]
        opp_last = opp_rep_hist[:, -1, :]
        own_last = (own_rep_hist[:, -1, :] if own_rep_hist is not None
                    else torch.full_like(opp_last, 0.5))
        return self.net(torch.cat([opp_last, own_last], dim=-1))


class MLPSignalEstimator(nn.Module):
    """φ̂^j (MLP, Markov-1): reads donor action + j's last own_rep, predicts σ ∈ [0,1]."""
    def __init__(self, hidden=16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, 1), nn.Sigmoid()
        )

    def forward(self, action, own_rep_hist, donor_rep_hist=None):
        # action: [B, 1], own_rep_hist: [B, seq, 1]
        own_last = (own_rep_hist[:, -1, :] if own_rep_hist is not None
                    else torch.full_like(action, 0.5))
        return self.net(torch.cat([action, own_last], dim=-1))


class TFActionEstimator(nn.Module):
    """π̂^j (Transformer): full rep history through encoder + tanh MLP head."""
    def __init__(self, emb_dim=16, hidden=16):
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
        self.register_buffer("_pe", sinusoidal_pe(256, emb_dim))

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
    """φ̂^j (Transformer): action scalar + own-rep history → signal ∈ [0,1]."""
    def __init__(self, emb_dim=16, hidden=16):
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
        self.register_buffer("_pe", sinusoidal_pe(256, emb_dim))

    def forward(self, action, own_rep_hist, donor_rep_hist=None):
        act_emb = self.action_proj(action)
        own_p = self.own_proj(own_rep_hist) + self._pe[:own_rep_hist.shape[1]].unsqueeze(0)
        own_emb = self.encoder(own_p)[:, -1, :]
        return self.head(torch.cat([act_emb, own_emb], dim=-1))


def build_estimators(arch: str, emb_dim: int = 16, hidden: int = 16,
                     factor_dim: int = 8):
    """Return (action_estimator, signal_estimator) of the requested architecture."""
    if arch == 'MLP':
        return MLPActionEstimator(hidden=hidden), MLPSignalEstimator(hidden=hidden)
    elif arch == 'TF':
        return TFActionEstimator(emb_dim=emb_dim, hidden=hidden), TFSignalEstimator(emb_dim=emb_dim, hidden=hidden)
    elif arch == 'Factored':
        return (FactoredMLPActionEstimator(factor_dim=factor_dim, hidden=hidden),
                FactoredMLPSignalEstimator(factor_dim=factor_dim, hidden=hidden))
    else:
        raise ValueError(f"Unknown estimator arch: {arch}")


# ══════════════════════════════════════════════════════════════════════════════
# Factored MLP (multiplicative-interaction architecture).
#   - Per-input nonlinear feature extractors f(a), g(r).
#   - Head MLP combines [f(a), g(r), f(a) ⊙ g(r)] (additive + multiplicative).
#   - General inductive bias for joint-assessment functions (leading-eight,
#     third-order norms, any bilinear / multiplicative opponent structure).
#   - Same family as Factorization Machines, bilinear pooling, dual encoders.
# ══════════════════════════════════════════════════════════════════════════════

class FactoredMLPActionEstimator(nn.Module):
    """π̂^j (Factored): per-input MLP features + Hadamard product combination.
    Reads (last_opp_rep, last_own_rep), predicts action ∈ [0,1]."""
    def __init__(self, factor_dim: int = 8, hidden: int = 16):
        super().__init__()
        self.f_opp = nn.Sequential(
            nn.Linear(1, factor_dim), nn.Tanh(),
            nn.Linear(factor_dim, factor_dim), nn.Tanh()
        )
        self.g_own = nn.Sequential(
            nn.Linear(1, factor_dim), nn.Tanh(),
            nn.Linear(factor_dim, factor_dim), nn.Tanh()
        )
        self.head = nn.Sequential(
            nn.Linear(factor_dim * 3, hidden), nn.Tanh(),
            nn.Linear(hidden, 1), nn.Sigmoid()
        )

    def forward(self, opp_rep_hist, own_rep_hist):
        opp_last = opp_rep_hist[:, -1, :]
        own_last = (own_rep_hist[:, -1, :] if own_rep_hist is not None
                    else torch.full_like(opp_last, 0.5))
        f = self.f_opp(opp_last)
        g = self.g_own(own_last)
        return self.head(torch.cat([f, g, f * g], dim=-1))


class FactoredMLPSignalEstimator(nn.Module):
    """φ̂^j (Factored): per-input MLP features + Hadamard product combination.
    Reads (action, last_own_rep), predicts signal ∈ [0,1]."""
    def __init__(self, factor_dim: int = 8, hidden: int = 16):
        super().__init__()
        self.f_a = nn.Sequential(
            nn.Linear(1, factor_dim), nn.Tanh(),
            nn.Linear(factor_dim, factor_dim), nn.Tanh()
        )
        self.g_r = nn.Sequential(
            nn.Linear(1, factor_dim), nn.Tanh(),
            nn.Linear(factor_dim, factor_dim), nn.Tanh()
        )
        self.head = nn.Sequential(
            nn.Linear(factor_dim * 3, hidden), nn.Tanh(),
            nn.Linear(hidden, 1), nn.Sigmoid()
        )

    def forward(self, action, own_rep_hist, donor_rep_hist=None):
        a = action
        r = (own_rep_hist[:, -1, :] if own_rep_hist is not None
             else torch.full_like(a, 0.5))
        f = self.f_a(a)
        g = self.g_r(r)
        return self.head(torch.cat([f, g, f * g], dim=-1))


# ══════════════════════════════════════════════════════════════════════════════
# Amortized estimators — single shared network conditioned on opp-ID embedding.
# Used in Exp 8 (scalability) to keep parameter count O(1) instead of O(N).
# ══════════════════════════════════════════════════════════════════════════════

class AmortizedActionEstimator(nn.Module):
    """One MLP shared across opponents, conditioned on a learned opp-ID embedding.

    Forward signature parallels MLPActionEstimator but takes an extra opp_id
    (int or [B] LongTensor) selecting which opponent we're predicting for.
    """
    def __init__(self, n_opps: int, opp_emb_dim: int = 8, hidden: int = 16):
        super().__init__()
        self.n_opps = n_opps
        self.opp_emb = nn.Embedding(n_opps, opp_emb_dim)
        self.net = nn.Sequential(
            nn.Linear(2 + opp_emb_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, 1), nn.Sigmoid()
        )

    def forward(self, opp_rep_hist, own_rep_hist, opp_id):
        opp_last = opp_rep_hist[:, -1, :]                                   # [B, 1]
        own_last = (own_rep_hist[:, -1, :] if own_rep_hist is not None
                    else torch.full_like(opp_last, 0.5))                    # [B, 1]
        if isinstance(opp_id, int):
            opp_id_t = torch.tensor(opp_id, dtype=torch.long, device=opp_last.device)
            emb = self.opp_emb(opp_id_t).unsqueeze(0).expand(opp_last.shape[0], -1)
        else:
            emb = self.opp_emb(opp_id)                                      # [B, emb]
        return self.net(torch.cat([opp_last, own_last, emb], dim=-1))


class AmortizedSignalEstimator(nn.Module):
    """One MLP shared across opponents, conditioned on opp-ID embedding."""
    def __init__(self, n_opps: int, opp_emb_dim: int = 8, hidden: int = 16):
        super().__init__()
        self.n_opps = n_opps
        self.opp_emb = nn.Embedding(n_opps, opp_emb_dim)
        self.net = nn.Sequential(
            nn.Linear(2 + opp_emb_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, 1), nn.Sigmoid()
        )

    def forward(self, action, own_rep_hist, opp_id, donor_rep_hist=None):
        own_last = (own_rep_hist[:, -1, :] if own_rep_hist is not None
                    else torch.full_like(action, 0.5))
        if isinstance(opp_id, int):
            opp_id_t = torch.tensor(opp_id, dtype=torch.long, device=action.device)
            emb = self.opp_emb(opp_id_t).unsqueeze(0).expand(action.shape[0], -1)
        else:
            emb = self.opp_emb(opp_id)
        return self.net(torch.cat([action, own_last, emb], dim=-1))


class AmortizedEstimatedOpponent(nn.Module):
    """Env-compatible wrapper around a shared amortized (action_est, signal_est).
    Each instance carries an opp_id that selects its slice of the shared net."""
    def __init__(self, name: str, opp_id: int,
                 action_est: nn.Module, signal_est: nn.Module,
                 signal_dim: int = 1, device: Optional[torch.device] = None):
        super().__init__()
        self.name = name
        self.opp_id = opp_id
        self.signal_dim = signal_dim
        self.second_order = True
        self.device = device if device is not None else torch.device("cpu")
        # Note: action_est and signal_est are SHARED references — do NOT register as submodules
        # (otherwise nn.Module would try to copy/own them per-instance).
        # Store as attributes directly.
        object.__setattr__(self, 'action_est', action_est)
        object.__setattr__(self, 'signal_est', signal_est)
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
        return self.action_est(rep_histories, own_rep_histories, self.opp_id).clamp(0, 1)

    def batched_send_signal(self, actions, own_reps, donor_reps=None):
        return self.signal_est(actions, own_reps, self.opp_id, donor_reps).clamp(0, 1)


def train_amortized_action_estimator(est: nn.Module, buffers: list,
                                      k_steps: int = 100, lr: float = 1e-2) -> float:
    """Train shared amortized action est on records from per-opp buffers.

    buffers : list of DataBuffer — index = opp_id within the shared net.
    """
    opt = optim.Adam(est.parameters(), lr=lr)
    final_loss = 0.0
    n_opps = len(buffers)
    # Pre-collect non-empty opp ids
    nonempty = [i for i in range(n_opps) if buffers[i].all_action_records()]
    if not nonempty:
        return float('nan')
    for _ in range(k_steps):
        opp_id = nonempty[np.random.randint(len(nonempty))]
        records = buffers[opp_id].all_action_records()
        rec = records[np.random.randint(len(records))]
        opp_rep, own_rep, obs_a = rec
        pred = est(opp_rep, own_rep, opp_id).clamp(0, 1)
        loss = ((pred - obs_a) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        final_loss = loss.item()
    return final_loss


def train_amortized_signal_estimator(est: nn.Module, buffers: list,
                                      k_steps: int = 100, lr: float = 1e-2) -> float:
    opt = optim.Adam(est.parameters(), lr=lr)
    final_loss = 0.0
    n_opps = len(buffers)
    nonempty = [i for i in range(n_opps) if buffers[i].all_signal_records()]
    if not nonempty:
        return float('nan')
    for _ in range(k_steps):
        opp_id = nonempty[np.random.randint(len(nonempty))]
        records = buffers[opp_id].all_signal_records()
        rec = records[np.random.randint(len(records))]
        action, own_rep, donor_rep, obs_s = rec
        pred = est(action, own_rep, opp_id, donor_rep).clamp(0, 1)
        loss = ((pred - obs_s) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        final_loss = loss.item()
    return final_loss


# ══════════════════════════════════════════════════════════════════════════════
# EstimatedOpponent: env-compatible wrapper.
# ══════════════════════════════════════════════════════════════════════════════

class EstimatedOpponent(nn.Module):
    """Acts like a real opponent in env.play_batch(), but routes batched_*
    calls through learned (action_est, signal_est). Used during the
    surrogate-env policy-update phase."""
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
# DataBuffer + RecordingWrapper
# ══════════════════════════════════════════════════════════════════════════════

class DataBuffer:
    """Sliding window of (input, observed output) pairs for one opponent.

    Each outer iter calls `new_iter()` to roll the window. Records appended
    during that iter go into the newest slot. Older slots fall off when the
    window fills.
    """
    def __init__(self, window: int):
        self.window = window
        self.action_data = deque(maxlen=window)   # each: list of (opp_rep_hist, own_rep_hist, obs_action)
        self.signal_data = deque(maxlen=window)   # each: list of (action, own_rep_hist, donor_rep_hist, obs_signal)

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
    """Wraps any opponent that exposes the env contract and records every
    batched call into a DataBuffer. Forwards to the underlying opponent for
    ground-truth output (used during real-env Phase-1 data collection)."""
    def __init__(self, inner: nn.Module, buffer: DataBuffer):
        super().__init__()
        self.inner = inner
        self.buffer = buffer
        # Forward attributes the env reads
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
# Estimator training (supervised MSE on buffered observations).
# ══════════════════════════════════════════════════════════════════════════════

def train_action_estimator(est: nn.Module, buf: DataBuffer,
                           k_steps: int = 100, lr: float = 1e-2) -> float:
    """Train action estimator for k_steps. Returns final-step MSE for logging."""
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
                           k_steps: int = 100, lr: float = 1e-2) -> float:
    """Train signal estimator for k_steps. Returns final-step MSE for logging."""
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
