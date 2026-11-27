import torch
import torch.nn as nn
from typing import Optional, Sequence


def build_mlp(in_features: int, hidden: Sequence[int], out_features: int,
              out_activation: Optional[nn.Module] = None) -> nn.Sequential:
    layers = []
    last = in_features
    for hs in hidden:
        layers.append(nn.Linear(last, hs))
        layers.append(nn.LayerNorm(hs))
        layers.append(nn.LeakyReLU(0.1))
        last = hs
    layers.append(nn.Linear(last, out_features))
    if out_activation is not None:
        layers.append(out_activation)
    return nn.Sequential(*layers)


def _make_transformer_encoder(emb_dim: int) -> nn.TransformerEncoder:
    layer = nn.TransformerEncoderLayer(
        d_model=emb_dim, nhead=1, dim_feedforward=32, batch_first=True)
    return nn.TransformerEncoder(layer, num_layers=1)


def _sinusoidal_pe(max_len: int, emb_dim: int) -> torch.Tensor:
    """Sinusoidal positional encoding (Vaswani et al. 2017)."""
    pe = torch.zeros(max_len, emb_dim)
    pos = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
    div = torch.exp(torch.arange(0, emb_dim, 2, dtype=torch.float32)
                    * (-torch.log(torch.tensor(10000.0)) / emb_dim))
    pe[:, 0::2] = torch.sin(pos * div)
    if emb_dim > 1:
        pe[:, 1::2] = torch.cos(pos * div[:emb_dim // 2])
    return pe  # [max_len, emb_dim]


class Agent(nn.Module):
    """
    Continuous agent with two INDEPENDENT strategy networks:
      - Action network pi^i: recipient's reputation history -> action a in [0,1]
      - Signal network phi^i: donor's action (+ own rep in 2nd order) -> signal sigma in [0,1]

    Each network has its own Transformer encoder. No shared parameters.
    """

    def __init__(
        self,
        name: str,
        action_hidden: Sequence[int] = (16,),
        signal_hidden: Sequence[int] = (16,),
        emb_dim: int = 16,
        signal_dim: int = 1,
        second_order: bool = False,
        device: Optional[torch.device] = None,
        is_demo: bool = False,
    ):
        super().__init__()
        self.name = name
        self.signal_dim = signal_dim
        self.second_order = second_order
        self.signal_history_embedding_dim = emb_dim
        self.device = device if device is not None else torch.device("cpu")
        self.is_demo = is_demo

        if self.is_demo:
            try:
                agent_idx = int(name.split("_")[-1])
            except:
                agent_idx = 0

            demo_action_hidden = (10 + agent_idx,)
            demo_signal_hidden = (20 + agent_idx,)

            self.action_net = build_mlp(
                self.signal_dim, demo_action_hidden, 1, out_activation=nn.Sigmoid())

            signal_in_dim = 1
            if self.second_order:
                signal_in_dim += self.signal_dim

            self.signal_net = build_mlp(
                signal_in_dim, demo_signal_hidden, self.signal_dim, out_activation=nn.Sigmoid())
        else:
            # === Action network: own Transformer encoder + MLP head ===
            self.action_input_proj = nn.Linear(self.signal_dim, emb_dim)
            self.action_encoder = _make_transformer_encoder(emb_dim)
            self.action_net = build_mlp(emb_dim, action_hidden, 1, out_activation=nn.Sigmoid())

            # === Signal network: own Transformer encoder + action proj + MLP head ===
            self.signal_input_proj = nn.Linear(self.signal_dim, emb_dim)
            self.signal_encoder = _make_transformer_encoder(emb_dim)
            self.signal_action_proj = nn.Linear(1, emb_dim)

            # Sinusoidal positional encoding (shared buffer, not learnable)
            max_seq_len = 256  # more than enough for any episode
            self.register_buffer("_pe", _sinusoidal_pe(max_seq_len, emb_dim))

            # Signal input: [action_emb, own_rep_emb], both emb_dim
            signal_in_dim = emb_dim + emb_dim
            self.signal_net = build_mlp(
                signal_in_dim, signal_hidden, self.signal_dim, out_activation=nn.Sigmoid())

        # Initial reputation token: fixed 0.5 = uninformative prior on [0,1]
        self.register_buffer(
            "init_token",
            torch.full((1, self.signal_dim), 0.5, dtype=torch.float32))
        self.register_buffer(
            "reputation_history",
            torch.full((1, self.signal_dim), 0.5, dtype=torch.float32))
        self.to(self.device)

    def reset_reputation(self, initial_reputation: torch.Tensor = None):
        if initial_reputation is not None:
            self.reputation_history = initial_reputation.view(1, self.signal_dim).to(
                dtype=torch.float32, device=self.device)
        else:
            self.reputation_history = self.init_token.clone()

    def append_reputation(self, signal_t: torch.Tensor):
        sig = signal_t.to(self.device).view(1, self.signal_dim)
        self.reputation_history = torch.cat([self.reputation_history, sig], dim=0)

    def get_observation(self) -> torch.Tensor:
        return self.reputation_history

    def _encode_for_action(self, history: torch.Tensor) -> torch.Tensor:
        """Encode reputation history using the ACTION encoder.
        Adds sinusoidal positional encoding so Transformer knows time order.
        Takes the last position output — most recent signal attends to full history."""
        hist = history.to(self.device).unsqueeze(0)   # [1, seq_len, signal_dim]
        proj = self.action_input_proj(hist)            # [1, seq_len, emb_dim]
        seq_len = proj.shape[1]
        proj = proj + self._pe[:seq_len].unsqueeze(0)  # add positional encoding
        enc = self.action_encoder(proj)                # [1, seq_len, emb_dim]
        return enc[:, -1, :].squeeze(0)                # [emb_dim] — last position

    def _encode_for_signal(self, history: torch.Tensor) -> torch.Tensor:
        """Encode reputation history using the SIGNAL encoder.
        Adds sinusoidal positional encoding. Takes the last position output."""
        hist = history.to(self.device).unsqueeze(0)
        proj = self.signal_input_proj(hist)
        seq_len = proj.shape[1]
        proj = proj + self._pe[:seq_len].unsqueeze(0)
        enc = self.signal_encoder(proj)
        return enc[:, -1, :].squeeze(0)

    def set_hardcode_action(self, decay: float = None):
        """Bypass action_net with exact aggregation. No Sigmoid bottleneck.
        decay=None: action = mean(rep).  decay=float: action = EMA(rep)."""
        self._hardcode_action = True
        self._hardcode_action_decay = decay

    def set_hardcode_signal(self, norm: str = None, beta: float = 5.0):
        """Bypass signal_net with exact function.
        norm=None: identity (σ = a).
        norm='L3' or 'L6': tanh spin-space formula."""
        self._hardcode_signal = True
        self._hardcode_signal_norm = norm
        self._hardcode_signal_beta = beta

    def take_action(self, recipient_reputation: torch.Tensor) -> torch.Tensor:
        """Action policy: pi^i(s_t^j) -> a in [0,1]."""
        if getattr(self, '_hardcode_action', False):
            decay = getattr(self, '_hardcode_action_decay', None)
            if decay is None:
                return recipient_reputation.mean(dim=0).view(1)
            seq_len = recipient_reputation.shape[0]
            exponents = torch.arange(seq_len - 1, -1, -1, dtype=torch.float32,
                                      device=recipient_reputation.device)
            weights = torch.pow(decay, exponents)
            weights = weights / weights.sum()
            return (recipient_reputation.squeeze(-1) * weights).sum().view(1)

        if self.is_demo:
            obs = recipient_reputation[-1].view(1, -1).to(self.device)
            return self.action_net(obs).squeeze(0)

        emb = self._encode_for_action(recipient_reputation)
        return self.action_net(emb.view(1, -1)).squeeze(0)

    def send_signal(self, donor_action: torch.Tensor,
                    own_reputation: Optional[torch.Tensor] = None,
                    donor_reputation: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Signal policy: phi^i(a, s_t^i) -> sigma in [0,1]."""
        if getattr(self, '_hardcode_signal', False):
            norm = getattr(self, '_hardcode_signal_norm', None)
            if norm is None:
                return donor_action.to(self.device).view(self.signal_dim)
            # Tanh spin-space signal
            a = donor_action.to(self.device).view(-1)
            if own_reputation is None:
                own_reputation = self.get_observation()
            r = own_reputation.mean()
            b = self._hardcode_signal_beta
            sx = torch.tanh(b * (a - 0.5))
            sy = torch.tanh(b * (r - 0.5))
            if norm == 'L3':
                sig = 1.0 - 0.25 * (1 - sx) * (1 + sy)
            elif norm == 'L6':
                sig = 0.5 * (1 + sx * sy)
            else:
                sig = a  # fallback to identity
            return sig.clamp(0, 1).view(self.signal_dim)

        a = donor_action.to(self.device).view(1, 1)

        if self.is_demo:
            if self.second_order:
                if own_reputation is None:
                    own_reputation = self.get_observation()
                own_rep = own_reputation[-1].view(1, -1).to(self.device)
                inp = torch.cat([a, own_rep], dim=-1)
            else:
                inp = a
            return self.signal_net(inp).squeeze(0)

        # Non-demo: project action + encode own rep, both to emb_dim
        if own_reputation is None:
            own_reputation = self.get_observation()
        action_emb = self.signal_action_proj(a)  # [1, emb_dim]
        own_emb = self._encode_for_signal(own_reputation).view(1, -1)  # [1, emb_dim]
        inp = torch.cat([action_emb, own_emb], dim=-1)  # [1, 2*emb_dim]
        return self.signal_net(inp).squeeze(0)

    def get_action_params(self) -> list:
        """All parameters in the action network (encoder + head)."""
        if self.is_demo:
            return list(self.action_net.parameters())
        return (list(self.action_input_proj.parameters())
                + list(self.action_encoder.parameters())
                + list(self.action_net.parameters()))

    def get_signal_params(self) -> list:
        """All parameters in the signal network (encoder + head)."""
        if self.is_demo:
            return list(self.signal_net.parameters())
        return (list(self.signal_action_proj.parameters())
                + list(self.signal_input_proj.parameters())
                + list(self.signal_encoder.parameters())
                + list(self.signal_net.parameters()))

