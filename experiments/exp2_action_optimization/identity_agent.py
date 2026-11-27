"""
Hard-coded identity agent for use as opponent.

No neural networks, no Sigmoid compression.
  take_action(rep) = exponentially-weighted rep (recent signals matter more)
  send_signal(a) = a  (exact honest reporting)

Fully differentiable — gradients flow through this agent.
"""
import torch
import torch.nn as nn
from typing import Optional


class IdentityAgent(nn.Module):
    """
    Fixed-strategy agent implementing identity mapping with recency bias.

    Action = exponentially weighted reputation:
      weight of signal at position k (from end) = decay^k
      Most recent signal has weight 1, init_token's weight decays to ~0.

    With decay=0.5, after 5 signals the init_token contributes < 3%.
    """

    def __init__(self, name: str, signal_dim: int = 1, decay: float = None,
                 memory_window: int = None,
                 device: Optional[torch.device] = None):
        """
        Args:
            decay: None for uniform mean, float (e.g. 0.5) for EMA.
            memory_window: if set, only use the last k entries of rep history.
                           None means use full history.
        """
        super().__init__()
        self.name = name
        self.signal_dim = signal_dim
        self.second_order = False
        self.decay = decay
        self.memory_window = memory_window
        self.device = device if device is not None else torch.device("cpu")

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

    def take_action(self, recipient_reputation: torch.Tensor) -> torch.Tensor:
        """
        action = aggregated reputation.
        decay=None: uniform mean (all signals equal weight)
        decay=float: EMA (recent signals dominate, init_token fades)
        memory_window: if set, only use last k entries
        """
        rep = recipient_reputation
        if self.memory_window is not None:
            rep = rep[-self.memory_window:]

        if self.decay is None:
            return rep.mean(dim=0).view(1)

        seq_len = rep.shape[0]
        exponents = torch.arange(seq_len - 1, -1, -1, dtype=torch.float32,
                                  device=rep.device)
        weights = torch.pow(self.decay, exponents)
        weights = weights / weights.sum()
        weighted = (rep.squeeze(-1) * weights).sum()
        return weighted.view(1)

    def send_signal(self, donor_action: torch.Tensor,
                    own_reputation: Optional[torch.Tensor] = None,
                    donor_reputation: Optional[torch.Tensor] = None) -> torch.Tensor:
        """signal = action. Exact identity."""
        return donor_action.to(self.device).view(self.signal_dim)
