"""
Two hard-coded opponent types for signal optimization experiments.

Both condition their ACTIONS on their own reputation, creating a gradient
pathway from Agent 0's signal → opponent's reputation → opponent's action → reward.

Type A — Proud Cooperator (Grim Trigger):
  action = sigmoid(T * (own_rep - 0.5)): cooperate when own rep is high
Type B — Guilty Defector:
  action = sigmoid(T * (0.5 - own_rep)): cooperate when own rep is low (redeem)

Optimal signal for Agent 0: honest reporting σ(a) ≈ a
  - Cooperators get high signal → keep cooperating
  - Defectors get low signal → guilt → switch to cooperating
"""
import torch
import torch.nn as nn
from typing import Optional


class ProudCooperator(nn.Module):
    """Hard-coded grim trigger: cooperate when own rep is high."""

    def __init__(self, name: str, temperature: float = 10.0,
                 signal_dim: int = 1, device: Optional[torch.device] = None):
        super().__init__()
        self.name = name
        self.signal_dim = signal_dim
        self.second_order = False
        self.temperature = temperature
        self.device = device if device is not None else torch.device("cpu")

        self.register_buffer("temp", torch.tensor(temperature))
        self.register_buffer("init_token",
                             torch.full((1, self.signal_dim), 0.5, dtype=torch.float32))
        self.register_buffer("reputation_history",
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
        own_rep = self.reputation_history[-1].mean()
        action = torch.sigmoid(self.temp * (own_rep - 0.5))
        return action.view(1)

    def send_signal(self, donor_action: torch.Tensor,
                    own_reputation: Optional[torch.Tensor] = None,
                    donor_reputation: Optional[torch.Tensor] = None) -> torch.Tensor:
        a = donor_action.to(self.device).view(-1)
        signal = torch.sigmoid(self.temp * (a - 0.5))
        return signal.view(self.signal_dim)


class GuiltyDefector(nn.Module):
    """Hard-coded defector with redemption: cooperate when own rep is low."""

    def __init__(self, name: str, temperature: float = 10.0,
                 signal_dim: int = 1, device: Optional[torch.device] = None):
        super().__init__()
        self.name = name
        self.signal_dim = signal_dim
        self.second_order = False
        self.temperature = temperature
        self.device = device if device is not None else torch.device("cpu")

        self.register_buffer("temp", torch.tensor(temperature))
        self.register_buffer("init_token",
                             torch.full((1, self.signal_dim), 0.5, dtype=torch.float32))
        self.register_buffer("reputation_history",
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
        own_rep = self.reputation_history[-1].mean()
        action = torch.sigmoid(self.temp * (0.5 - own_rep))
        return action.view(1)

    def send_signal(self, donor_action: torch.Tensor,
                    own_reputation: Optional[torch.Tensor] = None,
                    donor_reputation: Optional[torch.Tensor] = None) -> torch.Tensor:
        a = donor_action.to(self.device).view(-1)
        signal = torch.sigmoid(self.temp * (a - 0.5))
        return signal.view(self.signal_dim)
