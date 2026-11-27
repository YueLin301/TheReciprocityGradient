"""
Demo-only Agent with original Tanh MLP (no LayerNorm).

This reproduces the exact architecture used when the original computation graph
PDFs were generated, so the graph structure matches exactly.
"""
import torch
import torch.nn as nn
from typing import Optional, Sequence


def build_mlp(in_features: int, hidden: Sequence[int], out_features: int, out_activation: Optional[nn.Module] = None) -> nn.Sequential:
    layers = []
    last = in_features
    for hs in hidden:
        layers.append(nn.Linear(last, hs))
        layers.append(nn.Tanh())
        last = hs
    layers.append(nn.Linear(last, out_features))
    if out_activation is not None:
        layers.append(out_activation)
    return nn.Sequential(*layers)


class DemoAgent(nn.Module):
    """
    Demo agent with Markov approximation (memory window = 1).
    Uses Tanh activation to match the original computation graph.
    """

    def __init__(
        self,
        name: str,
        signal_dim: int = 1,
        second_order: bool = False,
        device: Optional[torch.device] = None,
    ):
        super().__init__()
        self.name = name
        self.signal_dim = signal_dim
        self.second_order = second_order
        self.device = device if device is not None else torch.device("cpu")
        self.is_demo = True

        try:
            agent_idx = int(name.split("_")[-1])
        except:
            agent_idx = 0

        demo_action_hidden = (10 + agent_idx,)
        demo_signal_hidden = (20 + agent_idx,)

        self.action_net = build_mlp(self.signal_dim, demo_action_hidden, 1, out_activation=nn.Sigmoid())

        signal_in_dim = 1
        if self.second_order:
            signal_in_dim += self.signal_dim

        self.signal_net = build_mlp(signal_in_dim, demo_signal_hidden, self.signal_dim, out_activation=nn.Sigmoid())

        self.register_buffer(
            "reputation_history",
            torch.empty((1, self.signal_dim), dtype=torch.float32)
        )
        self.to(self.device)

    def reset_reputation(self, initial_reputation: torch.Tensor):
        self.reputation_history = initial_reputation.view(1, self.signal_dim).to(
            dtype=torch.float32,
            device=self.device
        )

    def append_reputation(self, signal_t: torch.Tensor):
        sig = signal_t.to(self.device).view(1, self.signal_dim)
        self.reputation_history = torch.cat([self.reputation_history, sig], dim=0)

    def get_observation(self) -> torch.Tensor:
        return self.reputation_history

    def take_action(self, recipient_reputation: torch.Tensor) -> torch.Tensor:
        obs = recipient_reputation[-1].view(1, -1).to(self.device)
        action = self.action_net(obs).squeeze(0)
        return action

    def send_signal(self, donor_action: torch.Tensor, own_reputation: Optional[torch.Tensor] = None,
                    donor_reputation: Optional[torch.Tensor] = None) -> torch.Tensor:
        a = donor_action.to(self.device).view(1, 1)

        if self.second_order:
            if own_reputation is None:
                own_reputation = self.get_observation()
            own_rep = own_reputation[-1].view(1, -1).to(self.device)
            inp = torch.cat([a, own_rep], dim=-1)
        else:
            inp = a
        signal = self.signal_net(inp).squeeze(0)
        return signal
