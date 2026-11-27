"""Modified cooperator that conditions on BOTH own rep AND recipient rep.
This creates joint incentive for Agent 0's π (must get high rep) AND φ
(can boost HybridCoop's self-rep to increase its cooperation rate)."""

import torch
import torch.nn as nn

BETA = 5.0
T_ACTION = 10.0


def _tanh_L3_signal(action, recip_rep, beta=BETA):
    sx = torch.tanh(beta * (action - 0.5))
    sy = torch.tanh(beta * (recip_rep - 0.5))
    return (1.0 - 0.25 * (1 - sx) * (1 + sy)).clamp(0, 1)


class HybridCoopL3(nn.Module):
    """Cooperates if BOTH (own_rep + recip_rep)/2 is high.
    Uses L3 signal rule (same as ProudCoop).

    Creates joint incentive:
      - π must be discriminative (Agent 0 needs high rep → cooperate w/ good-rep)
      - φ can boost HybridCoop's own_rep → more cooperation toward Agent 0
    """

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
        recip_rep = recipient_reputation[-1].mean()
        score = 0.5 * own_rep + 0.5 * recip_rep
        return torch.sigmoid(T_ACTION * (score - 0.5)).view(1)

    def batched_take_action(self, recip_rep_histories, own_rep_histories=None):
        # recip_rep_histories: [B, T, D]
        # own_rep_histories is typically self.reputation_history for batched
        if own_rep_histories is None:
            own_rep_histories = self.reputation_history.unsqueeze(0).expand(
                recip_rep_histories.shape[0], -1, -1)
        own = own_rep_histories[:, -1].mean(dim=-1, keepdim=True)
        recip = recip_rep_histories[:, -1].mean(dim=-1, keepdim=True)
        score = 0.5 * own + 0.5 * recip
        return torch.sigmoid(T_ACTION * (score - 0.5))

    def send_signal(self, donor_action, own_reputation=None, donor_reputation=None):
        a = donor_action.to(self.device).view(-1)
        if own_reputation is not None:
            recip_rep = own_reputation[-1].mean()
        else:
            recip_rep = self.reputation_history[-1].mean()
        return _tanh_L3_signal(a, recip_rep).view(self.signal_dim)

    def batched_send_signal(self, donor_actions, own_rep_histories=None, donor_rep_histories=None):
        a = donor_actions.view(-1, 1)
        if own_rep_histories is not None:
            recip = own_rep_histories[:, -1].mean(dim=-1, keepdim=True)
        else:
            recip = self.reputation_history[-1].mean().expand(a.shape[0], 1)
        return _tanh_L3_signal(a, recip)
