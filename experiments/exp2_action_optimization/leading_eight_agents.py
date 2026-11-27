"""
All 8 leading-eight social norms (Ohtsuki & Iwasa, 2006) as differentiable agents.

Third-order assessment: signal depends on (donor_action, recipient_own_rep, donor_rep).
L3 (Simple Standing) and L6 (Stern Judging) are second-order (don't use donor_rep).
L1, L2, L4, L5, L7, L8 are genuinely third-order.

All use soft sigmoid thresholds for differentiability.
"""
import torch
import torch.nn as nn
from typing import Optional

# Assessment tables: (donor_rep, action, recip_rep) -> new_rep
LEADING_EIGHT = {
    "L1": {
        (True,True,True): True, (True,True,False): True,
        (True,False,True): False, (True,False,False): True,
        (False,True,True): True, (False,True,False): True,
        (False,False,True): False, (False,False,False): False,
    },
    "L2": {
        (True,True,True): True, (True,True,False): False,
        (True,False,True): False, (True,False,False): True,
        (False,True,True): True, (False,True,False): True,
        (False,False,True): False, (False,False,False): False,
    },
    "L3": {
        (True,True,True): True, (True,True,False): True,
        (True,False,True): False, (True,False,False): True,
        (False,True,True): True, (False,True,False): True,
        (False,False,True): False, (False,False,False): True,
    },
    "L4": {
        (True,True,True): True, (True,True,False): True,
        (True,False,True): False, (True,False,False): True,
        (False,True,True): True, (False,True,False): False,
        (False,False,True): False, (False,False,False): True,
    },
    "L5": {
        (True,True,True): True, (True,True,False): False,
        (True,False,True): False, (True,False,False): True,
        (False,True,True): True, (False,True,False): True,
        (False,False,True): False, (False,False,False): True,
    },
    "L6": {
        (True,True,True): True, (True,True,False): False,
        (True,False,True): False, (True,False,False): True,
        (False,True,True): True, (False,True,False): False,
        (False,False,True): False, (False,False,False): True,
    },
    "L7": {
        (True,True,True): True, (True,True,False): True,
        (True,False,True): False, (True,False,False): True,
        (False,True,True): True, (False,True,False): False,
        (False,False,True): False, (False,False,False): False,
    },
    "L8": {
        (True,True,True): True, (True,True,False): False,
        (True,False,True): False, (True,False,False): True,
        (False,True,True): True, (False,True,False): False,
        (False,False,True): False, (False,False,False): False,
    },
}

ACTION_RULES = {
    "L1": {(True,True): True, (True,False): False, (False,True): True, (False,False): True},
    "L2": {(True,True): True, (True,False): False, (False,True): True, (False,False): True},
    **{f"L{i}": {(True,True): True, (True,False): False, (False,True): True, (False,False): False}
       for i in range(3, 9)},
}

NORM_NAMES = {
    "L1": "L1", "L2": "Consistent Standing", "L3": "Simple Standing",
    "L4": "L4", "L5": "L5", "L6": "Stern Judging",
    "L7": "Staying", "L8": "Judging",
}


class LeadingEightAgent(nn.Module):
    """
    Leading-eight agent with two modes:

    continuous=False (default): Discrete lookup table softened by sigmoid(T*(x-0.5)).
        Output is approximately binary. Faithful to the classical formulation.

    continuous=True: Bilinear interpolation over the assessment table,
        with optional sharpness to control transition steepness.
        sharpness=0: pure bilinear (linear transitions).
        sharpness>0: apply sigmoid(sharpness*(x-0.5)) to inputs before bilinear.
        For second-order norms (L3, L6):
          Action (L3=L6): a = f(opp_rep)
          Signal L3: σ = 1 - f(recip_rep) * (1 - f(action))
          Signal L6: σ = f(action) * f(recip_rep) + (1-f(action)) * (1-f(recip_rep))
    """

    def __init__(self, name: str, norm: str = "L6", temperature: float = 10.0,
                 continuous: bool = False, decay: float = None, beta: float = 5.0,
                 signal_dim: int = 1, device: Optional[torch.device] = None):
        super().__init__()
        self.name = name
        self.norm = norm
        self.norm_name = NORM_NAMES[norm]
        self.assessment = LEADING_EIGHT[norm]
        self.action_rule = ACTION_RULES[norm]
        self.temperature = temperature
        self.continuous = continuous
        self.decay = decay
        self.beta = beta
        self.memory_window = None  # set via set_memory_window(k)
        self.signal_dim = signal_dim
        self.second_order = True
        self.device = device if device is not None else torch.device("cpu")

        # Pre-compute bilinear coefficients for continuous mode
        # Action rule: val[own_good][opp_good]
        self._action_vals = torch.zeros(2, 2)
        for (og, rg), coop in self.action_rule.items():
            self._action_vals[int(og)][int(rg)] = 1.0 if coop else 0.0

        # Signal rule: val[donor_good][action_coop][recip_good]
        self._signal_vals = torch.zeros(2, 2, 2)
        for (dg, coop, rg), result in self.assessment.items():
            self._signal_vals[int(dg)][int(coop)][int(rg)] = 1.0 if result else 0.0

        self.register_buffer("temp", torch.tensor(temperature))
        self.register_buffer("init_token",
                             torch.full((1, self.signal_dim), 0.5, dtype=torch.float32))
        self.register_buffer("reputation_history",
                             torch.full((1, self.signal_dim), 0.5, dtype=torch.float32))
        self.to(self.device)

    def _soft(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.temp * (x - 0.5))

    def _aggregate(self, rep: torch.Tensor) -> torch.Tensor:
        """Aggregate reputation history: mean or EMA, with optional memory window.
        rep: [seq_len, signal_dim] (unbatched) or [B, seq_len, signal_dim] (batched)."""
        if self.memory_window is not None:
            # Slice to last k entries along seq_len dimension
            rep = rep[..., -self.memory_window:, :] if rep.ndim == 3 else rep[-self.memory_window:]
        if self.decay is None:
            return rep.mean(dim=-2)
        # EMA: recent signals dominate
        dim = -2  # seq_len dimension
        seq_len = rep.shape[dim]
        exponents = torch.arange(seq_len - 1, -1, -1, dtype=torch.float32,
                                  device=rep.device)
        weights = torch.pow(self.decay, exponents)
        weights = weights / weights.sum()
        # Reshape weights for broadcasting
        shape = [1] * rep.ndim
        shape[dim] = seq_len
        weights = weights.view(shape)
        return (rep * weights).sum(dim=dim)

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

    def _cont_action(self, own_rep, opp_rep):
        """Continuous action: a = opp_rep (identity mapping).
        For L3-L8 the discrete rule is 'cooperate iff opp Good';
        the natural continuous generalization is simply a = opp_rep."""
        return opp_rep

    def _cont_signal(self, action, recip_rep, donor_rep):
        """Continuous signal using tanh spin-space formulation (β controls sharpness).

        L3 (Simple Standing):
            F = 1 - 0.25 * (1 - tanh(β(x-0.5))) * (1 + tanh(β(y-0.5)))

        L6 (Stern Judging):
            F = 0.5 * (1 + tanh(β(x-0.5)) * tanh(β(y-0.5)))

        Everywhere differentiable (C∞), rectangular decision boundaries,
        non-zero gradient at y=0.5 (unlike min/max or bilinear).
        """
        b = self.beta
        sx = torch.tanh(b * (action - 0.5))
        sy = torch.tanh(b * (recip_rep - 0.5))

        if self.norm == 'L3':
            return (1.0 - 0.25 * (1 - sx) * (1 + sy)).clamp(0, 1)
        elif self.norm == 'L6':
            return (0.5 * (1 + sx * sy)).clamp(0, 1)
        else:
            # General fallback: multilinear extension
            v = self._signal_vals.to(action.device)
            signal = action.new_zeros(action.shape)
            for di in range(2):
                for ci in range(2):
                    for ri in range(2):
                        w_d = donor_rep if di else (1 - donor_rep)
                        w_c = action if ci else (1 - action)
                        w_r = recip_rep if ri else (1 - recip_rep)
                        signal = signal + w_d * w_c * w_r * v[di, ci, ri]
            return signal

    def take_action(self, recipient_reputation: torch.Tensor) -> torch.Tensor:
        opp_rep = self._aggregate(recipient_reputation)
        own_rep = self._aggregate(self.reputation_history)

        if self.continuous:
            action = self._cont_action(own_rep, opp_rep)
        else:
            recip_good = self._soft(opp_rep)
            own_good = self._soft(own_rep)
            action = torch.tensor(0.0, device=self.device)
            for (og, rg), coop in self.action_rule.items():
                w_own = own_good if og else (1 - own_good)
                w_rec = recip_good if rg else (1 - recip_good)
                action = action + w_own * w_rec * (1.0 if coop else 0.0)

        return action.clamp(0, 1).view(1)

    def send_signal(self, donor_action: torch.Tensor,
                    own_reputation: Optional[torch.Tensor] = None,
                    donor_reputation: Optional[torch.Tensor] = None) -> torch.Tensor:
        a = donor_action.to(self.device).view(-1)

        if own_reputation is None:
            own_reputation = self.get_observation()
        r = self._aggregate(own_reputation)

        if donor_reputation is not None:
            d = self._aggregate(donor_reputation)
        else:
            d = torch.tensor(0.5, device=self.device)

        if self.continuous:
            signal = self._cont_signal(a, r, d)
        else:
            c = self._soft(a)
            g_recip = self._soft(r)
            g_donor = self._soft(d) if donor_reputation is not None else torch.tensor(0.5, device=self.device)
            signal = torch.tensor(0.0, device=self.device)
            for (dg, coop, rg), result_good in self.assessment.items():
                w_d = g_donor if dg else (1 - g_donor)
                w_c = c if coop else (1 - c)
                w_r = g_recip if rg else (1 - g_recip)
                signal = signal + w_d * w_c * w_r * (1.0 if result_good else 0.0)

        return signal.clamp(0, 1).view(self.signal_dim)

    # ── Batched methods (vectorized, no Python loop over batch) ──

    def batched_take_action(self, rep_histories: torch.Tensor,
                             own_rep_histories: torch.Tensor) -> torch.Tensor:
        """[B, seq, 1] → [B, 1]. Fully vectorized."""
        opp_rep = self._aggregate(rep_histories)      # [B, 1]
        own_rep = self._aggregate(own_rep_histories)   # [B, 1]

        if self.continuous:
            return self._cont_action(own_rep, opp_rep).clamp(0, 1)
        else:
            rg = self._soft(opp_rep)
            og = self._soft(own_rep)
            action = torch.zeros_like(rg)
            for (o, r), coop in self.action_rule.items():
                w_own = og if o else (1 - og)
                w_rec = rg if r else (1 - rg)
                action = action + w_own * w_rec * (1.0 if coop else 0.0)
            return action.clamp(0, 1)

    def batched_send_signal(self, actions: torch.Tensor,
                             own_reps: torch.Tensor,
                             donor_reps: torch.Tensor = None) -> torch.Tensor:
        """[B, 1] → [B, 1]. Fully vectorized."""
        recip_rep = self._aggregate(own_reps)                # [B, sd]
        donor_rep = self._aggregate(donor_reps) if donor_reps is not None \
                    else torch.full_like(actions, 0.5)

        if self.continuous:
            return self._cont_signal(actions, recip_rep, donor_rep).clamp(0, 1)
        else:
            c = self._soft(actions)
            g_r = self._soft(recip_rep)
            g_d = self._soft(donor_rep)
            signal = torch.zeros_like(c)
            for (dg, coop, rg), result_good in self.assessment.items():
                w_d = g_d if dg else (1 - g_d)
                w_c = c if coop else (1 - c)
                w_r = g_r if rg else (1 - g_r)
                signal = signal + w_d * w_c * w_r * (1.0 if result_good else 0.0)
            return signal.clamp(0, 1)
