"""
Batched version of ContinuousRandomMatchingDonationGame.

Runs multiple episodes IN PARALLEL as a single batched computation.
All agent forward passes become batched: [batch_size, ...] instead of [...].
This enables GPU acceleration and significantly faster training.

Usage:
    env = BatchedDonationGame(agents, b=10.0, continuation_prob=0.98, batch_size=128)
    cumulative_rewards, matching_sequence = env.play_batch()
    # cumulative_rewards[i] is a tensor of shape [batch_size]
"""
import torch
import torch.nn as nn
from typing import List, Tuple, Optional
from agent import Agent


class BatchedDonationGame:
    """
    Vectorized environment: runs batch_size episodes simultaneously.

    Key difference from ContinuousRandomMatchingDonationGame:
    - All episodes in the batch share the same matching sequence per timestep
      (simplification for vectorization; the randomness is across batches, not within)
    - Agent forward passes are batched: action_net/signal_net process [batch, dim]
    - Reputation histories are [batch, seq_len, signal_dim] per agent
    """

    def __init__(
        self,
        agents: List,
        b: float,
        c: float = 1.0,           # donor cost of full cooperation (default 1 for backward compat)
        batch_size: int = 128,
        continuation_prob: float = None,
        horizon: int = None,
        initial_reputation_dist: str = "uniform",
        initial_reputation_params: dict = {"low": 0.0, "high": 1.0},
        per_agent_init: List = None,
        allow_direct_reciprocity: bool = True,
        balanced_matching: bool = True,
        min_rounds: int = 1,
        gradient_depth: int = None,
        device: torch.device = None,
    ):
        self.agents = agents
        self.N = len(agents)
        self.b = float(b)
        self.c = float(c)
        self.batch_size = batch_size
        self.continuation_prob = continuation_prob
        self.horizon = horizon
        self.initial_reputation_dist = initial_reputation_dist
        self.initial_reputation_params = initial_reputation_params
        self.per_agent_init = per_agent_init  # list of dicts, one per agent (optional)
        self.allow_direct_reciprocity = allow_direct_reciprocity
        self.balanced_matching = balanced_matching
        self.min_rounds = min_rounds
        self.gradient_depth = gradient_depth
        self.device = device if device is not None else torch.device("cpu")

        if continuation_prob is None and horizon is None:
            raise ValueError("Either continuation_prob or horizon must be provided.")

        # Pre-compute one full round-robin for balanced matching
        if self.balanced_matching:
            self._all_pairs = [(i, j) for i in range(self.N) for j in range(self.N) if i != j]
            self._round_size = len(self._all_pairs)  # N*(N-1)

    def _sample_initial_rep(self, signal_dim: int) -> torch.Tensor:
        """Sample initial reputation: [batch_size, signal_dim]."""
        dist = self.initial_reputation_dist.lower()
        B = self.batch_size
        if dist == "constant":
            val = self.initial_reputation_params.get("value", 0.5)
            return torch.full((B, signal_dim), val, device=self.device)
        elif dist == "uniform":
            low = self.initial_reputation_params.get("low", 0.0)
            high = self.initial_reputation_params.get("high", 1.0)
            return (high - low) * torch.rand(B, signal_dim, device=self.device) + low
        elif dist == "normal":
            mean = self.initial_reputation_params.get("mean", 0.5)
            std = self.initial_reputation_params.get("std", 0.1)
            return (torch.randn(B, signal_dim, device=self.device) * std + mean).clamp(0, 1)
        elif dist == "empirical":
            # Sample from a saved empirical distribution (e.g. warm-up steady state)
            values = self.initial_reputation_params["values"]  # 1D tensor or list
            if not isinstance(values, torch.Tensor):
                values = torch.tensor(values, dtype=torch.float32)
            idx = torch.randint(0, len(values), (B,))
            return values[idx].view(B, 1).expand(B, signal_dim).to(self.device)
        else:
            raise ValueError(f"Unknown dist: {dist}")

    def _generate_round_robin(self) -> List[Tuple[int, int]]:
        """One shuffled round-robin: all N*(N-1) ordered pairs in random order."""
        perm = torch.randperm(self._round_size)
        return [self._all_pairs[perm[i].item()] for i in range(self._round_size)]

    def _generate_matching_sequence(self) -> List[Tuple[int, int]]:
        """
        Generate matching sequence.

        If balanced_matching=True:
            Complete round-robins, continuation die rolled after each round.
            Guarantees every agent is donor and recipient the same number of times.
            For N=3: each round = 6 interactions, every agent is donor 2x, recipient 2x.

        Otherwise:
            Original random pair sampling (legacy behavior).
        """
        if self.balanced_matching:
            seq = []

            if not self.allow_direct_reciprocity:
                # Indirect reciprocity: each ordered pair appears at most once.
                # Exactly ONE complete round-robin. Fixed length = N*(N-1).
                # No continuation probability needed — episode length is deterministic.
                # Every agent: exactly (N-1) times donor, (N-1) times recipient.
                seq.extend(self._generate_round_robin())
                return seq

            # Direct reciprocity: multiple round-robins, geometric over rounds.
            if self.continuation_prob is not None:
                # δ_round: probability of continuing after a complete round
                # E[rounds] = E[total_steps] / round_size = (1/(1-δ)) / N(N-1)
                # δ_round = 1 - round_size * (1 - δ)
                delta_round = 1.0 - self._round_size * (1.0 - self.continuation_prob)
                delta_round = max(delta_round, 0.0)
                # Play at least min_rounds, then roll dice each round
                for _ in range(self.min_rounds):
                    seq.extend(self._generate_round_robin())
                while torch.rand(1).item() < delta_round:
                    seq.extend(self._generate_round_robin())
            else:
                n_rounds = max(1, self.horizon // self._round_size)
                for _ in range(n_rounds):
                    seq.extend(self._generate_round_robin())
            return seq

        # Legacy: random pair sampling
        seq = []
        used_pairs = set()

        def sample_pair():
            if self.allow_direct_reciprocity:
                idx = torch.randperm(self.N)
                return (idx[0].item(), idx[1].item())
            else:
                if len(used_pairs) >= self.N * (self.N - 1):
                    return None
                for _ in range(100):
                    idx = torch.randperm(self.N)
                    p = (idx[0].item(), idx[1].item())
                    if p not in used_pairs:
                        used_pairs.add(p)
                        return p
                return None

        if self.continuation_prob is not None:
            while True:
                p = sample_pair()
                if p is None:
                    break
                seq.append(p)
                if torch.rand(1).item() > self.continuation_prob:
                    break
        else:
            for _ in range(self.horizon):
                p = sample_pair()
                if p is None:
                    break
                seq.append(p)
        return seq

    def _batched_take_action(self, agent, rep_histories: torch.Tensor,
                              own_rep_histories: torch.Tensor = None) -> torch.Tensor:
        """
        Batched action: process batch_size reputation histories at once.
        rep_histories: [batch_size, seq_len, signal_dim]  (recipient's rep)
        own_rep_histories: [batch_size, seq_len, signal_dim]  (donor's own rep, for fallback agents)
        Returns: [batch_size, 1]
        """
        # Apply memory window if agent has one (slice to last k entries)
        mw = getattr(agent, 'memory_window', None)
        if mw is not None:
            rep_histories = rep_histories[:, -mw:, :]
            if own_rep_histories is not None:
                own_rep_histories = own_rep_histories[:, -mw:, :]

        if getattr(agent, '_hardcode_action', False):
            decay = getattr(agent, '_hardcode_action_decay', None)
            if decay is None:
                return rep_histories.mean(dim=1)  # [B, signal_dim]
            # EMA: weight recent signals more, init_token fades
            seq_len = rep_histories.shape[1]
            exponents = torch.arange(seq_len - 1, -1, -1, dtype=torch.float32,
                                      device=rep_histories.device)
            weights = torch.pow(decay, exponents)
            weights = weights / weights.sum()
            # [B, seq_len, sd] × [seq_len] → [B, sd]
            return (rep_histories.squeeze(-1) * weights.unsqueeze(0)).sum(dim=1, keepdim=True)
        elif getattr(agent, 'is_demo', False):
            obs = rep_histories[:, -1, :]
            return agent.action_net(obs)
        elif hasattr(agent, 'action_input_proj'):
            # Transformer mode — BATCHED (action encoder + PE, last position)
            projected = agent.action_input_proj(rep_histories)  # [B, seq_len, emb_dim]
            seq_len = projected.shape[1]
            projected = projected + agent._pe[:seq_len].unsqueeze(0)
            encoded = agent.action_encoder(projected)           # [B, seq_len, emb_dim]
            last_out = encoded[:, -1, :]                        # [B, emb_dim]
            return agent.action_net(last_out)                   # [B, 1]
        elif hasattr(agent, 'batched_take_action'):
            # LeadingEightAgent or similar with batched methods
            return agent.batched_take_action(rep_histories, own_rep_histories)
        else:
            # Fallback: loop over batch elements
            B = rep_histories.shape[0]
            actions = []
            for i in range(B):
                if own_rep_histories is not None:
                    agent.reputation_history = own_rep_histories[i]
                actions.append(agent.take_action(rep_histories[i]))
            return torch.stack(actions, dim=0)

    def _batched_send_signal(self, agent, actions: torch.Tensor,
                              own_reps: torch.Tensor = None,
                              donor_reps: torch.Tensor = None) -> torch.Tensor:
        """
        Batched signal emission.
        actions: [batch_size, 1]
        own_reps: [batch_size, seq_len, signal_dim] or None
        donor_reps: [batch_size, seq_len, signal_dim] or None
        Returns: [batch_size, signal_dim]
        """
        if getattr(agent, '_hardcode_signal', False):
            norm = getattr(agent, '_hardcode_signal_norm', None)
            if norm is None:
                return actions.view(-1, agent.signal_dim)
            # Tanh spin-space signal (batched)
            b = agent._hardcode_signal_beta
            r = own_reps.mean(dim=1) if own_reps is not None else torch.full_like(actions, 0.5)
            sx = torch.tanh(b * (actions - 0.5))
            sy = torch.tanh(b * (r - 0.5))
            if norm == 'L3':
                sig = 1.0 - 0.25 * (1 - sx) * (1 + sy)
            elif norm == 'L6':
                sig = 0.5 * (1 + sx * sy)
            else:
                sig = actions
            return sig.clamp(0, 1).view(-1, agent.signal_dim)
        if getattr(agent, 'is_demo', False):
            if agent.second_order:
                own_last = own_reps[:, -1, :]
                inp = torch.cat([actions, own_last], dim=-1)
            else:
                inp = actions
            return agent.signal_net(inp)
        elif hasattr(agent, 'signal_input_proj'):
            # Transformer mode — BATCHED (signal encoder + PE last pos + action proj)
            if own_reps is None:
                own_reps = torch.full_like(actions.unsqueeze(-1), 0.5)
            action_emb = agent.signal_action_proj(actions)          # [B, emb]
            projected = agent.signal_input_proj(own_reps)           # [B, seq_len, emb]
            seq_len = projected.shape[1]
            projected = projected + agent._pe[:seq_len].unsqueeze(0)
            encoded = agent.signal_encoder(projected)               # [B, seq_len, emb]
            own_emb = encoded[:, -1, :]                             # [B, emb] — last pos
            inp = torch.cat([action_emb, own_emb], dim=-1)         # [B, 2*emb]
            return agent.signal_net(inp)                            # [B, signal_dim]
        elif hasattr(agent, 'batched_send_signal'):
            # LeadingEightAgent or similar with batched methods
            return agent.batched_send_signal(actions, own_reps, donor_reps)
        else:
            # Fallback: loop over batch elements
            B = actions.shape[0]
            signals = []
            for i in range(B):
                own_r = own_reps[i] if own_reps is not None else None
                donor_r = donor_reps[i] if donor_reps is not None else None
                signals.append(agent.send_signal(actions[i].squeeze(), own_reputation=own_r,
                                                  donor_reputation=donor_r))
            return torch.stack(signals, dim=0)

    def play_batch(self) -> Tuple[List[torch.Tensor], List[Tuple[int, int]]]:
        """
        Run batch_size episodes in parallel.

        Returns:
            cumulative_rewards: list of N tensors, each [batch_size]
            matching_sequence: the shared matching sequence
        """
        B = self.batch_size

        # Initialize per-agent reputation histories: list of [batch, 1, signal_dim]
        rep_histories = []
        for i, agent in enumerate(self.agents):
            sd = agent.signal_dim
            if self.per_agent_init is not None and i < len(self.per_agent_init):
                cfg = self.per_agent_init[i]
                old_dist, old_params = self.initial_reputation_dist, self.initial_reputation_params
                self.initial_reputation_dist = cfg.get("dist", old_dist)
                self.initial_reputation_params = cfg.get("params", old_params)
                init = self._sample_initial_rep(sd).unsqueeze(1)
                self.initial_reputation_dist, self.initial_reputation_params = old_dist, old_params
            else:
                init = self._sample_initial_rep(sd).unsqueeze(1)  # [B, 1, sd]
            rep_histories.append(init)

        # Cumulative rewards: [batch_size] per agent
        cum_rewards = [torch.zeros(B, device=self.device) for _ in range(self.N)]

        # Generate matching sequence
        matching_seq = self._generate_matching_sequence()

        for t, (donor_idx, recipient_idx) in enumerate(matching_seq):
            # Gradient depth truncation
            if self.gradient_depth is not None and t > 0 and t % self.gradient_depth == 0:
                rep_histories = [rh.detach() for rh in rep_histories]

            donor = self.agents[donor_idx]
            recipient = self.agents[recipient_idx]

            # 1. Donor observes recipient's reputation history
            recip_rep = rep_histories[recipient_idx]  # [B, seq_len, sd]

            # 2. Donor takes action (pass donor's own rep for fallback agents)
            actions = self._batched_take_action(
                donor, recip_rep,
                own_rep_histories=rep_histories[donor_idx])  # [B, 1]

            # 3. Rewards
            reward_donor = -self.c * actions.squeeze(-1)  # [B]
            reward_recipient = self.b * actions.squeeze(-1)  # [B]
            cum_rewards[donor_idx] = cum_rewards[donor_idx] + reward_donor
            cum_rewards[recipient_idx] = cum_rewards[recipient_idx] + reward_recipient

            # 4. Recipient sends signal (always pass own_reps and donor_reps)
            signals = self._batched_send_signal(
                recipient, actions,
                own_reps=rep_histories[recipient_idx],
                donor_reps=rep_histories[donor_idx])  # [B, sd]

            # 5. Append signal to donor's reputation
            rep_histories[donor_idx] = torch.cat(
                [rep_histories[donor_idx], signals.unsqueeze(1)], dim=1)

        return cum_rewards, matching_seq, rep_histories


def compute_per_interaction_payoff(cum_rewards, matching_seq, agent_idx):
    """
    Compute per-own-interaction payoff for a given agent.

    per_interaction = R^i / n_i
    where n_i = number of times agent i appeared (as donor or recipient).

    Full cooperation benchmark: (-c + b) / 2 = (b - 1) / 2

    Args:
        cum_rewards: list of [batch_size] tensors
        matching_seq: list of (donor, recipient) tuples
        agent_idx: which agent

    Returns:
        per_interaction: [batch_size] tensor (with grad)
        n_plays: int
    """
    n_plays = sum(1 for d, r in matching_seq if d == agent_idx or r == agent_idx)
    if n_plays == 0:
        return cum_rewards[agent_idx], 0
    return cum_rewards[agent_idx] / n_plays, n_plays
