import torch
from typing import List, Callable, Tuple
from agent import Agent


class ContinuousRandomMatchingDonationGame:
    """
    Continuous Random Matching Donation Game with Post-Gossip Mechanism
    as defined in the paper (Definition 2).
    
    This environment represents a fully differentiable formulation of the 
    random-matching donation game where:
    - Agents use continuous actions in [0, 1]
    - Agents emit continuous signals in [0, 1]^d
    - A differentiable computation graph is maintained across the episode 
      horizon to compute analytic, unbiased gradients of cumulative rewards.
    """

    def __init__(
        self,
        agents: List[Agent],
        b: float,
        continuation_prob: float = None,
        horizon: int = None,
        initial_reputation_dist: str = "constant",
        initial_reputation_params: dict = {"value": 0.5},
        allow_direct_reciprocity: bool = True,
        device: torch.device = None
    ):
        """
        Args:
            agents: List of agent instances (population N).
            b: The constant benefit multiplier.
            continuation_prob: Probability of the game continuing to the next round.
                               If provided, game length follows a geometric distribution.
            horizon: Fixed game horizon (T timesteps). Used if continuation_prob is None.
            initial_reputation_dist: String describing the distribution ("constant", "uniform", "normal").
            initial_reputation_params: Dictionary of parameters for the chosen distribution.
                - "constant": {"value": float}
                - "uniform": {"low": float, "high": float}
                - "normal": {"mean": float, "std": float}
            allow_direct_reciprocity: If True (default), same pair can interact multiple times.
                If False, each ordered pair (i,j) can appear at most once in the matching sequence.
                This forces pure indirect reciprocity — reputation is the ONLY channel of influence.
            device: Torch device for computations.
        """
        self.agents = agents
        self.N = len(agents)
        if self.N < 2:
            raise ValueError("The population must contain at least 2 agents.")

        if continuation_prob is None and horizon is None:
            raise ValueError("Either continuation_prob or horizon must be provided.")

        self.continuation_prob = continuation_prob
        self.horizon = horizon
        self.allow_direct_reciprocity = allow_direct_reciprocity
        self.b = float(b)
        self.device = device if device is not None else torch.device("cpu")

        # Config for initializing reputation
        self.initial_reputation_dist = initial_reputation_dist
        self.initial_reputation_params = initial_reputation_params

    def _sample_initial_reputation(self, signal_dim: int) -> torch.Tensor:
        """Helper to sample a single initial reputation tensor based on the specified distribution."""
        dist = self.initial_reputation_dist.lower()
        if dist == "constant":
            val = self.initial_reputation_params.get("value", 0.5)
            return torch.full((signal_dim,), val, dtype=torch.float32, device=self.device)
        elif dist == "uniform":
            low = self.initial_reputation_params.get("low", 0.0)
            high = self.initial_reputation_params.get("high", 1.0)
            # Uniform in [low, high)
            return (high - low) * torch.rand(signal_dim, dtype=torch.float32, device=self.device) + low
        elif dist == "normal":
            mean = self.initial_reputation_params.get("mean", 0.5)
            std = self.initial_reputation_params.get("std", 0.1)
            # Normal sampled and clamped to [0, 1] as signals are generally in this bound
            sampled = torch.randn(signal_dim, dtype=torch.float32, device=self.device) * std + mean
            return torch.clamp(sampled, 0.0, 1.0)
        else:
            raise ValueError(f"Unknown initial_reputation_dist: {dist}")

    def _sample_pair(self, used_pairs: set):
        """Sample a (donor, recipient) pair respecting direct reciprocity setting."""
        if self.allow_direct_reciprocity:
            indices = torch.randperm(self.N, device=self.device)
            return (indices[0].item(), indices[1].item())
        else:
            # Without direct reciprocity: no ordered pair can repeat
            max_pairs = self.N * (self.N - 1)
            if len(used_pairs) >= max_pairs:
                return None  # All pairs exhausted
            for _ in range(100):  # Try up to 100 times to find unused pair
                indices = torch.randperm(self.N, device=self.device)
                pair = (indices[0].item(), indices[1].item())
                if pair not in used_pairs:
                    used_pairs.add(pair)
                    return pair
            return None  # Give up (shouldn't happen unless nearly full)

    def reset(self):
        """Resets the game state and samples new initial reputation for all agents."""
        for agent in self.agents:
            init_rep = self._sample_initial_reputation(agent.signal_dim)
            agent.reset_reputation(init_rep)

    def step(self, donor_idx: int, recipient_idx: int) -> Tuple[torch.Tensor, torch.Tensor, dict]:
        """
        Executes a single timestep of the game interaction.
        """
        donor = self.agents[donor_idx]
        recipient = self.agents[recipient_idx]

        # 1. Donor observes recipient
        # Under vanilla (first-order) observation space, the donor views 
        # the recipient's historical reputation signals.
        reputation_recipient = recipient.get_observation()

        # 2. Donor chooses continuous action a_{i}^{t} in [0, 1]
        action = donor.take_action(reputation_recipient)

        # 3. Calculate immediate rewards
        # r_{i}^{t} = -c * a_{i}^{t}  (cost of donating, typically c=1)
        # r_{j}^{t} = b * a_{i}^{t}   (benefit of receiving)
        # Here we assume c=1, so reward_donor = -action
        reward_donor = -action
        reward_recipient = self.b * action

        # 4. Recipient emits a continuous signal \sigma
        # First-order: depends only on donor's action
        # Second-order: also conditions on recipient's own standing
        # Third-order: also conditions on donor's standing (for leading-eight opponents)
        if recipient.second_order:
            obs_self = recipient.get_observation()
            signal = recipient.send_signal(action, own_reputation=obs_self,
                                           donor_reputation=donor.get_observation())
        else:
            signal = recipient.send_signal(action)

        # 5. The signal becomes permanently attached to the donor for the remainder of the game
        donor.append_reputation(signal)

        info = {
            "donor_idx": donor_idx,
            "recipient_idx": recipient_idx,
            "recipient_reputation": reputation_recipient,
            "donor_action": action,
            "recipient_signal": signal,
        }

        return reward_donor, reward_recipient, info

    def play_episode(self, matching_sequence: List[Tuple[int, int]] = None,
                     gradient_depth: int = None) -> Tuple[List[torch.Tensor], List[Tuple[int, int]], List[dict]]:
        """
        Simulates the continuous game.

        Args:
            matching_sequence: Optional list of (donor_idx, recipient_idx) tuples for each timestep.
                               If None, a uniform random matching sequence is sampled.
            gradient_depth: If set, detach all agents' reputation histories every `gradient_depth`
                           timesteps. This truncates the gradient highway at depth k, analogous to
                           bounded-rationality agents with limited Theory of Mind.
                           None = full gradient (unlimited depth). 1 = myopic (no reputation influence).

        Returns:
            cumulative_rewards: The cumulative rewards for each agent, preserving the computation graph.
            matching_sequence: The actual matching sequence used during the episode.
            history: A list of dictionaries containing detailed records of each timestep.
        """
        self.reset()
        
        # Initialize cumulative rewards dynamically to maintain the gradient graph
        cumulative_rewards = [torch.tensor(0.0, device=self.device) for _ in range(self.N)]

        # Determine the matching sequence for the whole episode before execution
        if matching_sequence is None:
            self.current_matching_sequence = []
            used_pairs = set()  # Track used pairs for no-direct-reciprocity mode
            max_pairs = self.N * (self.N - 1)  # Total possible ordered pairs

            if self.continuation_prob is not None:
                t = 0
                while True:
                    pair = self._sample_pair(used_pairs)
                    if pair is None:
                        break  # No more valid pairs (only possible without direct reciprocity)
                    self.current_matching_sequence.append(pair)
                    t += 1
                    if torch.rand(1).item() > self.continuation_prob:
                        break
                actual_horizon = t
            else:
                actual_horizon = self.horizon
                for _ in range(actual_horizon):
                    pair = self._sample_pair(used_pairs)
                    if pair is None:
                        break  # No more valid pairs
                    self.current_matching_sequence.append(pair)
                actual_horizon = len(self.current_matching_sequence)
        else:
            self.current_matching_sequence = matching_sequence
            actual_horizon = len(matching_sequence)

        history = []
        for t in range(actual_horizon):
            # Gradient depth truncation: detach reputation histories periodically
            # This limits how far back the gradient can propagate through the
            # reputation chain, analogous to bounded Theory of Mind.
            if gradient_depth is not None and t > 0 and t % gradient_depth == 0:
                for agent in self.agents:
                    agent.reputation_history = agent.reputation_history.detach()

            donor_idx, recipient_idx = self.current_matching_sequence[t]

            r_donor, r_recipient, info = self.step(donor_idx, recipient_idx)
            history.append(info)

            # Accumulate rewards
            cumulative_rewards[donor_idx] = cumulative_rewards[donor_idx] + r_donor
            cumulative_rewards[recipient_idx] = cumulative_rewards[recipient_idx] + r_recipient

        return cumulative_rewards, self.current_matching_sequence, history
