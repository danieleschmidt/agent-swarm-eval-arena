"""
Agent implementations for the Agent Swarm Eval Arena.

Each agent receives an observation dict and returns an integer action.
The observation contents depend on the task — agents that inspect it
and reason about it will outperform agents that ignore it.
"""

from __future__ import annotations

import random
from abc import ABC, abstractmethod
from typing import Any


class Agent(ABC):
    """Base class for all arena agents."""

    name: str = "agent"

    @abstractmethod
    def act(self, obs: dict[str, Any], rng: random.Random | None = None) -> int:
        """
        Choose an action given the current observation.

        Parameters
        ----------
        obs : dict
            Task-specific observation (varies by task).
        rng : random.Random | None
            Source of randomness for stochastic agents.

        Returns
        -------
        int
            The chosen action index.
        """
        ...

    def reset(self) -> None:
        """Reset any per-episode internal state."""

    def __repr__(self) -> str:  # pragma: no cover
        return f"{self.__class__.__name__}(name={self.name!r})"


# ---------------------------------------------------------------------------
# Concrete agents
# ---------------------------------------------------------------------------

class RandomAgent(Agent):
    """
    Picks a uniformly-random action from the valid range.

    The valid range is inferred from the observation.  If no hint is found
    the agent defaults to picking 0.
    """

    def __init__(self, name: str = "random"):
        self.name = name

    def act(self, obs: dict[str, Any], rng: random.Random | None = None) -> int:
        rng = rng or random.Random()
        # Try to determine action space size from obs
        if "n_choices" in obs:
            return rng.randint(0, obs["n_choices"] - 1)
        if "n_strategies" in obs:
            return rng.randint(0, obs["n_strategies"] - 1)
        if "slot_values" in obs:
            return rng.randint(0, len(obs["slot_values"]) - 1)
        return 0


class GreedyAgent(Agent):
    """
    Maximises *immediate* individual reward with no regard for others.

    - ResourceAllocation: always picks the highest-value slot.
    - Coordination: picks the most common focal point (0).
    - Competitive: plays the strategy that beats 0 (strategy 1 in RPS).
    """

    def __init__(self, name: str = "greedy"):
        self.name = name

    def act(self, obs: dict[str, Any], rng: random.Random | None = None) -> int:
        if "slot_values" in obs:
            # Greedy: pick the slot with the highest value
            values = obs["slot_values"]
            return int(max(range(len(values)), key=lambda i: values[i]))

        if "n_choices" in obs:
            # In coordination, greedy anchors at 0 (simple focal point)
            return 0

        if "n_strategies" in obs:
            # In RPS, play the strategy that beats strategy 0
            return 1 % obs["n_strategies"]

        return 0


class CooperativeAgent(Agent):
    """
    Coordinates with other agents to maximise *collective* utility.

    - ResourceAllocation: distributes picks across slots using a deterministic
      hash of its own name so it avoids colliding with sibling agents when
      possible.
    - Coordination: converges on the same focal point as GreedyAgent (0).
    - Competitive: mirrors the most recent opponent (Tit-for-Tat in RPS).
    """

    def __init__(self, name: str = "cooperative"):
        self.name = name
        self._last_opp_action: int | None = None
        self._my_slot_offset: int = 0  # will be set on first resource obs

    def reset(self) -> None:
        self._last_opp_action = None

    def act(self, obs: dict[str, Any], rng: random.Random | None = None) -> int:
        if "slot_values" in obs:
            n_slots = len(obs["slot_values"])
            # Use agent name hash to consistently pick a different slot
            # than default greedy agents (who always pick the max-value slot).
            # This reduces overlap in multi-agent settings.
            offset = hash(self.name) % n_slots
            values = obs["slot_values"]
            # Sort slots by value descending; take the `offset`-th best
            sorted_slots = sorted(range(n_slots), key=lambda i: values[i], reverse=True)
            # Wrap around: if offset >= n_slots just pick the best remaining
            idx = min(offset, n_slots - 1)
            return sorted_slots[idx]

        if "n_choices" in obs:
            # Focal-point convergence: always choose 0
            return 0

        if "n_strategies" in obs:
            n = obs["n_strategies"]
            if self._last_opp_action is not None:
                # Tit-for-Tat: play what beats opponent's last move
                return (self._last_opp_action + 1) % n
            return 0  # start cooperatively

        return 0
