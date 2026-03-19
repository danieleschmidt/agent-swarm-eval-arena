"""
Task definitions for the Agent Swarm Eval Arena.

Each task defines an environment where agents compete or cooperate.
Tasks produce per-agent scores and a summary of the episode.
"""

from __future__ import annotations

import random
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


# ---------------------------------------------------------------------------
# Base
# ---------------------------------------------------------------------------

@dataclass
class TaskResult:
    """Outcome of one task episode."""
    task_name: str
    agent_scores: dict[str, float]   # agent_name -> score
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def ranked(self) -> list[tuple[str, float]]:
        """Return agents ranked best→worst."""
        return sorted(self.agent_scores.items(), key=lambda x: x[1], reverse=True)


class Task(ABC):
    """Abstract base for all evaluation tasks."""

    name: str = "task"

    @abstractmethod
    def run(self, agents: list[Any], rng: random.Random | None = None) -> TaskResult:
        """Run one episode with *agents* and return scores."""
        ...


# ---------------------------------------------------------------------------
# Task 1 – ResourceAllocationTask
# ---------------------------------------------------------------------------

class ResourceAllocationTask(Task):
    """
    Agents must allocate a shared, limited pool of resources across N slots.
    Each slot has a hidden value; maximising *collective* utility requires
    avoiding overlap (two agents picking the same slot split the reward).

    Rules
    -----
    - `total_resources` units are spread across `n_slots` slots with random values.
    - Each agent picks a slot each round; allocations across agents are resolved
      simultaneously.
    - If multiple agents pick the same slot they split its value equally.
    - Score = sum of resources received over `rounds` rounds.
    - Cooperation wins: non-overlapping picks maximise collective utility.
    """

    name = "resource_allocation"

    def __init__(self, n_slots: int = 5, total_resources: int = 100, rounds: int = 10):
        self.n_slots = n_slots
        self.total_resources = total_resources
        self.rounds = rounds

    def run(self, agents: list[Any], rng: random.Random | None = None) -> TaskResult:
        rng = rng or random.Random()
        scores = {a.name: 0.0 for a in agents}

        for _ in range(self.rounds):
            # Random slot values that sum to total_resources
            raw = [rng.random() for _ in range(self.n_slots)]
            total = sum(raw)
            slot_values = [v / total * self.total_resources for v in raw]

            # Each agent chooses a slot
            choices: dict[int, list[str]] = {}
            for agent in agents:
                obs = {"slot_values": list(slot_values), "n_agents": len(agents)}
                slot = agent.act(obs, rng)
                slot = max(0, min(self.n_slots - 1, int(slot)))
                choices.setdefault(slot, []).append(agent.name)

            # Resolve: split value among agents who chose the same slot
            for slot, names in choices.items():
                share = slot_values[slot] / len(names)
                for name in names:
                    scores[name] += share

        return TaskResult(
            task_name=self.name,
            agent_scores=scores,
            metadata={"n_slots": self.n_slots, "rounds": self.rounds},
        )


# ---------------------------------------------------------------------------
# Task 2 – CoordinationTask
# ---------------------------------------------------------------------------

class CoordinationTask(Task):
    """
    Agents must converge on the *same* action without direct communication.
    Each round every agent independently picks a number in [0, n_choices).
    If all agents agree, everyone scores +1 (cooperative reward).
    Otherwise nobody scores.

    This tests whether an agent type can adopt a focal-point strategy.
    """

    name = "coordination"

    def __init__(self, n_choices: int = 5, rounds: int = 20):
        self.n_choices = n_choices
        self.rounds = rounds

    def run(self, agents: list[Any], rng: random.Random | None = None) -> TaskResult:
        rng = rng or random.Random()
        scores = {a.name: 0.0 for a in agents}
        coordination_rounds = 0

        for _ in range(self.rounds):
            obs = {"n_choices": self.n_choices, "n_agents": len(agents)}
            picks = [max(0, min(self.n_choices - 1, int(a.act(obs, rng)))) for a in agents]

            if len(set(picks)) == 1:  # all agreed
                coordination_rounds += 1
                for a in agents:
                    scores[a.name] += 1.0

        return TaskResult(
            task_name=self.name,
            agent_scores=scores,
            metadata={
                "n_choices": self.n_choices,
                "rounds": self.rounds,
                "coordination_rate": coordination_rounds / self.rounds,
            },
        )


# ---------------------------------------------------------------------------
# Task 3 – CompetitiveTask
# ---------------------------------------------------------------------------

class CompetitiveTask(Task):
    """
    Zero-sum repeated game (Rock-Paper-Scissors variant with N choices).

    Each round agents pick a strategy; the winner beats the strategy directly
    below it (mod n_strategies). Pairwise matchups are held between all pairs;
    the total pot is redistributed to winners only.

    Score = total pairwise wins over `rounds` rounds.
    """

    name = "competitive"

    def __init__(self, n_strategies: int = 3, rounds: int = 30):
        self.n_strategies = n_strategies
        self.rounds = rounds

    def _beats(self, a: int, b: int) -> bool:
        """True if strategy *a* beats strategy *b* (RPS-like cycle)."""
        return (a - b) % self.n_strategies == 1

    def run(self, agents: list[Any], rng: random.Random | None = None) -> TaskResult:
        rng = rng or random.Random()
        scores = {a.name: 0.0 for a in agents}

        for _ in range(self.rounds):
            obs = {"n_strategies": self.n_strategies}
            picks: dict[str, int] = {}
            for a in agents:
                picks[a.name] = max(
                    0, min(self.n_strategies - 1, int(a.act(obs, rng)))
                )

            # All-pairs matchup
            names = list(picks.keys())
            for i in range(len(names)):
                for j in range(i + 1, len(names)):
                    ai, aj = names[i], names[j]
                    pi, pj = picks[ai], picks[aj]
                    if self._beats(pi, pj):
                        scores[ai] += 1.0
                    elif self._beats(pj, pi):
                        scores[aj] += 1.0
                    # tie: no points

        return TaskResult(
            task_name=self.name,
            agent_scores=scores,
            metadata={"n_strategies": self.n_strategies, "rounds": self.rounds},
        )
