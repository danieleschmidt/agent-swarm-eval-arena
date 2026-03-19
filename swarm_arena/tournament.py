"""
TournamentRunner: runs all agents across all tasks, aggregates scores,
and produces a ranked leaderboard.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .agents import Agent
from .arena import Arena
from .tasks import Task, TaskResult


@dataclass
class TournamentResult:
    """Full results for a tournament run."""

    # task_name -> TaskResult
    task_results: dict[str, TaskResult] = field(default_factory=dict)

    # agent_name -> normalised aggregate score  [0..1 per task, averaged]
    aggregate_scores: dict[str, float] = field(default_factory=dict)

    # Ordered list of (agent_name, score) best→worst
    leaderboard: list[tuple[str, float]] = field(default_factory=list)


class TournamentRunner:
    """
    Run every agent against every task, normalise scores per task, and
    produce an overall leaderboard.

    Parameters
    ----------
    agents : list[Agent]
        Agents to evaluate.
    tasks : list[Task]
        Tasks to run.
    n_episodes : int
        Episodes per (agent-set, task) pair.
    seed : int | None
        RNG seed (each task gets a derived seed for reproducibility).
    """

    def __init__(
        self,
        agents: list[Agent],
        tasks: list[Task],
        n_episodes: int = 5,
        seed: int | None = 42,
    ):
        self.agents = agents
        self.tasks = tasks
        self.n_episodes = n_episodes
        self.seed = seed

    def run(self) -> TournamentResult:
        task_results: dict[str, TaskResult] = {}
        # agent_name -> list of normalised scores (one per task)
        normalised: dict[str, list[float]] = {a.name: [] for a in self.agents}

        for i, task in enumerate(self.tasks):
            arena_seed = (self.seed or 0) + i * 1000
            arena = Arena(task, n_episodes=self.n_episodes, seed=arena_seed)
            result = arena.run(self.agents)
            task_results[task.name] = result

            # Normalise scores: min-max within this task
            scores = result.agent_scores
            lo = min(scores.values())
            hi = max(scores.values())
            span = hi - lo if hi != lo else 1.0

            for name, score in scores.items():
                normalised[name].append((score - lo) / span)

        # Average normalised scores
        agg = {
            name: (sum(ns) / len(ns) if ns else 0.0)
            for name, ns in normalised.items()
        }
        leaderboard = sorted(agg.items(), key=lambda x: x[1], reverse=True)

        return TournamentResult(
            task_results=task_results,
            aggregate_scores=agg,
            leaderboard=leaderboard,
        )
