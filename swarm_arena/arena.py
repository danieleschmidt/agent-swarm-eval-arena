"""
Arena: orchestrates agents running episodes on tasks.
"""

from __future__ import annotations

import random
from typing import Any

from .agents import Agent
from .tasks import Task, TaskResult


class Arena:
    """
    Runs a set of agents against a task for one or more episodes.

    Parameters
    ----------
    task : Task
        The task to evaluate agents on.
    n_episodes : int
        Number of independent episodes to average over.
    seed : int | None
        Global RNG seed for reproducibility.
    """

    def __init__(self, task: Task, n_episodes: int = 1, seed: int | None = None):
        self.task = task
        self.n_episodes = n_episodes
        self.rng = random.Random(seed)

    def run(self, agents: list[Agent]) -> TaskResult:
        """
        Run *agents* on the task for `n_episodes` episodes and return
        averaged scores.
        """
        if not agents:
            raise ValueError("Need at least one agent.")

        # Reset agent state before each run
        for a in agents:
            a.reset()

        cumulative: dict[str, float] = {a.name: 0.0 for a in agents}
        last_meta: dict[str, Any] = {}

        for _ in range(self.n_episodes):
            result = self.task.run(agents, rng=self.rng)
            for name, score in result.agent_scores.items():
                cumulative[name] += score
            last_meta = result.metadata

        avg_scores = {name: total / self.n_episodes for name, total in cumulative.items()}

        return TaskResult(
            task_name=self.task.name,
            agent_scores=avg_scores,
            metadata={**last_meta, "n_episodes": self.n_episodes},
        )
