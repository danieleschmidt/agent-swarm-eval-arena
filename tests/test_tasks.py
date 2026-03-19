"""Tests for task implementations."""

import random
import pytest

from swarm_arena.agents import GreedyAgent, RandomAgent, CooperativeAgent
from swarm_arena.tasks import (
    ResourceAllocationTask,
    CoordinationTask,
    CompetitiveTask,
    TaskResult,
)

RNG = random.Random(42)


def make_agents():
    return [RandomAgent("rand"), GreedyAgent("greedy"), CooperativeAgent("coop")]


# ---------------------------------------------------------------------------
# TaskResult
# ---------------------------------------------------------------------------

class TestTaskResult:
    def test_ranked_order(self):
        r = TaskResult("t", {"a": 10.0, "b": 5.0, "c": 20.0})
        names = [n for n, _ in r.ranked]
        assert names == ["c", "a", "b"]

    def test_ranked_ties(self):
        r = TaskResult("t", {"x": 1.0, "y": 1.0})
        assert len(r.ranked) == 2


# ---------------------------------------------------------------------------
# ResourceAllocationTask
# ---------------------------------------------------------------------------

class TestResourceAllocationTask:
    def test_returns_all_agents(self):
        task = ResourceAllocationTask(n_slots=4, total_resources=100, rounds=5)
        agents = make_agents()
        result = task.run(agents, RNG)
        assert set(result.agent_scores.keys()) == {a.name for a in agents}

    def test_total_score_at_most_total_resources_per_round(self):
        """Distributed resources ≤ total_resources * rounds (unselected slots are lost)."""
        task = ResourceAllocationTask(n_slots=5, total_resources=100, rounds=10)
        agents = make_agents()
        result = task.run(agents, RNG)
        total = sum(result.agent_scores.values())
        # Cannot exceed total budget (100 * 10 = 1000), must be > 0
        assert 0 < total <= 1000.0 + 1e-6

    def test_scores_non_negative(self):
        task = ResourceAllocationTask()
        for score in task.run(make_agents(), RNG).agent_scores.values():
            assert score >= 0.0

    def test_metadata_contains_rounds(self):
        task = ResourceAllocationTask(rounds=7)
        result = task.run(make_agents(), RNG)
        assert result.metadata["rounds"] == 7


# ---------------------------------------------------------------------------
# CoordinationTask
# ---------------------------------------------------------------------------

class TestCoordinationTask:
    def test_returns_all_agents(self):
        task = CoordinationTask(n_choices=3, rounds=5)
        agents = make_agents()
        result = task.run(agents, RNG)
        assert set(result.agent_scores.keys()) == {a.name for a in agents}

    def test_scores_non_negative(self):
        task = CoordinationTask()
        for score in task.run(make_agents(), RNG).agent_scores.values():
            assert score >= 0.0

    def test_coordination_rate_in_01(self):
        task = CoordinationTask(n_choices=3, rounds=20)
        result = task.run(make_agents(), RNG)
        rate = result.metadata["coordination_rate"]
        assert 0.0 <= rate <= 1.0

    def test_all_same_agent_achieves_perfect_coordination(self):
        """Three cooperative agents that always pick 0 should coordinate every round."""
        agents = [CooperativeAgent(f"c{i}") for i in range(3)]
        task = CoordinationTask(n_choices=5, rounds=20)
        result = task.run(agents, RNG)
        assert result.metadata["coordination_rate"] == 1.0


# ---------------------------------------------------------------------------
# CompetitiveTask
# ---------------------------------------------------------------------------

class TestCompetitiveTask:
    def test_returns_all_agents(self):
        task = CompetitiveTask(n_strategies=3, rounds=10)
        agents = make_agents()
        result = task.run(agents, RNG)
        assert set(result.agent_scores.keys()) == {a.name for a in agents}

    def test_scores_non_negative(self):
        task = CompetitiveTask()
        for score in task.run(make_agents(), RNG).agent_scores.values():
            assert score >= 0.0

    def test_beats_logic(self):
        task = CompetitiveTask(n_strategies=3)
        assert task._beats(1, 0)   # 1 beats 0
        assert task._beats(2, 1)   # 2 beats 1
        assert task._beats(0, 2)   # 0 beats 2 (wrap)
        assert not task._beats(0, 0)  # tie
        assert not task._beats(0, 1)  # 0 loses to 1

    def test_single_agent_scores_zero(self):
        """With only one agent there are no pairs, so zero wins."""
        task = CompetitiveTask(rounds=5)
        result = task.run([RandomAgent("solo")], RNG)
        assert result.agent_scores["solo"] == 0.0
