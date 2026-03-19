"""Tests for Arena and TournamentRunner."""

import pytest

from swarm_arena.agents import RandomAgent, GreedyAgent, CooperativeAgent
from swarm_arena.arena import Arena
from swarm_arena.report import EvalReport
from swarm_arena.tasks import ResourceAllocationTask, CoordinationTask, CompetitiveTask
from swarm_arena.tournament import TournamentRunner


def make_agents():
    return [RandomAgent("rand"), GreedyAgent("greedy"), CooperativeAgent("coop")]


def make_tasks():
    return [ResourceAllocationTask(), CoordinationTask(), CompetitiveTask()]


# ---------------------------------------------------------------------------
# Arena
# ---------------------------------------------------------------------------

class TestArena:
    def test_run_single_episode(self):
        arena = Arena(ResourceAllocationTask(), n_episodes=1, seed=0)
        result = arena.run(make_agents())
        assert result.task_name == "resource_allocation"
        assert len(result.agent_scores) == 3

    def test_run_multiple_episodes_averages(self):
        arena_1 = Arena(CoordinationTask(), n_episodes=1, seed=0)
        arena_10 = Arena(CoordinationTask(), n_episodes=10, seed=0)
        agents = make_agents()
        r1 = arena_1.run(agents)
        r10 = arena_10.run(agents)
        # Multiple episodes should average — scores needn't be identical
        assert set(r1.agent_scores.keys()) == set(r10.agent_scores.keys())

    def test_run_no_agents_raises(self):
        arena = Arena(ResourceAllocationTask())
        with pytest.raises(ValueError):
            arena.run([])

    def test_reproducible_with_same_seed(self):
        agents_a = [RandomAgent("r")]
        agents_b = [RandomAgent("r")]
        r_a = Arena(CompetitiveTask(), n_episodes=5, seed=77).run(agents_a)
        r_b = Arena(CompetitiveTask(), n_episodes=5, seed=77).run(agents_b)
        assert r_a.agent_scores == r_b.agent_scores


# ---------------------------------------------------------------------------
# TournamentRunner
# ---------------------------------------------------------------------------

class TestTournamentRunner:
    def test_leaderboard_length(self):
        runner = TournamentRunner(make_agents(), make_tasks(), n_episodes=3, seed=0)
        result = runner.run()
        assert len(result.leaderboard) == 3

    def test_leaderboard_sorted_descending(self):
        runner = TournamentRunner(make_agents(), make_tasks(), n_episodes=3, seed=0)
        result = runner.run()
        scores = [s for _, s in result.leaderboard]
        assert scores == sorted(scores, reverse=True)

    def test_all_tasks_present(self):
        runner = TournamentRunner(make_agents(), make_tasks(), n_episodes=3, seed=0)
        result = runner.run()
        assert set(result.task_results.keys()) == {"resource_allocation", "coordination", "competitive"}

    def test_normalised_scores_01(self):
        runner = TournamentRunner(make_agents(), make_tasks(), n_episodes=3, seed=0)
        result = runner.run()
        for score in result.aggregate_scores.values():
            assert 0.0 <= score <= 1.0

    def test_single_agent_tournament(self):
        """Single agent should still complete without errors."""
        runner = TournamentRunner([GreedyAgent("g")], make_tasks(), n_episodes=2, seed=0)
        result = runner.run()
        assert len(result.leaderboard) == 1


# ---------------------------------------------------------------------------
# EvalReport
# ---------------------------------------------------------------------------

class TestEvalReport:
    def _make_report(self):
        runner = TournamentRunner(make_agents(), make_tasks(), n_episodes=3, seed=0)
        result = runner.run()
        return EvalReport.from_tournament(result, n_episodes=3)

    def test_leaderboard_ranks(self):
        report = self._make_report()
        ranks = [e["rank"] for e in report.leaderboard]
        assert ranks == [1, 2, 3]

    def test_to_json_parseable(self):
        import json
        report = self._make_report()
        data = json.loads(report.to_json())
        assert "leaderboard" in data
        assert "task_breakdowns" in data

    def test_to_text_contains_leaderboard(self):
        report = self._make_report()
        text = report.to_text()
        assert "LEADERBOARD" in text
        assert "greedy" in text or "rand" in text or "coop" in text

    def test_save_creates_file(self, tmp_path):
        report = self._make_report()
        path = tmp_path / "report.json"
        report.save(str(path))
        assert path.exists()
        import json
        data = json.loads(path.read_text())
        assert data["n_agents"] == 3
