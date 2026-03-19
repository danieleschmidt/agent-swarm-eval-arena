"""Tests for agent implementations."""

import random
import pytest

from swarm_arena.agents import RandomAgent, GreedyAgent, CooperativeAgent


RNG = random.Random(0)


# ---------------------------------------------------------------------------
# RandomAgent
# ---------------------------------------------------------------------------

class TestRandomAgent:
    def setup_method(self):
        self.agent = RandomAgent("rand")

    def test_name(self):
        assert self.agent.name == "rand"

    def test_act_slot_values_in_range(self):
        obs = {"slot_values": [10.0, 20.0, 5.0, 30.0]}
        for _ in range(50):
            a = self.agent.act(obs, RNG)
            assert 0 <= a < 4

    def test_act_n_choices_in_range(self):
        obs = {"n_choices": 6}
        for _ in range(50):
            a = self.agent.act(obs, RNG)
            assert 0 <= a < 6

    def test_act_n_strategies_in_range(self):
        obs = {"n_strategies": 3}
        for _ in range(50):
            a = self.agent.act(obs, RNG)
            assert 0 <= a < 3

    def test_act_unknown_obs_returns_0(self):
        assert self.agent.act({}, RNG) == 0

    def test_reset_no_error(self):
        self.agent.reset()  # should not raise


# ---------------------------------------------------------------------------
# GreedyAgent
# ---------------------------------------------------------------------------

class TestGreedyAgent:
    def setup_method(self):
        self.agent = GreedyAgent("greedy")

    def test_picks_max_slot(self):
        obs = {"slot_values": [5.0, 99.0, 1.0]}
        assert self.agent.act(obs, RNG) == 1

    def test_picks_max_slot_last(self):
        obs = {"slot_values": [1.0, 2.0, 100.0]}
        assert self.agent.act(obs, RNG) == 2

    def test_coordination_picks_0(self):
        assert self.agent.act({"n_choices": 5}, RNG) == 0

    def test_competitive_picks_1_mod_n(self):
        # beats strategy 0
        assert self.agent.act({"n_strategies": 3}, RNG) == 1

    def test_competitive_wraps(self):
        # n_strategies=1: 1 % 1 == 0
        assert self.agent.act({"n_strategies": 1}, RNG) == 0


# ---------------------------------------------------------------------------
# CooperativeAgent
# ---------------------------------------------------------------------------

class TestCooperativeAgent:
    def setup_method(self):
        self.agent = CooperativeAgent("coop")

    def test_coordination_picks_0(self):
        assert self.agent.act({"n_choices": 5}, RNG) == 0

    def test_slot_in_range(self):
        obs = {"slot_values": [10.0, 20.0, 5.0]}
        action = self.agent.act(obs, RNG)
        assert 0 <= action < 3

    def test_competitive_no_history_picks_0(self):
        self.agent.reset()
        action = self.agent.act({"n_strategies": 3}, RNG)
        assert 0 <= action < 3

    def test_reset_clears_history(self):
        self.agent._last_opp_action = 2
        self.agent.reset()
        assert self.agent._last_opp_action is None

    def test_name_affects_slot_offset(self):
        """Different names should produce different slot choices on identical obs."""
        obs = {"slot_values": [100.0, 90.0, 80.0, 70.0, 60.0]}
        picks = set()
        for i in range(5):
            a = CooperativeAgent(f"coop-{i}")
            picks.add(a.act(obs, RNG))
        # Multiple agents should spread across slots
        assert len(picks) > 1
