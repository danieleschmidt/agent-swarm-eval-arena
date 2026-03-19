"""
Agent Swarm Eval Arena
======================
Benchmark and compare agent strategies on multi-agent coordination tasks.

Quick start::

    from swarm_arena import Arena, TournamentRunner, EvalReport
    from swarm_arena.agents import RandomAgent, GreedyAgent, CooperativeAgent
    from swarm_arena.tasks import ResourceAllocationTask, CoordinationTask, CompetitiveTask

    agents = [RandomAgent("rand"), GreedyAgent("greedy"), CooperativeAgent("coop")]
    tasks  = [ResourceAllocationTask(), CoordinationTask(), CompetitiveTask()]
    runner = TournamentRunner(agents, tasks, n_episodes=10)
    result = runner.run()
    report = EvalReport.from_tournament(result, n_episodes=10)
    print(report.to_text())
"""

from .agents import Agent, CooperativeAgent, GreedyAgent, RandomAgent
from .arena import Arena
from .report import EvalReport
from .tasks import (
    CompetitiveTask,
    CoordinationTask,
    ResourceAllocationTask,
    Task,
    TaskResult,
)
from .tournament import TournamentResult, TournamentRunner

__all__ = [
    # agents
    "Agent",
    "RandomAgent",
    "GreedyAgent",
    "CooperativeAgent",
    # tasks
    "Task",
    "TaskResult",
    "ResourceAllocationTask",
    "CoordinationTask",
    "CompetitiveTask",
    # arena
    "Arena",
    # tournament
    "TournamentRunner",
    "TournamentResult",
    # report
    "EvalReport",
]

__version__ = "0.2.0"
