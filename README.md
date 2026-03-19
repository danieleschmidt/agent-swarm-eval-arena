# Agent Swarm Eval Arena

A lightweight multi-agent evaluation framework for benchmarking and comparing agent strategies on coordination, cooperation, and competition tasks.

## What it does

Runs multiple agent types across a battery of tasks, normalises scores across tasks, and produces a ranked leaderboard and structured report. Useful for:

- Comparing greedy vs. cooperative vs. random strategies
- Studying emergent coordination behaviour
- Building intuition about multi-agent dynamics before scaling to heavier RL frameworks

## Tasks

| Task | Type | Description |
|------|------|-------------|
| `ResourceAllocationTask` | Cooperative | Agents allocate from a shared resource pool; collisions split rewards — spread out to maximise collective utility |
| `CoordinationTask` | Cooperative | Agents must independently converge on the same choice without communication — focal-point strategies win |
| `CompetitiveTask` | Competitive | Zero-sum Rock-Paper-Scissors variant; all-pairs matchups each round |

## Agents

| Agent | Strategy |
|-------|----------|
| `RandomAgent` | Uniform random actions — useful baseline |
| `GreedyAgent` | Maximises immediate individual reward every round |
| `CooperativeAgent` | Distributes choices to avoid overlap; converges on focal points |

## Quick start

```python
from swarm_arena import TournamentRunner, EvalReport
from swarm_arena.agents import RandomAgent, GreedyAgent, CooperativeAgent
from swarm_arena.tasks import ResourceAllocationTask, CoordinationTask, CompetitiveTask

agents = [RandomAgent("rand"), GreedyAgent("greedy"), CooperativeAgent("coop")]
tasks  = [ResourceAllocationTask(), CoordinationTask(), CompetitiveTask()]

runner = TournamentRunner(agents, tasks, n_episodes=10, seed=42)
result = runner.run()

report = EvalReport.from_tournament(result, n_episodes=10)
print(report.to_text())
report.save("results.json")
```

Example output:

```
============================================================
  AGENT SWARM EVALUATION REPORT
============================================================
  Generated : 2026-03-19T16:20:51+00:00
  Agents    : 3
  Tasks     : 3
  Episodes  : 10 per task

── OVERALL LEADERBOARD ─────────────────────────────
  🥇 # 1  greedy                0.6667
  🥈 # 2  coop                  0.3333
  🥉 # 3  rand                  0.1190

── TASK: resource_allocation ───────────────────────
       greedy                  166.00
       coop                    166.00
       rand                    139.85

── TASK: coordination ──────────────────────────────
       (all tied: 3.30)

── TASK: competitive ───────────────────────────────
       greedy                  39.10
       rand                    20.90
       coop                    10.80
============================================================
```

## Project layout

```
swarm_arena/
  __init__.py       # public API
  agents.py         # RandomAgent, GreedyAgent, CooperativeAgent
  tasks.py          # Task base + 3 concrete tasks
  arena.py          # Arena (single task runner)
  tournament.py     # TournamentRunner + leaderboard
  report.py         # EvalReport (JSON + text)
tests/
  test_agents.py
  test_tasks.py
  test_arena.py
```

## Running tests

```bash
python -m pytest tests/ -v
```

## Extending

**Add a new task** — subclass `Task`, implement `run(agents, rng) -> TaskResult`.

**Add a new agent** — subclass `Agent`, implement `act(obs, rng) -> int`.

The observation dict is task-specific; agents that inspect and reason about it will outperform those that don't.

## License

MIT
