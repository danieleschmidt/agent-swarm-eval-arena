"""
EvalReport: structured JSON + human-readable text report of tournament results.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any

from .tournament import TournamentResult


@dataclass
class EvalReport:
    """Structured evaluation report produced after a tournament."""

    generated_at: str
    n_agents: int
    n_tasks: int
    n_episodes_per_task: int
    leaderboard: list[dict[str, Any]]
    task_breakdowns: dict[str, Any]
    metadata: dict[str, Any]

    # ------------------------------------------------------------------ #
    # Factory                                                              #
    # ------------------------------------------------------------------ #

    @classmethod
    def from_tournament(
        cls,
        result: TournamentResult,
        n_episodes: int = 0,
        metadata: dict[str, Any] | None = None,
    ) -> "EvalReport":
        leaderboard = [
            {"rank": i + 1, "agent": name, "score": round(score, 4)}
            for i, (name, score) in enumerate(result.leaderboard)
        ]

        task_breakdowns: dict[str, Any] = {}
        for task_name, task_result in result.task_results.items():
            task_breakdowns[task_name] = {
                "ranked": [
                    {"agent": name, "score": round(score, 4)}
                    for name, score in task_result.ranked
                ],
                "metadata": task_result.metadata,
            }

        return cls(
            generated_at=datetime.now(timezone.utc).isoformat(),
            n_agents=len(result.leaderboard),
            n_tasks=len(result.task_results),
            n_episodes_per_task=n_episodes,
            leaderboard=leaderboard,
            task_breakdowns=task_breakdowns,
            metadata=metadata or {},
        )

    # ------------------------------------------------------------------ #
    # Serialisation                                                        #
    # ------------------------------------------------------------------ #

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    def to_text(self) -> str:
        """Human-readable text summary."""
        lines: list[str] = []
        lines.append("=" * 60)
        lines.append("  AGENT SWARM EVALUATION REPORT")
        lines.append("=" * 60)
        lines.append(f"  Generated : {self.generated_at}")
        lines.append(f"  Agents    : {self.n_agents}")
        lines.append(f"  Tasks     : {self.n_tasks}")
        lines.append(f"  Episodes  : {self.n_episodes_per_task} per task")
        lines.append("")
        lines.append("── OVERALL LEADERBOARD ─────────────────────────────")
        for entry in self.leaderboard:
            medal = {1: "🥇", 2: "🥈", 3: "🥉"}.get(entry["rank"], "  ")
            lines.append(f"  {medal} #{entry['rank']:2d}  {entry['agent']:<20s}  {entry['score']:.4f}")
        lines.append("")

        for task_name, breakdown in self.task_breakdowns.items():
            lines.append(f"── TASK: {task_name} " + "─" * max(0, 48 - len(task_name)))
            for entry in breakdown["ranked"]:
                lines.append(f"       {entry['agent']:<22s}  {entry['score']:.2f}")
            meta = breakdown.get("metadata", {})
            if meta:
                lines.append(f"       meta: {meta}")
            lines.append("")

        lines.append("=" * 60)
        return "\n".join(lines)

    def save(self, path: str) -> None:
        """Write JSON report to *path*."""
        with open(path, "w") as fh:
            fh.write(self.to_json())
