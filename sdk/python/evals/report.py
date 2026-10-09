# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 SecureAgentics

"""Scoring and the CI gate.

The headline number is deliberately not "percent passing". A gapped
path lets everything through, so it scores full marks on every case
that expects the tool to run, and a single percentage would flatter it.
What matters is counted separately:

* **Missed blocks** -- the tool ran when the verdict said stop. Each one
  is a bypass a customer could be harmed by.
* **Over-blocks** -- the tool was stopped when it should have run. Each
  one is pressure to turn block mode off.
* **Known gaps** -- missed blocks that are already written down in
  GAPS.md. Still failures, but not news.

The gate fails on any missed block that is *not* a known gap, on any
over-block, on any error, and on a known gap that has started passing,
because a fix nobody records is a fix that gets undone.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field

from .cases import Case
from .harness import Outcome


@dataclass(frozen=True)
class Result:
    """One case, run."""

    case: Case
    outcome: Outcome

    @property
    def passed(self) -> bool:
        """Whether the SDK did what the case says it should."""
        return self.outcome.error is None and (
            self.outcome.verdict_word == self.case.expected
        )

    @property
    def kind(self) -> str:
        """What sort of result this is, for the report's breakdown."""
        if self.outcome.error is not None:
            return "error"
        if self.passed:
            return "unexpected_pass" if self.case.known_gap else "pass"
        if self.case.expected == "blocked":
            return "known_gap" if self.case.known_gap else "missed_block"
        return "over_block"


@dataclass
class Scores:
    """The counts the gate reads."""

    total: int = 0
    passed: int = 0
    missed_blocks: int = 0
    over_blocks: int = 0
    known_gaps: int = 0
    unexpected_passes: int = 0
    errors: int = 0
    block_cases: int = 0
    run_cases: int = 0


@dataclass
class Report:
    """A scored run."""

    results: list[Result] = field(default_factory=list)
    scores: Scores = field(default_factory=Scores)

    @property
    def ok(self) -> bool:
        """Whether this run should pass CI. See the module docstring."""
        s = self.scores
        return (
            s.missed_blocks == 0
            and s.over_blocks == 0
            and s.errors == 0
            and s.unexpected_passes == 0
        )


def score(results: list[Result]) -> Report:
    """Count a run into a report."""
    s = Scores(total=len(results))
    for r in results:
        if r.case.expected == "blocked":
            s.block_cases += 1
        else:
            s.run_cases += 1
        if r.passed:
            s.passed += 1
        match r.kind:
            case "missed_block":
                s.missed_blocks += 1
            case "over_block":
                s.over_blocks += 1
            case "known_gap":
                s.known_gaps += 1
            case "unexpected_pass":
                s.unexpected_passes += 1
            case "error":
                s.errors += 1
            case _:
                pass
    return Report(results=results, scores=s)


_LABEL = {
    "missed_block": "MISSED BLOCK",
    "over_block": "OVER-BLOCK",
    "known_gap": "known gap",
    "unexpected_pass": "GAP NOW PASSES",
    "error": "ERROR",
}


def render(report: Report) -> str:
    """The human-readable report."""
    s = report.scores
    lines: list[str] = []
    lines.append("SDK enforcement eval")
    lines.append("=" * 68)
    lines.append(
        f"{s.total} cases  "
        f"({s.block_cases} expect a block, {s.run_cases} expect the tool to run)"
    )
    lines.append(f"passed            {s.passed}/{s.total}")
    lines.append(
        f"missed blocks     {s.missed_blocks}   (tool ran despite a halt verdict)"
    )
    lines.append(
        f"over-blocks       {s.over_blocks}   (tool stopped when it should have run)"
    )
    lines.append(f"known gaps        {s.known_gaps}   (documented in GAPS.md)")
    lines.append(f"errors            {s.errors}")
    if s.unexpected_passes:
        lines.append(
            f"gaps now passing  {s.unexpected_passes}   (fixed -- remove the marker in cases.py)"
        )

    by_surface: dict[str, list[Result]] = {}
    for r in report.results:
        by_surface.setdefault(r.case.surface, []).append(r)

    lines.append("")
    lines.append("By call path")
    lines.append("-" * 68)
    for surface, rows in by_surface.items():
        good = sum(1 for r in rows if r.passed)
        gaps = sum(1 for r in rows if r.kind == "known_gap")
        note = f"  [{gaps} known gaps]" if gaps else ""
        lines.append(f"  {surface:36s} {good}/{len(rows)}{note}")

    notable = [r for r in report.results if r.kind != "pass"]
    if notable:
        lines.append("")
        lines.append("Cases needing attention")
        lines.append("-" * 68)
        for r in notable:
            lines.append(f"  [{_LABEL[r.kind]}] {r.case.id}")
            lines.append(
                f"      expected {r.case.expected}, got {r.outcome.verdict_word}"
            )
            lines.append(f"      {r.case.why}")
            if r.case.known_gap:
                lines.append(f"      gap: {r.case.known_gap}")
            if r.outcome.error:
                lines.append(f"      error: {r.outcome.error}")

    lines.append("")
    lines.append("PASS" if report.ok else "FAIL")
    return "\n".join(lines)


def to_json(report: Report) -> str:
    """The machine-readable report, for comparing runs over time."""
    return json.dumps(
        {
            "scores": asdict(report.scores),
            "ok": report.ok,
            "results": [
                {
                    "id": r.case.id,
                    "surface": r.case.surface,
                    "mode": r.case.mode,
                    "verdict": r.case.verdict,
                    "expected": r.case.expected,
                    "got": r.outcome.verdict_word,
                    "kind": r.kind,
                    "known_gap": r.case.known_gap,
                    "error": r.outcome.error,
                }
                for r in report.results
            ],
        },
        indent=2,
    )
