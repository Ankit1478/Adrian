# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 SecureAgentics

"""Tests for the SDK enforcement eval itself.

The eval is a safety check, so a bug in its scoring is as bad as a bug
in the thing it checks: a harness that reports PASS whatever happens
protects nothing. These tests cover the scoring rules and the matrix's
own invariants, and one of them runs the real eval over the gated
LangChain paths end to end.
"""

from __future__ import annotations

from evals.cases import COMBOS, SURFACE_SPECS, Case, build_cases
from evals.harness import Outcome
from evals.report import Result, score
from evals.runner import quieten, run_all


def _case(expected: str, known_gap: str = "") -> Case:
    return Case(
        id="x",
        surface="langchain.toolnode_async",
        mode="block",
        verdict="halt",
        expected=expected,
        why="test",
        known_gap=known_gap,
    )


class TestScoring:
    def test_tool_running_despite_a_block_is_a_missed_block(self) -> None:
        report = score([Result(_case("blocked"), Outcome(ran=True))])
        assert report.scores.missed_blocks == 1
        assert not report.ok

    def test_tool_stopped_when_it_should_run_is_an_over_block(self) -> None:
        report = score([Result(_case("ran"), Outcome(ran=False))])
        assert report.scores.over_blocks == 1
        assert not report.ok

    def test_a_documented_gap_does_not_fail_the_gate(self) -> None:
        report = score([Result(_case("blocked", "some-gap"), Outcome(ran=True))])
        assert report.scores.known_gaps == 1
        assert report.scores.missed_blocks == 0
        assert report.ok

    def test_a_gap_that_starts_passing_fails_the_gate(self) -> None:
        """A fix nobody records is a fix that gets undone."""
        report = score([Result(_case("blocked", "some-gap"), Outcome(ran=False))])
        assert report.scores.unexpected_passes == 1
        assert not report.ok

    def test_an_error_fails_the_gate(self) -> None:
        """A hung or crashed case must never be read as enforcement."""
        report = score([Result(_case("blocked"), Outcome(ran=False, error="hung"))])
        assert report.scores.errors == 1
        assert not report.ok

    def test_a_clean_run_passes(self) -> None:
        report = score(
            [
                Result(_case("blocked"), Outcome(ran=False)),
                Result(_case("ran"), Outcome(ran=True)),
            ]
        )
        assert report.ok
        assert report.scores.passed == 2


class TestMatrix:
    def test_every_surface_is_run_against_every_situation(self) -> None:
        assert len(build_cases()) == len(SURFACE_SPECS) * len(COMBOS)

    def test_case_ids_are_unique(self) -> None:
        ids = [c.id for c in build_cases()]
        assert len(ids) == len(set(ids))

    def test_every_surface_named_in_the_matrix_exists(self) -> None:
        from evals.runner import load_surfaces

        load_surfaces()
        from evals.harness import SURFACES

        missing = {s.name for s in SURFACE_SPECS} - set(SURFACES)
        assert not missing, f"matrix names call paths with no surface: {missing}"

    def test_every_gap_is_documented(self) -> None:
        """A gap marker with no GAPS.md section is an undocumented bypass."""
        from pathlib import Path

        gaps = Path(__file__).parent.parent / "evals" / "GAPS.md"
        text = gaps.read_text()
        for spec in SURFACE_SPECS:
            if spec.gap:
                assert f"`{spec.gap}`" in text, f"{spec.gap} is not in GAPS.md"

    def test_a_gap_marker_only_applies_where_a_block_is_expected(self) -> None:
        """A path that enforces nothing still gets the allow cases right."""
        for case in build_cases():
            if case.known_gap:
                assert case.expected == "blocked"


class TestEndToEnd:
    async def test_the_gated_langchain_paths_still_enforce(self) -> None:
        """The real eval, over the paths that are supposed to work.

        Narrowed to the gated LangChain paths so the test stays quick;
        the full matrix including Anthropic and the gaps is what
        ``python -m evals`` runs.
        """
        quieten()
        gated = [
            c
            for c in build_cases()
            if c.surface in ("langchain.toolnode_async", "langchain.toolnode_sync")
        ]
        report = await run_all(gated)
        assert report.scores.missed_blocks == 0
        assert report.scores.over_blocks == 0
        assert report.scores.errors == 0
        assert report.ok
