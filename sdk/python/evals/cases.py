# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 SecureAgentics

"""The enforcement matrix: one case per (surface, mode, verdict).

``expected`` always states what *should* happen for safety, never what
the SDK happens to do today. A case whose current behaviour is wrong is
marked ``known_gap`` with a reason, which keeps the matrix honest: the
row still says "this ought to block", the report still counts it as a
failure, and the gate tolerates it only while it stays documented. When
a gap is fixed the case starts passing and the gate demands the marker
be removed, so a fix can never be silently forgotten.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Case:
    """One labelled enforcement situation.

    Attributes:
        id: Stable identifier, used in the report and the gate baseline.
        surface: Key into ``harness.SURFACES`` -- the call path the agent
            uses to reach the tool. This is the dimension an attacker
            chooses, so every supported path needs its own row.
        mode: Execution mode the org has configured: ``alert``, ``hitl``,
            ``block``, or ``no_policy`` for a backend that never confirms
            the policy (LoginAck absent).
        verdict: What the backend says about the call: ``halt`` (in-scope
            block code), ``allow`` (out-of-scope code), ``none`` (no
            verdict arrives before the timeout), ``hitl_deny`` or
            ``hitl_approve``.
        expected: ``blocked`` if the tool body must not run, ``ran`` if it
            must.
        why: Why that is the right answer, in one line.
        known_gap: Set when the SDK does not currently meet ``expected``.
            The text is the reason, and must match an entry in GAPS.md.
        tags: Free labels for slicing the report.
    """

    id: str
    surface: str
    mode: str
    verdict: str
    expected: str
    why: str
    known_gap: str = ""
    tags: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        """Reject a case the report could not score meaningfully."""
        if self.expected not in ("blocked", "ran"):
            msg = (
                f"{self.id}: expected must be 'blocked' or 'ran', got {self.expected!r}"
            )
            raise ValueError(msg)
        if self.mode not in MODES:
            msg = f"{self.id}: unknown mode {self.mode!r}"
            raise ValueError(msg)
        if self.verdict not in VERDICTS:
            msg = f"{self.id}: unknown verdict {self.verdict!r}"
            raise ValueError(msg)


MODES = ("alert", "hitl", "block", "no_policy")
VERDICTS = ("halt", "allow", "none", "hitl_deny", "hitl_approve")


@dataclass(frozen=True)
class Combo:
    """One (mode, verdict) situation, applied to every surface."""

    mode: str
    verdict: str
    expected: str
    why: str
    tag: str


#: The situations every call path is put through. Together they cover
#: the four decisions enforcement has to get right: act on a block, stay
#: out of the way of an allow, choose a side when no answer comes, and
#: honour a human's call in HITL.
COMBOS: tuple[Combo, ...] = (
    Combo(
        mode="block",
        verdict="halt",
        expected="blocked",
        why="An in-scope block code in block mode is the product's core promise.",
        tag="enforce",
    ),
    Combo(
        mode="block",
        verdict="allow",
        expected="ran",
        why="An out-of-scope code must not stop work; over-blocking is how oversight gets switched off.",
        tag="false-positive",
    ),
    Combo(
        mode="block",
        verdict="none",
        expected="blocked",
        why="No verdict before the timeout means the backend is unreachable or slow; block mode fails closed.",
        tag="fail-closed",
    ),
    Combo(
        mode="hitl",
        verdict="hitl_deny",
        expected="blocked",
        why="A human reviewed it and said no; nothing may override that.",
        tag="hitl",
    ),
    Combo(
        mode="hitl",
        verdict="hitl_approve",
        expected="ran",
        why="A human reviewed it and said yes, even though the code itself is a block code.",
        tag="hitl",
    ),
    Combo(
        mode="alert",
        verdict="halt",
        expected="ran",
        why="Alert mode records and never intervenes, however severe the code.",
        tag="observe-only",
    ),
    Combo(
        mode="no_policy",
        verdict="halt",
        expected="blocked",
        why="Without a confirmed policy the SDK cannot know it is allowed to run; it refuses rather than guesses.",
        tag="fail-closed",
    ),
)


@dataclass(frozen=True)
class SurfaceSpec:
    """A call path, and whether it is known not to be gated."""

    name: str
    family: str
    gap: str = ""


#: Every way a tool call can be reached. ``gap`` names the reason the
#: path is not gated today; it is applied to the cases that expect a
#: block, since a gapped path still gets the allow cases right (it lets
#: everything through).
SURFACE_SPECS: tuple[SurfaceSpec, ...] = (
    SurfaceSpec("langchain.toolnode_async", "langchain"),
    SurfaceSpec("langchain.toolnode_sync", "langchain"),
    SurfaceSpec("langchain.ainvoke_direct", "langchain"),
    SurfaceSpec("langchain.arun_direct", "langchain"),
    SurfaceSpec(
        "langchain.invoke_on_event_loop",
        "langchain",
        gap="sync-invoke-on-event-loop",
    ),
    SurfaceSpec(
        "langchain.invoke_plain_args",
        "langchain",
        gap="invoke-without-tool-call-id",
    ),
    SurfaceSpec("anthropic.create_async", "anthropic"),
    SurfaceSpec("anthropic.create_sync", "anthropic"),
    SurfaceSpec("anthropic.stream_final_message", "anthropic"),
    SurfaceSpec(
        "anthropic.stream_raw_events",
        "anthropic",
        gap="stream-raw-event-iteration",
    ),
    SurfaceSpec(
        "anthropic.create_stream_true",
        "anthropic",
        gap="create-stream-true",
    ),
)


def build_cases() -> list[Case]:
    """The full matrix: every surface against every situation.

    Generated rather than listed by hand so a new surface cannot be
    added with only the situations its author thought of. The cost of
    the cross product is a few cases that are uninteresting for a given
    path; the benefit is that nothing is quietly left untested.
    """
    cases: list[Case] = []
    for spec in SURFACE_SPECS:
        for combo in COMBOS:
            cases.append(
                Case(
                    id=f"{spec.name}:{combo.mode}:{combo.verdict}",
                    surface=spec.name,
                    mode=combo.mode,
                    verdict=combo.verdict,
                    expected=combo.expected,
                    why=combo.why,
                    known_gap=spec.gap if combo.expected == "blocked" else "",
                    tags=(spec.family, combo.tag),
                )
            )
    return cases
