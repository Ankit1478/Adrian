# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 SecureAgentics

"""Drives one enforcement case against the real, patched SDK.

Nothing here stubs the enforcement path. ``adrian.init`` runs, the
LangChain and Anthropic patches are applied for real, and the case is
driven through the same public call the application would make. The only
things standing in for production are the two ends the SDK talks to: the
backend (a ``WebSocketClient`` driven to a mode with a pre-resolved
verdict, exactly as ``tests/test_block_mode.py`` does) and the model
(a stub response, since the question is what the SDK does with a tool
call, not whether a model produces one).

The single observation each surface makes is whether the tool body ran.
That is the only thing a customer can be harmed by: a ``[BLOCKED]``
message in the transcript is cosmetic if the side effect already
happened.
"""

# pyright: reportPrivateUsage=false

from __future__ import annotations

import asyncio
import contextlib
import tempfile
from collections.abc import AsyncGenerator, Awaitable, Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

import adrian
from adrian.format.types import AgentContext, LlmPairData, PairedEvent
from adrian.proto import event_pb2 as pb
from adrian.ws import WebSocketClient

from .cases import Case

#: Verdict wait used for cases that are meant to resolve immediately.
#: Short, because a case that hangs here is a bug worth surfacing fast.
FAST_TIMEOUT = 1.0

#: Verdict wait for the ``none`` (no verdict arrives) cases. Long enough
#: to be a real wait, short enough that the suite stays interactive.
TIMEOUT_CASE_WAIT = 0.2

_MODES = {
    "alert": pb.MODE_ALERT,
    "hitl": pb.MODE_HITL,
    "block": pb.MODE_BLOCK,
}

TOOL_CALL_ID = "tc-eval-1"
EVENT_ID = "evt-eval-1"


@dataclass
class Probe:
    """The side effect a tool would have. Set only if the body runs."""

    ran: bool = False
    calls: list[str] = field(default_factory=list)

    def fire(self, arg: str) -> str:
        """Record that the real tool body executed."""
        self.ran = True
        self.calls.append(arg)
        return f"did:{arg}"


@dataclass
class Outcome:
    """What one surface observed."""

    ran: bool
    error: str | None = None

    @property
    def verdict_word(self) -> str:
        """``ran`` or ``blocked``, for comparison with a case's label."""
        return "ran" if self.ran else "blocked"


@dataclass
class World:
    """A live SDK wired to a backend in one mode with one verdict."""

    ws: WebSocketClient
    probe: Probe
    case: Case

    async def announce_tool_call(self) -> None:
        """Emit the LLM pair that asks for the tool call.

        In production a tool call always follows the LLM pair that
        produced it, and that pair is what tells the SDK which verdict a
        later tool call should wait on. The Anthropic surfaces emit one
        themselves by calling the patched client; the LangChain ones
        invoke a tool directly, so the pair is sent here.

        It goes through ``on_paired_event`` -- the SDK's own send path --
        so the tool-call map and the pending future are built by the SDK
        rather than written into its private fields by the harness.
        """
        await self.ws.on_paired_event(
            PairedEvent(
                event_id=EVENT_ID,
                invocation_id="eval-inv",
                session_id="eval-sess",
                run_id="eval-run",
                timestamp="2026-01-01T00:00:00Z",
                pair_type="llm",
                agent=AgentContext(agent_id="eval-agent"),
                parent=None,
                data=LlmPairData(
                    model="ChatAnthropic",
                    output="I will call the tool.",
                    tool_calls=[
                        {"id": TOOL_CALL_ID, "name": "probe", "args": {"x": "hi"}}
                    ],  # type: ignore[list-item]
                ),
            )
        )

    @property
    def timeout(self) -> float:
        """The block timeout this case was configured with."""
        return TIMEOUT_CASE_WAIT if self.case.verdict == "none" else FAST_TIMEOUT


def _policy(mode: int) -> pb.PolicySnapshot:
    """The policy snapshot a LoginAck would carry.

    ``policy_m3``/``policy_m4`` on and ``policy_m0``/``policy_m2`` off is
    the product default: block and escalate codes act, warn codes only
    notify. The ``allow`` verdict below is an M2, so it is genuinely
    out of scope rather than merely tolerated.
    """
    return pb.PolicySnapshot(
        mode=cast("Any", mode),
        policy_m0=False,
        policy_m2=False,
        policy_m3=True,
        policy_m4=True,
    )


class _Backend:
    """A stand-in worker backend that speaks the real wire protocol.

    It is an async iterable of serialised ``ServerFrame`` bytes, which is
    exactly what the SDK's own ``_recv_loop`` consumes from a websocket.
    So the LoginAck is decoded and applied by the SDK, and each verdict
    goes through ``_on_verdict_frame`` and resolves the pending future
    the way a real one does.

    An earlier version reached past all of that and set ``_mode``,
    ``_policy`` and the futures by hand. That tested enforcement from
    ``register_pending`` downwards and silently skipped the frame decode,
    the LoginAck application and the verdict dispatch -- the part of the
    SDK a real backend actually talks to.

    It answers reactively rather than from a script: the SDK mints the
    event id when it emits the pair, so a scripted frame would key on an
    id that does not exist yet. This waits for a pending verdict to
    appear and answers that one, which is what a backend does.
    """

    #: How often to look for a verdict the SDK is waiting on. Short
    #: relative to the shortest block timeout so it never adds a wait
    #: the case did not ask for.
    POLL = 0.005

    def __init__(
        self, ws: WebSocketClient, case: Case, policy: pb.PolicySnapshot
    ) -> None:
        self._ws = ws
        self._case = case
        self._policy = policy
        self._sent_login = False
        self._answered: set[str] = set()

    async def send(self, _data: bytes) -> None:
        """Accept a frame from the SDK.

        The backend's reply is driven by what the SDK is waiting on, not
        by what it sent, so the frame itself is not needed. Accepting it
        rather than failing is what keeps the send path real: the SDK
        buffers and retries when a send fails, which would change the
        timing the case is measuring.
        """
        return None

    def __aiter__(self) -> _Backend:
        """The recv loop iterates the connection; this is that connection."""
        return self

    async def __anext__(self) -> bytes:
        """The next frame the SDK would read off the socket."""
        if self._case.mode == "no_policy":
            # A backend that accepted the socket and then said nothing.
            # Never yielding is the honest shape of that: the SDK waits
            # for a LoginAck that is not coming.
            await _forever()

        if not self._sent_login:
            self._sent_login = True
            frame = pb.ServerFrame()
            frame.login_ack.policy.CopyFrom(self._policy)
            return frame.SerializeToString()

        if self._case.verdict == "none":
            # The backend is reachable but never rules on this call, so
            # the SDK's wait runs out for real and the fail-closed
            # choice is exercised rather than simulated.
            await _forever()

        while True:
            for event_id, pending in list(self._ws._pending_verdicts.items()):
                if event_id in self._answered or pending.done():
                    continue
                self._answered.add(event_id)
                frame = pb.ServerFrame()
                frame.verdict.CopyFrom(self._verdict_for(event_id))
                return frame.SerializeToString()
            await asyncio.sleep(self.POLL)

    def _verdict_for(self, event_id: str) -> pb.Verdict:
        """The ruling this case's backend returns for one event."""
        if self._case.verdict == "allow":
            verdict = pb.Verdict(event_id=event_id, mad_code="M2", policy=self._policy)
        else:
            verdict = pb.Verdict(
                event_id=event_id, mad_code="M4_a", policy=self._policy
            )
        if self._case.verdict == "hitl_deny":
            verdict.hitl.continue_execution = False
        elif self._case.verdict == "hitl_approve":
            verdict.hitl.continue_execution = True
        return verdict


async def _forever() -> bytes:
    """Block until cancelled, for a backend that sends nothing more."""
    await asyncio.Event().wait()
    raise AssertionError("unreachable")


@contextlib.asynccontextmanager
async def world_for(case: Case) -> AsyncGenerator[World]:
    """Start the SDK for one case and tear it down afterwards.

    The teardown matters as much as the setup: ``adrian.init`` installs
    global monkey-patches and a module-level client, so a case that
    leaked state would quietly decide the next one's result.
    """
    probe = Probe()
    with tempfile.TemporaryDirectory() as tmp:
        adrian.init(
            api_key="eval-key",
            log_file=str(Path(tmp) / "events.jsonl"),
            auto_instrument=True,
            ws_url="ws://eval.invalid",
            block_timeout=TIMEOUT_CASE_WAIT if case.verdict == "none" else FAST_TIMEOUT,
        )
        ws = adrian._ws_client
        if ws is None:
            msg = "adrian.init did not create a WebSocket client"
            raise RuntimeError(msg)

        policy = _policy(_MODES.get(case.mode, pb.MODE_BLOCK))

        # Hand the SDK the backend as its connection and let its own
        # receive loop read from it. The mode, the policy and every
        # verdict then arrive the way they do in production -- as
        # protobuf frames the SDK decodes itself -- rather than being
        # written into its private fields from outside.
        ws._ws = _Backend(ws, case, policy)  # type: ignore[assignment]
        ws._connected.set()
        ws._loop = asyncio.get_running_loop()
        recv = asyncio.create_task(ws._recv_loop())
        ws._recv_task = recv

        # Wait for the LoginAck to land so the case starts from a
        # logged-in SDK, exactly as a real one would. ``no_policy`` is
        # the case where it never comes, and waiting there is the point.
        if case.mode != "no_policy":
            await asyncio.wait_for(ws._login_ack_received.wait(), timeout=5.0)

        try:
            yield World(ws=ws, probe=probe, case=case)
        finally:
            recv.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await recv
            adrian.shutdown()


Surface = Callable[[World], Awaitable[Outcome]]

SURFACES: dict[str, Surface] = {}


def surface(name: str) -> Callable[[Surface], Surface]:
    """Register a call path under ``name`` for cases to reference."""

    def register(fn: Surface) -> Surface:
        SURFACES[name] = fn
        return fn

    return register
