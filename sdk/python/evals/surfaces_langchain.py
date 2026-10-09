# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 SecureAgentics

"""The LangChain / LangGraph call paths a tool can be reached through.

Each one is a genuinely different route into the same tool, and the gate
sits at a different place on each. ``ainvoke`` is gated by the async
gate; ``invoke`` from a worker thread bridges onto the WS loop;
``invoke`` from the event-loop thread cannot block and is skipped; and a
call made with plain arguments carries no tool_call_id for the gate to
key on. An attacker picks the route, so the matrix has to cover them
all rather than the one the happy path uses.
"""

# LangChain's tool helpers are only partially typed.
# pyright: reportUnknownMemberType=false
# pyright: reportUnusedParameter=false

from __future__ import annotations

from typing import Any

from langchain_core.messages import AIMessage
from langchain_core.runnables.config import RunnableConfig, ensure_config
from langchain_core.tools import StructuredTool
from langgraph._internal._constants import CONF, CONFIG_KEY_RUNTIME
from langgraph.prebuilt import ToolNode
from langgraph.runtime import Runtime

from .harness import TOOL_CALL_ID, Outcome, World, surface


def _runtime_config() -> RunnableConfig:
    """The config a modern ToolNode requires to dispatch."""
    return ensure_config({CONF: {CONFIG_KEY_RUNTIME: Runtime()}})


def _async_tool(world: World) -> StructuredTool:
    """An async tool whose body records that it ran."""

    async def probe(x: str) -> str:
        """Tool under test."""
        return world.probe.fire(x)

    return StructuredTool.from_function(coroutine=probe, name="probe")


def _sync_tool(world: World) -> StructuredTool:
    """A sync tool, which ToolNode dispatches on a worker thread."""

    def probe(x: str) -> str:
        """Tool under test."""
        return world.probe.fire(x)

    return StructuredTool.from_function(func=probe, name="probe")


def _tool_call_message() -> AIMessage:
    """The assistant turn that asks for the tool call."""
    return AIMessage(
        content="",
        tool_calls=[{"id": TOOL_CALL_ID, "name": "probe", "args": {"x": "hi"}}],
    )


@surface("langchain.toolnode_async")
async def toolnode_async(world: World) -> Outcome:
    """create_react_agent with an async tool: ToolNode -> BaseTool.ainvoke."""
    await world.announce_tool_call()
    node = ToolNode([_async_tool(world)])
    state: dict[str, Any] = {"messages": [_tool_call_message()]}
    await node.ainvoke(state, config=_runtime_config())  # pyright: ignore[reportUnknownMemberType]
    return Outcome(ran=world.probe.ran)


@surface("langchain.toolnode_sync")
async def toolnode_sync(world: World) -> Outcome:
    """create_react_agent with a sync tool: dispatched to a worker thread.

    The gate lands in ``BaseTool.invoke`` on a thread with no event loop
    and has to bridge back onto the WS loop to wait for the verdict.
    """
    await world.announce_tool_call()
    node = ToolNode([_sync_tool(world)])
    state: dict[str, Any] = {"messages": [_tool_call_message()]}
    await node.ainvoke(state, config=_runtime_config())  # pyright: ignore[reportUnknownMemberType]
    return Outcome(ran=world.probe.ran)


@surface("langchain.ainvoke_direct")
async def ainvoke_direct(world: World) -> Outcome:
    """Application code awaiting ``tool.ainvoke`` on a ToolCall itself."""
    await world.announce_tool_call()
    tool = _async_tool(world)
    await tool.ainvoke(
        {"id": TOOL_CALL_ID, "name": "probe", "args": {"x": "hi"}, "type": "tool_call"}
    )  # pyright: ignore[reportUnknownMemberType]
    return Outcome(ran=world.probe.ran)


@surface("langchain.arun_direct")
async def arun_direct(world: World) -> Outcome:
    """AgentExecutor's path: ``tool.arun`` with an explicit tool_call_id."""
    await world.announce_tool_call()
    tool = _async_tool(world)
    await tool.arun({"x": "hi"}, tool_call_id=TOOL_CALL_ID)  # pyright: ignore[reportUnknownMemberType]
    return Outcome(ran=world.probe.ran)


@surface("langchain.invoke_on_event_loop")
async def invoke_on_event_loop(world: World) -> Outcome:
    """A *sync* ``tool.invoke`` called straight from async code.

    The gate cannot block the event-loop thread without deadlocking the
    WS connection it is waiting on, so it steps aside. See GAPS.md.
    """
    await world.announce_tool_call()
    tool = _sync_tool(world)
    tool.invoke(
        {"id": TOOL_CALL_ID, "name": "probe", "args": {"x": "hi"}, "type": "tool_call"}
    )  # pyright: ignore[reportUnknownMemberType]
    return Outcome(ran=world.probe.ran)


@surface("langchain.invoke_plain_args")
async def invoke_plain_args(world: World) -> Outcome:
    """``tool.invoke({"x": "hi"})`` -- arguments, not a ToolCall.

    LangChain accepts both shapes. This one carries no id, so there is
    no key to look the verdict up by. See GAPS.md.
    """
    await world.announce_tool_call()
    tool = _sync_tool(world)
    await tool.ainvoke({"x": "hi"})  # pyright: ignore[reportUnknownMemberType]
    return Outcome(ran=world.probe.ran)
