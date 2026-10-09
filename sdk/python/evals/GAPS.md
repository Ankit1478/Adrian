# Known enforcement gaps

Four call paths reach a tool without passing the verdict gate. Each one
has a row in the matrix that still says `expected: blocked`, marked with
the `known_gap` key named below. They are counted as failures in the
report and tolerated by the gate only while they stay listed here. Fix
one and the case starts passing, the gate fails with `GAP NOW PASSES`,
and the marker has to come out — so a fix cannot be quietly lost.

**All four are still audited.** The eval confirms an event is recorded
for every one of them (`events recorded: 1`, against `0` for a path
that blocks successfully). The agent's behaviour is visible in the
dashboard and scored by the judge; what is missing is the stop. So in
`alert` mode — the default — these paths behave exactly like every
other. They matter to a customer who has turned `block` or `hitl` on and
assumes it covers everything.

---

## `sync-invoke-on-event-loop`

**Path:** a synchronous tool invoked with `tool.invoke(...)` directly
from `async` application code.

`langchain_handler._sync_gate` detects it is running on the event-loop
thread and returns without gating. That is not an oversight: the gate
has to wait for a verdict that arrives over the WebSocket, and that
socket is served by the very loop it would be blocking. Waiting there
deadlocks the connection the verdict has to come through, so the tool
would hang for ever instead of being blocked.

**Reach:** application code only. LangGraph's `ToolNode` dispatches a
sync tool through `run_in_executor`, landing the gate on a worker
thread where it *can* block — that path is covered and enforced
(`langchain.toolnode_sync`).

**Avoiding it:** `await tool.ainvoke(...)`, which routes to the async
gate and is enforced.

**A fix would have to** give the gate an answer without waiting on the
loop — for example refusing the call outright when policy is active and
the verdict is not already resolved. That trades a bypass for an
over-block, so it is a product decision rather than a bug fix, which is
why it is recorded here rather than patched.

---

## `invoke-without-tool-call-id`

**Path:** `tool.invoke({"x": "hi"})` — the tool's arguments, rather than
a LangChain `ToolCall` dict.

LangChain accepts both shapes. `_extract_tool_call_id` returns `None`
for the bare-argument one, and with no id there is no key to look a
verdict up by, so the gate is skipped. Both `invoke` and `ainvoke` are
affected; this is not the event-loop problem above.

**Reach:** hand-written orchestration. An agent loop built on `ToolNode`
or `AgentExecutor` always passes a `ToolCall`.

**A fix would have to** decide what to do with a tool call that has no
id while policy is active: fall back to the pending verdict for the
current run, or fail closed. Failing closed is the safe default and
would break any integration that legitimately calls tools this way, so
again it needs a decision, not just a patch.

---

## `stream-raw-event-iteration`

**Path:** `client.messages.stream(...)` consumed event by event, acting
on `content_block_stop` rather than calling `get_final_message()`.

The gate is attached to the stream's terminal methods
(`get_final_message`, `until_done`, and `get_final_text` which delegates
to them). A consumer that reads raw events has the tool call in hand
before any terminal method runs. This limit is already written down in
`adrian/anthropic_handler.py`'s module docstring; the eval turns it
from a comment into four failing rows.

**A fix would have to** gate during iteration, which means holding each
`tool_use` block until its verdict arrives — real work, and the reason
the docstring calls it a follow-up change.

---

## `create-stream-true`

**Path:** `client.messages.create(..., stream=True)`.

This returns an iterator, not a `Message`. The gate reads
`response.content`, finds it is not a list, and hands the response back
untouched. Unlike the case above there is no terminal method to attach
to at all, so this path is not gated anywhere.

**A fix would have to** wrap the returned iterator the way
`messages.stream` is wrapped, and shares the hard part of
`stream-raw-event-iteration`: a streamed tool call has to be held, not
just rewritten after the fact.
