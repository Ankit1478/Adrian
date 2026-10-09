# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 SecureAgentics

"""Runs the matrix, one case at a time.

Sequential on purpose. ``adrian.init`` installs module-level
monkey-patches and a single global client, so two cases running at once
would share a mode and a verdict and decide each other's result. The
whole suite is a few seconds; parallelism would buy nothing and cost
the one property the eval exists to have.
"""

from __future__ import annotations

import asyncio
import logging

from .cases import Case, build_cases
from .harness import SURFACES, Outcome, world_for
from .report import Report, Result, score

#: A case that has not finished by now is hung, not slow. The longest
#: legitimate wait is the ``no_policy`` LoginAck wait, hard-coded to 5s
#: inside the SDK.
CASE_TIMEOUT = 20.0


async def run_case(case: Case) -> Result:
    """Run one case against a freshly initialised SDK."""
    surface = SURFACES.get(case.surface)
    if surface is None:
        return Result(
            case, Outcome(ran=True, error=f"unknown surface {case.surface!r}")
        )

    try:
        async with world_for(case) as world:
            outcome = await asyncio.wait_for(surface(world), timeout=CASE_TIMEOUT)
    except TimeoutError:
        # Reported as an error, not a block: a hang is not enforcement,
        # and calling it one would let a deadlock score as a success.
        return Result(case, Outcome(ran=False, error=f"hung for {CASE_TIMEOUT}s"))
    except Exception as exc:  # noqa: BLE001
        return Result(case, Outcome(ran=False, error=f"{type(exc).__name__}: {exc}"))

    return Result(case, outcome)


async def run_all(cases: list[Case] | None = None) -> Report:
    """Run the whole matrix and score it."""
    load_surfaces()
    results = [await run_case(case) for case in (cases or build_cases())]
    return score(results)


def load_surfaces() -> None:
    """Import the surface modules, registering their call paths.

    The Anthropic stub has to go in before the first ``adrian.init``,
    so it is installed here rather than lazily per case.
    """
    from . import surfaces_anthropic, surfaces_langchain

    # Importing is the registration; naming the modules keeps that
    # visible to a reader and to the linters.
    assert surfaces_langchain.__name__
    surfaces_anthropic.install_model_stub()


def quieten() -> None:
    """Silence the SDK's own logging.

    The eval drives failure paths deliberately -- timeouts, missing
    LoginAcks, a WebSocket URL that will never connect -- so the SDK
    logs warnings for every one of them. They are expected here and
    would bury the report.
    """
    logging.getLogger("adrian").setLevel(logging.CRITICAL)
    for name in ("websockets", "httpx"):
        logging.getLogger(name).setLevel(logging.CRITICAL)
