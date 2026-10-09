# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 SecureAgentics

"""``python -m evals`` -- run the enforcement matrix and gate on it."""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

from .cases import build_cases
from .report import render, to_json
from .runner import quieten, run_all


def main() -> int:
    """Run the matrix, print the report, and return the exit code."""
    parser = argparse.ArgumentParser(
        prog="python -m evals",
        description="Check that the SDK enforces the verdicts it is given.",
    )
    parser.add_argument(
        "-surface",
        default="",
        help="only run call paths whose name contains this (e.g. 'anthropic')",
    )
    parser.add_argument(
        "-out",
        default="",
        help="write the JSON report here",
    )
    parser.add_argument(
        "-quiet",
        action="store_true",
        help="print only the verdict line",
    )
    args = parser.parse_args()

    quieten()
    cases = [c for c in build_cases() if args.surface in c.surface]
    if not cases:
        print(f"no call path matches {args.surface!r}", file=sys.stderr)
        return 2

    report = asyncio.run(run_all(cases))

    if args.out:
        path = Path(args.out)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(to_json(report))

    print(render(report) if not args.quiet else ("PASS" if report.ok else "FAIL"))
    return 0 if report.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
