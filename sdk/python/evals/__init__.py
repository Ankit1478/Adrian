# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 SecureAgentics

"""Enforcement eval for the Adrian SDK.

The judge eval (``backend/cmd/adrian-eval``) measures whether the
classifier reaches the right verdict. This one measures the step after:
given a verdict, does the SDK actually stop the tool?

Those are different failure modes. A perfect judge is worth nothing if a
BLOCK verdict arrives while the tool is already running, and a blocking
SDK is worth nothing if it blocks the wrong things. The two evals are
deliberately separate so a regression in either is attributable.

See ``README.md`` for how to run it and ``GAPS.md`` for the call paths
that are known not to be gated.
"""
