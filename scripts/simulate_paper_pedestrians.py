#!/usr/bin/env python3
"""Explicit AMP + Diffusion deployment; delegates to the existing evaluator."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from CrowdSim.humanoid.application.deployment import main  # noqa: E402 -- source script bootstrap


if __name__ == "__main__":
    raise SystemExit(main())
