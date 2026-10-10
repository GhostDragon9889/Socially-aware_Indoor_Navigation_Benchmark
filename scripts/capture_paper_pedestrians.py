#!/usr/bin/env python3
"""Capture live textured Diffusion + AMP pedestrians, optionally with depth."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

if __name__ == "__main__":
    from CrowdSim.humanoid.application.capture import main

    sys.exit(main())
