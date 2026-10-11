#!/usr/bin/env python3
"""Deprecated D6 filename: dispatch only technical preflight, never screening."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.g2_adapter_preflight import main

if __name__ == "__main__":
    raise SystemExit(main())
