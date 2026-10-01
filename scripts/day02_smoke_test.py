#!/usr/bin/env python3
"""MS-ILA Day-2 smoke-test runner.

Runs all Day-2 TV2 gates in one pytest invocation and writes:
    day02_report.json

Usage:
    python scripts/day02_smoke_test.py

Optional:
    python scripts/day02_smoke_test.py --project-root /path/to/repo
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


DEFAULT_TESTS = (
    "tests/test_day02_contracts.py",
    "tests/test_attention_fusion.py",
    "tests/test_msila_day02_head.py",
    "tests/test_gradient_flow.py",
    "tests/test_numerical_stability.py",
    "tests/test_multiview_forward.py",
    "tests/test_tv1_integration.py",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run all MS-ILA Day-2 architecture QA gates."
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=None,
        help=(
            "Repository root. Default: parent directory of scripts/."
        ),
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=None,
        help=(
            "JSON report path. Default: <project-root>/day02_report.json"
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if args.project_root is None:
        project_root = Path(__file__).resolve().parents[1]
    else:
        project_root = args.project_root.resolve()

    report_path = (
        args.report.resolve()
        if args.report is not None
        else project_root / "day02_report.json"
    )

    test_paths = [
        project_root / rel
        for rel in DEFAULT_TESTS
    ]

    missing = [
        str(path.relative_to(project_root))
        for path in test_paths
        if not path.is_file()
    ]

    started = datetime.now(
        timezone.utc
    ).isoformat()

    print("=" * 64)
    print("MS-ILA — DAY 02 ARCHITECTURE QA")
    print("=" * 64)

    if missing:
        print("\nPRE-FLIGHT: FAIL")
        print("Missing required test files:")
        for rel in missing:
            print(f"  - {rel}")

        report = {
            "day": 2,
            "status": "FAIL",
            "reason": "missing_test_files",
            "started_at_utc": started,
            "tests": list(DEFAULT_TESTS),
            "missing": missing,
        }

        report_path.write_text(
            json.dumps(
                report,
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

        print(f"\nReport: {report_path}")
        print("\nDAY 02: FAIL")
        return 2

    command = [
        sys.executable,
        "-m",
        "pytest",
        "-v",
        *DEFAULT_TESTS,
    ]

    print("\nRunning one pytest command:")
    print(" ".join(command))
    print()

    result = subprocess.run(
        command,
        cwd=project_root,
        text=True,
        capture_output=True,
    )

    # Preserve pytest output in terminal.
    if result.stdout:
        print(result.stdout, end="")

    if result.stderr:
        print(
            result.stderr,
            end="",
            file=sys.stderr,
        )

    finished = datetime.now(
        timezone.utc
    ).isoformat()

    status = (
        "PASS"
        if result.returncode == 0
        else "FAIL"
    )

    report = {
        "day": 2,
        "status": status,
        "returncode": result.returncode,
        "started_at_utc": started,
        "finished_at_utc": finished,
        "command": command,
        "tests": list(DEFAULT_TESTS),
        "stdout": result.stdout,
        "stderr": result.stderr,
    }

    report_path.write_text(
        json.dumps(
            report,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print()
    print("=" * 64)
    print(f"DAY 02: {status}")
    print(f"Report: {report_path}")
    print("=" * 64)

    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
