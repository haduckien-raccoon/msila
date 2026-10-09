"""Audit native MVTec AD 2 image/GT identities without using TEST for selection."""
import argparse
import json
from pathlib import Path
from src.data.loader import audit_mvtec_ad2

if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root", required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    report = audit_mvtec_ad2(a.data_root)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"status": report["status"], "counts": report["counts"]}, indent=2))
    raise SystemExit(0 if report["status"] == "PASS" else 1)
