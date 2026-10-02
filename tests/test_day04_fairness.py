"""
MS-ILA E9 — Day-04 configuration fairness audit.

Scientific question
-------------------
Are Day-04 adapter candidates identical in every controlled setting except:

    r = adapter reduction ratio
    d = fusion / projection dimension

The exact config paths for r and d are declared in a fairness manifest.

PASS
----
For every candidate config:
    1. r and d exist and are positive integers;
    2. the (r, d) pair is unique;
    3. recursive config differences are a subset of {r_path, d_path};
    4. after removing r and d, all canonical config hashes are identical.

Output
------
A machine-readable config diff report is written whenever the audit reaches
comparison stage.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import pytest


MANIFEST_SCHEMA_VERSION = "msila.e9.fairness_manifest.v1"
REPORT_SCHEMA_VERSION = "msila.e9.config_diff.v1"

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = PROJECT_ROOT / "configs" / "day04_fairness.json"
DEFAULT_REPORT = PROJECT_ROOT / "outputs" / "day04" / "fairness" / "config_diff.json"

_MISSING = object()


class FairnessAuditError(AssertionError):
    """Raised when Day-04 candidates violate the locked fairness contract."""


@dataclass(frozen=True)
class FairnessManifest:
    baseline_candidate: str
    config_root: str | None
    r_path: str
    d_path: str
    candidates: Mapping[str, Path]
    raw: Mapping[str, Any]
    sha256: str


def _canonicalize(value: Any) -> Any:
    """Type-preserving canonical representation for exact comparison."""
    if value is None:
        return {"__type__": "null", "value": None}
    if isinstance(value, bool):
        return {"__type__": "bool", "value": value}
    if isinstance(value, int):
        return {"__type__": "int", "value": value}
    if isinstance(value, float):
        if not math.isfinite(value):
            raise FairnessAuditError("Config contains NaN/Inf")
        return {"__type__": "float", "value": value}
    if isinstance(value, str):
        return {"__type__": "str", "value": value}
    if isinstance(value, Mapping):
        out: dict[str, Any] = {}
        for key in sorted(value.keys()):
            if not isinstance(key, str):
                raise FairnessAuditError(
                    f"Config mapping keys must be strings, got {type(key)!r}"
                )
            out[key] = _canonicalize(value[key])
        return {"__type__": "mapping", "value": out}
    if isinstance(value, (list, tuple)):
        return {
            "__type__": "sequence",
            "value": [_canonicalize(v) for v in value],
        }
    raise FairnessAuditError(
        f"Unsupported config value type {type(value)!r}"
    )


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        _canonicalize(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _nonempty_string(value: Any, *, field: str) -> str:
    value = str(value).strip()
    if not value:
        raise FairnessAuditError(f"{field} must be a non-empty string")
    return value


def _safe_relative_project_path(value: Any, *, field: str) -> Path:
    raw = _nonempty_string(value, field=field)
    path = Path(raw)
    if path.is_absolute() or ".." in path.parts:
        raise FairnessAuditError(
            f"{field} must be a safe project-relative path, got {raw!r}"
        )
    resolved = (PROJECT_ROOT / path).resolve()
    try:
        resolved.relative_to(PROJECT_ROOT.resolve())
    except ValueError as exc:
        raise FairnessAuditError(f"{field} escapes project root") from exc
    return resolved


def _validate_dotted_path(path: Any, *, field: str) -> str:
    path = _nonempty_string(path, field=field)
    parts = path.split(".")
    if any(not part for part in parts):
        raise FairnessAuditError(
            f"{field} must be a dotted mapping path without empty components"
        )
    if any("[" in part or "]" in part for part in parts):
        raise FairnessAuditError(
            f"{field} must address mapping keys only"
        )
    return path


def _load_yaml_or_json(path: Path) -> Mapping[str, Any]:
    if not path.is_file():
        raise FairnessAuditError(f"Candidate config does not exist: {path}")

    suffix = path.suffix.lower()
    if suffix == ".json":
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise FairnessAuditError(f"Cannot parse JSON config {path}") from exc
    elif suffix in {".yaml", ".yml"}:
        try:
            import yaml  # type: ignore
        except ImportError as exc:
            raise FairnessAuditError(
                f"PyYAML is required to read {path.name}"
            ) from exc
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    else:
        raise FairnessAuditError(
            f"Unsupported config extension {path.suffix!r}; use JSON/YAML"
        )

    if not isinstance(payload, Mapping):
        raise FairnessAuditError(
            f"Candidate config root must be a mapping: {path}"
        )

    _canonicalize(payload)
    return dict(payload)


def load_manifest(path: str | Path) -> FairnessManifest:
    path = Path(path).resolve()
    if not path.is_file():
        raise FileNotFoundError(path)

    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise FairnessAuditError(f"Invalid fairness manifest JSON: {path}") from exc

    if not isinstance(raw, Mapping):
        raise FairnessAuditError("Fairness manifest root must be an object")
    if raw.get("schema_version") != MANIFEST_SCHEMA_VERSION:
        raise FairnessAuditError(
            f"manifest.schema_version must be {MANIFEST_SCHEMA_VERSION!r}"
        )

    baseline = _nonempty_string(
        raw.get("baseline_candidate"),
        field="baseline_candidate",
    )

    config_root_raw = raw.get("config_root")
    config_root = (
        None
        if config_root_raw is None
        else _validate_dotted_path(config_root_raw, field="config_root")
    )

    allowed = raw.get("allowed_differences")
    if not isinstance(allowed, Mapping):
        raise FairnessAuditError("allowed_differences must be an object")
    if set(allowed.keys()) != {"r", "d"}:
        raise FairnessAuditError(
            "allowed_differences must contain exactly {'r','d'}"
        )

    r_path = _validate_dotted_path(
        allowed["r"], field="allowed_differences.r"
    )
    d_path = _validate_dotted_path(
        allowed["d"], field="allowed_differences.d"
    )
    if r_path == d_path:
        raise FairnessAuditError("r and d must map to different config paths")

    candidates_raw = raw.get("candidates")
    if not isinstance(candidates_raw, Mapping) or len(candidates_raw) < 2:
        raise FairnessAuditError(
            "candidates must map at least two IDs to config files"
        )

    candidates: dict[str, Path] = {}
    for candidate_id, config_path in candidates_raw.items():
        cid = _nonempty_string(candidate_id, field="candidate ID")
        candidates[cid] = _safe_relative_project_path(
            config_path,
            field=f"candidates.{cid}",
        )

    if baseline not in candidates:
        raise FairnessAuditError(
            f"baseline_candidate={baseline!r} not listed in candidates"
        )

    return FairnessManifest(
        baseline_candidate=baseline,
        config_root=config_root,
        r_path=r_path,
        d_path=d_path,
        candidates=candidates,
        raw=dict(raw),
        sha256=_canonical_sha256(raw),
    )


def _get_dotted(mapping: Mapping[str, Any], path: str) -> Any:
    current: Any = mapping
    for part in path.split("."):
        if not isinstance(current, Mapping) or part not in current:
            return _MISSING
        current = current[part]
    return current


def _require_subtree(
    mapping: Mapping[str, Any],
    config_root: str | None,
    *,
    candidate: str,
) -> Mapping[str, Any]:
    if config_root is None:
        return mapping

    value = _get_dotted(mapping, config_root)
    if value is _MISSING:
        raise FairnessAuditError(
            f"{candidate}: config_root {config_root!r} is missing"
        )
    if not isinstance(value, Mapping):
        raise FairnessAuditError(
            f"{candidate}: config_root {config_root!r} must resolve to mapping"
        )
    return dict(value)


def _positive_int_factor(
    value: Any,
    *,
    semantic_name: str,
    candidate: str,
) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise FairnessAuditError(
            f"{candidate}: {semantic_name} must be positive int, got {value!r}"
        )
    return int(value)


def _render_value(value: Any) -> Any:
    if value is _MISSING:
        return {"__missing__": True}
    return value


def _equal_exact(a: Any, b: Any) -> bool:
    if a is _MISSING or b is _MISSING:
        return a is b
    return _canonicalize(a) == _canonicalize(b)


def _recursive_diff(
    a: Any,
    b: Any,
    *,
    path: str = "",
) -> list[dict[str, Any]]:
    diffs: list[dict[str, Any]] = []

    if isinstance(a, Mapping) and isinstance(b, Mapping):
        keys = sorted(set(a.keys()) | set(b.keys()))
        for key in keys:
            if not isinstance(key, str):
                raise FairnessAuditError("Config mapping keys must be strings")
            child = f"{path}.{key}" if path else key
            av = a.get(key, _MISSING)
            bv = b.get(key, _MISSING)
            if av is _MISSING or bv is _MISSING:
                diffs.append(
                    {
                        "path": child,
                        "reference": _render_value(av),
                        "candidate": _render_value(bv),
                        "kind": "missing_key",
                    }
                )
            else:
                diffs.extend(_recursive_diff(av, bv, path=child))
        return diffs

    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        max_len = max(len(a), len(b))
        for i in range(max_len):
            child = f"{path}[{i}]"
            av = a[i] if i < len(a) else _MISSING
            bv = b[i] if i < len(b) else _MISSING
            if av is _MISSING or bv is _MISSING:
                diffs.append(
                    {
                        "path": child,
                        "reference": _render_value(av),
                        "candidate": _render_value(bv),
                        "kind": "sequence_length",
                    }
                )
            else:
                diffs.extend(_recursive_diff(av, bv, path=child))
        return diffs

    if not _equal_exact(a, b):
        diffs.append(
            {
                "path": path or "<root>",
                "reference": _render_value(a),
                "candidate": _render_value(b),
                "kind": "value_or_type",
            }
        )

    return diffs


def _delete_dotted(mapping: Mapping[str, Any], path: str) -> dict[str, Any]:
    result = copy.deepcopy(dict(mapping))
    parts = path.split(".")
    current: Any = result

    for part in parts[:-1]:
        if not isinstance(current, dict) or part not in current:
            raise FairnessAuditError(
                f"Cannot remove missing allowed path {path!r}"
            )
        current = current[part]

    leaf = parts[-1]
    if not isinstance(current, dict) or leaf not in current:
        raise FairnessAuditError(
            f"Cannot remove missing allowed path {path!r}"
        )
    del current[leaf]
    return result


def _controlled_hash(
    config: Mapping[str, Any],
    *,
    r_path: str,
    d_path: str,
) -> str:
    stripped = _delete_dotted(config, r_path)
    stripped = _delete_dotted(stripped, d_path)
    return _canonical_sha256(stripped)


def audit_configs(manifest: FairnessManifest) -> dict[str, Any]:
    configs: dict[str, Mapping[str, Any]] = {}
    config_hashes: dict[str, str] = {}
    factor_values: dict[str, dict[str, int]] = {}

    for candidate, path in manifest.candidates.items():
        raw_config = _load_yaml_or_json(path)
        config = _require_subtree(
            raw_config,
            manifest.config_root,
            candidate=candidate,
        )

        r_value = _get_dotted(config, manifest.r_path)
        d_value = _get_dotted(config, manifest.d_path)

        if r_value is _MISSING:
            raise FairnessAuditError(
                f"{candidate}: missing r path {manifest.r_path!r}"
            )
        if d_value is _MISSING:
            raise FairnessAuditError(
                f"{candidate}: missing d path {manifest.d_path!r}"
            )

        r = _positive_int_factor(
            r_value, semantic_name="r", candidate=candidate
        )
        d = _positive_int_factor(
            d_value, semantic_name="d", candidate=candidate
        )

        configs[candidate] = config
        config_hashes[candidate] = _canonical_sha256(config)
        factor_values[candidate] = {"r": r, "d": d}

    pair_to_candidates: dict[tuple[int, int], list[str]] = {}
    for candidate, factors in factor_values.items():
        pair = (factors["r"], factors["d"])
        pair_to_candidates.setdefault(pair, []).append(candidate)

    duplicate_pairs = {
        f"r={pair[0]},d={pair[1]}": sorted(ids)
        for pair, ids in pair_to_candidates.items()
        if len(ids) > 1
    }

    baseline = manifest.baseline_candidate
    reference = configs[baseline]
    allowed = {manifest.r_path, manifest.d_path}

    comparisons: dict[str, Any] = {}
    all_unexpected: list[dict[str, Any]] = []

    for candidate in manifest.candidates:
        if candidate == baseline:
            continue

        diffs = _recursive_diff(reference, configs[candidate])
        allowed_diffs = [
            diff for diff in diffs if diff["path"] in allowed
        ]
        unexpected_diffs = [
            diff for diff in diffs if diff["path"] not in allowed
        ]

        comparisons[candidate] = {
            "against": baseline,
            "all_differences": diffs,
            "allowed_differences": allowed_diffs,
            "unexpected_differences": unexpected_diffs,
            "status": "PASS" if not unexpected_diffs else "FAIL",
        }

        for diff in unexpected_diffs:
            all_unexpected.append(
                {"candidate_id": candidate, **diff}
            )

    controlled_hashes = {
        candidate: _controlled_hash(
            config,
            r_path=manifest.r_path,
            d_path=manifest.d_path,
        )
        for candidate, config in configs.items()
    }
    controlled_hash_match = len(set(controlled_hashes.values())) == 1

    status = (
        "PASS"
        if (
            not all_unexpected
            and not duplicate_pairs
            and controlled_hash_match
        )
        else "FAIL"
    )

    return {
        "schema_version": REPORT_SCHEMA_VERSION,
        "status": status,
        "manifest_sha256": manifest.sha256,
        "baseline_candidate": baseline,
        "config_root": manifest.config_root,
        "allowed_differences": {
            "r": manifest.r_path,
            "d": manifest.d_path,
        },
        "candidate_factors": factor_values,
        "candidate_config_sha256": config_hashes,
        "controlled_config_sha256_after_removing_r_d": controlled_hashes,
        "controlled_hashes_identical": controlled_hash_match,
        "duplicate_factor_pairs": duplicate_pairs,
        "comparisons": comparisons,
        "unexpected_differences": all_unexpected,
        "pass_rule": (
            "All exact recursive differences must be a subset of "
            "{r_path,d_path}; controlled hashes after removing r,d must match; "
            "all (r,d) pairs must be unique."
        ),
    }


def _atomic_json_dump(payload: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(
            payload,
            f,
            ensure_ascii=False,
            indent=2,
            sort_keys=False,
            allow_nan=False,
        )
        f.write("\n")
    os.replace(tmp, path)


def run_audit(
    *,
    manifest_path: str | Path,
    report_path: str | Path,
) -> dict[str, Any]:
    manifest = load_manifest(manifest_path)
    report = audit_configs(manifest)
    _atomic_json_dump(report, Path(report_path))

    if report["status"] != "PASS":
        unexpected_paths = sorted(
            {
                f"{item['candidate_id']}:{item['path']}"
                for item in report["unexpected_differences"]
            }
        )

        problems: list[str] = []
        if unexpected_paths:
            problems.append(
                "unexpected config differences="
                + ", ".join(unexpected_paths)
            )
        if report["duplicate_factor_pairs"]:
            problems.append(
                "duplicate (r,d) pairs="
                + json.dumps(
                    report["duplicate_factor_pairs"],
                    sort_keys=True,
                )
            )
        if not report["controlled_hashes_identical"]:
            problems.append("controlled hashes are not identical")

        raise FairnessAuditError(
            "E9 fairness audit FAILED: " + "; ".join(problems)
        )

    return report


def _resolve_pytest_manifest() -> Path | None:
    env = os.environ.get("DAY04_FAIRNESS_MANIFEST")
    if env:
        path = Path(env)
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        return path.resolve()

    if DEFAULT_MANIFEST.is_file():
        return DEFAULT_MANIFEST.resolve()

    return None


def _resolve_pytest_report() -> Path:
    env = os.environ.get("DAY04_FAIRNESS_REPORT")
    if env:
        path = Path(env)
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        return path.resolve()
    return DEFAULT_REPORT.resolve()


def test_day04_candidate_configs_differ_only_in_r_and_d() -> None:
    manifest_path = _resolve_pytest_manifest()
    if manifest_path is None:
        pytest.skip(
            "E9 fairness manifest not configured. "
            "Set DAY04_FAIRNESS_MANIFEST or create configs/day04_fairness.json. "
            "SKIP is not an E9 scientific PASS."
        )

    report = run_audit(
        manifest_path=manifest_path,
        report_path=_resolve_pytest_report(),
    )

    assert report["status"] == "PASS"
    assert report["controlled_hashes_identical"] is True
    assert report["unexpected_differences"] == []
    assert report["duplicate_factor_pairs"] == {}


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "MS-ILA E9 fairness audit: candidate configs may differ only in "
            "the manifest-declared r and d paths."
        )
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    try:
        report = run_audit(
            manifest_path=args.manifest,
            report_path=args.report,
        )
    except (FairnessAuditError, FileNotFoundError) as exc:
        print(f"[E9 FAIL] {exc}", file=sys.stderr)
        return 2

    print("[E9 PASS]")
    print(f"baseline={report['baseline_candidate']}")
    print(f"r_path={report['allowed_differences']['r']}")
    print(f"d_path={report['allowed_differences']['d']}")
    print(f"manifest_sha256={report['manifest_sha256']}")
    print(f"report={args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
