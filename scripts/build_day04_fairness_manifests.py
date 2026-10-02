#!/usr/bin/env python3
"""
Build real E9 fairness manifests from completed Day-04 experiment runs.

Scientific purpose
------------------
This script does NOT invent candidate configs and does NOT write a fairness
result. It only creates manifest JSON files that point to the config.yaml files
actually produced by completed Day-04 runs.

Expected completed-run layout
-----------------------------
By default:

    outputs/day04/{category}/{candidate}/
        config.yaml
        best.pt
        train_log.csv
        predictions/manifest.jsonl

Each config.yaml must contain:

    scientific_config:
      adapter:
        bottleneck_dim: <r>
        projection_dim: <d>
        ...

The manifest then audits only that scientific_config subtree and allows exactly:

    adapter.bottleneck_dim
    adapter.projection_dim

to vary.

Recommended usage
-----------------
Run AFTER the full screen has completed:

    python scripts/build_day04_fairness_manifests.py \
        --root outputs/day04 \
        --categories fabric,vial,wallplugs

Outputs:

    configs/day04_fairness.fabric.json
    configs/day04_fairness.vial.json
    configs/day04_fairness.wallplugs.json

These files are inputs to tests/test_day04_fairness.py. They are not E9 PASS
reports. E9 writes config_diff.json separately.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml


MANIFEST_SCHEMA_VERSION = "msila.e9.fairness_manifest.v1"

REQUIRED_RUN_ARTIFACTS = (
    "config.yaml",
    "best.pt",
    "train_log.csv",
    "predictions/manifest.jsonl",
)

R_PATH = "adapter.bottleneck_dim"
D_PATH = "adapter.projection_dim"
CONFIG_ROOT = "scientific_config"


class ManifestBuildError(RuntimeError):
    """Raised when completed-run evidence is insufficient or inconsistent."""


def _project_root() -> Path:
    # This script is intended to live at scripts/build_day04_fairness_manifests.py
    return Path(__file__).resolve().parents[1]


def _load_yaml(path: Path) -> Mapping[str, Any]:
    if not path.is_file():
        raise ManifestBuildError(f"Missing YAML file: {path}")

    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ManifestBuildError(f"{path} must contain a YAML mapping")

    return dict(payload)


def _get_dotted(mapping: Mapping[str, Any], path: str) -> Any:
    current: Any = mapping
    for part in path.split("."):
        if not isinstance(current, Mapping) or part not in current:
            raise ManifestBuildError(
                f"Missing dotted path {path!r}"
            )
        current = current[part]
    return current


def _delete_dotted(mapping: Mapping[str, Any], path: str) -> dict[str, Any]:
    result = copy.deepcopy(dict(mapping))
    parts = path.split(".")
    current: Any = result

    for part in parts[:-1]:
        if not isinstance(current, dict) or part not in current:
            raise ManifestBuildError(
                f"Cannot remove missing path {path!r}"
            )
        current = current[part]

    leaf = parts[-1]
    if not isinstance(current, dict) or leaf not in current:
        raise ManifestBuildError(
            f"Cannot remove missing path {path!r}"
        )

    del current[leaf]
    return result


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _positive_int(value: Any, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ManifestBuildError(
            f"{field} must be a positive integer, got {value!r}"
        )
    return int(value)


def _parse_categories(text: str) -> tuple[str, ...]:
    values = tuple(
        item.strip()
        for item in str(text).split(",")
        if item.strip()
    )
    if not values:
        raise ManifestBuildError("At least one category is required")
    if len(set(values)) != len(values):
        raise ManifestBuildError(
            f"Duplicate categories are not allowed: {values}"
        )
    return values


def _load_locked_grid(
    grid_path: Path,
) -> tuple[tuple[str, int, int], ...]:
    grid = _load_yaml(grid_path)

    candidates = grid.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        raise ManifestBuildError(
            "Grid must contain a non-empty candidates list"
        )

    rows: list[tuple[str, int, int]] = []
    seen_names: set[str] = set()
    seen_pairs: set[tuple[int, int]] = set()

    for index, raw in enumerate(candidates):
        if not isinstance(raw, Mapping):
            raise ManifestBuildError(
                f"grid.candidates[{index}] must be a mapping"
            )

        r = _positive_int(
            raw.get("bottleneck_dim"),
            field=f"grid.candidates[{index}].bottleneck_dim",
        )
        d = _positive_int(
            raw.get("projection_dim"),
            field=f"grid.candidates[{index}].projection_dim",
        )

        expected_name = f"adapter_r{r}_d{d}"
        run_name = str(
            raw.get("run_name", expected_name)
        )

        if run_name != expected_name:
            raise ManifestBuildError(
                f"grid candidate name drift: {run_name!r}; "
                f"expected {expected_name!r}"
            )

        if run_name in seen_names:
            raise ManifestBuildError(
                f"Duplicate candidate run_name: {run_name}"
            )
        if (r, d) in seen_pairs:
            raise ManifestBuildError(
                f"Duplicate candidate pair: r={r}, d={d}"
            )

        seen_names.add(run_name)
        seen_pairs.add((r, d))
        rows.append((run_name, r, d))

    expected = grid.get("expected_num_candidates")
    if expected is not None:
        expected = _positive_int(
            expected,
            field="grid.expected_num_candidates",
        )
        if expected != len(rows):
            raise ManifestBuildError(
                "Grid size mismatch: "
                f"expected_num_candidates={expected}, actual={len(rows)}"
            )

    return tuple(rows)


def _require_complete_run(run_dir: Path) -> None:
    missing = [
        rel
        for rel in REQUIRED_RUN_ARTIFACTS
        if not (run_dir / rel).is_file()
        or (run_dir / rel).stat().st_size <= 0
    ]

    if missing:
        raise ManifestBuildError(
            f"Incomplete Day-04 run: {run_dir}; "
            f"missing/empty artifacts={missing}"
        )


def _relative_to_project(
    path: Path,
    project_root: Path,
) -> str:
    path = path.resolve()
    root = project_root.resolve()

    try:
        relative = path.relative_to(root)
    except ValueError as exc:
        raise ManifestBuildError(
            "E9 currently requires project-relative candidate config paths. "
            f"Path is outside project root: {path}"
        ) from exc

    return relative.as_posix()


def _validate_one_run_config(
    *,
    config_path: Path,
    candidate_name: str,
    expected_r: int,
    expected_d: int,
    category: str,
) -> Mapping[str, Any]:
    payload = _load_yaml(config_path)

    scientific = payload.get(CONFIG_ROOT)
    if not isinstance(scientific, Mapping):
        raise ManifestBuildError(
            f"{config_path}: missing mapping {CONFIG_ROOT!r}. "
            "screen_adapter.py must save scientific_config before E9."
        )

    r = _positive_int(
        _get_dotted(scientific, R_PATH),
        field=f"{candidate_name}:{R_PATH}",
    )
    d = _positive_int(
        _get_dotted(scientific, D_PATH),
        field=f"{candidate_name}:{D_PATH}",
    )

    if r != expected_r or d != expected_d:
        raise ManifestBuildError(
            f"{candidate_name}: config contains (r={r}, d={d}) but "
            f"locked grid expects (r={expected_r}, d={expected_d})"
        )

    # Optional provenance checks against top-level runner metadata.
    candidate_meta = payload.get("candidate")
    if isinstance(candidate_meta, Mapping):
        if "run_name" in candidate_meta:
            actual_name = str(candidate_meta["run_name"])
            if actual_name != candidate_name:
                raise ManifestBuildError(
                    f"{config_path}: candidate.run_name={actual_name!r}, "
                    f"expected {candidate_name!r}"
                )

        if "r" in candidate_meta and int(candidate_meta["r"]) != expected_r:
            raise ManifestBuildError(
                f"{config_path}: top-level candidate.r disagrees with grid"
            )

        if "d" in candidate_meta and int(candidate_meta["d"]) != expected_d:
            raise ManifestBuildError(
                f"{config_path}: top-level candidate.d disagrees with grid"
            )

    if "category" in payload and str(payload["category"]) != category:
        raise ManifestBuildError(
            f"{config_path}: category={payload['category']!r}, "
            f"expected {category!r}"
        )

    return dict(scientific)


def _preaudit_scientific_configs(
    configs: Mapping[str, Mapping[str, Any]],
) -> str:
    """Ensure all scientific configs become identical after removing r,d.

    This is only a manifest-generation preflight. tests/test_day04_fairness.py
    remains the authoritative E9 audit and writes the actual config_diff report.
    """

    hashes: dict[str, str] = {}

    for candidate, config in configs.items():
        controlled = _delete_dotted(config, R_PATH)
        controlled = _delete_dotted(controlled, D_PATH)
        hashes[candidate] = _canonical_sha256(controlled)

    unique = set(hashes.values())
    if len(unique) != 1:
        details = ", ".join(
            f"{name}={digest[:12]}"
            for name, digest in sorted(hashes.items())
        )
        raise ManifestBuildError(
            "Scientific configs are already non-comparable before E9: "
            "controlled hashes differ after removing r,d. "
            f"{details}"
        )

    return next(iter(unique))


def _atomic_json_dump(
    payload: Mapping[str, Any],
    path: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")

    with tmp.open("w", encoding="utf-8") as file:
        json.dump(
            payload,
            file,
            ensure_ascii=False,
            indent=2,
            sort_keys=False,
            allow_nan=False,
        )
        file.write("\n")

    os.replace(tmp, path)


def build_manifests(
    *,
    project_root: Path,
    root: Path,
    grid_path: Path,
    output_dir: Path,
    categories: Sequence[str],
) -> list[Path]:
    candidates = _load_locked_grid(grid_path)
    baseline_candidate = candidates[0][0]

    written: list[Path] = []

    for category in categories:
        manifest_candidates: dict[str, str] = {}
        scientific_configs: dict[str, Mapping[str, Any]] = {}

        for candidate_name, r, d in candidates:
            run_dir = root / category / candidate_name
            _require_complete_run(run_dir)

            config_path = run_dir / "config.yaml"

            scientific_configs[candidate_name] = (
                _validate_one_run_config(
                    config_path=config_path,
                    candidate_name=candidate_name,
                    expected_r=r,
                    expected_d=d,
                    category=category,
                )
            )

            manifest_candidates[candidate_name] = (
                _relative_to_project(
                    config_path,
                    project_root,
                )
            )

        controlled_hash = _preaudit_scientific_configs(
            scientific_configs
        )

        manifest = {
            "schema_version": MANIFEST_SCHEMA_VERSION,
            "baseline_candidate": baseline_candidate,
            "config_root": CONFIG_ROOT,
            "allowed_differences": {
                "r": R_PATH,
                "d": D_PATH,
            },
            "candidates": manifest_candidates,
        }

        output_path = (
            output_dir
            / f"day04_fairness.{category}.json"
        )

        _atomic_json_dump(
            manifest,
            output_path,
        )

        print(
            "[E9 MANIFEST] "
            f"category={category} "
            f"candidates={len(candidates)} "
            f"controlled_sha256={controlled_hash} "
            f"output={output_path}"
        )

        written.append(output_path)

    return written


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Generate E9 fairness manifests from completed real Day-04 runs."
        )
    )

    parser.add_argument(
        "--root",
        type=Path,
        default=Path("outputs/day04"),
        help=(
            "Day-04 root containing <category>/<candidate>/... "
            "(default: outputs/day04)"
        ),
    )

    parser.add_argument(
        "--grid",
        type=Path,
        default=Path("configs/day04_adapter_grid.yaml"),
        help="Locked Day-04 Adapter grid.",
    )

    parser.add_argument(
        "--categories",
        type=str,
        default="fabric,vial,wallplugs",
        help="Comma-separated pilot categories.",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("configs"),
        help="Directory for generated fairness manifests.",
    )

    return parser


def main() -> int:
    args = _build_parser().parse_args()

    project_root = _project_root()

    def absolute_from_project(path: Path) -> Path:
        if path.is_absolute():
            return path.resolve()
        return (project_root / path).resolve()

    root = absolute_from_project(args.root)
    grid_path = absolute_from_project(args.grid)
    output_dir = absolute_from_project(args.output_dir)

    categories = _parse_categories(args.categories)

    try:
        written = build_manifests(
            project_root=project_root,
            root=root,
            grid_path=grid_path,
            output_dir=output_dir,
            categories=categories,
        )
    except (
        ManifestBuildError,
        FileNotFoundError,
        yaml.YAMLError,
        json.JSONDecodeError,
    ) as exc:
        print(
            f"[E9 MANIFEST FAIL] {exc}",
            file=__import__("sys").stderr,
        )
        return 2

    print(
        f"[E9 MANIFEST PASS] generated={len(written)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
