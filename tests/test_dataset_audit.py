import csv
import hashlib
import json

import numpy as np
import pytest
from PIL import Image

from src.data.loader import audit_mvtec_ad2, scan_mvtec_ad2


def image(root, name, *, mask=False, value=255):
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.full((7, 9) if mask else (7, 9, 3), value, dtype=np.uint8)).save(path)
    return path


def manifest(root, paths, output):
    with output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["relative_path", "file_size", "sha256", "split", "category"])
        writer.writeheader()
        for path in paths:
            relative = path.relative_to(root)
            writer.writerow(dict(relative_path=relative.as_posix(), file_size=path.stat().st_size,
                                 sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                                 split=relative.parts[1], category=relative.parts[0]))
    return output


def test_orphan_and_unrecognized_files_cannot_pass_inventory(tmp_path):
    image(tmp_path, "fabric/train/good/000.png")
    image(tmp_path, "fabric/test_public/ground_truth/bad/orphan_mask.png", mask=True)
    image(tmp_path, "fabric/unexpected/good/lost.png")
    image(tmp_path, "fabric/ground_truth/bad/unscoped_mask.png", mask=True)
    report = audit_mvtec_ad2(tmp_path)
    assert report["status"] == "FAIL"
    assert report["inventory"]["image_files"] == 2
    assert report["inventory"]["mask_files"] == 2
    assert report["inventory"]["orphan_gt"] == ["fabric/test_public/ground_truth/bad/orphan_mask.png"]
    assert {row["path"] for row in report["inventory"]["unrecognized"]} == {
        "fabric/unexpected/good/lost.png", "fabric/ground_truth/bad/unscoped_mask.png"}


def test_private_missing_gt_is_hidden_but_never_zero(tmp_path):
    image(tmp_path, "fabric/test_private/good/000.png")
    image(tmp_path, "fabric/test_private/bad/001.png")
    report = audit_mvtec_ad2(tmp_path)
    assert report["status"] == "PASS" and not report["errors"]
    assert report["pixel_evaluation"]["status"] == "PARTIAL"
    assert report["pixel_evaluation"]["eligible"] == 1
    abnormal = next(row for row in report["images"] if row["defect_type"] == "bad")
    assert abnormal["gt_status"] == "missing" and abnormal["gt_availability"] == "PRIVATE_HIDDEN"
    assert abnormal["mask"] is None and not abnormal["pixel_evaluation_eligible"]
    assert next(row for row in report["images"] if row["defect_type"] == "good")["pixel_evaluation_eligible"]
    assert next(row for row in scan_mvtec_ad2(tmp_path) if not row.is_normal).mask_path is None


def test_public_missing_gt_and_duplicate_gt_are_errors(tmp_path):
    image(tmp_path, "fabric/test_public/bad/000.png")
    assert audit_mvtec_ad2(tmp_path)["status"] == "FAIL"
    image(tmp_path, "fabric/test_public/ground_truth/bad/000.png", mask=True)
    image(tmp_path, "fabric/test_public/ground_truth/bad/000_mask.png", mask=True)
    report = audit_mvtec_ad2(tmp_path)
    assert len(report["inventory"]["duplicate_gt"]) == 1
    assert not report["images"][0]["pixel_evaluation_eligible"]


def test_explicit_category_and_split_coverage(tmp_path):
    image(tmp_path, "fabric/train/good/000.png")
    report = audit_mvtec_ad2(tmp_path, expected_categories=["fabric", "rice"],
                            expected_splits=["train", "validation"])
    assert report["status"] == "FAIL"
    assert report["coverage"]["missing_categories"] == ["rice"]
    assert report["coverage"]["missing_category_splits"] == ["fabric/validation", "rice/train", "rice/validation"]


def test_verified_manifest_detects_duplicate_raw_source_across_splits(tmp_path):
    root = tmp_path / "data"
    paths = [image(root, "fabric/train/good/000.png"), image(root, "fabric/validation/good/copy.png")]
    report = audit_mvtec_ad2(root, source_manifest=manifest(root, paths, tmp_path / "sha.csv"))
    assert report["source_integrity"]["status"] == "FAIL"
    assert report["source_integrity"]["verified_sources"] == 2
    duplicate = report["source_integrity"]["duplicate_content"][0]
    assert duplicate["cross_split"] and duplicate["train_overlap"]
    assert report["status"] == "FAIL"


def test_stale_and_incomplete_manifest_is_not_hash_evidence(tmp_path):
    root = tmp_path / "data"
    one = image(root, "fabric/train/good/000.png")
    image(root, "fabric/train/good/001.png", value=1)
    path = manifest(root, [one], tmp_path / "sha.csv")
    one.write_bytes(b"stale")
    report = audit_mvtec_ad2(root, source_manifest=path)
    assert report["status"] == "FAIL"
    assert report["source_integrity"]["verified_sources"] == 0
    assert report["source_integrity"]["duplicate_content"] == []
    assert {row["error"] for row in report["source_integrity"]["errors"]} >= {"manifest_hash_mismatch", "manifest_missing_source"}


def test_unrequested_hash_check_is_not_source_disjoint_pass(tmp_path):
    image(tmp_path, "fabric/train/good/000.png")
    report = audit_mvtec_ad2(tmp_path)
    assert report["source_integrity"]["status"] == "NOT_CHECKED"
    assert report["coverage"]["status"] == "NOT_REQUESTED"


def test_cli_json_csv_and_private_eligibility(tmp_path, capsys):
    from scripts.audit_dataset import main
    root = tmp_path / "data"
    image(root, "fabric/test_private/bad/000.png")
    output, table = tmp_path / "audit.json", tmp_path / "audit.csv"
    code = main(["--data-root", str(root), "--output", str(output), "--csv-output", str(table),
                 "--categories", "fabric", "--splits", "test_private"])
    assert code == 0
    report = json.loads(output.read_text())
    assert report["status"] == "PASS" and report["pixel_evaluation"]["eligible"] == 0
    with table.open() as stream:
        assert list(csv.DictReader(stream))[0]["pixel_evaluation_eligible"] == "False"
    assert json.loads(capsys.readouterr().out)["status"] == "PASS"


def test_cached_padding_validator_detects_replicated_foreground(tmp_path):
    from src.data.dataset_audit import audit_cached_tile_padding
    mask = np.zeros((512, 512), dtype=np.uint8)
    mask[6, 8] = 1
    np.save(tmp_path / "clean.npy", mask)
    contaminated = mask.copy()
    contaminated[6:, 8:] = 1
    np.save(tmp_path / "old.npy", contaminated)
    rows = [dict(image_id="raw0", source_hw=[7, 9], geometry=dict(local_box=[0, 0, 512, 512]),
                 mask_path="clean.npy"),
            dict(image_id="raw0", source_hw=[7, 9], local_box=[0, 0, 512, 512], mask_path="old.npy")]
    inputs = tmp_path / "tiles.json"
    inputs.write_text(json.dumps({"samples": rows}))
    result = audit_cached_tile_padding(inputs)
    assert result["status"] == "FAIL" and result["checked"] == 2
    assert result["tiles"][0]["padded_foreground_pixels"] == 0
    assert result["tiles"][1]["padded_foreground_pixels"] == (512 - 6) * (512 - 8) - 1
    assert np.array_equal(np.load(tmp_path / "old.npy"), contaminated)


def test_padding_without_source_mapping_is_blocked(tmp_path):
    from src.data.dataset_audit import audit_cached_tile_padding
    np.save(tmp_path / "mask.npy", np.zeros((512, 512), dtype=np.uint8))
    inputs = tmp_path / "tiles.json"
    inputs.write_text(json.dumps([dict(mask_path="mask.npy", image_id="raw0")]))
    result = audit_cached_tile_padding(inputs)
    assert result["status"] == "BLOCKED" and result["checked"] == 0


def test_archive_directory_cannot_be_silently_paired_as_category_split(tmp_path):
    image(tmp_path, "fabric/train/good/000.png")
    image(tmp_path, "fabric/archive/train/good/copy.png")
    report = audit_mvtec_ad2(tmp_path)
    assert report["status"] == "FAIL"
    assert report["inventory"]["recognized_images"] == 1
    assert report["inventory"]["unrecognized"][0]["path"] == "fabric/archive/train/good/copy.png"


@pytest.mark.parametrize("suffix", ["_seg", "_segmentation"])
def test_gt_suffixes_use_same_mask_classification_as_identity(tmp_path, suffix):
    image(tmp_path, "fabric/test_public/bad/000.png")
    image(tmp_path, f"fabric/test_public/bad/000{suffix}.png", mask=True)
    records = scan_mvtec_ad2(tmp_path, require_pixel_gt=True)
    assert len(records) == 1 and records[0].gt_status == "matched"


def test_duplicate_image_identity_is_reported_even_for_normal(tmp_path):
    path = image(tmp_path, "fabric/train/good/000.png")
    Image.open(path).save(path.with_suffix(".jpg"))
    report = audit_mvtec_ad2(tmp_path)
    assert report["status"] == "FAIL" and len(report["inventory"]["duplicate_images"]) == 1


def test_manifest_rejects_escaping_and_duplicate_paths(tmp_path):
    root = tmp_path / "data"
    path = image(root, "fabric/train/good/000.png")
    source_manifest = manifest(root, [path, path], tmp_path / "sha.csv")
    with source_manifest.open("a") as stream:
        stream.write("../outside.png,0," + "a" * 64 + ",train,fabric\n")
    report = audit_mvtec_ad2(root, source_manifest=source_manifest)
    assert {e["error"] for e in report["source_integrity"]["errors"]} == {
        "manifest_unsafe_path", "manifest_duplicate_path"}


@pytest.mark.parametrize("mode", ["bad_shape", "nonbinary", "negative_box"])
def test_padding_shape_binary_and_negative_bounds(tmp_path, mode):
    from src.data.dataset_audit import audit_cached_tile_padding
    mask = np.zeros((8, 8), dtype=np.uint8)
    if mode == "bad_shape":
        mask = mask[:7]
    elif mode == "nonbinary":
        mask[2, 2] = 2
    else:
        mask[0, 0] = 1  # outside source for a box with negative origin
    np.save(tmp_path / "mask.npy", mask)
    inputs = tmp_path / "tiles.json"
    inputs.write_text(json.dumps([dict(image_id="raw", source_hw=[7, 9],
                                      local_box=[-2, -2, 6, 6], mask_path="mask.npy")]))
    result = audit_cached_tile_padding(inputs)
    assert result["status"] == "FAIL"
    assert result["errors"][0]["error"] == ("foreground_in_padding" if mode == "negative_box" else "invalid_cached_mask")


def test_cli_exit_codes_and_padding_only_mode(tmp_path, capsys):
    from scripts.audit_dataset import main
    root = tmp_path / "data"
    image(root, "fabric/test_public/bad/000.png")
    args = ["--data-root", str(root), "--categories", "fabric", "--splits", "test_public",
            "--output", str(tmp_path / "audit.json")]
    assert main(args) == 1  # missing public GT
    args[1] = str(tmp_path / "missing")
    assert main(args) == 2
    inputs = tmp_path / "tiles.json"
    inputs.write_text(json.dumps([dict(image_id="raw", mask_path="mask.npy")]))
    assert main(["--padding-manifest", str(inputs), "--output", str(tmp_path / "padding.json")]) == 3
    report = json.loads((tmp_path / "padding.json").read_text())
    assert report["cached_tile_padding"]["status"] == "BLOCKED"
    assert len(capsys.readouterr().out.splitlines()) == 3


def test_audit_is_reproducible_and_uses_loader_nearest_contract(tmp_path):
    image(tmp_path, "fabric/test_public/bad/000.png")
    mask = image(tmp_path, "fabric/ground_truth/test_public/bad/000.png", mask=True)
    Image.fromarray(np.full((3, 4), 255, dtype=np.uint8)).save(mask)
    report = audit_mvtec_ad2(tmp_path)
    assert report == audit_mvtec_ad2(tmp_path)
    assert report["images"][0]["mask_requires_nearest_resize"]
    assert report["images"][0]["pixel_evaluation_eligible"]


def test_malformed_manifest_is_reported_without_crashing(tmp_path):
    root = tmp_path / "data"
    image(root, "fabric/train/good/000.png")
    source_manifest = tmp_path / "broken.csv"
    source_manifest.write_text("relative_path,file_size,sha256\nfabric/train/good/000.png\n")
    report = audit_mvtec_ad2(root, source_manifest=source_manifest)
    assert report["source_integrity"]["status"] == "FAIL"
    assert report["source_integrity"]["verified_sources"] == 0


@pytest.mark.parametrize("payload", [23, {"samples": [None]}, [None]])
def test_malformed_padding_json_returns_input_error(tmp_path, payload, capsys):
    from scripts.audit_dataset import main
    inputs, output = tmp_path / "tiles.json", tmp_path / "audit.json"
    inputs.write_text(json.dumps(payload))
    assert main(["--padding-manifest", str(inputs), "--output", str(output)]) == 2
    assert json.loads(output.read_text())["status"] == "ERROR"
    capsys.readouterr()


def test_recorded_day05_roles_are_checked_without_reassigning_sources(tmp_path):
    root = tmp_path / "data"
    paths = [image(root, "fabric/train/good/000.png"), image(root, "fabric/train/good/copy.png")]
    source_manifest = manifest(root, paths, tmp_path / "sha.csv")
    with source_manifest.open() as stream:
        rows = list(csv.DictReader(stream))
    with source_manifest.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=[*rows[0], "source_role"])
        writer.writeheader()
        writer.writerows(dict(row, source_role=role) for row, role in zip(rows, ["train_core", "dev_synthetic"]))
    report = audit_mvtec_ad2(root, source_manifest=source_manifest)
    assert report["source_integrity"]["status"] == "FAIL"
    assert report["source_integrity"]["duplicate_content"][0]["train_overlap"]
    assert report["source_integrity"]["source_disjoint_status"] == "FAIL"


def test_optional_normal_gt_zero_is_valid_and_foreground_is_an_error(tmp_path):
    image(tmp_path, "fabric/test_public/good/000.png")
    mask = image(tmp_path, "fabric/test_public/ground_truth/good/000_mask.png", mask=True, value=0)
    report = audit_mvtec_ad2(tmp_path)
    assert report["status"] == "PASS" and not report["inventory"]["orphan_gt"]
    assert report["images"][0]["gt_status"] == "normal_zero"
    assert report["images"][0]["mask"] is None
    Image.fromarray(np.ones((7, 9), dtype=np.uint8)).save(mask)
    report = audit_mvtec_ad2(tmp_path)
    assert report["status"] == "FAIL"
    assert not report["images"][0]["pixel_evaluation_eligible"]
    assert any(e["error"] == "normal_gt_nonzero" for e in report["errors"])


def test_declared_test_source_cannot_be_used_for_train_or_dev(tmp_path):
    root = tmp_path / "data"
    paths = [image(root, "fabric/train/good/000.png"), image(root, "fabric/test_public/good/001.png", value=1)]
    source_manifest = manifest(root, paths, tmp_path / "sha.csv")
    with source_manifest.open() as stream:
        rows = list(csv.DictReader(stream))
    with source_manifest.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=[*rows[0], "source_role"])
        writer.writeheader()
        writer.writerows(dict(row, source_role=role) for row, role in zip(rows, ["train_core", "dev_tiny"]))
    report = audit_mvtec_ad2(root, source_manifest=source_manifest)
    assert report["status"] == "FAIL"
    assert report["source_integrity"]["source_disjoint_status"] == "FAIL"
    assert any(e["error"] == "test_source_for_train_dev" for e in report["errors"])


def test_cli_output_io_error_has_documented_exit_code(tmp_path, capsys):
    from scripts.audit_dataset import main
    inputs = tmp_path / "tiles.json"
    inputs.write_text("[]")
    assert main(["--padding-manifest", str(inputs), "--output", str(tmp_path)]) == 2
    assert json.loads(capsys.readouterr().out)["status"] == "ERROR"
