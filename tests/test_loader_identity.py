import numpy as np
import pytest
from PIL import Image
from src.data.loader import scan_mvtec_ad2, MVTecAD2HighResDataset, audit_mvtec_ad2


def write(root, path, mask=False):
    p = root / path
    p.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.full((19, 23) if mask else (19, 23, 3), 255, dtype=np.uint8)).save(p)
    return p


def test_pairing_never_crosses_split_or_defect(tmp_path):
    for split in ("validation", "test_public"):
        for defect in ("good", "bad", "scratch"):
            write(tmp_path, f"fabric/{split}/{defect}/000.png")
    write(tmp_path, "fabric/test_public/ground_truth/bad/000_mask.png", True)
    write(tmp_path, "fabric/ground_truth/test_public/scratch/000.png", True)
    rows = scan_mvtec_ad2(tmp_path)
    assert len(rows) == 6
    assert all(r.mask_path is None for r in rows if r.is_normal or r.split == "validation")
    assert all(r.gt_status == "matched" for r in rows if not r.is_normal and r.split == "test_public")
    ds = MVTecAD2HighResDataset(tmp_path, split="test_public")
    good = next(ds[i] for i, r in enumerate(ds.records) if r.is_normal)
    assert good["mask"].shape == (19, 23) and not good["mask"].any()
    with pytest.raises(ValueError, match="missing pixel GT"):
        MVTecAD2HighResDataset(tmp_path, split="validation")


def test_ambiguous_identity_and_unscoped_mask_are_errors(tmp_path):
    write(tmp_path, "fabric/test_public/bad/000.png")
    write(tmp_path, "fabric/ground_truth/bad/000_mask.png", True)
    assert scan_mvtec_ad2(tmp_path)[0].gt_status == "missing"
    write(tmp_path, "fabric/test_public/ground_truth/bad/000.png", True)
    write(tmp_path, "fabric/test_public/ground_truth/bad/000_mask.png", True)
    report = audit_mvtec_ad2(tmp_path)
    assert report["counts"]["fabric/test_public"]["ambiguous"] == 1
    with pytest.raises(ValueError, match="ambiguous pixel GT"):
        scan_mvtec_ad2(tmp_path, require_pixel_gt=True)


def test_nested_identity_is_preserved(tmp_path):
    for camera in ("cam1", "cam2"):
        write(tmp_path, f"fabric/test_public/bad/{camera}/000.png")
        write(tmp_path, f"fabric/test_public/ground_truth/bad/{camera}/000_mask.png", True)
    rows = scan_mvtec_ad2(tmp_path, require_pixel_gt=True)
    assert len(rows) == 2 and rows[0].mask_path != rows[1].mask_path
