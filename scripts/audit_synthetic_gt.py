"""Audit existing DEV GT before choosing size bins or inspecting predictions."""
import argparse
import json
from pathlib import Path
import numpy as np
from PIL import Image
from src.eval.region_stats import component_geometry
from src.data.tiling import generate_tile_records, crop_with_padding
import torch


def audit(inputs, mask_root=None):
    obj = json.loads(Path(inputs).read_text())
    rows = obj if isinstance(obj, list) else obj["samples"]
    result = []
    for row in rows:
        raw = row.get("gt_mask", row.get("mask_path"))
        if raw is None:
            if row.get("is_anomaly") is not False:
                raise ValueError(f"Missing GT: {row['image_id']}")
            continue
        p = Path(raw)
        if not p.is_absolute():
            p = (Path(mask_root) if mask_root else Path(inputs).parent) / p
        mask = np.load(p, allow_pickle=False) if p.suffix == ".npy" else np.array(Image.open(p))
        mask = mask.squeeze() > 0
        base = dict(image_id=row["image_id"], split=row.get("split", row.get("meta", {}).get("split")),
                    hw=list(mask.shape), gt_path=str(p), components=component_geometry(mask))
        base["tiles"] = []
        for tile in generate_tile_records(*mask.shape):
            local = crop_with_padding(torch.from_numpy(mask.astype(np.float32))[None],
                                      tile.local_xyxy, pad_mode="constant")[0].numpy()
            base["tiles"].append(dict(local_xyxy=list(tile.local_xyxy), hw=list(local.shape),
                                      components=component_geometry(local)))
        result.append(base)
    areas = [c["area"] for row in result for c in row["components"]]
    return dict(schema="msila.gt_geometry_audit.v2", input=str(inputs), images=result,
                component_count=len(areas), area_quantiles=None if not areas else
                np.quantile(areas, [0, .25, .5, .75, 1]).tolist(),
                metrics_inspected=False)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--inputs", required=True)
    p.add_argument("--mask-root")
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(audit(a.inputs, a.mask_root), indent=2) + "\n")
