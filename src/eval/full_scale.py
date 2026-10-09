"""Predeclared DEV groups, using the project's exact AU-PRO@0.05 evaluator."""
import numpy as np
from pathlib import Path
import hashlib
import tempfile
from scipy.ndimage import label
from src.eval.tiny_analysis import region_aupro
from src.eval.region_stats import component_geometry
from src.eval.boundary_analysis import contour_counts, contour_metrics
from src.metrics.aupro import aupro_from_parts


class DEVMetricAccumulator:
    def __init__(self, protocol, *, per_region=True, disk_backed=False):
        self.protocol = protocol
        self.groups = {}
        self.regions = []
        self.per_region = per_region
        self._scratch = tempfile.TemporaryDirectory(prefix='msila_metric_') if disk_backed else None

    def add(self, score, gt, *, split, image_id):
        score, gt = np.asarray(score, dtype=np.float32), np.asarray(gt).astype(bool)
        if score.shape != gt.shape or gt.ndim != 2 or not np.isfinite(score).all():
            raise ValueError(f"Invalid DEV score/GT shape or values: {image_id}")
        if split not in {"dev_tiny", "dev_mixed", "test_public"}:
            raise ValueError("Full-scale evaluation accepts declared DEV groups or selected TEST_PUBLIC")
        labels, n = label(gt, structure=np.ones((3,3),dtype=np.uint8))
        normal = score[~gt]
        if self._scratch is not None:
            path=Path(self._scratch.name)/(hashlib.sha256(f'{split}|{image_id}|{len(self.groups)}|{self.groups.get("all",{}).get("images",0)}'.encode()).hexdigest()+'.npy')
            np.save(path,np.sort(normal))
            normal=np.load(path,mmap_mode='r',allow_pickle=False)
        geometries = component_geometry(gt, self.protocol["boundary_band_px"])
        buckets = {"all": [], split: [], "tiny_regions": [], "image_boundary_regions": []}
        buckets.update({"size_bin/"+b: [] for b in self.protocol["size_bins"]})
        for c in geometries:
            values = score[labels == c["region_id"]]
            size = next((b for b, bounds in self.protocol["size_bins"].items()
                         if bounds[0] <= c["area"] <= bounds[1]), "outside_declared_bins")
            buckets["all"].append(values)
            buckets[split].append(values)
            if size in self.protocol["size_bins"]:
                buckets["size_bin/"+size].append(values)
            if size in self.protocol["dev_tiny_bins"]:
                buckets["tiny_regions"].append(values)
            if c["is_boundary"]:
                buckets["image_boundary_regions"].append(values)
            if self.per_region:
                self.regions.append(dict(image_id=image_id, split=split, size_bin=size, **c,
                    mean_score=float(values.mean()), max_score=float(values.max()),
                    overlap_at_locked_threshold=float((values >= self.protocol["prediction_threshold"]).mean())))
        for group, parts in buckets.items():
            state = self.groups.setdefault(group, dict(normals=[], regions=[], images=0,
                                                       contour_counts=np.zeros(4,dtype=np.int64)))
            state["normals"].append(normal)
            state["regions"].extend(parts)
            state["images"] += 1
        # Contour localization is distinct from image-edge region performance.
        for group in ("all", split):
            self.groups[group]["contour_counts"] += contour_counts(
                gt, score >= self.protocol["prediction_threshold"], self.protocol["contour_tolerance_px"])

    def result(self):
        out = {}
        for group, state in self.groups.items():
            metric = aupro_from_parts(state['normals'],state['regions'],sorted_normals=self._scratch is not None)
            value, status = metric['aupro'],metric['status']
            p, r, f1, fstatus = contour_metrics(state["contour_counts"])
            out[group] = dict(aupro_0_05=value, status=status, images=state["images"],
                              regions=len(state["regions"]), contour_boundary_f1=f1,
                              contour_status=fstatus, contour_precision=p, contour_recall=r)
        tiny, mixed = out.get("dev_tiny", {}), out.get("dev_mixed", {})
        values = [tiny.get("aupro_0_05"), mixed.get("aupro_0_05")]
        selection = None if any(v is None for v in values) else float(sum(values)/2)
        return dict(schema="msila.full_scale.metrics.v2", groups=out, dev_aupro=selection,
                    selection_rule="0.5*DEV_tiny_AU_PRO005+0.5*DEV_mixed_AU_PRO005",
                    score_normalization="sigmoid_no_per_image_rescaling", per_region=self.regions,
                    prediction_threshold=self.protocol["prediction_threshold"])
