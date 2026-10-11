#!/usr/bin/env python3
"""D9: 45 CPU mock forward/backward probes. No dataset, AU-PRO or selection."""
from __future__ import annotations

import argparse
from copy import deepcopy
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
from torch import nn
import yaml

from src.models.contracts import MULTIVIEW_FEATURE_KEYS
from src.models.fusion_factory import FUSION_METHODS
from src.models.msila import E5, resolve_g2_config
from src.train.screen_representation import save_json, sha256_file, sha256_json

PAIRS = tuple((r, d) for r in (64, 128, 256) for d in (256, 512, 768))
FIELDS = ("r", "d", "fusion", "status", "shape", "params", "adapter_params", "fusion_params",
          "logits_shape", "attention_logits_shape", "config_sha256", "gpu_status",
          "peak_vram_if_measured", "error")


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


class MockPairedFeatures(nn.Module):
    """Frozen six-feature fixture on a small grid; never pretrained evidence."""
    model_name, feature_mode = "dinov3_vitb16", "multilayer"
    out_channels, depth, blocks, patch_size = 768, 12, (4, 8, 12), 16

    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(1.0))
        self.register_buffer("features", torch.randn(6, 768, 4, 4))

    def extract_local_context(self, local, context, *, strategy):
        return {f"{view}{block}": (self.features[i:i+1] * self.scale).expand(local.shape[0], -1, -1, -1)
                for i, (view, block) in enumerate((v, b) for v in ("L", "C") for b in self.blocks)}


def candidate_config(config, pair, method):
    cfg = deepcopy(config)
    cfg["adapter"].update(r=pair[0], d=pair[1])
    cfg["fusion"]["method"] = method
    cfg.update(stage="E5", mode="technical_preflight", adapter_pair_role="technical_candidate",
               selection_status="pending_joint_selection_D10", selection_lock=None)
    cfg = resolve_g2_config(cfg, root=ROOT)
    require(cfg == yaml.safe_load(yaml.safe_dump(cfg)), "Config YAML round-trip drift")
    return cfg


def mock_model(cfg):
    a = cfg["adapter"]
    return E5(MockPairedFeatures(), fusion_method=cfg["fusion"]["method"],
              adapter_bottleneck_dim=a["r"], adapter_projection_dim=a["d"],
              adapter_kernel_size=a["kernel_size"], gamma_init=a["gamma_init"], adapter_bias=a["bias"],
              fusion_dim=cfg["fusion"]["dim"], hidden_channels=cfg["decoder"]["hidden_channels"],
              deterministic_resize=True, pair_strategy=cfg["paired_views"]["pair_strategy"],
              deterministic_sampling=cfg["paired_views"]["deterministic_sampling"])


def mock_batch():
    matrix = torch.tensor([[2/3, 0, 256/3], [0, 2/3, 256/3], [0, 0, 1]], dtype=torch.float64)
    return dict(image=torch.zeros(1, 3, 512, 512), context=torch.ones(1, 3, 512, 512),
                view_meta=[dict(local_sample_id="one-native-fixture", context_sample_id="one-native-fixture",
                                geometry=dict(local_to_context=matrix, local_input_hw=(512, 512),
                                              context_input_hw=(512, 512)))])


def module_digest(module):
    digest = hashlib.sha256()
    for key, value in module.state_dict().items():
        digest.update(key.encode())
        digest.update(str((tuple(value.shape), value.dtype)).encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def check_model(model, cfg):
    model.train()
    require(model.num_sources == 6 and tuple(model.source_block_map) == MULTIVIEW_FEATURE_KEYS, "Source order/count drift")
    require(not model.extractor.training, "Backbone must remain eval/frozen")
    before = {name: module_digest(getattr(model, name)) for name in ("extractor", "adapters", "projection", "decoder")}
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(trainable, lr=.01)
    require({id(p) for group in optimizer.param_groups for p in group["params"]} == {id(p) for p in trainable},
            "Optimizer membership drift")
    batch, dim = mock_batch(), cfg["fusion"]["dim"]
    evidence = {}
    for step in (1, 2):
        optimizer.zero_grad(set_to_none=True)
        logits, trace = model(batch, return_trace=True)
        require(tuple(logits.shape) == (1, 1, 512, 512) and bool(torch.isfinite(logits).all()), "Invalid logits")
        require(tuple(trace["projected"]) == MULTIVIEW_FEATURE_KEYS, "Projected source order drift")
        require(all(tuple(x.shape) == (1, dim, 4, 4) for x in trace["projected"].values()), "Projected shape drift")
        require(tuple(trace["decoder_feature"].shape) == (1, dim, 4, 4), "Fusion width/grid drift")
        for value in trace["projected"].values():
            value.retain_grad()
        if step == 1:
            require(all(torch.equal(trace["adapted"][key], trace["dino"][f"{key.split('_')[0][0].upper()}{block}"])
                        for key, block in model.source_block_map.items()), "gamma=0 must give exact identity")
            with torch.no_grad():
                expected = model.fusion(trace["projected"])
                reordered = model.fusion(dict(reversed(list(trace["projected"].items()))))
                require(torch.equal(expected, reordered), "Insertion-order-dependent Fusion")
            if model.fusion_method == "attention":
                attention = trace["fusion_trace"]
                require(tuple(attention["logits"].shape) == (1, 6, 4, 4), "Attention must be spatial [B,6,H,W]")
                require(torch.allclose(attention["attention"].sum(1), torch.ones(1, 4, 4)), "Wrong attention softmax axis")
                evidence["attention_logits_shape"] = [1, 6, 4, 4]
        # A technical gradient probe, not a dataset training objective or metric.
        loss = logits.square().mean()
        require(bool(torch.isfinite(loss)), "Nonfinite technical loss")
        loss.backward()
        require(all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in trainable), "Missing/nonfinite parameter gradient")
        require(all(x.grad is not None and bool(torch.isfinite(x.grad).all()) and bool(x.grad.abs().sum() > 0)
                    for x in trace["projected"].values()), "Disconnected Fusion source gradient")
        for adapter in model.adapters.values():
            require(bool(adapter.gamma.grad.abs() > 0), "Disconnected Adapter gamma")
            branch = [p for name, p in adapter.named_parameters() if name != "gamma"]
            require(all(bool(torch.count_nonzero(p.grad) == 0) for p in branch) if step == 1
                    else all(bool(torch.count_nonzero(p.grad) > 0) for p in branch), "Adapter Conv gradient/gate drift")
        require(all(bool(torch.count_nonzero(p.grad) > 0) for p in model.fusion.parameters()), "Disconnected learned Fusion")
        if step == 1:
            optimizer.step()  # Activate gamma for the second backward probe only.
    require(all(not p.requires_grad and p.grad is None for p in model.extractor.parameters()), "Backbone gradient")
    require(module_digest(model.extractor) == before["extractor"], "Frozen backbone state changed")
    adapters = sum(p.numel() for p in model.adapters.parameters())
    require(all((a.r, a.d, a.down_proj.in_channels) == (cfg["adapter"]["r"], cfg["adapter"]["d"], 768)
                for a in model.adapters.values()), "Adapter r/d/C drift")
    evidence.update(shape=[1, dim, 4, 4], logits_shape=[1, 1, 512, 512], params=sum(p.numel() for p in trainable),
                    adapter_params=adapters, fusion_params=sum(p.numel() for p in model.fusion.parameters()),
                    base_initial_sha256=before, finite_gradients=True, source_gradients=6,
                    forward_backward_probes=2, frozen_backbone=True)
    return evidence


def gpu_plan():
    return dict(status="NOT RUN", execution="Colab only; no local GPU", backbone="dinov3_vitb16",
                dtype="float32", local_input=["B", 3, 512, 512], context_source_fov=[768, 768],
                context_input=["B", 3, 512, 512], dino_features=[6, "B", 768, 32, 32],
                projected_features=[6, "B", 64, 32, 32], attention_logits=["B", 6, 32, 32],
                decoder_logits=["B", 1, 512, 512], batch_probes=[1, 2, 4, 8, 16],
                pairs=[list(pair) for pair in PAIRS], methods=list(FUSION_METHODS),
                record=["GPU model", "r", "d", "Fusion", "batch", "PASS/FAIL/OOM/NOT RUN", "peak allocated/reserved bytes", "error"],
                oom_policy="Record each failed combination/batch; do not silently remove any candidate",
                budget_policy="Technical feasibility only; D10 must declare a shared training budget before selection",
                pair_strategy="concat packs 2B; sequential uses two B forwards with the same frozen backbone")


def run_cpu(config, output_root):
    cfg = resolve_g2_config(config, root=ROOT)
    require((cfg["backbone"]["name"], cfg["backbone"]["channels"], cfg["backbone"]["feature_blocks"])
            == ("dinov3_vitb16", 768, [4, 8, 12]), "D9 matrix requires verified ViT-B/16 grid")
    require(cfg["adapter"]["gamma_init"] == 0, "Preflight requires identity initialization gamma=0")
    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=True)
    threads, rng = torch.get_num_threads(), torch.get_rng_state()
    rows, records, errors = [], [], []
    try:
        torch.set_num_threads(1)
        for pair in PAIRS:
            for method in FUSION_METHODS:
                row = dict(r=pair[0], d=pair[1], fusion=method, status="FAIL", gpu_status="NOT RUN",
                           peak_vram_if_measured=None, error="")
                record = dict(verification_scope="CPU_mock_4x4", experiment_runs=0)
                try:
                    candidate = candidate_config(cfg, pair, method)
                    row["config_sha256"] = sha256_json(candidate)
                    record["config"] = candidate
                    torch.manual_seed(candidate["training"]["seed"])
                    record.update(check_model(mock_model(candidate), candidate))
                    row.update(status="PASS", **{key: record[key] for key in FIELDS if key in record})
                except (ValueError, TypeError, RuntimeError, KeyError) as exc:
                    row["error"] = f"{type(exc).__name__}: {exc}"
                    errors.append({"r": pair[0], "d": pair[1], "fusion": method, "error": row["error"]})
                record.update(row)
                save_json(record, root / "cpu" / f"r{pair[0]}_d{pair[1]}_{method}.json")
                records.append(record)
                rows.append(row)
    finally:
        torch.set_num_threads(threads)
        torch.set_rng_state(rng)
    # Keep all 45 rows, including explicit FAIL reasons; no ranking or scores.
    with (root / "fusion_preflight_45.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(v) if isinstance(v, list) else v for k, v in row.items()})
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    sources = ("scripts/g2_fusion_preflight.py", "scripts/run_g2.py", "configs/g2_experiments.yaml",
               "src/models/msila.py", "src/models/fusion_factory.py", "src/models/position_attention_fusion.py",
               "src/models/fusions/weighted_sum.py", "src/models/fusions/concat.py", "src/models/fusions/gated.py",
               "src/models/mean_fusion.py")
    summary = dict(status="PASS" if not errors else "FAIL", cpu_pass=sum(row["status"] == "PASS" for row in rows),
                   expected_combinations=45, gpu_status="NOT RUN", experiment_runs=0,
                   selection_status="pending_joint_selection_D10", selection_lock=None,
                   commit=commit, branch=subprocess.check_output(["git", "branch", "--show-current"], cwd=ROOT, text=True).strip(),
                   git_dirty=bool(subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no"], cwd=ROOT, text=True)),
                   integration_commit=subprocess.check_output(["git", "rev-parse", "origin/g2/integration"], cwd=ROOT, text=True).strip(),
                   seed=cfg["training"]["seed"], verification_scope="CPU_mock_4x4", errors=errors,
                   source_code_sha256={name: sha256_file(ROOT / name) for name in sources})
    save_json(summary, root / "run_manifest.json")
    save_json(gpu_plan(), root / "gpu_preflight_plan.json")
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/g2_experiments.yaml")
    parser.add_argument("--output-root", default="outputs/G2/D9")
    args = parser.parse_args(argv)
    summary = run_cpu(yaml.safe_load(Path(args.config).read_text()), args.output_root)
    print(json.dumps(summary, indent=2))
    return 0 if summary["cpu_pass"] == 45 else 1


if __name__ == "__main__":
    raise SystemExit(main())
