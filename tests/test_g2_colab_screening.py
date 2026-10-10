"""CPU fixtures only; no GPU jobs or real screening measurements are produced."""
from copy import deepcopy
import ast
import csv
import json
from pathlib import Path

import pytest
import torch

from scripts import g2_colab_screening as colab
from src.models.msila import build_g2_model
from src.train import g2_e2
from src.train.screen_adapter import seed_everything
from tests.test_g2_tv1_model import BackboneFixture
from tests.test_g2_tv1_runner import setup, cpu_determinism  # noqa: F401


def test_batch_probe_oom_and_headroom_backoff(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    visited = []
    def probe(batch):
        visited.append(batch)
        if batch == 32:
            raise torch.cuda.OutOfMemoryError("unit fixture")
        return {"peak_reserved_bytes": batch * 100}
    batch, trials = colab.choose_batch([32, 16, 8, 4], probe, memory_limit=1000)
    assert batch == 8 and visited == [32, 16, 8]
    assert [row["status"] for row in trials] == ["OOM", "HEADROOM", "FIT"]
    with pytest.raises(ValueError, match="scientific failure"):
        colab.choose_batch([4, 2], lambda b: (_ for _ in ()).throw(ValueError("scientific failure")), memory_limit=1000)
    with pytest.raises(colab.runner.G2Blocked, match="no batch fits"):
        colab.choose_batch([1], lambda b: {"peak_reserved_bytes": 100}, memory_limit=99)


@pytest.mark.parametrize("gib,cap", [(15, 16), (23, 32), (40, 64), (80, 128)])
def test_vram_caps_and_candidate_bounds(gib, cap):
    assert colab.vram_batch_cap(gib * 2**30, 128) == cap
    assert colab.vram_batch_cap(gib * 2**30, 8) == 8
    candidates = colab.batch_candidates(cap)
    assert candidates[0] == cap and candidates[-1] == 1
    with pytest.raises(ValueError): colab.batch_candidates(0)


def test_cached_backbone_replays_rng_and_only_shares_frozen_parameters(setup, monkeypatch):
    study = setup["study"]("dinov3_vitb16")
    cfg = colab.runner.make_context(study, "rice", (64, 256), "adapter_screen", smoke=True)[0]["config"]
    loads = []
    def load(**kwargs):
        loads.append(kwargs["model"])
        torch.rand(13)  # Emulate RNG consumption while initializing real DINO.
        return BackboneFixture(kwargs["model"])
    monkeypatch.setattr(torch.hub, "load", load)
    seed_everything(cfg["training"]["seed"], deterministic=True, warn_only=False)
    original = build_g2_model(cfg, experiment="E2")
    cache = g2_e2.FrozenE2Factory(cfg, device="cpu")
    seed_everything(cfg["training"]["seed"], deterministic=True, warn_only=False)
    first = cache(cfg)
    second = cache(cfg)
    assert len(loads) == 2  # One uncached reference, one shared cached load.
    assert first.extractor is second.extractor
    assert first.adapter is not second.adapter and first.decoder is not second.decoder
    for group in ("adapter", "decoder"):
        expected = getattr(original, group).state_dict()
        assert all(torch.equal(v, expected[k]) for k, v in getattr(first, group).state_dict().items())
        assert {id(p) for p in getattr(first, group).parameters()}.isdisjoint(
            id(p) for p in getattr(second, group).parameters())
    assert not any(p.requires_grad for p in cache.extractor.parameters())
    changed = deepcopy(cfg); changed["training"]["seed"] += 1
    with pytest.raises(ValueError, match="provenance"): cache(changed)
    changed = deepcopy(cfg); changed["backbone"]["checkpoint_sha256"] = "changed"
    with pytest.raises(ValueError, match="provenance"): cache(changed)
    with pytest.raises(ValueError, match="Only a matching"):
        build_g2_model(cfg, experiment="E3", extractor=cache.extractor)
    decoder_hashes = set()
    for r, d in study["pairs"]:
        candidate = deepcopy(cfg); candidate["adapter"].update(r=r, d=d)
        decoder_hashes.add(colab.module_sha256(cache(candidate).decoder))
    assert len(loads) == 2 and len(decoder_hashes) == 1


def test_preflight_subset_keeps_full_grid_and_fair_budget(setup):
    study = setup["study"]("dinov3_vitb16")
    selected = colab.requested_pairs(study, ["256:768", "64:256"])
    assert selected == [(64, 256), (256, 768)] and len(study["pairs"]) == 9
    for invalid in (["64:999"], ["64:256", "64:256"], ["bad"], []):
        with pytest.raises(ValueError): colab.requested_pairs(study, invalid)
    plan = colab.preflight_plan(study)
    assert len(plan) == 8 and {row["seed"] for row in plan} == {2026}
    for category in colab.runner.CATEGORIES:
        contexts = [colab.runner.make_context(study, category, pair, "adapter_screen")[0] for pair in selected]
        assert contexts[0]["expected_steps"] == contexts[1]["expected_steps"]
        a, b = [deepcopy(c["config"]) for c in contexts]
        for cfg in (a, b): cfg["adapter"].pop("r"); cfg["adapter"].pop("d")
        assert a == b


def test_checkpoint_callback_backup_and_cached_resume(setup):
    study = setup["study"]()
    context, pools = colab.runner.make_context(study, "rice", (32, 128), "adapter_screen", smoke=True)
    cfg = context["config"]
    directory = colab.runner.run_directory(study, "rice", (32, 128), "adapter_screen", True)
    backup = directory.parents[3] / "drive_backup"
    cache = g2_e2.FrozenE2Factory(cfg, device="cpu")
    checkpoints = []
    def on_checkpoint(path):
        payload, _ = colab.runner.load_checkpoint_payload(path, require_sha256=True)
        checkpoints.append(payload["training_state"]["global_step"])
        colab.copy_tree(directory, backup)
    result = g2_e2.train_e2(context, pools, directory, device="cpu", model_factory=cache, on_checkpoint=on_checkpoint)
    colab.copy_tree(directory, backup)
    assert result["status"] == "PASS" and checkpoints
    assert colab.runner.read_valid_result(backup, context)
    assert not colab.runner.real_evidence(result)
    (directory / "metrics.json").unlink()
    resumed = g2_e2.train_e2(context, pools, directory, device="cpu", resume=True, model_factory=cache)
    assert resumed["status"] == "PASS" and resumed["global_step"] == 2
    assert result["last_checkpoint_sha256"] == resumed["last_checkpoint_sha256"]


def test_report_has_72_blank_rows_9_blank_macros_and_no_lock(setup):
    study = setup["study"]("dinov3_vitb16")
    result = colab.export_results(study)
    assert result["status"] == "NOT RUN" and result["valid_runs"] == 0
    directory = study["root"] / "full/colab_report"
    rows = list(csv.DictReader((directory / "screening_72.csv").open()))
    assert len(rows) == 72 and {r["status"] for r in rows} == {"NOT RUN"}
    assert all(r["synthetic_dev_aupro_0_05"] == "" for r in rows)
    macros = json.loads((directory / "macro_9.json").read_text())
    assert len(macros) == 9 and all(r["macro_synthetic_dev_aupro_0_05"] is None for r in macros)
    assert (directory / "rd_heatmap.png").is_file()
    assert not list(study["root"].rglob("adapter_selection_lock.json"))


def test_preflight_probe_uses_real_loss_api_on_cpu_fixture(setup, monkeypatch):
    args = colab.runner.parse_args(['--config', str(setup['path']), '--stage', 'adapter_screen',
                                   '--backbone', 'dinov3_vitb16', '--device', 'cpu'])
    cfg, _, _ = colab.runner.load_config(args)
    cfg['category'] = 'rice'; cfg['adapter'].update(r=256, d=768)
    cfg['synthetic_protocol'] = colab.runner.validate_native_protocol(colab.read_yaml(
        colab.runner.absolute(cfg['synthetic_protocol'])))
    pools, _ = colab.discover_sources(cfg)
    native = colab.G1NativeDataset(pools['train'], cfg['synthetic_protocol'], seed=2026, role='train', variants=2, fixed=False)
    tiles = colab.G1TileDataset(native, tile_size=512, overlap=128)
    factory = g2_e2.FrozenE2Factory(cfg, device='cpu')
    before = colab.module_sha256(factory.extractor)
    # Stub memory bookkeeping only; forward/backward/loss remain real CPU code.
    for name in ('empty_cache', 'reset_peak_memory_stats', 'synchronize'):
        monkeypatch.setattr(torch.cuda, name, lambda *a: None)
    for name in ('max_memory_reserved', 'max_memory_allocated'):
        monkeypatch.setattr(torch.cuda, name, lambda *a: 0)
    result = colab.probe_training_batch(factory, cfg, tiles, 2, 'cpu')
    assert result['forward_backward_steps'] == 2 and result['finite_loss'] and result['frozen_backbone']
    assert colab.module_sha256(factory.extractor) == before


def test_notebook_syntax_guard_and_archive_boundary(tmp_path):
    import hashlib
    import io
    import os
    import shutil
    import tarfile
    import re
    from PIL import Image
    root = Path(__file__).resolve().parents[1]
    nb = json.loads((root / 'notebooks/G2_02_Adapter_Screening.ipynb').read_text())
    assert nb['nbformat'] == 4
    for cell in nb['cells']:
        if cell['cell_type'] == 'code':
            source = ''.join(cell['source'])
            assert not cell['outputs'] and cell['execution_count'] is None
            for node in ast.walk(ast.parse(source)):
                if isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant):
                    if any(isinstance(t, ast.Name) and t.id.endswith('_script') for t in node.targets):
                        ast.parse(node.value.value)
    assert 'RUN_FULL_SCREENING = False' in ''.join(nb['cells'][2]['source'])
    assert 'if RUN_FULL_SCREENING:' in ''.join(nb['cells'][12]['source'])
    archive_dir = tmp_path / 'drive_data'; archive_dir.mkdir()
    local_work = tmp_path / 'local'; local_work.mkdir()
    local_archives = local_work / 'archives'; local_archives.mkdir()
    local_data = local_work / 'dataset'; local_data.mkdir()
    drive_out = tmp_path / 'drive_out'; drive_out.mkdir()
    weights = tmp_path / 'cpu_fixture.pth'; weights.write_bytes(b'copy fixture; not real pretrained weights')
    expected = {}
    categories = list(colab.runner.CATEGORIES)
    with tarfile.open(archive_dir / 'all_categories.tar.gz', 'w:gz') as archive:
        for index, category in enumerate(categories):
            for split in ('TRAIN', 'VALIDATION', 'TEST_PUBLIC'):
                buffer = io.BytesIO(); Image.new('RGB', (32, 48), (index, 50, len(split))).save(buffer, 'PNG')
                payload = buffer.getvalue(); name = f'{category}/{split}/good/0.png'
                member = tarfile.TarInfo(name); member.size = len(payload)
                archive.addfile(member, io.BytesIO(payload))
                if split != 'TEST_PUBLIC': expected[name] = payload
    ns = dict(Path=Path, hashlib=hashlib, shutil=shutil, os=os, tarfile=tarfile, re=re, json=json,
              ARCHIVE_DIR=archive_dir, ARCHIVE_PATTERNS=['*.tar.gz'], LOCAL_WORK=local_work,
              LOCAL_ARCHIVES=local_archives, LOCAL_DATA=local_data, CATEGORIES=categories,
              CHECKPOINT_ON_DRIVE=weights, LOCAL_CHECKPOINT=local_work / 'weights.pth',
              DRIVE_OUTPUT_ROOT=drive_out, previous_setup=drive_out / 'setup_manifest.json',
              RESOLVED_COMMIT='CPU_FIXTURE_ONLY', RESOLVED_DINO_COMMIT='CPU_FIXTURE_ONLY')
    exec(''.join(nb['cells'][5]['source']), ns)
    for name, payload in expected.items(): assert (local_data / name).read_bytes() == payload
    assert len(list(local_data.rglob('*.png'))) == 16
    assert not list(local_data.rglob('TEST_PUBLIC'))
    assert ns['good_target']('rice/TEST_PUBLIC/TRAIN/good/0.png', 'rice.tar.gz') is None
    with pytest.raises(ValueError): ns['good_target']('../rice/TRAIN/good/0.png', 'rice.tar.gz')
    exec(''.join(nb['cells'][5]['source']), ns)  # Same inputs are safe to rerun.
    assert not list(drive_out.rglob('adapter_selection_lock.json'))


def test_drive_checkpoint_torn_copy_recovers_previous_generation(tmp_path):
    def write(path, data):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        Path(str(path) + '.sha256').write_text(colab.runner.sha256_file(path))
    local = tmp_path / 'local/last.pt'
    drive = tmp_path / 'drive/last.pt'
    write(local, b'first verified generation')
    colab.copy_tree(local.parent, drive.parent)
    write(local, b'second verified generation')
    colab.copy_tree(local.parent, drive.parent)
    assert drive.read_bytes() == local.read_bytes()
    previous = Path(str(drive) + '.previous')
    assert previous.read_bytes() == b'first verified generation'
    drive.write_bytes(b'torn copy from a killed runtime')
    restored = tmp_path / 'restored'
    colab.copy_tree(drive.parent, restored)
    assert (restored / 'last.pt').read_bytes() == previous.read_bytes()
    assert colab.checkpoint_digest(restored / 'last.pt')
    previous.write_bytes(b'corrupt previous generation')
    with pytest.raises(ValueError, match='checksums invalid'):
        colab.copy_tree(drive.parent, tmp_path / 'bad_restore')


def test_preflight_without_cuda_is_not_run_and_never_loads_model(setup, tmp_path, monkeypatch):
    config = colab.read_yaml(setup['path'])
    config['backbone'] = 'dinov3_vitb16'
    colab.save_yaml(config, setup['path'])
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)
    monkeypatch.setattr(colab, 'FrozenE2Factory', lambda *a, **k: pytest.fail('Must not load a model without CUDA'))
    with pytest.raises(FileNotFoundError, match='NOT RUN'):
        colab.main(['--action', 'preflight', '--config', str(setup['path']),
                    '--local-root', str(tmp_path / 'local_runs'), '--drive-root', str(tmp_path / 'drive_runs')])
    assert not list((tmp_path / 'local_runs').rglob('*.pt'))
    assert not list((tmp_path / 'local_runs').rglob('adapter_selection_lock.json'))
