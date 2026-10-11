"""D6 CPU/API fixtures only: never evidence of real GPU performance."""
from copy import deepcopy
import csv
import io
import json
from pathlib import Path
import shutil
import tarfile

import numpy as np
from PIL import Image
import pytest
import torch
from torch import nn
import yaml

from scripts import eval_g2 as g2
from src.data.loader import G1NativeDataset
from src.models.msila import build_g2_model
from src.train import g1_e1, g2_e2
from src.utils.resume import load_checkpoint_payload
from tests.test_g1_e1 import config_and_dataset


@pytest.fixture(autouse=True)
def cpu_contract():
    threads, rng = torch.get_num_threads(), torch.get_rng_state()
    deterministic = torch.are_deterministic_algorithms_enabled()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(threads)
    torch.set_rng_state(rng)
    torch.use_deterministic_algorithms(deterministic)


@pytest.fixture
def trained(tmp_path, monkeypatch):
    cfg, data = config_and_dataset(tmp_path, monkeypatch)
    cfg["training"].update(mode="train", epochs=1, max_steps=2)
    cfg["data"].update(tile_size=512, overlap=128, max_train_sources=2, dev_variants_per_image=6)
    cfg["evaluation"]["example_limit"] = 0
    folder = tmp_path / "E1/rice"
    folder.mkdir(parents=True)
    g1_e1.run(cfg, device="cpu", output_dir=folder)
    payload, _ = load_checkpoint_payload(folder/"best.pt", require_sha256=True)
    return dict(cfg=payload["config"], payload=payload, path=folder/"best.pt", root=folder.parent, data=data)


def args_for(trained, tmp_path, *extra):
    return g2.parse_args(["--device","cpu","--categories","rice","--e1-root",str(trained["root"]),
                         "--output-root",str(tmp_path/"eval"),*extra])


def observe_restore(monkeypatch):
    built = []
    def factory(*args, **kwargs):
        model = build_g2_model(*args, **kwargs)
        built.append(model)
        return model
    monkeypatch.setattr(g2,"build_g2_model",factory)
    return built


def test_trained_e1_restore_and_checkpoint_provenance(trained, tmp_path, monkeypatch):
    payload = g2.load_trained_checkpoint(trained["path"],"E1","rice")
    cfg, pools = g2.relocate_config(payload["config"],args_for(trained,tmp_path))
    g2.validate_e1_budget(trained["path"],payload,cfg,pools)
    mismatched = deepcopy(cfg)
    mismatched["training"]["epochs"] += 1
    with pytest.raises(g2.Blocked,match="budget/provenance"):
        g2.validate_e1_budget(trained["path"],payload,mismatched,pools)
    built = observe_restore(monkeypatch)
    with pytest.raises(g2.Blocked,match="fixture"):
        g2.restore_model(payload,cfg,"E1","cpu")
    assert all(torch.equal(v,built[0].decoder.state_dict()[k]) for k,v in payload["model_state"].items())
    assert all(not p.requires_grad for p in built[0].extractor.parameters())
    assert len(pools["dev"]) == 1
    with pytest.raises(g2.Blocked,match="category mismatch"):
        g2.load_trained_checkpoint(trained["path"],"E1","can")
    with pytest.raises(g2.Blocked,match="reserved"):
        g2.load_trained_checkpoint(trained["path"],"E3","rice")
    corrupt = tmp_path/"corrupt.pt"
    shutil.copy2(trained["path"],corrupt)
    corrupt.with_suffix(".pt.sha256").write_text("0"*64)
    with pytest.raises(RuntimeError):
        g2.load_trained_checkpoint(corrupt,"E1","rice")
    payload["training_state"]["global_step"] = 0
    torch.save(payload,corrupt)
    corrupt.with_suffix(".pt.sha256").write_text(g2.file_sha256(corrupt))
    with pytest.raises(g2.Blocked,match="no optimizer updates"):
        g2.load_trained_checkpoint(corrupt,"E1","rice")


def test_actual_e2_checkpoint_without_selection_lock_and_fixture_rejected(trained,tmp_path,monkeypatch):
    cfg = deepcopy(trained["cfg"])
    cfg.update(stage="E2",mode="full",study_sha256=g2.sha256_json("unit protocol"),expected_steps=1)
    cfg["adapter"] = dict(r=32,d=128,kernel_size=3,gamma_init=0.,bias=True)
    cfg["training"]["checkpoint_interval_steps"] = 1
    context = dict(config=cfg,config_sha256=g2.sha256_json(cfg),expected_steps=1)
    pools,_ = g1_e1.discover_sources(cfg)
    folder = tmp_path/"E2/rice"
    result = g2_e2.train_e2(context,pools,folder,device="cpu")
    assert result["training_complete"] and result["verification_scope"] == "fixture"
    payload,_ = load_checkpoint_payload(folder/"best.pt",require_sha256=True)
    built = observe_restore(monkeypatch)
    with pytest.raises(g2.Blocked,match="fixture"):
        g2.restore_model(payload,cfg,"E2","cpu")
    head = nn.ModuleDict({"adapter":built[0].adapter,"decoder":built[0].decoder})
    assert all(torch.equal(v,head.state_dict()[k]) for k,v in payload["model_state"].items())
    assert any(k.startswith("adapter.") for k in payload["model_state"])
    accepted = g2.load_trained_checkpoint(folder/"best.pt","E2","rice")
    assert accepted["config"] == payload["config"]
    assert not list(tmp_path.rglob('*selection_lock*'))
    payload["config"]["adapter"]["r"] += 1
    torch.save(payload,folder/"best.pt")
    (folder/"best.pt.sha256").write_text(g2.file_sha256(folder/"best.pt"))
    with pytest.raises(g2.Blocked,match="config/r/d provenance"):
        g2.load_trained_checkpoint(folder/"best.pt","E2","rice")


def test_legacy_e1_best_only_keeps_scientific_protocol(trained,tmp_path):
    original = trained["path"]
    original.with_name("last.pt").unlink()
    original.with_name("metrics.json").unlink()
    path,payload,cfg,pools,identity = g2.prepare_job(args_for(trained,tmp_path),"E1","rice")
    assert path == original and payload["training_state"]["global_step"] > 0
    assert not identity["final_training_budget_verified"]
    assert identity["scientific_protocol"] == g2.scientific_identity(trained["cfg"])


def test_relocation_fair_protocol_sources_and_test_exclusion(trained,tmp_path):
    moved = tmp_path/"relocated"
    shutil.copytree(trained["data"],moved)
    (moved/"rice/TEST_PUBLIC/bad/0.png").write_bytes(b"TEST must not be read")
    args = args_for(trained,tmp_path,"--data-root",str(moved))
    cfg,pools = g2.relocate_config(trained["cfg"],args)
    assert g2.scientific_identity(cfg) == g2.scientific_identity(trained["cfg"])
    assert all(r.split == "validation" and r.defect_type == "good" for r in pools["dev"])
    for section,key,value in [("training","dev_seed",999),("training","seed",42),("training","epochs",9),
        ("data","overlap",64),("backbone","name","dinov3_vitb16"),("decoder","deterministic_resize",True),
        ("loss","dice_weight",2.)]:
        other = deepcopy(cfg)
        other[section][key] = value
        assert g2.scientific_identity(other) != g2.scientific_identity(cfg)
    Image.fromarray(np.zeros((96,128,3),dtype=np.uint8)).save(moved/"rice/VALIDATION/good/0.png")
    with pytest.raises(g2.Blocked,match="sources/order/content"):
        g2.relocate_config(trained["cfg"],args)


class PixelProbability(nn.Module):
    def forward(self,image):
        p = (image[:,:1]*.229+.485).clamp(.01,.99)
        return torch.logit(p)


@pytest.mark.parametrize("hw",[(529,677),(71,127),(512,768)])
def test_native_orientation_hann_overlap_padding_sigmoid(hw):
    p = torch.linspace(.1,.4,hw[0])[:,None] + torch.linspace(.02,.42,hw[1])[None,:]
    image = p.expand(3,*hw).clone()
    cfg = dict(data=dict(tile_size=512,overlap=128),evaluation=dict(tile_batch_size=2))
    actual = g2.predict_native(PixelProbability(),image,cfg,"cpu")
    assert actual.shape == hw
    torch.testing.assert_close(actual,p,rtol=1e-5,atol=1e-6)
    assert actual[0,-1] > actual[0,0] and actual[-1,0] > actual[0,0]


def test_exact_reused_metrics_p99_dice_native_exports(trained,tmp_path):
    cfg,pools = g2.relocate_config(trained["cfg"],args_for(trained,tmp_path))
    native = G1NativeDataset(pools["dev"],cfg["synthetic_protocol"],seed=cfg["training"]["dev_seed"],
                             role="dev",variants=6,fixed=True)
    folder = tmp_path/"native"
    capture = g2.CaptureNative(native,folder)
    actual = g1_e1.evaluate_dev(PixelProbability(),native,cfg,"cpu",folder,predict_fn=capture)
    expected = g1_e1.evaluate_dev(PixelProbability(),native,cfg,"cpu")
    assert actual == expected
    assert len(capture.rows) == 7 and actual["dev_tiny"]["regions"] == 4
    assert actual["dev_mixed"]["regions"] == 6
    normal = np.load(folder/capture.rows[0]["score_path"])
    assert actual["normal_score_p99"] == pytest.approx(float(np.quantile(normal,.99)))
    dice = g2.dice_from_exports(folder,capture.rows,"rice",.5)
    from src.metrics.segf1 import aggregate_seg_f1
    samples = [dict(category="rice",anomaly_map=np.load(folder/r["score_path"]),
                    gt_mask=np.load(folder/r["mask_path"])) for r in capture.rows]
    assert dice == aggregate_seg_f1(samples,.5)["per_category"]["rice"]
    assert dice["fp"] > 0
    assert all(np.load(folder/r["score_path"]).shape == (96,128) for r in capture.rows)
    assert not any("TEST" in row["source"]["path"] for row in capture.rows)


def hand_computable_dev(scores, masks, bins):
    """Metric fixtures only; no generated experiment/checkpoint or acceptance claim."""
    protocol = yaml.safe_load((g2.ROOT/"configs/full_scale_synthetic.yaml").read_text())
    native = []
    for index,(score,mask,size_bin) in enumerate(zip(scores,masks,bins)):
        native.append(dict(image=torch.full((3,*mask.shape),float(index)),
                           mask=torch.from_numpy(mask[None].astype(np.float32)),
                           meta=dict(sample_id=f"metric_fixture_{index}",original_hw=list(mask.shape),
                                     synthetic=dict(is_anomaly=bool(mask.any()),size_bin=size_bin))))
    cfg = dict(category="rice",synthetic_protocol=protocol,evaluation=dict(example_limit=0))
    predict = lambda model,image,cfg,device: torch.from_numpy(scores[int(image[0,0,0])].astype(np.float32))
    return g2.evaluate_dev(None,native,cfg,"cpu",predict_fn=predict)


def test_hand_computed_native_tiny_mixed_normal_p99_and_pooled_dice(tmp_path):
    # 4-pixel tiny region, 129-pixel mixed region, one normal-only false positive.
    # All foreground scores outrank every background -> both AU-PROs exactly 1.
    normal = np.array([[0.,.1],[.2,.6]],dtype=np.float32)
    tiny = np.array([[1,1,0],[1,1,0]],dtype=np.uint8)
    mixed = np.zeros((13,11),dtype=np.uint8)
    mixed.flat[:129] = 1
    masks = [np.zeros_like(normal,dtype=np.uint8),tiny,mixed]
    scores = [normal,np.where(tiny,.9,.1),np.where(mixed,.9,.1)]
    protocol = yaml.safe_load((g2.ROOT/"configs/full_scale_synthetic.yaml").read_text())
    metrics = hand_computable_dev(scores,masks,[None,protocol["dev_tiny_bins"][0],"mixed_control"])
    assert metrics["qa_status"] == "PASS" and metrics["aupro_max_fpr"] == .05
    assert metrics["dev_tiny"]["regions"] == 1 and metrics["dev_mixed"]["regions"] == 2
    assert metrics["synthetic_dev_aupro_0_05"] == pytest.approx(1.)
    assert metrics["dev_tiny"]["aupro_0_05"] == pytest.approx(1.)
    assert metrics["dev_mixed"]["aupro_0_05"] == pytest.approx(1.)
    # Linear P99 on [0,.1,.2,.6]: .2 + .97*(.6-.2), not anomalous pixels.
    assert metrics["normal_score_p99"] == pytest.approx(.588)
    rows = []
    for index,(score,mask) in enumerate(zip(scores,masks)):
        np.save(tmp_path/f"score{index}.npy",score.astype(np.float32))
        np.save(tmp_path/f"mask{index}.npy",mask)
        rows.append(dict(score_path=f"score{index}.npy",mask_path=f"mask{index}.npy"))
    dice = g2.dice_from_exports(tmp_path,rows,"rice",.5)
    assert (dice["tp"],dice["fp"],dice["fn"]) == (133,1,0)
    assert dice["f1"] == pytest.approx(266/267)


@pytest.mark.parametrize("case,expected",[("perfect",1.),("inverted",0.),("tie",.025),("one_fp",.5)])
def test_hand_computed_aupro_cutoff_and_orientation(case,expected):
    # Exactly 40 normal pixels; one high background costs FPR=1/40=.025.
    mask = np.zeros((4,11),dtype=np.uint8)
    mask[0,:4] = 1
    score = np.where(mask,.8,.1)
    if case == "inverted":
        score = 1-score
    elif case == "tie":
        score.fill(.5)  # PRO=FPR; integral[0,.05]/.05 = .025.
    elif case == "one_fp":
        score[-1,-1] = .9  # PRO=1 only after .025: AU-PRO=(.05-.025)/.05.
    protocol = yaml.safe_load((g2.ROOT/"configs/full_scale_synthetic.yaml").read_text())
    metrics = hand_computable_dev([score],[mask],[protocol["dev_tiny_bins"][0]])
    assert metrics["aupro_max_fpr"] == .05
    assert metrics["synthetic_dev_aupro_0_05"] == pytest.approx(expected)
    assert metrics["dev_tiny"]["aupro_0_05"] == pytest.approx(expected)
    assert metrics["normal_score_p99"] is None  # no normal-only image in this fixture


def test_locked_dice_threshold_cannot_be_retuned(trained,tmp_path):
    cfg = deepcopy(trained["cfg"])
    cfg["synthetic_protocol"]["prediction_threshold"] = .7
    with pytest.raises((g2.Blocked,ValueError),match="threshold|protocol"):
        g2.relocate_config(cfg,args_for(trained,tmp_path))


def test_dice_includes_normal_fp_and_threshold_is_inclusive(tmp_path):
    # At the locked >= .5 rule: one TP, one FN, one FP -> Dice=2/(2+1+1).
    masks = [np.array([[1,1],[0,0]],dtype=np.uint8),np.zeros((1,2),dtype=np.uint8)]
    scores = [np.array([[.5,.499],[0.,0.]],dtype=np.float32),np.array([[.5,0.]],dtype=np.float32)]
    rows = []
    for index,(score,mask) in enumerate(zip(scores,masks)):
        np.save(tmp_path/f"s{index}.npy",score)
        np.save(tmp_path/f"m{index}.npy",mask)
        rows.append(dict(score_path=f"s{index}.npy",mask_path=f"m{index}.npy"))
    dice = g2.dice_from_exports(tmp_path,rows,"rice",.5)
    assert (dice["tp"],dice["fp"],dice["fn"]) == (1,1,1)
    assert dice["f1"] == pytest.approx(.5)


def valid_result(category="rice",score=.2):
    return dict(status="PASS",category=category,acceptance_eligible=True,**{k:score for k in g2.METRICS},
                qa_status="PASS",native_resolution=True,aupro_max_fpr=.05,
                dev_tiny={"regions":1,"aupro_0_05":score},dev_mixed={"regions":1,"aupro_0_05":score},
                dice_threshold=.5,score_normalization="sigmoid_no_rescaling",
                score_orientation="higher_is_more_anomalous",n_samples=2,qa_samples=2,
                pair_protocol_sha256="unit protocol",dev_samples_sha256="unit DEV")


def test_coverage_no_partial_macro_false_zero_or_unfair_comparison(tmp_path):
    results = {("E1",c):valid_result(c,.2) for c in g2.CATEGORIES}
    results.update({("E2",c):valid_result(c,.3) for c in g2.CATEGORIES[:-1]})
    summary = g2.write_comparison(tmp_path,results)
    assert summary["coverage"] == {"E1":8,"E2":7} and summary["macro_mean_8categories"] is None
    with (tmp_path/"metrics_E1_E2.csv").open() as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 8 and rows[-1]["E2_synthetic_dev_aupro_0_05"] == ""
    results["E2",g2.CATEGORIES[-1]] = valid_result(g2.CATEGORIES[-1],.3)
    summary = g2.write_comparison(tmp_path,results)
    assert summary["status"] == "PASS" and summary["paired_pass"] == 8
    assert summary["macro_mean_8categories"]["E2_minus_E1_aupro_0_05"] == pytest.approx(.1)
    results["E2","rice"]["dev_samples_sha256"] = "different DEV"
    blocked = g2.write_comparison(tmp_path,results)
    assert blocked["paired_pass"] == 7 and blocked["macro_mean_8categories"] is None
    assert blocked["records"][3]["pair_status"] == "BLOCKED"


@pytest.mark.parametrize("status",["SMOKE_PASS","CPU_PASS","READY","BLOCKED","NOT RUN"])
def test_non_acceptance_statuses_never_count(status,tmp_path):
    record = valid_result()
    record.update(status=status,acceptance_eligible=False)
    summary = g2.write_comparison(tmp_path,{("E1","rice"):record,("E2","rice"):record})
    assert summary["coverage"] == {"E1":0,"E2":0} and summary["paired_pass"] == 0
    assert summary["macro_mean_8categories"] is None


def test_independent_e1_all8_succeeds_without_e2_or_any_lock(tmp_path,monkeypatch):
    calls = []
    def evaluate(args,experiment,category):
        calls.append((experiment,category))
        return valid_result(category,.2)
    monkeypatch.setattr(g2,"evaluate_category",evaluate)
    code = g2.main(["--experiments","E1","--device","cpu","--output-root",str(tmp_path)])
    summary = json.loads((tmp_path/"summary.json").read_text())
    assert code == 0 and summary["status"] == "PASS"
    assert summary["coverage"] == {"E1":8,"E2":0}
    assert summary["comparison_status"] == "BLOCKED" and not summary["complete"]
    assert summary["macro_mean_8categories"] is None
    assert summary["per_experiment_macro_mean_8categories"]["E1"][g2.METRICS[0]] == pytest.approx(.2)
    assert all(row["E2_status"] == "NOT RUN" for row in summary["records"])
    assert calls == [("E1",category) for category in g2.CATEGORIES]


def test_missing_e2_does_not_prevent_requested_e1(tmp_path,monkeypatch):
    def evaluate(args,experiment,category):
        if experiment == "E2":
            raise g2.Blocked("Missing E2 checkpoint; awaits D10 joint selection and main training")
        return valid_result(category,.2)
    monkeypatch.setattr(g2,"evaluate_category",evaluate)
    monkeypatch.setattr(g2,"prepare_job",lambda *a: (_ for _ in ()).throw(g2.Blocked("Missing E2 checkpoint")))
    code = g2.main(["--categories","rice","--device","cpu","--output-root",str(tmp_path)])
    summary = json.loads((tmp_path/"summary.json").read_text())
    assert code == 2 and summary["coverage"] == {"E1":1,"E2":0}
    rice = summary["records"][3]
    assert rice["E1_status"] == "PASS" and rice["E2_status"] == "BLOCKED"
    assert rice["E2_synthetic_dev_aupro_0_05"] is None
    assert all(row["E1_status"] == row["E2_status"] == "NOT RUN"
               for row in summary["records"] if row["category"] != "rice")


def test_missing_metric_blocked_and_nonfinite_metric_failed():
    result = valid_result()
    result["normal_score_p99"] = None
    with pytest.raises(g2.Blocked,match="metric unavailable"):
        g2.validate_metrics(result)
    result["normal_score_p99"] = float("nan")
    with pytest.raises(ValueError,match="nonfinite"):
        g2.validate_metrics(result)


def test_output_cannot_overwrite_training_metrics(tmp_path):
    with pytest.raises(SystemExit):
        g2.parse_args(["--output-root",str(tmp_path),"--e1-root",str(tmp_path/"E1")])


def test_cli_missing_inputs_preserves_all8_and_null_metrics(tmp_path):
    code = g2.main(["--device","cpu","--output-root",str(tmp_path),
                    "--e1-root",str(tmp_path/"absent E1"),"--e2-root",str(tmp_path/"absent E2")])
    summary = json.loads((tmp_path/"summary.json").read_text())
    assert code == 2 and summary["status"] == "BLOCKED"
    assert summary["coverage"] == {"E1":0,"E2":0} and len(summary["records"]) == 8
    assert all(r["E1_status"] == r["E2_status"] == "BLOCKED" for r in summary["records"])
    assert all(r["E2_minus_E1_aupro_0_05"] is None for r in summary["records"])
    assert len(json.loads((tmp_path/"coverage.json").read_text())["missing"]) == 16


def test_resume_checks_native_files_and_input_signature(tmp_path):
    checkpoint = tmp_path/"best.pt"
    checkpoint.write_bytes(b"unit checkpoint bytes; no inference")
    np.save(tmp_path/"score.npy",np.zeros((2,3),dtype=np.float32))
    np.save(tmp_path/"mask.npy",np.zeros((2,3),dtype=np.uint8))
    rows = [dict(score_path="score.npy",mask_path="mask.npy",original_hw=[2,3],
                 score_sha256=g2.file_sha256(tmp_path/"score.npy"),mask_sha256=g2.file_sha256(tmp_path/"mask.npy"))]
    g2.write_metrics_json(rows,tmp_path/"native_maps.json")
    g2.write_metrics_json({"status":"PASS"},tmp_path/"qa_report.json")
    g2.write_metrics_json({"unit":True},tmp_path/"evaluation_config.json")
    example = tmp_path/"examples/unit/comparison.png"
    example.parent.mkdir(parents=True)
    example.write_bytes(b"unit artifact, no GPU inference")
    record = valid_result()
    record.update(n_samples=1,qa_samples=1,evaluation_sha256="unit signature",checkpoint_sha256=g2.file_sha256(checkpoint),
                  native_maps_sha256=g2.sha256_json(rows),qa_report_sha256=g2.file_sha256(tmp_path/"qa_report.json"),
                  evaluation_config_sha256=g2.file_sha256(tmp_path/"evaluation_config.json"),
                  examples_sha256={"examples/unit/comparison.png":g2.file_sha256(example)})
    g2.write_metrics_json(record,tmp_path/"metrics.json")
    (tmp_path/"metrics.json.sha256").write_text(g2.file_sha256(tmp_path/"metrics.json"))
    assert g2.completed_result(tmp_path,"unit signature",checkpoint) == record
    assert g2.completed_result(tmp_path,"different signature",checkpoint) is None
    np.save(tmp_path/"score.npy",np.ones((3,2),dtype=np.float32))
    assert g2.completed_result(tmp_path,"unit signature",checkpoint) is None


def test_copy_all8_archives_before_good_only_extract(tmp_path,monkeypatch):
    source,local,output = tmp_path/"drive",tmp_path/"local",tmp_path/"dataset"
    source.mkdir()
    names = {c:f"{c}.tar.gz" for c in g2.CATEGORIES}
    for category in g2.CATEGORIES:
        with tarfile.open(source/names[category],"w:gz") as archive:
            for split in ("train","validation","test_public"):
                data = b"transport fixture only"
                info = tarfile.TarInfo(f"{category}/{split}/good/0.png")
                info.size = len(data)
                archive.addfile(info,io.BytesIO(data))
    original = tarfile.open
    def observe(path,*args,**kwargs):
        assert Path(path).parent == local
        assert all((local/name).is_file() for name in names.values())
        return original(path,*args,**kwargs)
    monkeypatch.setattr(tarfile,"open",observe)
    assert g2.prepare_archives(source,names,local,output) == output
    assert len(list(output.rglob("*.png"))) == 16
    assert {p.relative_to(output).parts[1] for p in output.rglob("*.png")} == {"TRAIN","VALIDATION"}


@pytest.mark.parametrize("name",["../rice/train/good/0.png","/rice/train/good/0.png","can/train/good/0.png"])
def test_archive_traversal_and_wrong_category_rejected(name):
    with pytest.raises(ValueError,match="BLOCKED"):
        g2.good_member_path(tarfile.TarInfo(name),"rice")


def test_complete_evaluation_pipeline_cpu_fixture_never_accepted(trained,tmp_path,monkeypatch):
    """Exercise real checkpoint/DEV/exports end to end with a test-only restorer."""
    args = args_for(trained,tmp_path,"--experiments","E1")
    def fixture_restorer(payload,cfg,experiment,device):
        cfg = deepcopy(cfg)
        cfg["adapter"] = dict(r=1,d=1,kernel_size=3,gamma_init=0.,bias=True)
        model = build_g2_model(cfg,experiment=experiment)
        model.decoder.load_state_dict(payload["model_state"],strict=True)
        return model.eval()
    monkeypatch.setattr(g2,"restore_model",fixture_restorer)
    result = g2.evaluate_category(args,"E1","rice")
    assert result["status"] == "CPU_PASS" and result["verification_scope"] == "fixture"
    assert not result["acceptance_eligible"] and result["n_samples"] == 7
    assert result["dev_seed"] == trained["cfg"]["training"]["dev_seed"]
    assert result["examples_sha256"]
    output = Path(args.output_root)/"E1/rice"
    assert (output/"metrics.json.sha256").is_file()
    assert len(json.loads((output/"native_maps.json").read_text())) == 7
    assert g2.write_comparison(args.output_root,{("E1","rice"):result})["coverage"]["E1"] == 0
    args.resume = True
    saved = valid_result()
    monkeypatch.setattr(g2,"completed_result",lambda *a: saved)
    monkeypatch.setattr(g2,"restore_model",lambda *a: pytest.fail("Valid resume must not load/infer again"))
    assert g2.evaluate_category(args,"E1","rice") == saved


def test_preflight_blocks_protocol_mismatch_before_e2_inference(tmp_path,monkeypatch):
    monkeypatch.setattr(g2,"evaluate_category",lambda args,exp,cat: (
        dict(status="READY",pair_protocol_sha256="E1 protocol") if exp == "E1"
        else pytest.fail("Different E2 protocol must be rejected before inference")))
    monkeypatch.setattr(g2,"prepare_job",lambda *a: (None,None,None,None,{"scientific_protocol":"other protocol"}))
    code = g2.main(["--device","cpu","--categories","rice","--preflight","--output-root",str(tmp_path)])
    summary = json.loads((tmp_path/"preflight/summary.json").read_text())
    assert code == 2 and summary["coverage"] == {"E1":0,"E2":0}
    rice = summary["records"][3]
    assert rice["E1_status"] == "READY" and rice["E2_status"] == "BLOCKED"
    assert "budget differs" in rice["E2_reason"]


