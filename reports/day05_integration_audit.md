# Day 05 integration audit — 2026-10-08

**CODE PASS: 634 passed, 0 failed, 18 skipped** on CPU fixtures. **REAL-DATA: BLOCKED_MISSING_DATA. FULL-TRAIN / FULL-EVALUATION: not executed.** The workspace has no configured real full cache/records, DINO source/checkpoint or CUDA. Tiny/boundary scientific locks were not produced by the attached Day 04 notebook and remain unresolved; two explicit skeletons are supplied, not fabricated locks.

The user requested integration with `vits16` and supplied **Day04_REAL_Pipeline_FULL_CACHE_v3 (3).ipynb** as the actual training reference. Its cell 7 protocol overlay plus cell 28 overrides lock 150 epochs, batch64, AdamW lr0.001/weight_decay0, scheduler null, AMP bfloat16, deterministic warn-only. Adapter r32/d384, downstream projection64, frozen dinov3_vits16 blocks4/8/12, tile512/context768→512/overlap128, MeanFusion, BCE+Dice, primary AU-PRO0.05, seed42 remain fixed. Checkpoint minimum val_loss remains the rule; val_total_loss is its log-column alias.

## Reused existing components

TV1: configs/day05_representation.yaml, configs/day04_train_protocol.yaml (unchanged baseline), src/train/screen_representation.py, FeatureSelector, ContextToLocalAligner, SixFeatureProjection, FeatureCacheWriter/Reader, CachedFeatureDataset/dataloader, ResidualAdapter2d, MeanFusion, BasicDecoder and AnomalySegmentationLoss. The existing Day05_TV1C notebook is updated, not replaced with a different training framework.

TV2: scripts/build_evaluation_manifest.py, day05_full_inference.py, eval_day05_representation.py; src/eval/evaluator.py, tiny_analysis.py, boundary_analysis.py, region_stats.py, compare_representation.py, efficiency.py; scripts/aggregate_week06_multiscale.py and create_representation_lock.py. Tiling, padded crops, Hann stitching and geometry use existing utilities. The older src/tools/build_feature_cache.py has a different per-record store, so the new FULL-v3 builder calls the requested **existing sharded FeatureCacheWriter/Reader** directly.

## Actual defects and minimal corrections

| Cause | File(s) | Correction |
|---|---|---|
| Cache validation searched serialized text; a vits16 filename could pass without a backbone field | screen_representation.py, day05_contract.py | Require exact backbone metadata, SHA checkpoint identity, blocks/preprocessing/FOV/full-source flags; reject missing/wrong metadata and the unverified-cache flag |
| Actual Day04 FULL notebook hyperparameters differed from repo template | new day05_day04_full_v3_protocol.yaml, full_train_template.yaml, notebook | Extract exact notebook protocol and overrides; prefer persisted Day04 seed42 YAML on Drive; remove unresolved hard-coded paths from template |
| Runner ignored AMP and deterministic warn-only flags | screen_representation.py | Apply locked bfloat16 autocast to preflight/train/validation and respect deterministic settings; preserve loss epsilon |
| TV1 wrote resolved_config.yaml, TV2 looked for config.yaml and assumed nonexistent schema fields | build_evaluation_manifest.py, day05_full_inference.py | Read resolved config/manifest, persist original day05 architecture config in TV1, strict-load exact class and architecture; verify schema/config/checkpoint hashes |
| Handoff demanded one candidate and three distinct seeds | build_evaluation_manifest.py | Exactly R0/R1/R2, common seed42 and recomputed controlled protocol hash; validate complete budget/artifacts |
| Handoff and pixel metric manifest had unrelated, conflicting schemas | both manifest/inference scripts | Separate msila.day05.handoff.v1 and msila.day05.metric_manifest.v1; bind them by hash |
| TV2 constructor omitted mandatory DINO weights; adapter dimensions silently defaulted to r128/d512 | day05_full_inference.py | Explicit local checkpoint with producer hash comparison; use r32/d384 and ViT-S/16 with real 384-channel/patch16 validation. The old model-name default already was vits16; it was not a ViT-B default defect |
| Official loader has image_norm but no dev_synthetic split | inference script, FULL-v3 builder | Confirm actual loader API; explicit native synthetic input manifest with matching locked DEV IDs, source identities, generator metadata and GT hashes |
| Local tensors were copied as Context, identity geometry, whole-image extraction | day05_full_inference.py | Real independent Local/Context crops, bicubic-antialias Context resize as notebook, real geometry in padded Context frame, full native tiling/Hann stitching |
| Generated maps used category/index filenames and lacked ID/checkpoint provenance | inference/eval scripts | Native float32 .npy maps, per-map source/checkpoint/preprocess/stitch/GT/hash metadata; reject missing/mismatched IDs, maps and checkpoints |
| Day04 cache only stored cropped masks/features, no native synthetic image | new build_day05_cache.py | Reuse exact FULL-v3 source-disjoint hash80/20 plan, crop, seeds and anomaly types; synthesize before extraction; export native images with explicit evaluation unit. Replay existing cache requires equal local masks, generator metadata and all six DINO features |
| COMPLETE was skipped without validating artifacts; mask changes did not enter protocol fingerprints | screen_representation.py | Check COMPLETE hashes, add actual mask fingerprints, export selection_record.json with selected epoch/value/checkpoint hash |
| Resume loaded RNG tensors onto CUDA and could retain duplicate incomplete logs / change shuffle via persistent workers | screen_representation.py | CPU checkpoint loading for RNG restoration; epoch rollback of uncommitted logs; independent sampler generator using seed42+epoch−1. Fixture interrupted/resumed model equals uninterrupted weights |
| Comparison expected image_path, new native synthetic images require lossless float arrays | inference/compare_representation.py | Publish actual image_path contract; support HWC float32 .npy in image display only; no map normalization |
| Several merged tests referenced nonexistent root-level modules or old adapter constructor arguments | benchmark/dinov3/region/comparison/gradient/identity tests | Import canonical src modules; use current explicit in_dim/bottleneck_dim/projection_dim API in isolated fixtures |
| A tolerance test stored float32(1e-5), which is slightly below 1e-5 | test_feature_cache.py | Test first representable value above the unchanged tolerance and a separate exact binary equality boundary; metric/cache acceptance tolerance unchanged |
| Unconfigured external integration gates failed before checking absent assets | real-DINO / representation / ownership tests | SKIP missing default external assets; explicit invalid overrides continue to FAIL; no synthetic replacement |

## New files and purpose

- src/train/day05_contract.py: shared provenance/scientific-lock validators.
- scripts/build_day05_cache.py: real FULL-v3 builder plus verified Day04 native-DEV replay/export, using the existing cache format.
- scripts/run_day05_pipeline.py: audit/preflight/train/inference/evaluate/all/dry-run orchestration, stop on failed gate, stage resume with checksums and preserved incomplete attempts.
- configs/day05_day04_full_v3_protocol.yaml: exact actual Day04 notebook settings; no tuning.
- configs/day05_tiny_protocol.example.json and day05_boundary_protocol.example.json: unresolved skeletons requested after user confirmed they did not know where these locks were; never treated as scientific evidence.
- tests/test_day05_pipeline.py: software acceptance regressions (33 test cases including parameterizations).
- pytest.ini: declare existing integration marker.
- docs/DAY05_PIPELINE_RUNBOOK.md, this audit, status JSON, test log, code-overlay ZIP/SHA: reproducible execution and evidence handoff.

## Verification

Command actually executed:

```bash
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONPATH=. timeout 180s \
  /tmp/msila-day05-qa/bin/python -m pytest -q tests --tb=short -rs
```

Result: **634 PASS / 0 FAIL / 18 SKIP, 19.15 seconds**. PyTorch 2.14.1+cpu, torchvision0.29.1+cpu, NumPy1.26.4, SciPy1.11.4. QA used an isolated /tmp environment reusing existing installed PyTorch with the compatible system NumPy/SciPy; no scientific dependency/config override was used to obtain a real-data PASS.

Skipped: real-cache step (1), E9 external manifest (1), CUDA gate (1), real DINO feature gates (9), real DINO extractor (1), real cache-vs-online (1), full real DINO forwards (2), real image representation gate (1), real backbone ownership gate (1).

Fixtures cover correct/wrong/missing producer backbone, strict TV1→TV2 load for all candidates, seed/protocol/source identity, independent R2 Context/alignment, finite loss/gradient/weight updates, BF16 preflight, finite tiled native outputs including small/non-divisible images, pixel coordinate reconstruction, maps/GT IDs and provenance, reference metrics, actual tiny/boundary/region/comparison modules, rejection of missing/mismatched aggregation/lock evidence, interrupted epoch resume, preservation of COMPLETE, mask drift, cache rebuild/replay, archive wrappers and final TEST normal-label GT semantics. Fixtures never constitute industrial anomaly performance evidence.

Additional checks: Python compilation of source/scripts/tests and notebook code cells; git diff --check; CLI help for four entrypoints; dry-run returns BLOCKED_MISSING_DATA with no best.pt generated; code ZIP contents and notebook SHA verified.

Day04 metric sources are byte-for-byte unchanged from HEAD:

- evaluator.py: d89d413fd1e77ba754adf8f0fab5a0a8d9cd125893dc462ab933ffe87a751215
- aupro.py: b80fa72a464ff9508efc8729f1eebc6ce40d5f061b316cb30950a8c8abb2b2d6
- segf1.py: 3225701e440156a58e1d402c31b1200e6466ac164d2db8b02e29c48f4e41812e

No aggregate_week06_multiscale.py or create_representation_lock.py formula/rule changes were needed.

## Execution and remaining gates

Use updated notebooks/Day05_TV1C_Full_Train_Fabric_Colab.ipynb with reports/day05_code_overlay.zip, or the exact Bash commands in docs/DAY05_PIPELINE_RUNBOOK.md. Set COMMON once from real Day04 artifacts, then run:

```bash
python -m scripts.run_day05_pipeline --stage preflight "${COMMON[@]}"
python -m scripts.run_day05_pipeline --stage train "${COMMON[@]}"
python -m scripts.run_day05_pipeline --stage inference "${COMMON[@]}" --input-manifest "$INPUTS" --seg-f1-threshold 0.5
python -m scripts.run_day05_pipeline --stage evaluate "${COMMON[@]}" --seg-f1-threshold 0.5 --tiny-protocol "$TINY" --boundary-protocol "$BOUNDARY"
```

BLOCKED: real dataset/full cache/records/DINO checkout and weights unavailable in configured workspace; CUDA unavailable; native synthetic export requires matching raw sources; tiny area/boundary width/tolerance locks unresolved. No full scientific training or evaluation was started. Phase 2 remains gated on real Phase 1 PASS and needs its explicitly scoped category protocol; this runner enforces Phase 1 Fabric and does not auto-start Phase 2.

The pre-existing untracked configs/day04_selection_report.drive.json was inspected, preserved unchanged and excluded from the code overlay. No repository commit/push was performed.
