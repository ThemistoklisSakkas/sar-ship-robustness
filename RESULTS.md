# Results log

Every metric produced in this project is recorded here, with the config and
environment that produced it.

## Environment

| Item | Value |
|---|---|
| Training hardware | Google Colab, Tesla T4 (16 GB) |
| Python | 3.13.16 |
| torch / CUDA | 2.11.0+cu130 / 13.0 |
| ultralytics | 8.4.202 |
| Seed | 42 |

Recorded 2026-11-04. Unit tests on Colab: 22 passed.

## Phase 1 — Dataset

Source: official SSDD release, `BBox_SSDD/voc_style` (horizontal boxes). Official split:
928 train / 232 test (46 inshore, 186 offshore). Val = 15% of train, stratified by scene, seed 42.

**Test set (official, unchanged — final):**

| subset | images | ships | small | medium | large | max ships/img |
|---|---|---|---|---|---|---|
| test | 232 | 546 | 339 (62%) | 188 (34%) | 19 (3%) | 24 |
| test/inshore | 46 | 172 | 100 | 62 | 10 | 24 |
| test/offshore | 186 | 374 | 239 | 126 | 9 | 19 |

Size bins are COCO-style on the original image scale: small < 32², medium < 96², large ≥ 96² px².

**Findings:**
- **Leakage in the official split:** 12 train images are byte-identical to test images.
  Their train copies are removed from the train/val pool (`drop_train_test_duplicates: true`);
  the test set stays official so results remain comparable with the literature.
- No empty images in any split: the model never sees a ship-free scene during training,
  so false-alarm behaviour on pure sea/land is untested by this dataset.
- Inshore images are denser (3.7 ships/img in test vs 2.0 offshore).
- Image sizes vary (≈190–670 px per side); training at imgsz 640 upsamples most images.
- Only exact duplicates are detected; near-duplicates (overlapping crops of one scene) are
  not, so a residual optimistic bias is possible. Documented as a limitation.

**Train/val (after duplicate removal, final):**

| subset | images | ships | small | medium | large | max ships/img |
|---|---|---|---|---|---|---|
| train | 778 | 1688 | 1055 | 591 | 42 | 29 |
| train/inshore | 151 | 452 | 248 | 180 | 24 | 11 |
| train/offshore | 627 | 1236 | 807 | 411 | 18 | 29 |
| val | 138 | 307 | 199 | 103 | 5 | 27 |
| val/inshore | 27 | 59 | 34 | 21 | 4 | 10 |
| val/offshore | 111 | 248 | 165 | 82 | 1 | 27 |

**Conversion cross-check** against the release's own COCO files: train.json 928 images /
2041 ships (ours before exclusion: 1760 + 281 = 2041), test.json 232 / 546,
test_inshore.json 46 / 172, test_offshore.json 186 / 374 — all identical. 0 boxes dropped.

**Duplicate pairs (train == test):** 000093=000001, 000227=000011, 000456=000229,
000033=000389, 000637=000389, 000502=000501, 000006=000631, 000237=000639, 000634=000639,
000243=000641, 001108=000739, 001122=001121. 10 distinct test images are affected; two of
them have two copies in train. Changing the train pool changes the val draw (same seed).

**Colour images:** SAR is single-channel, but 25 of 1148 used images (2.2%) contain
pseudo-colour pixels (channel spread > 16 grey levels in > 0.1% of pixels):
train 19, val 2, test 4 (000051, 000061, 000069, 000911). Coloured share ranges
0.2%–13.0% of pixels (000069 is the strongest). All 25 are exported as single-channel
lossless PNG (`to_grayscale: true`); grey images are copied byte-for-byte. Detection
was checked visually on the raw images (`figures/colour_images_raw.png`).

**Scene-leakage indicator:** 22 of the 25 colour images have consecutive ids
000046–000070, i.e. they are very likely crops of the same acquisition, and they are
spread over train, val and test by the official index-based split. This supports the
near-duplicate limitation above: neighbouring crops of one scene can sit on both sides
of the train/test boundary, so test results are likely somewhat optimistic.
SSDD provides no scene ids, so a scene-grouped split is not possible here.

## Phase 2 — Baseline

Model YOLO11n (2.59 M params, COCO-pretrained), imgsz 640, 100 epochs, batch 16,
optimizer auto (AdamW lr 0.002), seed 42, Ultralytics default augmentation with hue and
saturation jitter disabled. Run `baseline_yolo11n`, ultralytics 8.4.208, torch 2.14.1+cu130,
Tesla T4. Weights: `best.pt` (selected on val).

**Detection metrics** (conf 0.001, NMS IoU 0.7):

| subset | P | R | mAP@0.5 | mAP@0.5:0.95 |
|---|---|---|---|---|
| val | 0.971 | 0.979 | 0.993 | 0.743 |
| **test** | **0.956** | **0.952** | **0.987** | **0.709** |
| test/inshore | 0.878 | 0.913 | 0.949 | 0.636 |
| test/offshore | 0.984 | 0.991 | 0.993 | 0.738 |

P/R are Ultralytics' values at the max-F1 confidence. Timings printed by `val` are not
reported here: images were read from mounted Google Drive, so I/O dominates; inference speed
is measured properly in phase 6.

**Operating point** on test (conf ≥ 0.25, TP if IoU ≥ 0.5): TP 534, FP 56, FN 12 →
recall 0.978, precision 0.905, 0.24 false alarms per image.

**Training:** 100 epochs ran, best val mAP@0.5:0.95 = 0.742 at epoch 93; final box loss
train 0.855 / val 0.890; val box loss was still at its minimum at epoch 100 → no sign of
overfitting, the model may even gain slightly from longer training.

**Training curves** (`evaluation/training_curves.png`): train and val loss track each other
for all 100 epochs (no divergence). mAP@0.5 saturates near 0.98–0.99 by epoch ~20, while
mAP@0.5:0.95 keeps rising slowly to the end: detection is learnt early, box precision late.
The visible drop in train loss after epoch 90 is `close_mosaic` (no mosaic → easier batches).
Val P/R fluctuate by ±1–2 points between epochs (307 val ships: one ship = 0.3 points),
so "best epoch 93" vs epoch 100 is within noise.

**Reproducibility check:** an earlier run with the same config and seed (lost after epoch 94,
see below) had val mAP@0.5 0.993 / mAP@0.5:0.95 0.742 at epoch 93; this run has
0.993 / 0.742 at epoch 93. GPU non-determinism did not change the result at 3 decimals.

**Observations from the example grids** (`evaluation/examples_good.png`, `examples_bad.png`):
- Errors are mostly false alarms, not misses (56 FP vs 12 FN at conf 0.25).
- False alarms cluster on bright coastal structures (quays, breakwaters, buildings) —
  inshore precision 0.878 vs offshore 0.984.
- Misses happen at dense berths, where ships moored side by side get one merged box
  (e.g. test 001089, 000219) — the NMS/overlap case from phase 0.
- Some "false alarms" sit on bright objects at the image border or small isolated points;
  some may be unlabelled vessels (annotation omissions). To be inspected in phase 3.
- val → test drop (0.743 → 0.709 mAP@0.5:0.95) is expected: val was used for model selection.

**Incident:** the first baseline run (94/100 epochs) was overwritten when a fresh run was
started on a CPU-only runtime in the same folder. `train.py` now refuses to overwrite a run
with weights (needs `--resume`, a new `--name` or `--overwrite`) and refuses to train without
a GPU. The run above is a clean, uninterrupted 100-epoch run.

## Phase 3 — Error analysis

Model `baseline_yolo11n/best.pt` on the official test set. Predictions cached at conf 0.001;
mAP computed by our own COCO-style implementation (`analysis/ap.py`, 101-point interpolation,
COCO size-range ignore rules), sizes on the original image scale.

**Implementation check** against Ultralytics on the same test set: mAP@0.5 0.982 vs 0.987,
mAP@0.5:0.95 0.713 vs 0.709 (differences ≤ 0.005: interpolation/integration details, matching
across IoU thresholds, and letterboxing in `val` vs `predict`).

**mAP by scene and ship size:**

| subset | ships | mAP@0.5 | mAP@0.5:0.95 |
|---|---|---|---|
| all | 546 | 0.982 | 0.713 |
| all / small | 339 | 0.978 | 0.677 |
| all / medium | 188 | 0.986 | 0.774 |
| all / large | 19 | 1.000 | 0.708 |
| inshore | 172 | 0.948 | 0.645 |
| inshore / small | 100 | 0.936 | **0.599** |
| inshore / medium | 62 | 0.962 | 0.730 |
| inshore / large | 10 | 1.000 | 0.651 |
| offshore | 374 | 0.989 | 0.739 |
| offshore / small | 239 | 0.989 | 0.707 |
| offshore / medium | 126 | 0.998 | 0.792 |
| offshore / large | 9 | 1.000 | 0.764 |

**Bootstrap 95% CIs** (1000 resamples of images, seed 42):

| subset | images | mAP@0.5 | mAP@0.5:0.95 |
|---|---|---|---|
| all | 232 | 0.982 [0.968, 0.993] | 0.713 [0.685, 0.740] |
| inshore | 46 | 0.948 [0.909, 0.982] | 0.645 [0.595, 0.711] |
| offshore | 186 | 0.989 [0.984, 1.000] | 0.739 [0.716, 0.765] |

**Operating point** (conf ≥ 0.25, IoU ≥ 0.5) — recall: offshore 371/374 (0.992), inshore
163/172 (0.948); isolated ships 486/494 (0.984), crowded ships (box overlaps another ship's box)
48/52 (0.923); small 331/339, medium 184/188, large 19/19.

| scene | TP | FN merged | FN localisation | FN low score | FN missed |
|---|---|---|---|---|---|
| inshore | 163 | 2 | 4 | 3 | 0 |
| offshore | 371 | 0 | 0 | 2 | 1 |

| scene | FP duplicate | FP merged box | FP localisation | FP border | FP background |
|---|---|---|---|---|---|
| inshore | 8 | 1 | 8 | 6 | 19 |
| offshore | 0 | 0 | 0 | 5 | 9 |

**Findings:**
1. **The coast is the problem, not ship size.** Offshore recall is 99.2% and offshore small ships
   still reach mAP@0.5 0.989. Inshore loses ~10 points of mAP@0.5:0.95; the inshore/offshore
   CIs for mAP@0.5:0.95 do not overlap ([0.595, 0.711] vs [0.716, 0.765]), so the gap is real.
2. **Small ships are found but boxed loosely:** mAP@0.5 barely changes with size, mAP@0.5:0.95
   drops from 0.774 (medium) to 0.677 (small) — the few-pixel IoU sensitivity.
   The large-ship value (0.708) rests on 19 ships and is too noisy to interpret.
3. **Only 5 of the 12 misses would be recovered by a lower threshold** (low score). The other 6
   inshore misses are merged with a neighbour or boxed with the wrong extent at dense berths —
   a box/NMS problem, not a confidence problem. One offshore ship is truly missed.
4. **17 of the 42 inshore false alarms are box-quality errors next to real ships** (duplicates,
   wrong extent, one box over two ships), all at berths; the rest are clutter (19) and
   cut-off objects at the image border (6).
5. **Uncertainty:** with this test set, mAP@0.5:0.95 is known to about ±0.03 overall and
   ±0.06 for inshore. Phase 5 improvements smaller than that are not resolvable on inshore.

**Visual review of the most confident "false alarms"** (`analysis/false_alarms_top.png`,
judged by eye, no ground truth available):
- Several open-water detections (e.g. 000379 0.63, 000589 0.67, 000211 0.53, 000771 0.58) are
  compact or elongated bright targets that look like small vessels: likely **unlabelled ships**.
- **Border** detections (000739, 000379, 001069) sit on ships cut off by the image edge, which
  the annotators did not label.
- 000391 (0.81): the model boxed a ship *with its azimuth smear*, the annotation only the bright
  hull point → IoU < 0.1, so one ship becomes one FP plus one FN. Annotation convention, not
  detector failure.
- On land/coast (001019, 001159, 000619 — a pier-like line) the detections are real clutter.
- The test labels are **not** edited: relabelling the test set after seeing the model's output
  would bias the evaluation. Consequence: measured precision is a pessimistic estimate.
  Operationally these are exactly the "SAR contact without AIS" candidates that AIS
  correlation would resolve.

## Phase 4 — Robustness of the baseline

Model `baseline_yolo11n/best.pt`, official test set (232 images), 5 degradations × 5 severity
levels + clean = 26 conditions. Each test image degraded with a per-(condition, image) seed and
saved losslessly (PNG); boxes unchanged. mAP by our COCO-style code; CI = bootstrap over images
(300 resamples); recall and FP/image at conf 0.25, IoU 0.5.
Figures: `robustness/robustness_curves.png`, `robustness/degradation_examples.png`.

Hypothesis written before the run: resolution and blur hurt first (small ships, mAP@0.5:0.95);
speckle L=1 increases false alarms; JPEG holds until quality ≈ 20.

| condition | mAP@0.5 [95% CI] | mAP@0.5:0.95 [95% CI] | Δ vs clean | inshore | offshore | recall | FP/img |
|---|---|---|---|---|---|---|---|
| clean | 0.982 [0.968, 0.993] | 0.713 [0.687, 0.737] | +0.0% | 0.645 | 0.739 | 0.978 | 0.24 |
| speckle L=16 | 0.977 [0.959, 0.989] | 0.699 [0.671, 0.724] | −2.1% | 0.630 | 0.726 | 0.969 | 0.24 |
| speckle L=8 | 0.968 [0.954, 0.983] | 0.686 [0.660, 0.711] | −3.9% | 0.615 | 0.714 | 0.945 | 0.20 |
| speckle L=4 | 0.964 [0.939, 0.980] | 0.668 [0.642, 0.691] | −6.4% | 0.593 | 0.696 | 0.932 | 0.15 |
| speckle L=2 | 0.933 [0.900, 0.958] | 0.622 [0.591, 0.652] | −12.9% | 0.533 | 0.657 | 0.857 | 0.12 |
| speckle L=1 | 0.776 [0.717, 0.833] | 0.482 [0.440, 0.522] | −32.5% | 0.340 | 0.544 | 0.617 | 0.08 |
| blur σ=0.5 | 0.982 [0.967, 0.994] | 0.714 [0.688, 0.738] | +0.1% | 0.644 | 0.742 | 0.976 | 0.21 |
| blur σ=1 | 0.966 [0.946, 0.984] | 0.658 [0.628, 0.686] | −7.8% | 0.560 | 0.698 | 0.951 | 0.15 |
| blur σ=1.5 | 0.948 [0.918, 0.973] | 0.601 [0.562, 0.638] | −15.7% | 0.481 | 0.651 | 0.921 | 0.14 |
| blur σ=2 | 0.923 [0.877, 0.960] | 0.547 [0.500, 0.591] | −23.4% | 0.426 | 0.597 | 0.892 | 0.19 |
| blur σ=3 | 0.837 [0.757, 0.900] | 0.436 [0.382, 0.488] | −38.9% | 0.331 | 0.481 | 0.808 | 0.32 |
| contrast ×0.8 | 0.982 [0.969, 0.994] | 0.713 [0.688, 0.737] | −0.0% | 0.647 | 0.741 | 0.978 | 0.22 |
| contrast ×0.6 | 0.979 [0.966, 0.988] | 0.706 [0.681, 0.730] | −1.0% | 0.641 | 0.731 | 0.967 | 0.18 |
| contrast ×0.4 | 0.964 [0.946, 0.981] | 0.686 [0.659, 0.715] | −3.9% | 0.613 | 0.717 | 0.934 | 0.14 |
| contrast ×0.3 | 0.950 [0.925, 0.973] | 0.658 [0.627, 0.688] | −7.8% | 0.576 | 0.692 | 0.901 | 0.18 |
| contrast ×0.2 | 0.924 [0.892, 0.956] | 0.603 [0.568, 0.637] | −15.5% | 0.514 | 0.641 | 0.844 | 0.12 |
| resolution 1.5× | 0.971 [0.955, 0.985] | 0.683 [0.659, 0.707] | −4.3% | 0.592 | 0.720 | 0.963 | 0.19 |
| resolution 2× | 0.967 [0.944, 0.984] | 0.656 [0.627, 0.685] | −8.0% | 0.562 | 0.694 | 0.952 | 0.13 |
| resolution 3× | 0.937 [0.895, 0.968] | 0.586 [0.546, 0.624] | −17.9% | 0.468 | 0.635 | 0.908 | 0.11 |
| resolution 4× | 0.894 [0.833, 0.940] | 0.514 [0.470, 0.559] | −28.0% | 0.379 | 0.572 | 0.844 | 0.14 |
| resolution 6× | 0.762 [0.683, 0.833] | 0.366 [0.319, 0.413] | −48.7% | 0.259 | 0.416 | 0.723 | 0.25 |
| JPEG q=50 | 0.982 [0.970, 0.993] | 0.704 [0.679, 0.726] | −1.4% | 0.646 | 0.727 | 0.978 | 0.23 |
| JPEG q=30 | 0.981 [0.963, 0.988] | 0.706 [0.681, 0.729] | −1.0% | 0.637 | 0.734 | 0.967 | 0.18 |
| JPEG q=20 | 0.977 [0.960, 0.989] | 0.691 [0.665, 0.715] | −3.2% | 0.621 | 0.721 | 0.952 | 0.16 |
| JPEG q=10 | 0.952 [0.933, 0.970] | 0.649 [0.619, 0.674] | −9.1% | 0.560 | 0.685 | 0.908 | 0.13 |
| JPEG q=5 | 0.929 [0.903, 0.953] | 0.586 [0.551, 0.618] | −17.9% | 0.469 | 0.636 | 0.892 | 0.28 |

(inshore / offshore columns: mAP@0.5:0.95.)

**Breaking points** (first level losing > 10% of clean):

| degradation | on mAP@0.5 | on mAP@0.5:0.95 |
|---|---|---|
| speckle | L = 1 | L = 2 |
| blur | σ = 3 | σ = 1.5 |
| contrast | not reached | ×0.2 |
| resolution | 6× | 3× |
| JPEG | not reached | q = 5 |

**Findings:**
1. **Ranking at the worst level (mAP@0.5:0.95):** resolution −49% > blur −39% > speckle −33%
   > JPEG −18% > contrast −16%. Hypothesis confirmed for resolution and blur.
2. **Localisation breaks long before detection.** Blur σ=1 costs 1.6% of mAP@0.5 but 7.8% of
   mAP@0.5:0.95; resolution 3× costs 4.6% vs 17.9%. Ships are still found, but their boxes —
   and any length/class estimate derived from them — degrade early.
3. **Speckle fails silently (hypothesis refuted).** At L=1 recall falls from 0.978 to 0.617 while
   false alarms *fall* from 0.24 to 0.08 per image: the detector becomes less confident
   everywhere instead of hallucinating ships. An operator watching the false-alarm rate would
   see a "quieter" system, not a broken one. In contrast, blur σ=3, resolution 6× and JPEG q=5
   raise false alarms (0.32, 0.25, 0.28 per image): blobs and compression blocks look ship-like.
4. **Inshore is hit harder.** Resolution 6×: inshore −60% vs offshore −44%; blur σ=3: −49% vs
   −35%. Separating ships from quays and from each other needs exactly the fine detail that is
   lost.
5. **Harmless levels:** blur σ=0.5, contrast ×0.8, JPEG q=50 and q=30 are inside the clean CI.
   **JPEG q=20 costs only 3%**: chips can be compressed hard on a datalink.
6. Mild levels (e.g. speckle L=16) are not distinguishable from clean with these independent
   CIs; a paired comparison (same images) is more sensitive and is used in phase 5.

**Operational reading:** the detector should not be used on sensor modes more than ~2× coarser
than the training data, nor on products with clearly stronger speckle (fewer looks), without
retraining; input image quality (e.g. resolution, estimated number of looks) should be checked
before trusting a "no ship here" output. JPEG quality ≥ 20 is a safe datalink setting.

**Limitations:** SSDD pixel values are 8-bit display amplitudes with unknown scaling (possibly
log-scaled), and they already contain speckle, so the added speckle lowers an unknown
equivalent number of looks; L=1 on top of it is harsher than a real single-look product.
Contrast reduction is linear around the image mean (ranking of pixels preserved); a real low
target-to-clutter situation (rough sea) adds clutter energy and would likely be harder. Each
degradation is applied alone; real conditions combine them.

## Phase 5 — Robust training

_TBD_

## Phase 6 — Model comparison & deployment

_TBD_
