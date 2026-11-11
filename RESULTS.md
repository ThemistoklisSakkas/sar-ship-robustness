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

_TBD_

## Phase 5 — Robust training

_TBD_

## Phase 6 — Model comparison & deployment

_TBD_
