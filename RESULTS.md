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

_TBD_

## Phase 4 — Robustness of the baseline

_TBD_

## Phase 5 — Robust training

_TBD_

## Phase 6 — Model comparison & deployment

_TBD_
