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

**Train/val (after duplicate removal):** _to be filled from the re-run._

## Phase 2 — Baseline

_TBD_

## Phase 3 — Error analysis

_TBD_

## Phase 4 — Robustness of the baseline

_TBD_

## Phase 5 — Robust training

_TBD_

## Phase 6 — Model comparison & deployment

_TBD_
