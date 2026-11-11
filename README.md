# SAR Ship Detection & Field-Robustness Study

Train a YOLO ship detector on satellite SAR imagery, measure how it degrades under
realistic field conditions (speckle, blur, low contrast, lower resolution, compression),
and make it more robust with degradation-aware training.

> Status: **work in progress** — see the phase checklist below. All metrics live in
> [RESULTS.md](RESULTS.md).

## Why this matters

SAR sees through cloud and at night, which makes it a primary sensor for maritime
domain awareness and for finding "dark" vessels that switch off AIS. A detector is only
useful operationally if we know how it behaves when image quality is worse than in the
training set. This project quantifies that.

## Phases

- [x] 0. Concepts (detection, YOLO, IoU, NMS, P/R, mAP, SAR specifics)
- [x] 1. Data: SSDD download, licence, VOC→YOLO conversion, visual checks, statistics, leakage checks
- [x] 2. Baseline training (YOLO11n), curves, held-out test evaluation
- [x] 3. Error analysis by ship size, inshore/offshore, dense groups, false alarms
- [x] 4. Field-robustness study: degradation functions × severity → robustness curves
- [x] 5. Robust training with degradation-aware augmentation
- [x] 6. Small vs larger model, ONNX export, inference timing
- [ ] 7. Final write-up, figures, limitations

## Repository layout

```
configs/        experiment parameters (YAML)
src/sarship/    all logic: data, degradations, training, evaluation, analysis
scripts/        thin command-line entry points
notebooks/      thin Colab notebooks (one per phase)
tests/          pytest unit tests
docs/           dataset licence notes
```

## Setup

Local (CPU, for development and tests):

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r requirements-dev.txt
pytest
```

Google Colab (GPU, for training): open `notebooks/00_colab_setup.ipynb`.

## Reproducibility

Seeds are fixed (`configs/default.yaml`), all parameters live in config files, and
package versions plus GPU type are recorded in RESULTS.md for every run. GPU training
is only approximately deterministic (some CUDA kernels are not bit-exact).
