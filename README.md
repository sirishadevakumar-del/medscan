# MedScan — adaptive, uncertainty-aware, explainable analysis for all diagnostic image types

Generalises the chest-X-ray pipeline of the STCR-2026 paper to X-ray, CT, MRI, ultrasound, fundus,
dermoscopy and histopathology. One shared DenseNet121 + EfficientNet-B0 ensemble with a classifier head per
task and an auxiliary modality-routing head.

## Pipeline (paper section → code)

| Paper | Here |
|---|---|
| III-B quality assessment (blur / contrast / artefacts) | `medscan/quality.py` — thresholds **calibrated per task** from training data |
| III-C DenseNet121 + EfficientNet-B0, MC-Dropout head (Dropout-512-BN-ReLU-Dropout) | `medscan/models.py` |
| III-D loss / Adam / LR halving / early stop / grad-clip 1.0 | `train.py` (CE for multi-class, BCE for multi-label) |
| III-E Monte Carlo Dropout, confidence tiers 0.85 / 0.70 | `medscan/uncertainty.py`, `medscan/policy.py` |
| III-F Grad-CAM | `medscan/gradcam.py` (+ quantitative deletion test) |
| Calibration (ref. [13]) | temperature scaling on the validation set + ECE / reliability diagram |
| Audit, clinician feedback, longitudinal comparison | `medscan/audit.py`, `app.py` |
| Tables II–IV, Figs 1,3,4,5 | `evaluate.py` → `report.md`, `results.json`, PNGs |

## Quick start

```bash
pip install -r requirements.txt
python tests/smoke_test.py                       # 1–2 min, CPU, synthetic data: verifies everything runs

python train.py --out runs/exp1                  # downloads MedMNIST (6 tasks) and trains
python evaluate.py --ckpt runs/exp1/best.pt      # writes runs/exp1/eval/report.md + figures
python infer.py --ckpt runs/exp1/best.pt --image scan.png          # auto-detects modality
streamlit run app.py -- --ckpt runs/exp1/best.pt                    # demo UI with feedback logging
```

For an 8 GB GPU the defaults (batch 32, AMP, 224 px) fit. First run: try `--epochs 15` to check timing
(6 tasks × 100 epochs is long). Use `--image-size 128` to speed up.

## Using your own data (any modality, incl. MRI and your CheXpert subset)

1. Arrange `root/{train,val,test}/<class_name>/*.png|jpg|dcm|nii.gz`
   (`scripts/prepare_folder_dataset.py` does a 70/15/15 split, **by patient** if you give `--patient-regex`).
2. Add an entry to `configs/tasks.json` with `"source": "folder"`, a unique `name`, a `modality`, `"path"`.
   Set `"hflip": false` where left/right matters.
3. Re-run `train.py`. No code changes are needed.

DICOM is windowed automatically. NIfTI volumes are analysed as a 2D **middle slice** — true 3D / multi-slice
aggregation is not implemented.

## Things worth knowing

- **Confidence.** The paper's `1 − σ/σmax` ignores the predicted probability (a model that always outputs 0.5
  would get confidence 1). Default here is `p(pred) × (1 − σ/σmax)`; set `confidence_mode: "paper"` in the config to
  reproduce the original definition. σ includes disagreement between the two ensemble members.
- **MC Dropout speed.** Dropout exists only in the heads, so the backbones run once and T=20 head passes are sampled.
- **Ensembling.** Member *probabilities* are averaged (paper Eq. 1 averages outputs; its text also says "concatenated").
- **Quality thresholds** are the 5th/95th percentile of each task's training images, so ~95 % pass on in-distribution
  data by construction. Hand-set thresholds in `quality_thresholds.json` if you want clinically defined ones.
- **MedMNIST** images are low-resolution (≤224 px) benchmark data. Good for developing and comparing, not a substitute
  for full-resolution clinical datasets.
- **Not a medical device.** Needs external validation, patient-level splits and clinical review before any clinical claim.
- Set `MEDSCAN_SALT` for the audit-log patient-id hashing.
