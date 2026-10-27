# RSNA Knee Abnormality Detection — Bronze Solution

**Competition:** [RSNA Knee Abnormality Detection](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection)
**Result:** Bronze medal · best public score **0.958** (final pair **v16 + v21g**)
**Our Kaggle notebook:** https://www.kaggle.com/code/basselashraf/rsna-knee-abnormality-detection
**1st-place solution:** https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/writeups/1st-place-solution

## Task

Multimodal ML for 12 knee MRI findings (ACL, MCL, medial/lateral meniscus, medial/lateral OA, PF OA, effusion, synovitis, Baker's cyst, contusion, fracture) from paired DICOM volumes + radiology reports. 4,407 train studies (only 58 expert-labeled), ~24k series. Metric = macro-averaged ROC-AUC over 12 targets. Runtime ≤9h GPU, Internet OFF, notebook submits `submission.csv`.

## Our solution (v21g)

v21g = the **v16 public-frontier base** (goodpjw2008 v27, embedded verbatim) + the **dkim54 v11i independent member** blended at **w = 0.10**, fail-soft.

v16 base is a 4-stage ensemble:
1. **Raptor** — nartaa SWA Raptor CoAtNet (384 + 224, mirror TTA, 65/35 of the Raptor slot)
2. **Compact** — heliosli R6 (4 folds + stackers, flip TTA, 2-GPU worker)
3. **Reader** — goodpjw 2.5D ConvNeXt reader at 1/10
4. **Per-finding weights** — heliosli's per-finding weights

Appended leg (v21g): dkim54's independent convnext_small line (2 runs × 5 folds, 128-slice stacks at 384 px, per-finding attention pooling, flip TTA) at w = 0.10, run as two study-shard processes, with a budget gate and fail-soft fallback to the v16 submission.

All members are SHA-pinned. Final submission is an equal-weight rank vote, then `DataFrame.rank(pct=True)`.

## 1st-place solution (reference)

Team Kangaroo treated the competition as a **labeling** problem: LLM agents produced 12 finding probabilities, 3D ROI boxes, and per-slice lesion boxes from reports + images. A ConvNeXt-T with ROI box pooling + in-box attention + auxiliary lesion/report heads reached **0.964 public / 0.954 private** (efficiency track, single model) and **0.967** ensemble. Our v21g (0.958) is a public-ingredient ensemble; closing the +0.008 gap needs private models / real labels.

## Repository layout

```
.
├── README.md
├── .gitignore
├── rsna-knee-abnormality-detection.ipynb   # final Kaggle notebook (v21g)
├── user_nb/
│   ├── rsna-knee-abnormality-detection-v21g.ipynb
│   └── rsna-knee-abnormality-detection-v16.ipynb
├── notebooks/
│   ├── v16/build_v16.py
│   ├── v19/build_v19.py
│   └── v21/build_v21_variants.py
└── analysis/
    ├── test_v16_blend.py
    └── test_v19_leg.py
```

## Reproduction

```bash
python notebooks/v16/build_v16.py
python notebooks/v19/build_v19.py
python notebooks/v21/build_v21_variants.py   # writes notebooks/v21/variants/v21g-dkim10/
```

Offline checks (CPU, no DICOMs, no GPU):

```bash
python analysis/test_v16_blend.py
python analysis/test_v19_leg.py
```

Kaggle execution: T4×2, Internet OFF. Attach the datasets listed in the notebook metadata (Raptor, heliosli compact, reader, dkim54, pylibjpeg wheels) + competition data.
