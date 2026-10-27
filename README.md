# RSNA Knee Abnormality Detection — Kaggle

Multimodal ML for 12 knee MRI findings (ACL/MCL/Meniscus/OA/Effusion/Synovitis/Baker's/Contusion/Fracture) from paired DICOM volumes + radiology reports.

> **Competition:** [RSNA Knee Abnormality Detection](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection)
> 4407 train studies (58 expert-labeled), ~24k series. Metric = macro-averaged ROC-AUC over 12 targets. Runtime ≤9h GPU, Internet OFF, Notebook submit `submission.csv`.

## Structure

```
.
├── README.md               — this file
├── PROGRESS.md             — version history + LB scores
├── .gitignore
├── data/                   — competition CSVs (train.csv, train_series.csv, test.csv, sample_submission.csv); images ignored via .gitignore
│   └── peek_data.py        → moved to src/
├── aux_data/               — report-derived labels
│   ├── report_labels_v2.csv
│   └── steven/llm_labels_{v2,full,v4_blend}.csv  # v4_blend 0.893 vs gold (best)
├── notebooks/
│   ├── original/           — v0.587 frozen DINOv2 MIL baseline (user_nb original)
│   ├── v2/                 — end-to-end fine-tune draft
│   ├── v3/                 — v3 improved (crashed, LB 0.500)
│   ├── v4/                 — v4 (crashed, LB 0.500)
│   ├── v5/                 — v5: pick_slots fix + preflight/dry-run (LB 0.793)
│   ├── v7/                 — v7: 256px, 8-slice wide-band, focal loss (LB 0.811)
│   ├── v8/                 — v8: grouped + gold-stratified folds, hybrid loss
│   ├── v11/                — v11 trainer builder, source dump, and manifest sample
│   ├── v11_submit/         — v11 submitter builder, source dump, and modular segments
│   └── ref/original_ref_kernels/ — public kernels for reference
├── src/
│   ├── test_v11_contract.py — Seam integration: trainer artifact -> submitter loader
│   ├── check_labels2.py     — label AUC vs 58 gold studies (v4_blend = 0.893)
│   ├── test_v8_pipeline.py  — offline pipeline regression tests
│   ├── test_v8_model.py     — offline model architecture regression tests
│   └── to_ipynb.py          — py:percent -> Kaggle .ipynb
├── analysis/               — forensic analyses, bug reports, and offline test suites
│   ├── test_v11_train.py   — 18 offline trainer tests
│   ├── test_v11_submit.py  — 121 offline submitter tests
│   ├── v11_train_notes.md  — trainer budget, runtime model, and attach list
│   ├── v11_submit_notes.md — submitter merge, decode groups, and verification log lines
│   ├── budget_and_bugs.md  — v10 budget, governor, and bug forensics
│   ├── fold_forensics.md   — fold split forensics
│   └── slot_forensics.md   — DICOM slot & degenerate CSV flag forensics
├── docs/
│   └── v11_member_contract.md — frozen contract between own members and public 20-member package
├── ref_kernels/            — 0-899-let-me-cook.py (0.891 public reference kernel)
├── user_nb/                — Kaggle-ready upload notebooks (v11-train, v11-submit, etc.)
└── experiments/            — local runs / logs
```

## Quickstart (v11 Kaggle Execution)

v11 is split into **two coordinated notebooks** targeting **~0.92 LB**:

### Step 1: Train Own Members (`user_nb/rsna-knee-abnormality-detection-v11-train.ipynb`)
1. **Inputs:**
   - Competition: `rsna-knee-abnormality-detection`
   - Soft labels: `stevenleehans/rsna-knee-llm-report-labels` (`llm_labels_v4_blend.csv`)
   - Backbone: DINOv3 model directory (preferred) or DINOv2 (`facebook/dinov2` or `girishbose/dinov2-vits14-rsna-knee`)
2. **Settings:** GPU T4×2, Internet OFF.
3. **Execution:** ~2–3.5 h wall time. Writes `ckpt_own_f1.pt` … `ckpt_own_f5.pt`, `own_manifest.json`, `oof_own.npy`, `oof_uids.json`.
4. **Create Dataset:** Save the notebook version and create a Kaggle dataset from the output (e.g., `rsna-knee-v11-own`).

### Step 2: Score Ensemble (`user_nb/rsna-knee-abnormality-detection-v11-submit.ipynb`)
1. **Inputs:**
   - Competition: `rsna-knee-abnormality-detection`
   - Public package: `rsna-knee-weights` (20-member DINOv2 package with `manifest.json`)
   - Own package: `rsna-knee-v11-own` (5-member trained output from Step 1)
   - Backbone: Same DINOv2 / DINOv3 model directories used in Step 1
2. **Settings:** GPU T4×2, Internet OFF.
3. **Execution:** ~55–72 min total. Decodes public members (3 slices) and own members (6 slices = 4 windows) in separate cached passes, performs frontier diagnosis pooling (max/top2/mean), and ranks them together in an equal-weight vote. Writes `submission.csv`.

Local verification (no GPU / no DICOM downloads required):
```bash
python src/test_v11_contract.py    # seam integration: trainer artifact -> submitter loader
python analysis/test_v11_train.py   # 18 offline trainer tests
python analysis/test_v11_submit.py  # 121 offline submitter tests
python notebooks/v11/sample_manifest.py # generates sample manifest via stub run
```

## Key Ideas (v11)

| Area | Prior Baseline (v9 / v10) | v11 Improvement |
|------|---------------------------|-----------------|
| Architecture | Single monolithic script | Two decoupled notebooks: independent trainer + joint multi-package rank voter |
| Member Ensemble | Single model or 2 models | Blends **20 public DINOv2 members** + **5 own trained members** into one rank vote |
| Image Resolution | 256px | **336px** resolution with 130mm FOV crop |
| Depth Sampling | 8 slices | Dual groups: 3 slices (1 window) for public, **6 slices (4 overlapping windows)** for own |
| Slots | 5 slots (dropped SAG_FLUID_NOFS) | **6 recovered slots** using DICOM header recovery (bypassing degenerate CSV flags) |
| Feature Pooling | CLS token only | **`cls_mean_focal`** patch pooling (1152-dim representation) |
| Window Pooling | Simple mean | **Frontier diagnosis-specific pooling** (max for tears/fracture, top2 for ACL/MCL, mean for OA) |
| Fail-Soft Guard | Silent 0.500 fallback | Strict member validation, per-member fail-soft, dual-group caching, never emits 0.500 unless 0 members survive |


## Evaluation

Macro AUC: `score = mean(AUC_i), i=0..11`. Only rank matters → submission uses `DataFrame.rank(pct=True)`. See `PROGRESS.md` for per-version scores.

## Timeline / Prizes

- Oct 15 2026 — Entry & Team Merger Deadline
- Oct 22 2026 — Final Submission
- Efficiency track: `eff = (S_best - S)/(S_best - S_benchmark) * t` (minimize)

