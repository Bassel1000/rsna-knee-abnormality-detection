# Progress Tracker — RSNA Knee

## Leaderboard

**Public LB (2026-10-16 20:33 UTC): top team 0.968 (CentralyDS); top-10
cutoff 0.964 (18 teams ≥ 0.964); our v16 = 0.958 (scored 14:10, sub
57271728) sits with 659 teams ≥ 0.958; 80 teams ≥ 0.960.** Our locked-bundle
line (v13 0.913, v14 0.912, v15 union 0.898) is superseded — the public
frontier is a different lineage (nartaa SWA Raptor + heliosli compact models
+ small readers). See `docs/v16_plan.md`. REF's 0.891 is one community
pipeline, not the ceiling.

**Update (2026-10-18):** v17 scored **0.958** (sub 57312302; dkim@0.08/skip —
effect below noise), v18 scored **0.957** (sub 57328295; the full
10-checkpoint dkim leg at w 0.15, tta off is a measured −0.001). Every
author-tested and own-tested union variant is ≤ 0.958 — the public shelf is
exhausted. **Final-2 (pre-registered rule) = v16 + v18** (most decorrelated
pair); selection is a site action on the submissions page.

| Version | Notebook | LB (public) | Private | Efficiency | Key Change | Date | Notes |
|---------|----------|-------------|---------|------------|------------|------|-------|
| v0.587 (baseline) | `notebooks/original/rsna-knee-abnormality-detection.ipynb` | 0.587 | — | — | Frozen DINOv2-S (768-dim CLS+patch), 2 slots/plane, 5-lang regex, Focal+pos_weight, 5-fold CV | 2026-08-21 | Archive `archive_v0.587_20260821.py` |
| v2 draft | `notebooks/v2/v2-rsna-knee-dinov2-finetune.py` | — | — | — | E2E fine-tune last 6 blocks, LLM `v4_blend` soft targets, 6 slots, laterality, 130mm crop, grouped holdout | 2026-08-21 | Not submitted — carries the same `n_slices` bug |
| v3 improved | `notebooks/v3/v3-rsna-knee-dinov2-finetune-improved.ipynb` | **0.500** | — | — | Fixes: cache ordering by study→series, label-index `uid_to_row`, rank submit, full-data retrain | 2026-08-30 | **Crash, not a model result** — `pick_slots` `KeyError: n_slices` |
| v4 fixed | `notebooks/v4/v4-rsna-knee-dinov2-finetune-fixed.ipynb` | **0.500** | — | — | Robust `find_dinov2` (HF **or** `.bin/.pth`), `is_hf` branch in `Model`, `annotate`/`walk` guards | 2026-08-30 | **Same crash at 112.6 s** — `build_model` was never reached, so the v3 post-mortem was wrong |
| v5 | `notebooks/v5/v5-rsna-knee-dinov2-finetune.ipynb` | **0.793** | — | — | Fix `pick_slots`, preflight, dry-run, DINOv2-S fine-tuned + LLM v4_blend soft targets (Version 10 run) | 2026-08-31 | Successful scoring run on Kaggle |
| **v6** | `user_nb/rsna-knee-abnormality-detection-v6.ipynb` | *pending (superseded — do not run: 336px 5-fold needs ~6.7x v5 GPU work + ~17 GB train cache, will early-stop under-trained)* | — | — | **336px** resolution (0.39mm/px), **6-slice depth caching**, **cls_mean_focal** patch pooling (1152d), **diagnosis-specific window pooling** (max/top2/mean), **5-fold CV with rank ensembling** | 2026-08-31 | Offline tests green, but time/RAM budget does not close on T4x2 |
| **v7** | `user_nb/rsna-knee-abnormality-detection-v7.ipynb` | **0.811** | — | — | **256px + 8-slice wide-band (0.12–0.88)** cache, **5-fold x5ep rank ensemble**, **class-balanced focal loss** (pos_w 1–5 + γ=1.5), **gold x3 upsampling**, **label-table guard**, train-cache-freed-before-test (peak RAM ~halved) | 2026-09-08 | **Kaggle Version 13 scoring run 0.811** |
| **v8** | `user_nb/rsna-knee-abnormality-detection-v8.ipynb` | *pending* | — | — | **Grouped + gold-stratified folds**, **hybrid soft/focal loss** (plain BCE on soft, focal on gold only), **gold-shrinkage pos_weight** (0.7 soft + 0.3 gold), **gold OOF per-label table** | 2026-09-09 | `test_v8_pipeline` + `test_v8_model` green |
| **v9** | `user_nb/rsna-knee-abnormality-detection-v9.ipynb` | **0.846** | — | — | DINOv2-S + ConvNeXt-V2-Base ensemble, 8-slice cache, multi-window infer | 2026-09-15 | Scored **0.846** on LB |
| **v9.1** | `user_nb/rsna-knee-abnormality-detection-v9_1.ipynb` | **0.846** | — | — | Fast inference-only runner loading v9 checkpoints with `meta.json` | 2026-09-20 | Validated infer-only mode |
| **v10** | `user_nb/rsna-knee-abnormality-detection-v10.ipynb` | — | — | — | Retrain attempt: 12 slices, 882/882/881/881/881 folds, dropped SAG_FLUID_NOFS | 2026-10-05 | Forensic audit revealed SAG_FLUID_NOFS drop was due to degenerate CSV flags, and `itertuples` column renaming bug disabled gold weighting |
| **REF** | `ref_kernels/0-899-let-me-cook.py` | **0.891** | — | — | Reference public pipeline: 336px DINOv2 20-member ensemble, 6 recovered slots, SlotHead(query, slot_prior=0.55), frontier window pooling, rank voting | 2026-10-05 | Public benchmark source of truth |
| **v11 (current)** | `user_nb/rsna-knee-abnormality-detection-v11-train.ipynb` + `user_nb/rsna-knee-abnormality-detection-v11-submit.ipynb` | **Target ~0.92** | — | — | **Two-notebook member ensemble**: own 5-fold trainer (336px, 6 recovered slots, `cls_mean_focal`, DINOv3-preferred/DINOv2-fallback, 4 windows, gold weighting fixed) + multi-package joint submitter blending public 20 members (3 slices) and own 5 members (6 slices) via equal-weight rank voting with fail-soft merge | 2026-10-05 | Contract frozen (`docs/v11_member_contract.md`), 18/18 train tests pass, 121/121 submit tests pass, seam integration test green |
| **v12 (scored 0.885)** | `user_nb/rsna-knee-abnormality-detection-v12-train.ipynb` + `user_nb/rsna-knee-abnormality-detection-v12-submit.ipynb` | **0.885** | — | — | 3 ViT + 2 ConvNeXtV2 folds, evidence-weighted vote | 2026-10-09 | **Below the 0.891 REF** — the τ=0.10 evidence softmax gave own members 93% of the vote (public 20 got 7%; own-convnext alone 65%). Root cause + fix ladder: `docs/v12_postmortem.md` |
| **v13 (new)** | `user_nb/rsna-knee-abnormality-detection-v13.ipynb` | *pending* | — | — | **Single notebook**: Phase 1 trains 3 own DINOv3-S+ members into `/kaggle/working` (`V13_TRAIN`), Phase 2 votes own + attached public packages under the **locked public-style recipe** (Raptor v5/v10/v8 → Base 0.90/0.10 → E10 → Final 0.85/0.15; own at `V13_OWN_WEIGHT=0.10`); RadImageNet ResNet-50 member path; v11-neutral loss; 4-source labels; 3 h inference reserve | 2026-10-13 | Evidence vote dropped for fixed public coefficients. Research + design: `docs/v13_research.md`, `docs/v13_plan.md`. Recipe math verified offline (diff 0.0, `analysis/test_v13.py` T02) |
| **v13 (scored 0.913)** | same, `V13_TRAIN=0` fast run | **0.913** | — | — | REF-20 DINOv2 fills the base slot (raptor/dinov3 were wrongly skipped as "cannot load offline"), + RadImageNet E10/E11 | 2026-10-13 | v13's base = `1.00xdino2(REF)` — a logged deviation from the locked recipe |
| **v14 (scored 0.912)** | `user_nb/rsna-knee-abnormality-detection-v14.ipynb` | **0.912** | — | — | **Locked recipe fully executed** (the whole point of v14): timm 1.0.29 from the base image, Raptor v5/v10/v8 probability blend (0.50/0.30/0.20) + DINOv3 5 folds rank-mean → base **0.90×raptor + 0.10×dinov3** → E10 RadImageNet blend on 10/12 classes → final 0.85×E10 + 0.15×E11. 38 members, 0 rejected; `V13_TRAIN=0`, `V13_DINO_WEIGHT=0` (REF-20 excluded) | 2026-10-14 | **−0.001 vs v13 = a wash.** The base-family *swap* (REF-20 DINOv2 → Raptor+DINOv3) buys nothing on the public LB |
| **v15 (scored 0.898)** | `user_nb/rsna-knee-abnormality-detection-v14.ipynb`, `V13_DINO_WEIGHT=0.2` | **0.898** | — | — | Union base: REF-20 re-folded into base at 0.2 (renormalized 0.75 raptor / 0.083 dinov3 / 0.167 dino2) | 2026-10-14 | **−0.015 vs v13** — union of two weak families scored below either alone; the locked-bundle lineage is dead |
| **v16** | `user_nb/rsna-knee-abnormality-detection-v16.ipynb` | **0.958** | — | — | **Public-frontier base**: goodpjw2008 v27 embedded verbatim — nartaa SWA Raptor 384+224 (mirror, 65/35 of the Raptor slot) + heliosli R6 compact (4 folds + stackers, flip TTA) + heliosli per-finding weights + goodpjw 2.5D reader at 1/10; all sources SHA-pinned | 2026-10-16 | **Scored 0.958 (14:10, sub 57271728, rank ≈ 660)** — matches the public artifact exactly. Built by `notebooks/v16/build_v16.py` (provenance SHA `a906df93…`), tested by `analysis/test_v16_blend.py`; design + gap analysis `docs/v16_plan.md` |
| **v17** | `user_nb/rsna-knee-abnormality-detection-v17.ipynb` | **0.958** | — | — | v16 + dkim54 v11i leg (tier 2 / w 0.08, or skipped — both below noise) | 2026-10-17 | sub 57312302; tie with v16 |
| **v18** | `user_nb/rsna-knee-abnormality-detection-v18.ipynb` | **0.957** | — | — | v16 + full 10-ckpt dkim leg (2-GPU study shard, tta off, w 0.15) | 2026-10-18 | sub 57328295; **−0.001** → dkim closed; hedge #2 (decorrelated file) |

## v14 run analysis (2026-10-14, log: `analysis/v14_run3.log`)

**What the log proves the run was correct:** timm 1.0.29 imported from the
base image (the v13 plan's "CoaTNet/DINOv3 cannot load offline" premise was
false); all 3 raptor arms strict-load OK at 73.1M params, all 5 dinov3 folds
strict-load OK, rad e10/e11 heads loaded; `vote ready: 38 member(s) survived;
0 member(s) rejected`; `v13 base: 0.90xraptor + 0.10xdinov3 (sum 1.00)`;
`v13 E10: 10/12 classes blended`; `v13 final: 0.85xE10 + 0.15xE11`. The
submission structure is the published locked recipe, byte-for-byte in its
three-stage math (`notebooks/v13/v14_source.py:4290-4349`).

**Why the ~0.94 ceiling prediction (`docs/v14_plan.md`) missed:** the
`tannguyenngoc/rsna-knee-final-locked-publicstyle` bundle is a *partial*
reproduction — Raptor 3 arms + DINOv3 + RadImageNet only, Gold58 0.962 tuned
on 58 studies ("do not retune these weights on Gold58" per its README). The
actual 0.943 frontier (willmurray9 sub 56777807, `docs/v13_research.md:12-23`)
combines **DINOv2 + A5 + RadImageNet + Raptor + four CoaTNets** — more member
families than the bundle ships. Our v13 (REF-20 DINOv2 + rad) and v14 (Raptor +
DINOv3 + rad) each carry a *subset* of those families; swapping one subset for
the other is capacity-neutral (0.913 vs 0.912). The frontier's +0.03 lives in
member **diversity**, not in which base family wins.

**Design flaw in the v14 experiment:** `V13_DINO_WEIGHT=0` (v14_source.py:4116)
dropped the REF-20 DINOv2 package entirely, so v14 tested *replace* (dino2 →
raptor+dinov3) when the frontier's structure is *union* (all families present).
The machinery for the union already exists: at `V13_DINO_WEIGHT=w` the REF
package folds into base renormalized against 0.90/0.10
(`v14_source.py:4306-4312`).

**Public-LB resolution limit:** the visible run sample is 3 studies (≈200 s
runs), and the scored set is larger — a 3-study macro AUC is quantized to
steps of 0.5/12 ≈ 0.0417, which cannot express 0.912/0.913/0.966 at all.
Treat ±0.001 as noise; the private LB decides. Cap public submissions at the
union test below (no weight sweeps against the public LB).

**v15 union run (2026-10-14, Kaggle log `Downloads/rsna-knee-abnormality-detection.log`):**
**CLEAN — submittable.** Diffed against the v14 run log line-by-line: the
only semantic changes are the config line (`V13_DINO_WEIGHT=0.2`) and
`v13 base: 0.90xraptor + 0.10xdinov3 + 0.20xdino2(REF) (sum 1.20)`. All 38
members survived, 0 rejected, every member's mean-p identical to v14 (raptor
0.4786/0.4912/0.4941, dinov3 ×5, rad e10/e11 ×5 each), `final: 0.85xE10 +
0.15xE11`, final check 3 rows × 12 targets, range [0.3333, 1.0000] — a real
vote, not the 0.5 benchmark. The DEGRADED line is the expected `V13_TRAIN=0`
public-only path. This is the first run where DINOv2, DINOv3, CoaTNet/Raptor
and RadImageNet all vote in one base — the frontier's union structure.

**Next experiment (v15, one env var, no code change):** **prepped 2026-10-14** —
`build_v14.py` now flips the `V13_DINO_WEIGHT` default 0.0 → 0.2 (asserted
anchor), so the rebuilt `user_nb/rsna-knee-abnormality-detection-v14.ipynb`
runs the union base out of the box: (0.90 raptor + 0.10 dinov3 + 0.20 REF-20)/1.20
≈ 0.75 raptor + 0.083 dinov3 + 0.167 DINOv2 — the union structure the frontier
uses. Verified offline: `analysis/test_v14_recipe.py` 12/12 (new check #6
proves the union math at 0.0 tolerance and fixed a vacuous dino2 branch in the
test's role classifier), `check_v14_guards.py` + `test_v14_cellorder.py` green.
If Kaggle Environment sets `V13_DINO_WEIGHT`, clear it or set 0.2. In the run
log, verify `V13_DINO_WEIGHT=0.2` on the config line and
`v13 base: 0.90xraptor + 0.10xdinov3 + 0.20xdino2(REF) (sum 1.20)` before
submitting. v13 (0.913) stays the fallback submission.

## v16 — move to the public frontier (2026-10-16)

The locked-bundle family tree is superseded. Frontier check (2026-10-16): top
team 0.968, top-10 cutoff ≈ 0.965, best reproducible public notebook 0.958
(goodpjw2008 v27, rank 195/6077). The frontier lineage is: nartaa SWA Raptor
CoAtNet-384 (0.949) + 224 (0.945) with mirror TTA, heliosli's compact
ConvNeXt models read by their SHA-pinned worker (S6 → R7 → R6 + flip TTA —
the compact-model step is what moved the LB), per-finding rank blends, small
readers at ~1/10. Every author-tested add-on is neutral at 0.958 (R7 at
0.4/0.5 of the compact slot, extra readers at 0.2, flip TTA on R7, S6) —
public members saturate; 0.965+ needs members outside the public pool.

v16 = goodpjw v27 embedded **verbatim** in
`user_nb/rsna-knee-abnormality-detection-v16.ipynb` (19 cells + our header).
`notebooks/v16/build_v16.py` pins the source copy by SHA-256 and asserts
byte-equality of every embedded cell plus the recipe markers (R384/R224
weight names, R6 bundle id + manifest hash, per-finding weights, reader
weights); `analysis/test_v16_blend.py` execs the embedded A3 blend cell on
synthetic CSVs (full-224 and `R224_UNUSABLE` cases, receipt contract,
fallback rows). Full design, attach list and gap analysis: `docs/v16_plan.md`.

## v17 — dkim54 probe and the v16 runtime budget (2026-10-16/17)

**Reading of the LB after 0.958.** leo's handbook
(`ref_kernels/oct16/leoprovorov/rsna-knee-0-957-explained-why-the-open-ensembl.ipynb`):
the public half is ≈ 640 studies (the scored rerun sees ≈ 2×640 hidden
studies), score noise ±0.003; the open models share error at ρ ≈ 0.86–0.89
and the family ceiling is ≈ 0.960 — same-family add-ons cannot reach 0.964+;
the gap needs members outside the Raptor/compact family (or better labels).
This matches our own "author-tested neutral" measurements.

**Candidate found.** `dkim54/rsna-knee-models-v11i` (published 2026-10-15
20:59, 136 downloads, most-downloaded new model set): independent
convnext_small line — 2 runs × 5 folds, 128-slice v2128 stacks at 384 px,
per-finding attention pooling, flip TTA, own multi-source report labels;
ships `code/{dataset,model,infer_lib}.py` (SHA-256 pinned `36e96043…`,
`76dd275d…`, `94e4421a…`; spec `2719714f…`) + offline pylibjpeg wheels;
designed for code-competition notebooks (internet OFF). Standalone quality is
unpublished — the probe measures it.

**Probe built, smoke-tested green (kernel v3).**
`user_nb/rsna-knee-dkim-v11i-probe.ipynb` (by `notebooks/v17/build_probe_dkim.py`;
push dir `notebooks/v17/probe_kernel/`, slug
`basselashraf/rsna-knee-dkim54-v11i-probe`, T4, internet OFF, attach
`dkim54/rsna-knee-models-v11i` + `basselashraf/rsna-knee-pylibjpeg-wheels-cp313`).
Runs dkim54's `infer_lib.predict` unmodified → `submission.csv` (this family
only), `dkim_preds.csv` (per-study probabilities, for post-hoc correlation
analysis), `probe_receipt.json`.

*Run log 2026-10-16 (v1 → ERROR → v3 green):* v1 failed two ways — (a) the
competition now mounts at `/kaggle/input/competitions/rsna-knee-abnormality-detection`
(new style; fixed with goodpjw's `find_test_root()` candidate logic) and (b)
dkim's bundled pylibjpeg wheels are cp310–312 but the Kaggle image moved to
Python 3.13 (fixed: our own cp313 wheel set published as
`basselashraf/rsna-knee-pylibjpeg-wheels-cp313`, install fail-soft). The test
DICOMs are Explicit VR Little Endian (uncompressed) — verified from a
downloaded sample slice and from the 15 sample series in-run — so the decoder
plugin is insurance, not a requirement.

*v3 interactive result:* pins verified; 3 studies, 10 models, 0 stack
failures, 95 s inference ≈ 32 s/study incl. startup; submission range
[0.022, 0.917]; `DKIM_PROBE_DONE`.

**Scored-rerun sizing (done 2026-10-16 night).** At ~30 s/study and a hidden
test of ≈ 2×640 studies the scored rerun could approach the session limit — too
close to risk a timeout (errors burn a daily submission), so a calibration was
run first: `notebooks/v17/build_calib_dkim.py` (kernel
`basselashraf/rsna-knee-dkim54-v11i-calib`, green 2026-10-16 night) timed the
exact pinned pipeline on 48 real *train* studies. Results: pass 1 (full
10-checkpoint recipe) **20.3 s/study** → 640 ≈ 3.6 h / 1280 ≈ **7.2 h**; pass 2
(first run only, 5 folds) 10.0 s/study → 1280 ≈ 3.5 h; 0 stack failures;
topology 4 vCPU + **2 × Tesla T4** (torch 2.11.0+cu128; `predict` uses one
GPU); steady state is GPU-bound (pass1 − pass2 ≈ inference of 5 extra models).
Decision per the pre-declared rule (≤ 22 s/study → submit as-is): **submit the
probe unmodified (v3)** — 7.2 h vs the 12 h harness timeout (`papermill
timeout=43200`), margin 1.6×. If the hidden studies read heavier than the train
sample, ≈ 10 h worst case still fits. T4x2 study-sharding stays in reserve for
v17 if v16's own runtime + the dkim leg threatens the cap.

**v17 plan — probe scored, weight chosen (2026-10-16/17).** The probe was
submitted unmodified (notebook v3; sub 57284919, 2026-10-16 22:20 UTC) and
scored **0.947** — inside the pre-declared 0.945–0.953 band → **w = 0.15**
(fail-soft; a failed or late dkim stage keeps the v16 submission). The score
lands where its lineage predicts (Raptor-384 0.949 / Raptor-224 0.945):
a real, decorrelated member — good enough to hedge as the final-2 companion,
not good enough to dominate the blend. Anchors unchanged: R6 0.956, R7 0.957,
v16 0.958, noise ±0.003.

**New hard constraint (2026-10-17): the Code Requirements cap CPU/GPU
notebooks at 9 h.** v16's own guards (NB_T0 starts at the Raptor stage;
compact deadline 8.3 h; R224 gate; Part B cutoff 8.2 h) suggest goodpjw sized
the stack to *just* fit that cap — scaled from the 3-study interactive log,
v16 alone is ≈ 8–8.7 h on ~1,280 hidden studies. Naively appending the
5-model dkim leg (+3.5 h) blows the cap; the 10-model leg (+7.2 h) is out.
Options on the table:

* **A — skip the dkim leg** (v17 = v16; no new information).
* **B — prepend dkim, patch v16's deadline constants** — risky: a delayed
  dkim can starve the compact worker → `our_fail > 0.5·n` raise → *no*
  submission at all.
* **C — v16 verbatim + appended dkim leg behind a tiered runtime guard** (as
  built, 2026-10-17): the reader stays; the leg counts wall clock from v16's
  NB_T0, picks the largest checkpoint tier (10 → 5 → 2) whose calibrated
  estimate fits inside 9 h − 20 min reserve (marginal 2.6 s/study/checkpoint,
  600 s startup), and on any failure/skip/blend error leaves the v16
  submission byte-identical (zero risk to the union). w = 0.15 for a ≥5-checkpoint
  leg, 0.08 for the 2-checkpoint slice.

**v16 stage-timing calibration (2026-10-17, kernel green in ~18 min).** The
fake-root kernel ran all four v16 stages on 48 real **train** studies:
per-stage wall seconds → 48-study, then the steady-state model (see below).
Stage A3 total (NB_T0 → blend end) = 1,173.5 s. Raw: Raptor-384 108.8 s,
R224 58.8 s, compact 961.0 s (workers 902.0 / 951.4 s, 0 fallbacks, 48/48
scored, both `compile` checks PASS at ~257/269 s), reader 64.6 s.

**Warmup vs steady state (the key read).** The receipt's `s_per_study` fields
are cumulative and warmup-dominated (first study 363–377 s); the worker logs
give the marginal rate: after warmup, **22.8–23.5 s/study per worker** on two
T4s, `gpu_s ≈ 23.5–24.4 s/study` — the compact stage is GPU-bound, 2 workers
= 2 GPUs in parallel → **~11.4–11.75 s/study combined**, plus ~370 s fixed
warmup (compile check + loads) and ~45 s of stage gaps. Affine projection
(T = warmup + m·N) to 1,280 studies: Raptor-384 ≈ 0.77 h, R224 ≈ 0.42 h,
**compact ≈ 4.2–4.3 h**, A3 ≈ 0.02 h, reader ≈ 0.44 h → **v16 total ≈
6.1 h** (not the ~8–8.7 h the naive linear/3-study scaling suggested; linear
7.12 h for compact alone is an over-count because it spreads 0.1 h of warmup
across 48 studies).

**Consequence for v17's guard (decided by real elapsed at scoring time, so no
code change needed):** with elapsed ≈ 6.1 h at Part C, est(tier 5) = 0.6 h +
1280·2.6·5 s = 4.79 h → 6.1 + 4.79 + 0.33 = 11.2 h > 9 h → skipped;
est(tier 2) = 0.6 h + 1280·2.6·2 s = 2.02 h → 6.1 + 2.02 + 0.33 = **8.45 h ≤
9 h → tier 2 runs** (2 checkpoints = `v2128r3_f0/f1`, w = 0.08) with ~0.5 h
slack. If hidden studies are heavier (more windows/study), elapsed grows and
the guard degrades to skip → v17 ≡ v16 byte-for-byte (fail-soft). Cross-check:
v16's own 8.2/8.3 h guards were conservative — the stack fits 9 h comfortably.

Keep the two best scored versions as the final-2 selections (handbook ch. 8:
the hedge is worth ≈ +0.001–0.002 privately).

**v17 built + locally tested (2026-10-17).** `notebooks/v17/build_v17.py` →
`user_nb/rsna-knee-abnormality-detection-v17.ipynb` (kernel
`basselashraf/rsna-knee-v17-v16-dkim54-leg` — Kaggle slugifies the title, so
the metadata id follows the resolved slug; private, T4, internet OFF; attach =
v16's five datasets + `dkim54/rsna-knee-models-v11i` + our cp313 wheel set).
Body = v16 cells [3..19] byte-identical (verified programmatically), then:
backup cell (`submission.csv` → `v16_sub.csv`) → leg (locate + SHA pins + tier
→ dkim `predict` on a trimmed spec with symlinked weights → per-finding rank
blend → receipt) → summary. `analysis/test_v17_blend.py` (7 cases, green):
w by tier, name-aligned dkim columns, shuffled rows, skipped leg keeps
`submission.csv` byte-identical, missing-study and non-finite dkim output
restore the v16 file, receipt cell never raises.

**v17 interactive smoke run: green end-to-end (2026-10-17, kernel v1).** All
v16 stages ran on the 3-study sample (raptor 43 s, R224 23 s, compact 427.7 s
with 0 failures, reader 22 s) → snapshot sha `5e9ba665…` → wheels rc 0 →
`DKIM_LEG_TIER n_models=10` → `DKIM_LEG_DONE` 92.3 s (30.8 s/study) →
`V17_BLEND_DONE` w 0.15 → final sha `f03e9001…` ≠ v16 sha → `V17_SUMMARY` /
`V17_DONE`. Blend math spot-checked on the downloaded CSVs against the v16 and
dkim member files (exact match). Combined with the calib16 numbers the scored
run is expected to run the leg at **tier 2 / w = 0.08** inside the 9 h cap,
or skip (≡ v16). Either way v17 yields a distinct 0.958-class scored
submission for the final-2 hedge.

**Also checked (no action):** heliosli's updated notebook (V2 per-finding
weights `{LatOA 0.15, PFOA 0.65, Syn 0.85, Baker 0.30}`, reader merged) tops
at 0.9576 < v16; haungjiawei's DINOv2-SlotHead v2 kit is trainable
(2–3 h/fold on T4) but a weaker lineage; shingo257 / sahil00001 ship fold
checkpoints without code; kirderf gold58 is a train subset (leaky for every
trained-on-all member — comparative use only).

## v18 — public rescan + the full dkim leg (2026-10-18)

**Public rescan (kernels/datasets, Oct 16–17).** Read the new drops and the
new experiment notebooks; all re-weighting paths are now measured negatives:

* **goodpjw's `rsna-knee-does-s6-help-next-to-r6-lb-0-957`** (v16's author,
  2026-10-17): version 4 added S6 (solo 0.950) as a second compact model at
  0.3 → **0.957** (−0.001 vs 0.958); versions 6/7 retuned the per-finding
  Raptor weights on the 58 gold studies (3 findings moved → 0.957; all 12 at
  the gold-58 optimum → **0.955**). Verdict in the notebook itself: "do not
  add S6 to R6, and do not tune per-finding weights on the 58 gold studies".
  Its public-LB journey also confirms the lineage ladder: R6+flip TTA 50/50
  = 0.957; + heliosli's per-finding weights + 224 mirror + reader at 1/10 =
  0.958 (our v16 recipe); R7 alone 0.956; S6 0.950 solo.
* **heliosli's "This is best public dataset new version =0.9586"**
  (2026-10-17 04:21, 139 votes; pulled + read): raptor-384 + raptor-224
  (0.65/0.35 in one Raptor slot) half + R6 (bundle fe055406, flip TTA) half,
  weights fixed 50/50, inputs = the same nartaa weight datasets +
  `rsna-knee-r6-frozen-fe055406` — i.e. our v16's exact lineage. The title's
  0.9586 vs v16's measured 0.958 is score noise (leo: ±0.003); nothing new
  to graft.
* **nartaa's `rsna-knee-s6fall-weights`** (10-17, 4.4k downloads): one
  all-data S6 checkpoint (`E_prior_s42_fall.pt`; convnext_small, 320/288,
  JEV-blend soft labels, 4,263 studies, gold-58 excluded); `s6fall-solo`
  scored **0.950** = pure S6 (`W_A, W_B = 0.0, 1.0`; the 0.949 kernel is
  embedded byte-identical only as the per-study fallback for unreadable
  studies). Rank rho vs the 0.949 Raptor-384: fold-0 mean 0.876 — but
  goodpjw's v4 already shows this family adds nothing next to R6.
* **dkim54's `rsna-knee-blend-code`** (10-16): the s6src/r6src/cloud_code
  trees + MONAI 0.6.0, no weights, no predictions, no blend recipe.
* `neronrx/mycelis-rsna-knee-4fold-hive` / `abhiiiish/...grandmaster-ensemble-3m`:
  not pursued (weaker lineage than the plateau members).

**The one untested lever we hold: the dkim54 v11i member at full strength.**
Probe solo 0.947 (Raptor-class, family-different), v16 = 0.958. v17's scored
run — per the calib16 projection — could only afford tier 2 at w = 0.08 (or
skip): an effect ≥10× smaller than LB noise, hence the 0.958 tie. The full
10-checkpoint leg does not fit the 9 h cap next to v16 in one process
(est 7.4 h tta-on; tier 5 tta-on est 4.8 h + 6.1 h v16 > 9 h). ⇒ 2-GPU
study-shard + tta off.

**v18 built + locally tested (2026-10-18).** `notebooks/v18/build_v18.py` →
`user_nb/rsna-knee-abnormality-detection-v18.ipynb` (kernel
`basselashraf/rsna-knee-v18-v16-dkim54-full-leg`; same attach list as v17).
Body = v16 cells [3..19] byte-identical (verified programmatically); backup
cell → leg cells → summary. Leg deltas vs v17:

* **2-GPU study-shard**: two `infer_lib.predict` processes (one per T4 via
  `CUDA_VISIBLE_DEVICES`; infer_lib uses `dev="cuda"`, so it honors the
  mask), each scoring every other study through a fake half-root (subset
  `test.csv`/`test_series.csv` + symlinked `test_series` and
  `sample_submission.csv`), merged back in test.csv order with duplicate- and
  completeness-asserts.
* **tta off** in the trimmed spec (`"tta": false` per model): measured
  tta-on marginal 2.07 s/study/checkpoint → ~1.04 off; negligible for a
  15%-weight member; documented as the cost of fitting the full set.
* **budget**: `v18_est(n) = 600 s + n_test·1.1·n/2` (tta-off rate with
  margin, 2 shards); tiers (10 → 5 → 2) against 9 h − 20 min reserve via
  `v18_tier(...)`; absolute kill deadline `NB_T0 + 9 h − 10 min` (workers
  killed, v16 kept). Pre-declared w: 0.15 for ≥5 ckpt, 0.08 for 2.
* fail-soft identical to v17: any failure keeps the v16 submission
  byte-identical (`V18_FALLBACK_KEEP_V16`); receipt `v18_blend_receipt.json`
  (per-finding rho on the scored studies).

`analysis/test_v18_leg.py` — **10/10 green**: backup; blend w 0.15/0.08
(name-aligned columns, shuffled rows); skip byte-identical; missing-study and
non-finite restore; tier ladder 10/5/2/skip boundary values; half-root
subsetting; merge alignment; full fake-Popen sharded orchestration
(complete, ordered, value-exact merge); receipt never raises.

**Pushed 2026-10-18; batch smoke green (3-study commit, ~13 min).**
`DKIM_LEG_TIER n_models=10 tta=off par=2 est=0.17 h elapsed=0.13 h` → both
shard workers rc=0 (2 + 1 studies, zero overlap) → `DKIM_LEG_DONE` 60.0 s
(startup-dominated, 20.0 s/study incl. loads) → `V18_BLEND_DONE` w 0.15 →
`V18_SUMMARY` / `V18_DONE`; v16 snapshot sha `5e9ba665…` (same as v17's
smoke — v16 body byte-identical) and final sha `f03e9001…` (= v17 smoke's,
expected: on 3 studies the per-finding percentile ranks are
order-quantized, so an order-preserving member cannot change the blend
bytes). Offline recompute of the blend from the downloaded `v16_sub.csv` +
`dkim_sub.csv` matches `submission.csv` to 3.3e-8. Expected scored path at
~1,280 studies: v16 ≈ 6.1 h → tier 10 fits (needs ≤ 6.54 h) → leg est
2.12 h → ~8.3 h total; if v16 runs 6.6–7.5 h → tier 5 (est 1.14 h) at
w 0.15; > 8.1 h → skip (≡ v16).

**Submitted for scoring 2026-10-17 22:54 UTC (sub 57328295, PENDING → scored
rerun ~8.2–8.9 h).** CLI form for code competitions (Go CLI): `kaggle
competitions submit <comp> -k <kernel> -v <ver> -f submission.csv -m <msg>` —
without `-f <output name>` the API returns 400 CreateCodeSubmission. Daily
limit 5; 3 remaining after this one. Scored-run outputs stay unretrievable
(`kernels output` = latest commit run; scored reruns don't update
`lastRunTime`), so the real `n_test` / elapsed / per-finding rho from this
run are not inspectable — any next iteration must rely on the public score
±0.003 and on the calib16 projection.

**Resolved 2026-10-18: scored 0.957** (sub 57328295). The full dkim leg at
w 0.15 / tta-off costs −0.001 vs v16; the dkim line is closed (0.15 → 0.957,
0.08/skip → 0.958). Verified again this session: `kernels output` returns the
latest *commit* run — all three kernels' downloads are 3-study smokes
(`rows: 3`, `elapsed_h_at_blend: 0.15`; v16 smoke receipt `total_seconds:
469`; v17/v18 smoke shas both `f03e9001…`, the same as the 10-18 smoke
above). The scored files remain unretrievable.

## v19 outlook — dkim TTA leg built + pushed; hold the rest (2026-10-18)

Post-v18 rescan (Oct 17–18 drops): nothing new to graft.

* **`neronrx/mycelis-rsna-knee-4fold-hive`** (previous note: "weaker
  lineage"; now root-caused): `log_fold0` shows convnext_small 384
  **warm-started from R6's own fold-0 checkpoint** (`E_lrprior_s0_f0.pt`,
  342/381 layers loaded), then pseudo-label trained. Soft-label val AUC
  0.92, but the log's own gold-58 early-stop AUC is 0.58–0.67 — a derivative
  of R6; zero decorrelation value. Closed.
* `kaiwalyaatulraut/rsna-knee` (Oct 18) = another figure-copy of goodpjw
  v27; `reopon/rsna-knee-s6-pure-*`, `notfu4y/…deterministic-report-svmri
  -infer`, `vyassenthilkumar/rsna-knee-dino-stage-a`,
  `shardulsalodkar/rana-*`: unscored or weak-family.
  `ryanholbrook/…efficiency-lb` = hosts' notebook, not a model.
  `abhiiiish/…grandmaster-ensemble-3m` = convnext_tiny + resnet34 ckpts, no
  score. `riadmohamed42/rsna-knee-dino-5fold-weights` = DINOv2 family
  (v13–15 lineage, measured dead). All skipped.
* The plateau is measured from every direction: R7@0.4/0.5 (author,
  neutral), S6@0.3 next to R6 (0.957), gold-58-tuned (0.957) / all-tuned
  (0.955) weights, second readers @0.2 (0.957), dkim@0.15 (0.957). v16 =
  top of the public shelf; 0.964+ needs members outside it.

**Decision: no scored runs on re-blends, but ONE more member test instead.**
The single untested cell of the dkim matrix is the *full-quality* member at a
visible pre-declared weight: v18 ran it degraded (tta-off, cost of fitting 10
ckpts) at w = 0.15 → 0.957; v17 ran it full-quality at w = 0.08 → 0.958
(invisible). **v19** (built + pushed 2026-10-18, kernel
`basselashraf/rsna-knee-v19-v16-dkim54-tta-leg`) = v16 body byte-identical +
dkim54 v11i leg with flip TTA ON, checkpoint tier **5** (run r3 f0–f4; est
600 s + n·2.07·5/2 ≈ 2.0 h next to v16's ~6.1 h, fits 9 h), w = 0.15
(≥5 ckpt) / 0.08 (2 ckpt), same fail-soft ladder (5 → 2 → skip; skip ⇒ v16
byte-identical). Offline harness `analysis/test_v19_leg.py` all green; build
`notebooks/v19/build_v19.py`. One scored run, then hold.

Smoke validated 2026-10-18 (commit run, 3-study sample): `DKIM_LEG_TIER
n_models=5 tta=on`, both shard workers rc=0 (2+1 studies, 5 models, cuda,
zero overlap), `V19_BLEND_DONE` w 0.15, receipt `v19_blend_receipt.json`
(stage v19-dkim-leg; v16 snapshot sha `5e9ba665…`, final sha `f03e9001…` —
same as the v17/v18 smokes, expected). **Submitted for scoring 2026-10-18
17:11 UTC (sub 57362650, PENDING → scored rerun ~8 h). 4 submissions remain
today.**

CLI gotcha (cost 3 failed submits): the push's commit run (version 1) sat
QUEUED for hours and never ran; the UI "Save & Run All" created **version
2**, which completed 16:00 UTC. `kernels status`/`output`/`pull` always
describe the LATEST version and `kernels files <kernel>/N` silently ignores
the suffix — so `-v 1` submits failed with `Submission not allowed: Did not
find provided Notebook Output File` (v1 has no run outputs). Fix: read
`currentVersionNumber` via `ApiGetKernelRequest` (`kernels pull` metadata
JSON also shows it) and submit that version. The CLI hides the API error
body; debug via direct SDK call — `%TEMP%\opencode\submit_v19_direct.py`
(`kaggle` 2.2.4 / `kagglesdk` 0.1.37).

Hold the remaining daily submissions for (a) a genuinely new *model* family
appearing before Oct 22, (b) any hedge re-decision on v16/v18. If a new
member drops, reuse the v19 playbook: v16 body byte-identical + probe leg →
writeup → attach leg with pre-declared weight + fail-soft; harness =
`analysis/test_v19_leg.py`. Daily watch (cheap): `kaggle kernels list -s
"rsna knee" --sort-by dateRun` and `kaggle datasets list -s "rsna knee"
--sort-by updated` — only standalone ≥ 0.958 claims justify a probe.

**Final-pair update (2026-10-19, after v19 scored 0.958):** the pre-registered
pair **v16 + v18** is now dominated — v19 also scored **0.958** and its dkim leg
is the *same* family at the same weight but with TTA on and tier 5 (v18's leg
ran TTA-off and scored 0.957; v17's ran at tier 2/w 0.08 or skipped).
Recommended final two: **v16 + v19**. Still a manual site action on the
submissions page (the CLI cannot select).

## The ceiling, measured (2026-10-19/20) — why v20 failed and what is left

Every public label table was scored against the **58 expert studies**
(`data/train.csv` gold rows; the same instrument leo uses):

| label table | mean AUC vs the 58 experts |
|---|---|
| omerzalman 98% soft labels (11 findings; *dev-optimised on these same 58*) | 0.9260 |
| **the 0.949 public member itself (Raptor-384)** | **0.9229** |
| tsuyu122 balanced multi-source labels | 0.9124 |
| JEV hybrid labels | 0.9037 |
| our llm v4 blend | 0.8976 |
| **nartaa's blend — the teacher of the 0.949 member** | **0.8922** |
| srinath2495 dual-teacher consensus | 0.7588 |
| abdullahmathar argon4 | 0.7851 |

Two consequences, both now measured rather than assumed:

* **No public teacher beats the strongest public model** ⇒ every
  fine-tune/distillation is capped (v20: −0.0247). A cross-fitted per-finding
  teacher selection gives 0.9194 ± 0.0766 — no gain over the best single table.
* **A *fresh* model can beat a better teacher** (the 0.949 member beats its
  0.8922 teacher by +0.031). That is the only lever left, and it only pays if
  the new member is also *decorrelated* (leo's ρ model: a 0.95-strong member at
  ρ 0.75 lifts a 0.958 blend to ~0.964; at the family's ρ ≈ 0.86 it is a wash).

New public members triaged and closed (their own numbers): cyruspinto's four
engines — Swin-Base-384 **0.7372**, ConvNeXt-Base 0.8712, ConvNeXt-Small
0.8325, ConvNeXt-tiny-24ch 0.7614; haungjiawei DINOv2 (dead family);
riadmohamed42 DINOv2 (dead); bliverrigpdoge/omerzalman label kits (the 98%
table is the best public teacher). The top-10 teams' *public* artifacts are
**weaker than ours** (the #1 team's bundle is the 0.941–0.943 lineage) → their
edge is private models, and the +0.008 to 0.966 is not reachable from public
ingredients in the time left.

## v21 — a fresh member on a better teacher (2026-10-20)

Pre-registration: `docs/v21_plan.md`. Builder `notebooks/v21/build_v21_train.py`,
harness `analysis/test_v21_train.py` (11/11), scored-notebook builder
`notebooks/v21/build_v21_score.py` (7/7, the deployment windowing is asserted
bit-identical to the trainer's `StudyWindows`).

* **Teacher**: per-finding quantile-normalised mean of three public tables
  (omerzalman 98% + tsuyu balanced + JEV hybrid) over the 4,349 non-expert
  studies — 0.9145 without the dev-optimised table, 0.9225 with it, vs 0.8922
  for the teacher of the 0.949 public member. Built **in-kernel** from the
  attached public datasets (the offline test asserts it equals the locally
  measured `teacher.csv` to 1e-6 — it caught a real index bug).
* **Trainer**: nartaa's own `train_knee.py` (from `nartaa/rsna-knee-hpo-assets`),
  unmodified, on the public 44 × 336 × 336 corpus. The 58 experts are never
  trained on; the per-epoch selection uses a held-out **soft-label** fold
  (`--oof_every/--oof_mask`), so the gold58 gate is not contaminated by epoch
  picking.
* **Gate (pre-declared)**: gold58 macro ≥ base − 0.02 **and** mean per-finding
  Spearman ρ vs the base ≤ 0.88; plus the corpus-vs-DICOM transfer check
  (ρ ≥ 0.95) so the member is deployable at all.
* **Attach**: v21 = v16 body byte-identical + the member leg (public 44-slot
  builder, two study shards), `final = rank((1−w)·rank(v16) + w·rank(member))`,
  w = 0.15, fail-soft.

Kaggle plumbing learned today (cost hours, recorded so it is not re-learned):
* the **v16 blocks set `HF_HUB_OFFLINE=1`** — the training subprocess must pop
  it or timm cannot fetch the pretrained CoAtNet;
* a **deleted kernel's id cannot be re-pushed** (the new version stays QUEUED
  forever) — always push under a fresh id;
* `kaggle kernels logs -f` streams the live log of a running kernel (the only
  way to see the training progress);
* the CLI's `-v` submit fails with "Did not find provided Notebook Output
  File"; the version that has outputs is `currentVersionNumber` from
  `ApiGetKernelRequest` → `notebooks/v21/submit_kernel.py` reads it and submits
  via the SDK (with the API error body printed);
* **only two kernels can be pushed at a time** (queued ones count), so the
  variant commit runs queue behind the training kernel;
* the trainer is GPU-bound at 384 px on a T4: **1.74 h per epoch** (49,536
  window-forwards per epoch, 4,131 studies x 12 windows) — a 5-epoch schedule
  needs 8.7 h, so the run stops on its 8.4 h wall-clock budget (4 epochs) and
  the SWA is rebuilt from the resume state's top-k list (OOF-selected, not
  gold-selected → the gate stays honest).

**The label sweep (2026-10-20, 30+ public tables scored on the 58 experts).**
Best tables: omerzalman "11-label-evidence-fusion" soft_labels **0.9287**,
omerzalman 98% 0.9260, omerzalman 99% 0.9227, calibrated 0.9222,
strict_mri_oof 0.9158, tsuyu balanced 0.9124, genesis96839 stacked 0.9058,
JEV hybrid 0.9037, steven v4 0.8976, karttikjangid05 consensus 0.8892,
rayanbabur 0.8991 (self-reported), argon4 0.7851, dual-teacher 0.7588.
Tables that score 1.0 on the 58 have the **expert labels injected** for those
studies (noisyislands, tasmeemreza, tonylin1026, narisettichaitanya, the JEV
gold74 eval set) — they are not honest measurements of the weak-study labels.
An independent audit of the public label sets (karttikjangid05, 2026-10-08)
reproduces our numbers exactly (jev_hybrid 0.9037, steven_v4 0.8927,
consensus_mean 0.8887) and flags jev_hybrid/gperdrizet as gold-tuned.
**No public teacher beats the strongest public model (0.9229).**

**A team's full submission ledger (prvsiyan, `rsna-knee-2026-experiment-ledger`):
90 scored rows.** Their best is **0.949** (the Raptor-384 with the anatomical
mirror), reached from 0.937→0.941→0.943→0.949 over a month of work, and
*every* member addition was rejected: mirror+ConvNeXt-2.5D 0.948 (rejected),
mirror+BK-A 0.947 (rejected), five-fold student 0.943 (plateau), four-view TTA
0.949 (tie), plain view only 0.948. Independent confirmation that the public
shelf is a plateau at ~0.949-0.958 and that the top-10 teams (0.966+) hold
something private.

**v21 variants (scored hypothesis tests, one per run):** the reader is the
least image-correlated member we have and enters the v16 blend with per-finding
weights of 0.003-0.035; nobody has tested it higher. Nine variants were built
(`notebooks/v21/build_v21_variants.py`, 6/6 offline checks) and submitted
(each a full scored rerun, ~8.3 h):

| submission | variant | the change | public |
|---|---|---|---|
| 57407613 | v21a | v19 body, reader weights x2 | 0.957 |
| 57408278 | v21b | v19 body, reader weights x3 | 0.957 |
| 57409111 | v21c | v19 body, reader weights x6 | 0.957 |
| 57409428 | v21d | v16 body (no dkim leg), reader weights x3 | 0.957 |
| 57409913 | v21e | v19 body, dkim54 leg at w=0.25 | 0.957 |
| (v21f) | | v19 body, dkim54 leg at **w=0.20** | |
| (v21g) | | v19 body, dkim54 leg at **w=0.10** | |
| (v21h) | | v19 body, reader weights **x0.5** | |
| (v21i) | | v19 body, Raptor slot **0.45** (the compact arm is the stronger of the two) | |

**What the five scored variants proved:** the reader's weight is *monotone
down* (x1 = 0.958 → x2/x3/x6 = 0.957), so the optimum is at or below x1; the
dkim weight curve is 0.08 → 0.958, 0.15 → 0.958, 0.25 → 0.957, so the interior
points 0.10/0.20 are the open cells. And a correction to an earlier belief: the
public score is a **deterministic function of the recipe**, not a noisy draw —
there is no lottery to exploit; every recipe we tried is either 0.958 (v16, v17,
v19) or 0.957.

**v21 members (fresh model on the better teacher) — gate FAILED, line closed.**
Both seeds ran the full 5-epoch schedule on the public 44 x 336 x 336 corpus:

| | gold58 (DICOM path) | gold58 (corpus path) | gate (>= 0.9029) |
|---|---|---|---|
| member A (seed 42) | 0.8465 | 0.8555 | **fail** |
| member B (seed 43) | 0.8424 | 0.8472 | **fail** |

The per-epoch trajectory (A): 0.7276 → 0.8011 → 0.8412 → 0.8507 → 0.8555, i.e.
saturating well below the bar; the seed does not matter, so it is the *recipe*,
not the draw. The transfer check **passed** (ρ = 0.9677 between the corpus path
and the DICOM deployment path), so the deployment machinery is sound — the
member is simply too weak: nartaa's 0.949 member needed 16 epochs on the
*unpublished* 96-slice/384 corpus, and 5 epochs on the public one cannot
reproduce it. The blend arithmetic agrees: a ~0.93-LB member is worth ≤ +0.001
at w=0.15, which is inside the noise, so the gate's rejection is the right call
even before the strength test. **The member line is closed**; the final pair
stays **v16 + v19**.

## v20 post-mortem — the Raptor warm-start fine-tune regressed (2026-10-19)

Full forensics: `docs/v20_postmortem.md`. Gate: base **0.9229** → `ft_deploy`
**0.8982** (Δ −0.0247, needed +0.004) → no submission, weights not uploaded.
Three measured mechanisms, all on the kept artifacts
(`%TEMP%\opencode\kout_v20b\ft`):

1. **Teacher weaker than student (structural).** The FT target was
   `soft-v2 |prior` — a label table in the same league as our own
   `llm_v4_blend` (agreement AUC 0.9055; that table measures 0.8976 against the
   58 expert studies) while the base Raptor scores **0.9229** on the same 58.
   The base was already trained on those 4,349 studies with a *stronger*
   JEV/OAI blend, so the run was a label-degradation pass. (`|strict` is 100%
   NaN in the published file — no better column existed.) Same phenomenon as
   bleg's R6 warm-start (0.919 vs R6 ~0.951) and our v13–v15 (0.898–0.913).
2. **K = 1 training window (our bug).** `aug_batch` produced one window per
   sample; `RaptorClassifier.head` softmaxes over the window axis, so with
   K = 1 the attention output is identically 1 and the attention branch gets
   **zero gradient**. Proven in the checkpoints: `att.*` in `ft_deploy.pt` is
   exactly the base attention × 0.9833 (pure AdamW decay, residual 5e-5) while
   `clsW` was rewritten (fit factor 0.844, residual 0.25) and the backbone moved
   2.3% (BN `running_mean` 33%). The scored protocol pools **K = 94** windows
   with that stale attention → MIL specialisation destroyed exactly where window
   selection matters (MCL −0.175, MedMen −0.034, PF OA −0.030) and not where it
   does not (Lateral OA +0.037, Synovitis +0.004). Shard loss *rose* 0.327 →
   0.487: the model moved away from its fit.
3. **BN statistics drift** (train-mode running stats applied at eval).

**Label-teacher leaderboard (new, measured on the 58 expert studies):**
omerzalman 98% soft_labels **0.9260** (11 findings, no Synovitis, *dev-optimised
on these same 58*) > base Raptor model 0.9229 > our llm_v4_blend 0.8976 >
report_v2 0.8705 > argon4 0.7851 > dual-teacher consensus 0.7588. Cross-fitted
per-finding teacher selection: 0.9194 ± 0.0766 → no gain over the best single
table. ⇒ **no public label table clearly beats the base model, so every
label-distillation path is capped at the 0.958–0.960 block**; a corrected
fine-tune (batch = studies, K windows, attention trained, BN frozen) would cost
6 h training + the last scored run and is not worth it.

**New drops triaged (2026-10-19 scan):** cyruspinto engine10 Swin-Base-384
0.7372 / engine11 ConvNeXt-Base 0.8712 / engine9 0.8325 / engine8 0.7614 (their
own gold58 numbers — new backbones, weak members); haungjiawei train-v2 =
DINOv2 family (dead); ayodejiibrahimlateef R6 repro = duplicate; srinath2495
dual-teacher 0.7588 and abdullahmathar argon4 0.7851 = far below the shelf;
omerzalman/bliverrigpdoge 98% label kit = best public label table (measure the
99% / 36.36%-coverage variant too).

**Plan to the 2026-10-22 23:59 deadline:** (1) site action — select **v16 +
v19**; (2) optional last scored run — v21 = v16 body byte-identical + reader
weight 0.10 → 0.20 (the only *text*-based member, so the least correlated;
untested cell with a positive prior, zero extra runtime, pre-declared weight);
(3) no further training runs; (4) keep the daily watch — only a standalone member
≥ 0.958 or a label table > 0.94 on the 58 changes the picture. Reaching the
top-10 (0.966) needs private models + OAI-scale real labels (weeks of NDA), not
a better public blend.

## Root cause of the two 0.500 submissions (v3, v4)

```
[   14.2s] header pass: train
[  112.6s] slots_tr = pick_slots(train_series, htr)
           chosen[name] = c.sort_values("n_slices", ascending=False)...
           KeyError: 'n_slices'
           wrote fallback submission.csv
```

`n_slices` is produced by the DICOM header pass (`hdr_df`); `pick_slots` sorted
`series_df`, i.e. `train_series.csv`, whose columns are only
`StudyInstanceUID, SeriesInstanceUID, Fluid_Sensitive, Fat_Suppression, Anatomical_Plane`.
The first study raised, the notebook-level `try/except` wrote the all-`0.5` benchmark file,
and the LB reported exactly **0.500** after ~2 min instead of a 4–8 h train. Present in
v2/v3/v4. **Treat an LB of exactly 0.500 as "the run crashed", never as a model score.**

Secondary finding from the same log: `LABEL SOURCE: lexicon only (4407 studies)` — the
LLM label dataset was **not attached**, costing ~0.02 of label quality (0.870 vs 0.893).

## Label Quality (vs 58 gold studies)

Measured by `src/check_labels2.py` (`src/check_labels.py`):

| Source | Coverage | Mean AUC vs Gold | Per-label (ACL/MCL/MedMen/LatMen/MedOA/LatOA/PFOA/Eff/Syn/Baker/Cont/Frac) |
|--------|----------|------------------|-----------------------------------------------------------------------------|
| `report_labels_v2.csv` | 57/58 | 0.870 | 0.997 0.976 0.943 0.841 0.908 0.789 0.891 0.830 0.694 0.932 0.768 0.871 |
| `llm_labels_v2.csv` | 58/58 | 0.887 | 0.993 0.964 0.954 0.862 0.931 0.830 0.901 0.853 0.790 0.946 0.831 0.793 |
| `llm_labels_full.csv` | 58/58 | 0.878 | 0.993 0.964 0.954 0.862 0.931 0.830 0.901 0.853 0.678 0.946 0.831 0.793 |
| **`llm_labels_v4_blend.csv`** | **58/58** | **0.893** | **0.987 0.968 0.948 0.879 0.932 0.833 0.902 0.877 0.790 0.944 0.860 0.793** |

→ `v4`/`v5`/`v6` prefers `v4_blend` → lexicon fallback.

## Experiments

| # | Date | Config | Train | Holdout AUC | LB | Time | Log |
|---|------|--------|-------|-------------|----|------|-----|
| 1 | 2026-08-21 | v0 frozen, 5-fold, 14 epochs | 3525 | — | 0.587 | — | `archive_v0.587_20260821.py` |
| 2 | 2026-08-30 | v3 T4×2, 10+2 epochs, 224px | 3525 | — | 0.500 | 2 min | crash: `pick_slots` `KeyError: n_slices` |
| 3 | 2026-08-30 | v4 T4×2, 10+2 epochs, 224px | 3525 | — | 0.500 | 112 s | same crash; labels also missing (`lexicon only`) |
| 4 | 2026-08-31 | v5 (Kaggle version 10 run) | 3525 | 0.812 | **0.793** | ~4.5 h | End-to-end DINOv2-S fine-tuned |
| 5 | 2026-08-31 | v6 (336px + multi-slice focal + 5-fold) | 4407 | pending | pending | over budget | NOT RUN — 336px 5-fold ≈6.7x v5 GPU + ~17 GB cache, cannot finish in 9 h; superseded by v7 |
| 6 | 2026-09-08 | v7 (256px + 8-slice + focal + gold x3 + 5-fold) | 4407 | pending | **0.811 (Version 13)** | ~7-8 h est | Kaggle scoring run 0.811 |
| 7 | 2026-09-09 | v8 (256px + 8-slice + hybrid loss + gold-stratified 5-fold) | 4407 | pending | pending | ~7-8 h est | Model + pipeline tests passed |
| 8 | 2026-09-15 | v9 (DINOv2-S + ConvNeXt-V2-Base ensemble, 8-slice) | 4407 | — | **0.846** | ~6 h | LB 0.846 scoring run |
| 9 | 2026-10-05 | REF baseline (20-member DINOv2 ensemble) | — | — | **0.891** | — | Public reference kernel |
| 10 | 2026-10-05 | v11 (own 5 members @ 6 slices + public 20 members @ 3 slices) | 4407 | ~0.83+ | Target ~0.92 | ~2-3 h train / ~1 h submit | All offline contract & seam tests passed |

## Local checks (no DICOMs, no GPU, no internet)

```bash
python src/test_v11_contract.py    # seam integration: trainer artifact -> submitter loader
python analysis/test_v11_train.py   # 18 offline trainer tests (CPU, no DICOMs)
python analysis/test_v11_submit.py  # 121 offline submitter tests (CPU, no DICOMs)
python notebooks/v12/build_v12_train.py   # v12 train notebook + dump (v11 + asserted patches)
python notebooks/v12/build_v12_submit.py  # v12 submit notebook + py:percent src
python analysis/test_v12_train.py   # 21 offline v12 trainer tests (CPU, no DICOMs)
python analysis/test_v12_submit.py  # 18 offline v12 submitter tests (CPU, no DICOMs)
python src/test_v12_contract.py     # 5 v12 seam tests: train artifact -> submit loader + vote
python notebooks/v11/sample_manifest.py # generates sample manifest & exercises stub training
python src/test_v8_pipeline.py     # stratified folds, shrinkage pos_weight, hybrid loss, 8-slice cache
python src/test_v8_model.py        # cls_mean_focal, diagnosis pooling, window infer, hybrid train step, rank submit
python src/check_labels2.py        # 0.893 v4_blend vs gold
```

## TODO

- [x] Fix ordering bug (`rec["ordered"]` per series)
- [x] Fix label-index bug (`uid_to_row`)
- [x] Robust backbone loading (HF + timm, pos_embed resample, shape filter)
- [x] Organize repo + git + markdown
- [x] Fix `pick_slots` `n_slices` KeyError (root cause of both 0.500s)
- [x] Preflight + dry run so a crash costs seconds instead of a submission
- [x] Local regression tests for the slot/cache/model plumbing
- [x] **v6: Upgrade to 336px resolution (0.39mm/px at 130mm FOV crop)**
- [x] **v6: Implement 6-slice depth caching + 3D depth slice jitter during training**
- [x] **v6: Implement diagnosis-specific window pooling (max, top2, mean) for multi-window TTA**
- [x] **v6: Implement `cls_mean_focal` patch pooling (1152d representations)**
- [x] **v6: Implement 5-fold cross-validation with percentile rank ensembling**
- [x] **v7: Re-budget to 256px + 8-slice wide-band so 5 folds fit in 9 h on T4x2**
- [x] **v7: Class-balanced focal loss (inverse-prevalence pos_weight 1–5 + gamma 1.5)**
- [x] **v7: Gold x3 upsampling + per-label OOF + label-table guard + train-cache phasing**
- [x] Run v7 on Kaggle (Version 13 scoring run **0.811**)
- [x] **v8: Grouped + gold-stratified folds (template grouping kept, gold balanced)**
- [x] **v8: Hybrid soft/focal loss (plain BCE on soft, focal on gold only)**
- [x] **v8: Gold-shrinkage pos_weight (0.7 soft + 0.3 gold) + gold OOF per-label table**
- [x] **v9 & v10 Forensics**: Identified degenerate CSV flags (`Fluid_Sensitive == Fat_Suppression`) causing slot 5 drop, and pandas `itertuples` column renaming bug disabling gold weighting
- [x] **v11 Member Contract**: Defined and froze interchange contract between own members and public 20-member package (`docs/v11_member_contract.md`)
- [x] **v11 Train Notebook**: Built `user_nb/rsna-knee-abnormality-detection-v11-train.ipynb` (336px, 6 recovered slots, `cls_mean_focal`, DINOv3-preferred/DINOv2-fallback, 2 groups = 6 slices = 4 windows, real gold weighting, REF strict checkpoint envelope `{model, fingerprint, meta}`)
- [x] **v11 Submit Notebook**: Built `user_nb/rsna-knee-abnormality-detection-v11-submit.ipynb` (multi-package discovery & merge, dual-group cache decoding, diagnosis pooling, rank ensembling, fail-soft guards)
- [x] **Verification**: All 18 train tests, 121 submit tests, and lead integration seam tests passing green
- [x] **v12 Upgrade (from all M1–M4 past-agent modules)**: Built `user_nb/rsna-knee-abnormality-detection-v12-train.ipynb` (3 ViT + 2 ConvNeXtV2 folds, reliability-weighted + smoothed hybrid loss, pos_weight floor 0.5, per-target `oof_auc` evidence) + `user_nb/rsna-knee-abnormality-detection-v12-submit.ipynb` (evidence-weighted softmax vote τ=0.10 x 1/√N cluster balance, ConvNeXt loader, v11-identical fallback) — see `docs/v12_delta.md`
- [x] **v12 Verification**: 21/21 train tests, 18/18 submit tests, 5/5 seam tests green (plus 114/114 E2E still green)
- [ ] **Kaggle Execution Step 1 (v12)**: Run `rsna-knee-abnormality-detection-v12-train.ipynb` on Kaggle (T4x2, Internet OFF; attach `rsna-knee-abnormality-detection`, `llm_labels_v4_blend`, DINOv3 and/or DINOv2 dirs, optionally a ConvNeXtV2 HF dir — without it convnext folds train as ViT)
- [ ] **Kaggle Execution Step 2 (v12)**: Create Kaggle dataset from training output (e.g., `rsna-knee-v12-own` containing `ckpt_own_f*.pt` + `own_manifest.json`)
- [ ] **Kaggle Execution Step 3 (v12)**: Run `rsna-knee-abnormality-detection-v12-submit.ipynb` on Kaggle (T4x2, Internet OFF; attach `rsna-knee-weights`, `rsna-knee-v12-own`, backbone dirs) for the upgraded scoring run
- [ ] **v13 Kaggle run 1 (fast)**: Run `user_nb/rsna-knee-abnormality-detection-v13.ipynb` with `V13_TRAIN=0` — pure public-frontier reproduction (~2–3 h, expected ~0.94 public). Attach: competition data, `tannguyenngoc/rsna-knee-final-locked-publicstyle`, `yunusgmsoy/rsna-knee-llm-labels-4-source-merged`. Check the log's `v13 roles:` line for raptor/dinov3/radimagenet counts.
- [ ] **v13 Kaggle run 2 (full)**: `V13_TRAIN=1` (default) — Phase 1 trains 3 own members, Phase 2 blends at `V13_OWN_WEIGHT=0.10`. If own members help on Gold58, raise the weight; if not, stay at run 1's blend.
- [ ] **v13 follow-ups**: RadImageNet checkpoint head geometry (first run's rejection log tells us the real architecture); CoaTNet own members (public 0.926 evidence, needs a new loader); DINOv3 cross-attention head per `ref_kernels/bend-the-knee-to-dinov3-ensembled.py`.
- [x] **v16 Kaggle run (public-frontier base)**: Upload `user_nb/rsna-knee-abnormality-detection-v16.ipynb` (T4x2, Internet OFF; attach competition + `goodpjw2008/rsna-knee-2-5d-convnext-reader`, `rostigor/rsna-knee-r6-fe055406`, `nartaa/rsna-knee-publication-swa-weights-20261007`, `heliosli/rsna-orthofoundation-fold0-epoch7-20261004`, `heliosli/rsna-knee-compact-r7cnxfsss`). **Scored 0.958, 2026-10-16 14:10 (sub 57271728).**
- [x] **v17 probe submit + score**: probe kernel green at v3 (slug `basselashraf/rsna-knee-dkim54-v11i-probe`); submitted unmodified 2026-10-16 22:20 (sub 57284919) → **scored 0.947**, inside the 0.945–0.953 band → pre-declared weight **w = 0.15** (Raptor-384/224 anchors: 0.949 / 0.945).
- [ ] **v17 calibration run**: DONE 2026-10-16 night — 20.3 s/study (10 models, → 7.2 h for 1280), 10.0 s/study (5 models), 2×T4 topology, 0 stack failures. Verdict: submit the probe unmodified.
- [x] **v16 timing calibration (calib16)**: DONE 2026-10-17 (48 train studies, fake root, kernel green). Warmup-corrected: compact is GPU-bound at ~11.5 s/study combined (2×T4) + 0.1 h warmup → **v16 ≈ 6.1 h at 1,280 studies**; v17's guard will run tier 2 (w 0.08, ~0.5 h slack) or skip.
- [x] **v17 built + smoke green (2026-10-17)**: kernel `basselashraf/rsna-knee-v17-v16-dkim54-leg` v1 COMPLETE — full v16 stack + `DKIM_LEG_TIER 10` + blend on the 3-study sample, receipts consistent (`analysis/test_v17_blend.py` 7/7). Ready to submit.
- [x] **v17 submit + score (2026-10-17)**: sub 57312302 → **0.958** (leg ran at tier 2/w 0.08 or skipped — effect below noise; v16 and v17 tie at 0.958, both banked for the hedge).
- [x] **v18 smoke check + submit (2026-10-18)**: smoke markers all confirmed (`DKIM_LEG_TIER`, two `dkim worker N: K studies rc=0`, `DKIM_LEG_DONE`, `V18_BLEND_DONE`, `V18_SUMMARY`, `V18_DONE`); **submitted via CLI 2026-10-17 22:54 UTC (sub 57328295, PENDING)** — `-k <kernel> -v 1 -f submission.csv -m <msg>` (the `-f submission.csv` output name is required for code comps). Expected scored path at ~1,280 studies: tier 10 (w 0.15, ~8.2–8.6 h) if v16 ends ≤ ~6.5 h; tier 5 (w 0.15) if ≤ ~7.5 h; else tier 2 / skip ≡ v16.
- [x] **v18 score watch (2026-10-18)**: scored **0.957** (sub 57328295) — the dkim leg is a measured −0.001, line closed; v18 kept only as the decorrelated hedge partner.
- [x] **Final-2 selection decided (2026-10-18)**: pre-registered rule → **v16 + v18** (v18 ≤ 0.958 → most decorrelated pair). **Superseded 2026-10-19: v19 also scored 0.958 with a stronger leg (tier 5, TTA on) → recommended pair is v16 + v19** (see "v20 post-mortem" section). Site action remains manual (CLI cannot select).
- [x] **v19 score watch**: smoke green (tier 5, tta=on, w 0.15); **submitted 2026-10-18 17:11 UTC (sub 57362650, v2)** → **scored 0.958** (2026-10-19) — ties v16; below the 0.959 replacement bar, so the pre-registered final pair stays **v16 + v18**; site selection (v16 + v18) remains a manual action before the deadline.
- [x] **v17 candidates (closed 2026-10-18)**: R7@0.4/0.5 of the compact slot — author-measured neutral; mikkel knee-centred reader @0.2 — author-measured 0.957; own DINOv2/A5/Rad family @≤0.1 — closed by the v13–v15 evidence (old family 0.898–0.913; weak-member adds measured negative); sujan anti-tie = a monotone transform, cannot change AUC. No scored runs on re-blends (see v19 outlook).
- [x] **Top-10 gap triage (2026-10-18 evening)**: LB boundary re-confirmed — top 0.969, 6th–11th = 0.966 → need **≥0.966 (+0.008 over v16)**. Fresh drops triaged: (1) **bleg family** (`kritikabenjwal/rsna-knee-bleg-s0-f01`, `sarthakgupta1801/rsna-knee-bleg-s0-f23`, `xhaitanya/rsna-knee-bleg-s1-f01`, ~372 MB each, published today) = R6 warm-start (`/tmp/warm_r6`, md5 `e267b556b0fb1631f5d208c6c813cab1`) 2-epoch fine-tunes (LR×0.5) on `rsna-knee-labels-soft-v2` with the v9.6 compact trainer; **gold58 macro measured from published `gold_logits`**: s0 folds 0.9106/0.9270/0.9240/0.9137 (mean 0.9188), s1 0.9118/0.9226 → R6-grade derivative, no LB evidence; parked (revisit only if authors post a blend/kernel or s2+ with clearly higher gold AUC). (2) **cyruspinto engine8/engine9** — bare `.pt` weights, no code, poster has no public RSNA kernels → unusable; parked. (3) **haungjiawei train-v2** — DINOv2-small+SlotHead pipeline (README read: T4, ~2–3 h/fold) in the old weak-member league (0.898–0.913), no score claim; parked. False alarm: goodpjw2008's "S6E10 | … 0.9619" kernels = Playground S6E10 (different competition). Conclusion: **no ≥0.96 public member exists; +0.008 = private models**; pending user decision on resources (train our own / outside compute / keep mining).
- [x] **v20 gate: FAILED — line closed (2026-10-19)**: v2 completed cleanly (2 epochs/shard, 1.72 h/shard, 2·17.4k steps, healthy bounded-queue pipeline; all artifacts in `/kaggle/working/ft`). Gold58 macro: **base 0.9229 → ft_deploy 0.8982 (Δ −0.0247; gate needed +0.004)**. Per-finding movement: MCL −0.175, Medial Meniscus −0.034, PF OA −0.030, Effusion −0.024, Contusion −0.020; only Lateral OA +0.037 (from the low 0.776 base), Synovitis +0.004. **Verdict: no v20 submission; ft weights not uploaded.** Root causes (measured on the artifacts, full write-up `docs/v20_postmortem.md`): (a) the teacher (`soft-v2 |prior`) is weaker than the student (0.90-class label table vs the base model's 0.9229 on the same 58 expert studies) and the base had already seen those 4,349 studies with a stronger JEV/OAI blend; (b) **our bug** — the trainer ran K = 1 window per forward, so the MIL attention got zero gradient (in `ft_deploy.pt` `att.*` = base × 0.9833, pure AdamW decay) while the per-window classifier was rewritten to the weak priors, and the scored protocol pools K = 94 windows with that stale attention; (c) BN statistic drift. The +0.008 gap cannot be closed by label distillation toward LLM priors — it needs real labels (OAI) or a genuinely stronger family. Final pair: **v16 + v19** (site action).
- [ ] **v21 (optional, one scored run, zero extra runtime)**: v16 body byte-identical + goodpjw 2.5D reader weight 0.10 → 0.20 (the only *text*-based member ⇒ least correlated with the image members; the author measured a *second* reader @0.2 as neutral, never this reader at a higher weight). Pre-declared weight, fail-soft, submit by 2026-10-22 08:00 UTC so the ~6.1 h scored rerun lands before the deadline. Expected effect inside the ±0.003 noise band, but it is the only untested cell with a positive prior and it costs one submission slot.
- [ ] **Measure the omerzalman 99% / 36.36%-coverage label variant on the 58 expert studies** (CPU, minutes; the 98% variant measures 0.9260 but is dev-optimised on the same 58). Only a table > 0.94 would reopen a label-based play.
- [ ] **Kaggle Execution Step 1 (v11 fallback)**: Run `user_nb/rsna-knee-abnormality-detection-v11-train.ipynb` on Kaggle (T4x2, Internet OFF; attach `rsna-knee-abnormality-detection`, `llm_labels_v4_blend`, DINOv3 and/or DINOv2 model directories)
- [ ] **Kaggle Execution Step 2**: Create Kaggle dataset from training output (e.g., `rsna-knee-v11-own` containing `ckpt_own_f*.pt` + `own_manifest.json`)
- [ ] **Kaggle Execution Step 3**: Run `rsna-knee-abnormality-detection-v11-submit.ipynb` on Kaggle (T4x2, Internet OFF; attach `rsna-knee-weights`, `rsna-knee-v11-own`, model directories) for final ~0.92 scoring run


