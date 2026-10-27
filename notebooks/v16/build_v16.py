"""Build `user_nb/rsna-knee-abnormality-detection-v16.ipynb` from the pinned
public notebook `goodpjw2008/rsna-knee-coatnet-r6-2-5d-mil-lb-0-958` (v27).

Why v16 exists
--------------

The locked-bundle lineage is dead on the public LB: v13 scored 0.913,
v14 0.912 (base-family swap = a wash), v15 union 0.898 (union of two weak
families scored below either alone).  Meanwhile the public frontier moved to a
different lineage (checked 2026-10-16):

- nartaa's SWA Raptor CoAtNet-384 (0.949 alone) + 224 (0.945) retrained on all
  data + JEV/OAI weak labels, with anatomical-mirror TTA;
- heliosli's compact ConvNeXt models read by their SHA-pinned worker
  (S6 0.954 -> R7 0.956 -> R6 + flip TTA 0.957-0.958 with nartaa's Raptor);
- per-finding rank blends (heliosli's weights) and small readers at ~1/10.

goodpjw2008's notebook is the best *reproducible* public artifact at 0.958
(rank 195/6077, 2026-10-16).  v16 embeds its version 27 verbatim — every cell
below our header is byte-for-byte the pinned copy (sources untouched; outputs
and execution counts cleared).  Our header documents provenance, the attach
list, success signals, and the honest gap to the 0.965 top-10 cutoff.

Runtime pin: the copy was pulled 2026-10-16 12:48 UTC (lastRunTime of v27).

Usage:  python notebooks/v16/build_v16.py
Writes: user_nb/rsna-knee-abnormality-detection-v16.ipynb
"""
from __future__ import annotations

import ast
import hashlib
import json
from copy import deepcopy
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SRC_NB = REPO / "ref_kernels" / "oct16" / "goodpjw" / "rsna-knee-coatnet-r6-2-5d-mil-lb-0-958.ipynb"
SRC_META = REPO / "ref_kernels" / "oct16" / "goodpjw" / "kernel-metadata.json"
OUT_NB = REPO / "user_nb" / "rsna-knee-abnormality-detection-v16.ipynb"

SRC_SHA256 = "a906df9328b6a53d7e0fe4e18962c655867c8617c54e6745a8af647033b2399c"

CELL_TYPES = ["code", "markdown", "markdown", "code", "code", "markdown",
              "code", "code", "markdown", "code", "code", "markdown",
              "code", "markdown", "code", "code", "code", "code", "code"]

# every cell source must still contain its pipeline marker (recipe drift guard)
MARKERS = {
    1: "0.958",
    3: "blendjev_oai_d96_r384_swa.pt",
    6: "c08_r224_swa.pt",
    7: "R224_OK",
    9: "fe055406fad6a12f",
    12: "W_RAPTOR_C",
    18: "OWN_W_PER_TARGET",
}
MARKER_9_EXTRA = "6508ce6921eda986c885b2226fa55dd4d2aa8e5fc570b4c007c9a60c05bfc648"

DATASET_SOURCES = [
    "goodpjw2008/rsna-knee-2-5d-convnext-reader",
    "heliosli/rsna-knee-compact-r7cnxfsss",
    "rostigor/rsna-knee-r6-fe055406",
    "nartaa/rsna-knee-publication-swa-weights-20261007",
    "heliosli/rsna-orthofoundation-fold0-epoch7-20261004",
]
COMPETITION = "rsna-knee-abnormality-detection"

V16_HEADER = '''# RSNA Knee Abnormality Detection — v16 (public-frontier union base, LB 0.958)

**This notebook is [goodpjw2008's public notebook](https://www.kaggle.com/code/goodpjw2008/rsna-knee-coatnet-r6-2-5d-mil-lb-0-958)
"RSNA Knee | CoAtNet + R6 + 2.5D MIL | LB 0.958" (version 27, 2026-10-16), embedded verbatim.** Our repo's header cell is this one; every cell below it is
byte-for-byte the pinned copy (only outputs / execution counts are cleared).
Verified at build time by `notebooks/v16/build_v16.py` against the pinned copy
`ref_kernels/oct16/goodpjw/rsna-knee-coatnet-r6-2-5d-mil-lb-0-958.ipynb`
(SHA-256 `a906df9328b6a53d7e0fe4e18962c655867c8617c54e6745a8af647033b2399c`).

## Why v16 exists (our v13/v14/v15 line is superseded)

| Our run | LB | What it did |
|---|---|---|
| v13 | 0.913 | REF-20 DINOv2 (0.891) + RadImageNet, locked-bundle subset |
| v14 | 0.912 | Raptor v5/v10/v8 + DINOv3 + RadImageNet, still a subset |
| v15 (union) | 0.898 | REF-20 re-folded at 0.2 — union of two weak families lost to either alone |

The public frontier (checked 2026-10-16) is a *different* lineage and no longer
uses the locked bundle: nartaa's SWA Raptor CoAtNet-384 (0.949 alone) + 224
(0.945) with anatomical-mirror TTA, and heliosli's compact ConvNeXt models (the
S6 -> R7 -> R6 + flip-TTA step is what moved the LB) read by their SHA-pinned
worker. goodpjw's version 27 is the best public artifact at **0.958 (rank
195/6077)**.

## What runs (stage map)

1. **A1** — nartaa Raptor CoAtNet-384 SWA (`raptor_ft_alldata_t16_blendjev_oai_d96_r384_swa.pt`,
   all-data training + JEV/OAI weak-label blend) with the anatomical mirror view.
2. **A1b** — nartaa Raptor-224 (`..._c08_r224_swa.pt`, same dataset) with the
   mirror view, at **0.35 of the Raptor slot** (varshithakolupuri's addition).
3. **A2** — heliosli's compact **R6** (bundle `fe055406fad6a12f`, 4 fold models
   + stackers) scored with their worker `probe_dual.py` using **flip TTA**;
   every source file and the bundle manifest are SHA-256 checked
   (manifest `6508ce6921eda986...`).
4. **A3** — per-finding rank blend: `Raptor = 0.65*rank(384) + 0.35*rank(224)`,
   final `= w*rank(Raptor) + (1-w)*rank(R6)` with heliosli's per-finding `w`
   (Synovitis 0.8, Lateral OA 0.2, PF OA 0.6, Baker's 0.35, else 0.5).
5. **B** — goodpjw's own 2.5D ConvNeXt reader (3 public folds), rank-blended
   into A3's submission at **1/10** of its v6 per-finding weights; fail-soft
   (if it fails or runs late, the A3 submission is kept).

## Attach list (Kaggle settings)

- GPU: **T4 x2** (single T4 works), Internet: **OFF** (every artifact is pinned).
- Competition: `rsna-knee-abnormality-detection`
- Datasets (exact slugs):
  - `goodpjw2008/rsna-knee-2-5d-convnext-reader`
  - `heliosli/rsna-knee-compact-r7cnxfsss` (kept for compatibility; this version uses R6)
  - `rostigor/rsna-knee-r6-fe055406`
  - `nartaa/rsna-knee-publication-swa-weights-20261007`
  - `heliosli/rsna-orthofoundation-fold0-epoch7-20261004`

Do **not** substitute heliosli's original `rsna-knee-r6-frozen-fe055406`
dataset: its current version holds R7, and the notebook asserts the R6 manifest
SHA. `rostigor/rsna-knee-r6-fe055406` is the versioned R6 copy that matches.

## Success signals

`RAPTOR_STAGE_DONE` -> `R224_STAGE rc=0 ok=True` -> `ASSETS_READY` ->
`OURS_STAGE_COMPLETE` -> `NOTEBOOK_COMPLETE`; outputs `submission.csv` +
`blend_receipt.json`. A hard stage failure raises (no silent single-model
submission); stage B is fail-soft by design.

## Expected result and honest ceiling

- Expected: **~0.958** public LB (the best reproducible public artifact).
- Top-10 cutoff is ~0.965 today (top team 0.968). All public add-ons are
  author-tested **neutral** at 0.958 (R7 at 0.4/0.5 of the compact slot,
  extra readers at 0.2, flip TTA on R7, S6). Public members saturate at
  0.958; 0.965+ needs members outside the public pool.
- v17 candidates (one variable per run, all variance bets, not expected
  gains): R7 at 0.4 of the compact slot (`BUNDLE_ID 8c7b666e27dc0e8f`),
  mikkel's knee-centred reader at <= 0.2, our own DINOv2/A5/Rad family at
  <= 0.1, sujan's anti-tie monotonic calibration. See `docs/v16_plan.md`.
'''


def cell_source(cell: dict) -> str:
    src = cell["source"]
    return "".join(src) if isinstance(src, list) else src


def check_cell_python(idx: int, src: str) -> None:
    """Parse a code cell for syntax. Magics and shell escapes are stripped
    (the notebook is meant for Kaggle, not for local execution)."""
    attempts = [src]
    lines = src.splitlines()
    if lines and lines[0].lstrip().startswith("%%"):
        attempts.append("\n".join(lines[1:]))
    attempts.append("\n".join(
        ln for ln in lines
        if not ln.lstrip().startswith(("%", "!"))))
    err = None
    for text in attempts:
        try:
            ast.parse(text, filename=f"cell{idx}")
            return
        except SyntaxError as exc:
            err = exc
    raise SyntaxError(f"cell {idx} does not parse: {err}") from None


def main() -> int:
    raw = SRC_NB.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    assert digest == SRC_SHA256, (
        f"pinned copy drifted: {digest} != {SRC_SHA256} — re-pin deliberately "
        f"or restore ref_kernels/oct16/goodpjw/rsna-knee-coatnet-r6-2-5d-mil-lb-0-958.ipynb")
    src_nb = json.loads(raw)

    cells = src_nb["cells"]
    assert [c["cell_type"] for c in cells] == CELL_TYPES, "cell layout drifted"
    for i, marker in MARKERS.items():
        assert marker in cell_source(cells[i]), f"recipe marker {marker!r} missing from cell {i}"
    assert MARKER_9_EXTRA in cell_source(cells[9]), "R6 manifest hash missing from cell 9"

    meta = json.loads(SRC_META.read_text(encoding="utf-8"))
    assert meta["dataset_sources"] == DATASET_SOURCES, "attach list drifted from the pinned metadata"
    assert meta["competition_sources"] == [COMPETITION], "competition source drifted"
    assert meta["enable_gpu"] and not meta["enable_internet"], "GPU/internet flags drifted"

    out_cells = [{
        "cell_type": "markdown",
        "id": "v16-header",
        "metadata": {},
        "source": V16_HEADER.splitlines(keepends=True),
    }]
    for i, cell in enumerate(cells):
        cell = deepcopy(cell)
        if cell["cell_type"] == "code":
            check_cell_python(i, cell_source(cell))
            cell["outputs"] = []
            cell["execution_count"] = None
        out_cells.append(cell)

    out_nb = {
        "cells": out_cells,
        "metadata": deepcopy(src_nb["metadata"]),
        "nbformat": src_nb.get("nbformat", 4),
        # the pulled cells carry per-cell `id`s (nbformat >= 4.5); declare 4.5
        # so the notebook validates cleanly (the pulled file mislabels itself)
        "nbformat_minor": max(src_nb.get("nbformat_minor", 4), 5),
    }

    # provenance guard: every embedded cell source must equal the pinned copy
    for i, cell in enumerate(cells):
        assert cell_source(out_nb["cells"][i + 1]) == cell_source(cell), f"cell {i} mutated in assembly"
    assert out_nb["cells"][0]["source"] == V16_HEADER.splitlines(keepends=True)

    text = json.dumps(out_nb, indent=1)
    json.loads(text)  # roundtrip
    assert "a906df9328b6a53d7e0fe4e18962c655867c8617c54e6745a8af647033b2399c" in V16_HEADER

    OUT_NB.parent.mkdir(parents=True, exist_ok=True)
    OUT_NB.write_text(text, encoding="utf-8")

    n_code = sum(c["cell_type"] == "code" for c in out_cells)
    print(f"wrote {OUT_NB}")
    print(f"cells: {len(out_cells)} ({n_code} code) = 1 header + {len(cells)} embedded from the pinned copy")
    print(f"pinned source sha256 verified: {digest}")
    print("embedded cell sources: byte-for-byte equal to goodpjw2008 v27 (outputs cleared)")
    print(f"attach: competition {COMPETITION} + {len(DATASET_SOURCES)} datasets; T4, internet OFF")
    print("expected LB ~0.958; honest top-10 gap documented in the header and docs/v16_plan.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
