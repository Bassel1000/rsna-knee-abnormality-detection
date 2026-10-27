"""Offline test of the v16 build's embedded blend cell (goodpjw2008 v27 A3).

Execs the embedded per-finding rank-blend cell from
`user_nb/rsna-knee-abnormality-detection-v16.ipynb` against synthetic CSVs in a
temp directory, with no DICOMs, no GPU and no Kaggle.  It proves, on this box:

  1. the extracted notebook's A3 cell executes and honours its guards —
     per-finding Raptor weights (heliosli's 0.958 list), the 224-in-Raptor-slot
     mix (A224=0.35, `R224_UNUSABLE` fallback when the file is absent), the
     raptor-failed / ours-failed row fallbacks, `BLEND_FALLBACK` reporting and
     the `blend_receipt.json` contract;
  2. the emitted submission covers every study with finite values in [0, 1].

Usage:  python analysis/test_v16_blend.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
NB = REPO / "user_nb" / "rsna-knee-abnormality-detection-v16.ipynb"

LABELS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA",
          "Lateral OA", "PF OA", "Effusion", "Synovitis", "Baker's",
          "Contusion", "Fracture"]
N = 40  # 1 / 40 = 2.5 % < the 3 % Raptor failure guard


def load_blend_cell() -> str:
    nb = json.loads(NB.read_text(encoding="utf-8"))
    hits = [c for c in nb["cells"]
            if c["cell_type"] == "code" and "W_RAPTOR_C" in "".join(c["source"])]
    assert len(hits) == 1, f"expected exactly one blend cell, found {len(hits)}"
    src = "".join(hits[0]["source"])
    for anchor in ("W_RAPTOR_C", "A224 = 0.35", "raptor224_sub.csv",
                   "blend_receipt.json", "BLEND_FALLBACK"):
        assert anchor in src, f"blend cell lost {anchor!r}"
    return src


def make_env(tmp: Path, with_224: bool) -> dict:
    rng = np.random.default_rng(0)
    uids = [f"S{i:02d}" for i in range(N)]
    tmp.mkdir(parents=True, exist_ok=True)
    (tmp / "test.csv").write_text(
        "StudyInstanceUID\n" + "\n".join(uids) + "\n", encoding="utf-8")
    pd.DataFrame(columns=["StudyInstanceUID"] + LABELS,
                 data=np.hstack([np.array(uids)[:, None], np.zeros((N, 12))])) \
        .to_csv(tmp / "sample_submission.csv", index=False)

    def frame(values):
        df = pd.DataFrame(values, columns=LABELS)
        df.insert(0, "StudyInstanceUID", uids)
        return df

    ours = rng.random((N, 12))
    frame(ours).to_csv(tmp / "ours_sub.csv", index=False)
    raw = rng.random((N, 12))
    raw[-1] = 0.5                      # study S39: Raptor failed (all-column 0.5)
    frame(raw).to_csv(tmp / "raptor_raw.csv", index=False)
    rsub = rng.random((N, 12))
    rsub[-1] = 0.5
    frame((rsub >= raw).astype(float) * 0.0 + raw + 1e-3).to_csv(  # near-identical rank source
        tmp / "raptor_sub.csv", index=False)
    if with_224:
        frame(rng.random((N, 12))).to_csv(tmp / "raptor224_sub.csv", index=False)

    pred = tmp / "prediction"
    (pred / "worker0").mkdir(parents=True)
    keep = [i for i in range(N) if i != 38]        # S38 not scored by us
    scored = pd.DataFrame(ours[keep], columns=LABELS)
    scored.insert(0, "StudyInstanceUID", [uids[i] for i in keep])
    scored.to_csv(pred / "worker0" / "predictions.csv", index=False)
    return {"DATA": tmp, "WORK": tmp, "R224_OK": True, "RAPTOR_SECONDS": 12.0,
            "NB_T0": 0.0, "pred": pred}


def run_case(with_224: bool, tmp: Path) -> tuple:
    import json as _json
    import time as _time
    src = load_blend_cell()
    env = make_env(tmp, with_224)
    prints = []
    ns = {
        "__name__": "v16_blend_test",
        "print": lambda *a, **k: prints.append(" ".join(str(x) for x in a)),
        "display": lambda *a, **k: None,
        "time": _time,
        "json": _json,
        **env,
    }
    exec(compile(src, "v16_blend_cell", "exec"), ns)

    sub = pd.read_csv(tmp / "submission.csv")
    assert sub["StudyInstanceUID"].tolist() == [f"S{i:02d}" for i in range(N)]
    vals = sub[LABELS].to_numpy()
    assert np.isfinite(vals).all() and (vals >= 0).all() and (vals <= 1).all()
    assert (sub[LABELS].nunique() > 1).all(), "constant column"

    receipt = json.loads((tmp / "blend_receipt.json").read_text(encoding="utf-8"))
    assert receipt["mode"] == "blend" and receipt["rows"] == N
    assert receipt["raptor_failed"] == 1 and receipt["ours_failed"] == 1
    assert receipt["both_failed"] == 0
    assert receipt["weights"]["raptor_per_class"]["Synovitis"] == 0.8
    assert any("BLEND_FALLBACK" in p for p in prints)
    if with_224:
        assert not any("R224_UNUSABLE" in p for p in prints)
    else:
        assert any("R224_UNUSABLE" in p for p in prints)
    return prints


def main() -> int:
    import shutil
    import tempfile
    root = Path(tempfile.mkdtemp(prefix="v16_blend_"))
    try:
        prints = run_case(True, root / "full")
        print("case full-224: OK")
        run_case(False, root / "no224")
        print("case missing-224 (R224_UNUSABLE fallback): OK")
        for line in prints[-3:]:
            print("  tail:", line[:120])
    finally:
        shutil.rmtree(root, ignore_errors=True)
    print("test_v16_blend: all green")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
