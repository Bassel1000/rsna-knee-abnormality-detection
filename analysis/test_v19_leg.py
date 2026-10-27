"""Offline tests for the v19 leg cells (build_v19.py output).

Execs the embedded `v19-v16-backup`, `v19-leg-locate`, `v19-leg-run`,
`v19-leg-blend` and `v19-leg-receipt` cell sources on synthetic CSVs in a temp
dir (CPU only, no DICOMs, no Kaggle paths): verifies the fail-soft contracts
and the v19 sharding logic -

  * backup cell copies submission.csv -> v16_sub.csv,
  * a good blend writes the weighted-rank submission + receipt (w by tier),
  * a failed/skipped leg leaves submission.csv byte-identical (v16 kept),
  * a mid-blend error restores the v16 file,
  * the tier ladder (5 -> 2 -> skip, tta-on rate) matches the budget math,
  * `_write_half_root` subsets test.csv/test_series.csv and links test_series,
  * `_leg_merge` concatenates shard files, aligns rows to test.csv order,
  * the full sharded orchestration (fake Popen workers) merges to a complete,
    test.csv-ordered dkim_sub.csv and flags success,
  * the receipt cell never raises.

Usage: python analysis/test_v19_leg.py
"""
from __future__ import annotations

import hashlib
import json
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
NB = REPO / "user_nb" / "rsna-knee-abnormality-detection-v19.ipynb"

LABELS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA",
          "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]


def sha(p: Path) -> str:
    h = hashlib.sha256()
    h.update(Path(p).read_bytes())
    return h.hexdigest()


def cell_src(cid: str) -> str:
    nb = json.loads(NB.read_text(encoding="utf-8"))
    for c in nb["cells"]:
        if c.get("id") == cid:
            return "".join(c["source"])
    raise KeyError(cid)


def run_cell(cid: str, g: dict) -> None:
    src = cell_src(cid)
    exec(compile(src, cid, "exec"), g)


def make_ws(tmp: Path, n: int = 9, seed: int = 7) -> tuple[Path, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    ids = [f"1.2.840.{i:04d}" for i in range(1, n + 1)]
    v16 = pd.DataFrame(rng.uniform(0, 1, (n, 12)), columns=LABELS)
    v16.insert(0, "StudyInstanceUID", ids)
    (tmp / "v16_sub.csv").write_text(v16.to_csv(index=False), encoding="utf-8")
    (tmp / "submission.csv").write_text(v16.to_csv(index=False), encoding="utf-8")

    rho = 0.75
    z1 = rng.normal(size=(n, 12))
    z2 = rng.normal(size=(n, 12))
    dk = rho * z1 + np.sqrt(1 - rho**2) * z2
    dk = pd.DataFrame((dk - dk.min(0)) / (dk.max(0) - dk.min(0)) * 0.9 + 0.05, columns=LABELS)
    cols = list(rng.permutation(LABELS))
    dk = dk[cols].copy()
    dk.insert(0, "StudyInstanceUID", ids)
    dk = dk.iloc[rng.permutation(n)].reset_index(drop=True)
    (tmp / "dkim_sub.csv").write_text(dk.to_csv(index=False), encoding="utf-8")
    return tmp / "submission.csv", v16


def base_globals(tmp: Path, *, leg_ok: bool, leg_n: int) -> dict:
    return {"WORK": tmp, "Path": Path, "shutil": shutil, "json": json, "time": time,
            "NB_T0": time.time() - 2 * 3600, "LEG_N": leg_n, "LEG_SECONDS": 123.4,
            "DKIM_LEG_OK": leg_ok, "_sha": lambda p: sha(Path(p))}


def make_fake_comp(tmp: Path, n: int = 10) -> Path:
    comp = tmp / "comp"
    comp.mkdir()
    ids = [f"9.{i:03d}" for i in range(n)]
    with open(comp / "test.csv", "w", encoding="utf-8", newline="") as f:
        f.write("StudyInstanceUID,extra\n")
        for u in ids:
            f.write(f"{u},x\n")
    with open(comp / "test_series.csv", "w", encoding="utf-8", newline="") as f:
        f.write("StudyInstanceUID,SeriesInstanceUID,Fluid_Sensitive,Fat_Suppression,Anatomical_Plane\n")
        for i, u in enumerate(ids):
            f.write(f"{u},{u}-s0,1,0,Sagittal\n")
            f.write(f"{u},{u}-s1,0,1,Coronal\n")
    (comp / "test_series").mkdir()
    (comp / "test_series" / "dummy.txt").write_text("x", encoding="utf-8")
    with open(comp / "sample_submission.csv", "w", encoding="utf-8", newline="") as f:
        f.write("StudyInstanceUID," + ",".join(LABELS) + "\n")
    return comp


def _c2_globals(tmp: Path, comp: Path, n_test: int, subprocess_mod) -> dict:
    return {"WORK": tmp, "COMP": str(comp), "TRIM": tmp / "trim", "CODE": str(tmp / "code"),
            "LEG_READY": False, "LEG_N": 5, "n_test": n_test,
            "LEG_DEADLINE_T": time.time() + 3600, "Path": Path, "csv": __import__("csv"),
            "json": json, "os": __import__("os"), "sys": sys, "time": time,
            "subprocess": subprocess_mod, "shutil": shutil}


class FakeSubprocess:
    """Minimal stand-in for the subprocess module used by the v19-leg-run cell."""

    STDOUT = -2
    master: dict = {}

    class Popen:
        def __init__(self, args, env=None, stdout=None, stderr=None, **kw):
            self.returncode = 0
            snippet = args[-1]
            out = re.search(r"out_csv=(['\"])(.*?)\1", snippet).group(2)
            half_dir = re.search(r"comp_dir=(['\"])(.*?)\1", snippet).group(2)
            ids = pd.read_csv(Path(half_dir) / "test.csv", dtype=str)["StudyInstanceUID"].tolist()
            cols = list(LABELS)
            if env.get("CUDA_VISIBLE_DEVICES") == "1":
                cols = cols[::-1]   # different column order per shard: merge must align by name
            df = pd.DataFrame([FakeSubprocess.master[u] for u in ids], columns=LABELS)
            df = df[cols].copy()
            df.insert(0, "StudyInstanceUID", ids)
            df.to_csv(out, index=False)
            if stdout is not None:
                stdout.write(f"fake worker {env.get('CUDA_VISIBLE_DEVICES')} wrote {len(ids)} rows\n")

        def poll(self):
            return self.returncode

        def kill(self):
            self.returncode = -9

        def wait(self, timeout=None):
            return self.returncode


def main() -> int:
    tmp = Path(tempfile.mkdtemp(prefix="v19_test_"))
    sub, v16 = make_ws(tmp)
    v16_sha = sha(tmp / "v16_sub.csv")
    dkim_full = (tmp / "dkim_sub.csv").read_text(encoding="utf-8")

    # 1. backup cell copies submission.csv -> v16_sub.csv
    (tmp / "v16_sub.csv").unlink()
    run_cell("v19-v16-backup", base_globals(tmp, leg_ok=False, leg_n=0))
    assert (tmp / "v16_sub.csv").is_file() and sha(tmp / "v16_sub.csv") == v16_sha
    print("ok 1: backup cell")

    # 2. good blend, LEG_N=5 -> w 0.15, weighted-rank output, receipt written
    g = base_globals(tmp, leg_ok=True, leg_n=5)
    run_cell("v19-leg-blend", g)
    out = pd.read_csv(sub, dtype={"StudyInstanceUID": str})
    assert out["StudyInstanceUID"].tolist() == v16["StudyInstanceUID"].tolist()
    assert set(out.columns) == set(v16.columns)
    v16s = pd.read_csv(tmp / "v16_sub.csv").set_index("StudyInstanceUID")
    dks = pd.read_csv(tmp / "dkim_sub.csv").set_index("StudyInstanceUID")
    for c in LABELS:
        a = pd.Series(v16s.loc[out["StudyInstanceUID"], c].values.astype(float)).rank(pct=True).values
        b = pd.Series(dks.loc[out["StudyInstanceUID"], c].values.astype(float)).rank(pct=True).values
        want = 0.85 * a + 0.15 * b
        got = out[c].values.astype(float)
        assert np.allclose(got, want, atol=1e-5), c
    rec = json.loads((tmp / "v19_blend_receipt.json").read_text())
    assert rec["stage"] == "v19-dkim-leg" and rec["w"] == 0.15 and rec["n_models"] == 5
    assert all(-1.0 <= v <= 1.0 for v in rec["spearman_v16_vs_dkim"].values())
    print("ok 2: blend w=0.15 (columns aligned by name, rows shuffled in dkim csv)")

    # 3. LEG_N=2 -> w 0.08
    (tmp / "submission.csv").write_text(v16.to_csv(index=False), encoding="utf-8")
    run_cell("v19-leg-blend", base_globals(tmp, leg_ok=True, leg_n=2))
    rec = json.loads((tmp / "v19_blend_receipt.json").read_text())
    assert rec["w"] == 0.08 and rec["n_models"] == 2
    print("ok 3: blend w=0.08 for a 2-checkpoint slice")

    # 4. skipped leg -> submission untouched
    (tmp / "submission.csv").write_text(v16.to_csv(index=False), encoding="utf-8")
    before = sha(sub)
    run_cell("v19-leg-blend", base_globals(tmp, leg_ok=False, leg_n=0))
    assert sha(sub) == before
    print("ok 4: skipped leg keeps submission byte-identical")

    # 5. missing scored study -> blend fails, v16 restored
    dks = pd.read_csv(tmp / "dkim_sub.csv", dtype={"StudyInstanceUID": str})
    dks.iloc[:-1].to_csv(tmp / "dkim_sub.csv", index=False)
    before = sha(sub)
    run_cell("v19-leg-blend", base_globals(tmp, leg_ok=True, leg_n=5))
    assert sha(sub) == before, "blend failure must restore the v16 file"
    print("ok 5: missing study -> restore")

    # 6. non-finite dkim value -> blend fails, v16 restored (full file first: test 5 truncated it)
    (tmp / "dkim_sub.csv").write_text(dkim_full, encoding="utf-8")
    dks = pd.read_csv(tmp / "dkim_sub.csv", dtype={"StudyInstanceUID": str})
    dks.loc[0, LABELS[0]] = float("nan")
    dks.to_csv(tmp / "dkim_sub.csv", index=False)
    before = sha(sub)
    run_cell("v19-leg-blend", base_globals(tmp, leg_ok=True, leg_n=5))
    assert sha(sub) == before
    print("ok 6: non-finite value -> restore")

    # 7. tier ladder matches the budget math (n_test=1280 studies, tta-on rate)
    comp = make_fake_comp(tmp)
    g1 = {"DATA": str(comp), "WORK": tmp, "NB_T0": time.time(), "Path": Path,
          "csv": __import__("csv"), "glob": __import__("glob"), "hashlib": hashlib,
          "json": json, "os": __import__("os"), "subprocess": subprocess, "sys": sys, "time": time}
    run_cell("v19-leg-locate", g1)
    assert g1["LEG_READY"] is False, "locate must fail-soft when no dkim dir is mounted"
    assert "LEG_DEADLINE_T" in g1
    tier = g1["v19_tier"]
    assert tier(1280, 5 * 3600) == 5
    assert tier(1280, 6.0 * 3600) == 5
    assert tier(1280, 6.6 * 3600) == 5          # tier-5 boundary: 23760 + 7224 + 1200 <= 32400
    assert tier(1280, 6.7 * 3600) == 2          # one minute past it -> tier 2
    assert tier(1280, 7.6 * 3600) == 2
    assert tier(1280, 7.9 * 3600) == 0          # tier-2 boundary: 28440 + 3249.6 + 1200 > 32400
    assert g1["v19_est"](1280, 5) == 600.0 + 1280 * 2.07 * 5 / 2
    print("ok 7: tier ladder 5 -> 2 -> skip; locate fail-soft")

    # 8. _write_half_root + _leg_merge unit checks
    g2 = _c2_globals(tmp, comp, 10, FakeSubprocess)
    run_cell("v19-leg-run", g2)                      # LEG_READY False -> defs only, run skipped
    assert g2["DKIM_LEG_OK"] is False
    uids = [f"9.{i:03d}" for i in range(10)]
    half = set(uids[0::2])
    n_rows = g2["_write_half_root"](tmp / "halfA", half, str(comp))
    assert n_rows == 5
    hc = pd.read_csv(tmp / "halfA" / "test.csv", dtype=str)
    assert hc["StudyInstanceUID"].tolist() == [u for u in uids if u in half]
    hs = pd.read_csv(tmp / "halfA" / "test_series.csv", dtype=str)
    assert set(hs["StudyInstanceUID"]) == half and len(hs) == 10
    assert (tmp / "halfA" / "test_series").exists() or (tmp / "halfA" / "test_series").is_symlink()
    assert (tmp / "halfA" / "sample_submission.csv").exists()

    rng = np.random.default_rng(3)
    rows0 = pd.DataFrame(rng.uniform(0, 1, (5, 12)), columns=LABELS[::-1])
    rows0.insert(0, "StudyInstanceUID", [u for u in uids if u in half])
    rows1 = pd.DataFrame(rng.uniform(0, 1, (5, 12)), columns=LABELS)
    rows1.insert(0, "StudyInstanceUID", [u for u in uids if u not in half])
    rows0.to_csv(tmp / "h0.csv", index=False)
    rows1.to_csv(tmp / "h1.csv", index=False)
    n_merged = g2["_leg_merge"]([tmp / "h0.csv", tmp / "h1.csv"], uids, tmp / "merged.csv")
    assert n_merged == 10
    m = pd.read_csv(tmp / "merged.csv", dtype={"StudyInstanceUID": str})
    assert m["StudyInstanceUID"].tolist() == uids
    assert [c for c in m.columns if c in LABELS] and set(LABELS).issubset(m.columns)
    print("ok 8: half-root subsetting + merge alignment")

    # 9. full sharded orchestration with fake Popen workers
    master = {}
    rng = np.random.default_rng(11)
    for u in uids:
        master[u] = rng.uniform(0, 1, 12)
    FakeSubprocess.master = master
    g3 = _c2_globals(tmp, comp, 10, FakeSubprocess)
    g3["LEG_READY"] = True
    run_cell("v19-leg-run", g3)
    assert g3["DKIM_LEG_OK"] is True, "sharded run should succeed with fake workers"
    dm = pd.read_csv(tmp / "dkim_sub.csv", dtype={"StudyInstanceUID": str})
    assert dm["StudyInstanceUID"].tolist() == uids
    for _, r in dm.iterrows():
        assert np.allclose(r[LABELS].values.astype(float), master[r["StudyInstanceUID"]], atol=1e-6)
    assert g3["LEG_SECONDS"] >= 0
    print("ok 9: sharded orchestration merges to a complete, ordered dkim_sub.csv")

    # 10. receipt cell never raises (even with nothing set)
    (REPO / "analysis" / "_tmp_receipt_cell.py").write_text(cell_src("v19-leg-receipt"), encoding="utf-8")
    try:
        r = subprocess.run([sys.executable, str(REPO / "analysis" / "_tmp_receipt_cell.py")],
                           capture_output=True, text=True, check=False)
        assert r.returncode == 0, r.stderr
        assert "V19_DONE" in r.stdout, r.stdout
    finally:
        (REPO / "analysis" / "_tmp_receipt_cell.py").unlink()
    print("ok 10: receipt cell standalone, never raises")

    shutil.rmtree(tmp, ignore_errors=True)
    print("ALL V19 LEG CELL TESTS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
