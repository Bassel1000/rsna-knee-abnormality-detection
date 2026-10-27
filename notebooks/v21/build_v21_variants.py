"""Build the v21 SCORED variants: the same v16/v19 bodies with one pre-declared
blend knob moved.

The only untested knob with a positive prior is the weight of goodpjw's 2.5D
reader: in the v16 body it enters the final blend with per-finding weights of
0.003-0.035 (mean ~0.017). It is the least image-correlated member we have (a
different architecture, a different training recipe), and nobody has ever
tested it at a higher weight - the author only tested a *second* reader at 0.2
(measured 0.957). Raising it is the cheapest positive-prior cell left, and it
costs no extra runtime.

Each variant is a full notebook (v16 body verbatim + optional dkim leg), so each
is a separate scored run and a separate lottery ticket.

Usage:  python notebooks/v21/build_v21_variants.py
"""
from __future__ import annotations

import ast
import importlib.util
import json
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
V16_NB = REPO / "user_nb" / "rsna-knee-abnormality-detection-v16.ipynb"
V19_BUILDER = REPO / "notebooks" / "v19" / "build_v19.py"
OUT_ROOT = REPO / "notebooks" / "v21" / "variants"

COMPETITION = "rsna-knee-abnormality-detection"
V16_DATASETS = [
    "goodpjw2008/rsna-knee-2-5d-convnext-reader",
    "rostigor/rsna-knee-r6-fe055406",
    "nartaa/rsna-knee-publication-swa-weights-20261007",
    "heliosli/rsna-orthofoundation-fold0-epoch7-20261004",
    "heliosli/rsna-knee-compact-r7cnxfsss",
]
DKIM_DATASETS = ["dkim54/rsna-knee-models-v11i",
                 "basselashraf/rsna-knee-pylibjpeg-wheels-cp313"]


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# name -> (body, reader scale, dkim weight, note)
RAPTOR_SLOT = {"v21i-raptor045": 0.45}
A224 = {"v21j-r224-025": 0.25}


def scale_raptor_slot(src: str, w: float) -> str:
    """Shift the Raptor slot (and its per-finding overrides) to a new global weight."""
    m = re.search(r"W_RAPTOR, W_OURS = ([0-9.]+), ([0-9.]+)", src)
    assert m, "W_RAPTOR not found"
    base = float(m.group(1))
    f = w / base
    src = src.replace(m.group(0), "W_RAPTOR, W_OURS = %r, %r" % (w, round(1.0 - w, 4)))
    m2 = re.search(r"W_RAPTOR_C = (\{[^}]*\})", src)
    assert m2, "W_RAPTOR_C not found"
    d = ast.literal_eval(m2.group(1))
    d2 = {k: round(v * f, 4) for k, v in d.items()}
    src = src[:m2.start(1)] + repr(d2) + src[m2.end(1):]
    return src


VARIANTS = {
    "v21a-reader-x2": ("v19", 2.0, None,
                       "the v19 body (v16 + dkim54 TTA leg) with the reader's per-finding "
                       "weights doubled (0.006-0.07)"),
    "v21b-reader-x3": ("v19", 3.0, None,
                       "the v19 body with the reader's per-finding weights tripled (0.009-0.105)"),
    "v21c-reader-x6": ("v19", 6.0, None,
                       "the v19 body with the reader's per-finding weights x6 (0.018-0.21)"),
    "v21d-reader-x3-v16": ("v16", 3.0, None,
                       "the v16 body (no dkim leg) with the reader's per-finding weights x3"),
    "v21e-dkim25": ("v19", 1.0, 0.25,
                       "the v19 body with the dkim54 leg at w=0.25 (0.15 and 0.08 are the only "
                       "measured weights)"),
    "v21f-dkim20": ("v19", 1.0, 0.20,
                       "the v19 body with the dkim54 leg at w=0.20 (the weight curve: 0.08 and "
                       "0.15 scored 0.958, 0.25 scored 0.957)"),
    "v21g-dkim10": ("v19", 1.0, 0.10,
                       "the v19 body with the dkim54 leg at w=0.10 (the other side of the "
                       "measured optimum)"),
    "v21h-reader-x0.5": ("v19", 0.5, None,
                       "the v19 body with the reader's per-finding weights halved (x2/x3/x6 all "
                       "scored 0.957, so the optimum is at or below x1)"),
    "v21i-raptor045": ("v19", 1.0, None,
                       "the v19 body with the Raptor slot at 0.45 (the compact arm is the "
                       "stronger of the two, so the mix is slightly rebalanced)"),
    "v21j-r224-025": ("v19", 1.0, None,
                       "the v19 body with the 224-crop arm at 0.25 of the Raptor slot instead "
                       "of 0.35 (the public Raptor recipe uses 0.20; the 384 is the stronger "
                       "of the two)"),
}


def scale_reader(src: str, factor: float) -> str:
    m = re.search(r"OWN_W_PER_TARGET = (\{[^}]*\})", src)
    assert m, "OWN_W_PER_TARGET not found"
    d = ast.literal_eval(m.group(1))
    d2 = {k: round(v * factor, 4) for k, v in d.items()}
    note = "  # v21 variant: reader weights x%g" % factor
    new = repr(d2) + note
    src = src[:m.start(1)] + new + src[m.end(1):]
    src = re.sub(r"OWN_W = 0\.15", "OWN_W = %.4f" % (0.15 * factor), src, count=1)
    return src


def build_variant(name: str, body_kind: str, reader_scale: float, dkim_w, note: str) -> dict:
    v16 = json.loads(V16_NB.read_text(encoding="utf-8"))
    v16_cells = v16["cells"]
    assert len(v16_cells) == 20
    body = [dict(c) for c in v16_cells[3:20]]
    assert "%%writefile /tmp/raptor_run.py" in "".join(body[1]["source"])

    # patch the reader weight in the v16 body (cell 19 is the reader leg)
    reader_cell = None
    for i, c in enumerate(body):
        if "OWN_W_PER_TARGET" in "".join(c["source"]):
            reader_cell = i
    assert reader_cell is not None, "reader cell not found"
    body[reader_cell]["source"] = scale_reader("".join(body[reader_cell]["source"]),
                                               reader_scale).splitlines(keepends=True)
    if name in A224:
        for i, c in enumerate(body):
            s = "".join(c["source"])
            if "A224 = 0.35" in s:
                body[i]["source"] = s.replace("A224 = 0.35", "A224 = %r" % A224[name]).splitlines(keepends=True)
                break
        else:
            raise SystemExit("A224 not found for " + name)
    if name in RAPTOR_SLOT:
        for i, c in enumerate(body):
            s = "".join(c["source"])
            if "W_RAPTOR, W_OURS" in s:
                body[i]["source"] = scale_raptor_slot(s, RAPTOR_SLOT[name]).splitlines(keepends=True)
                break
        else:
            raise SystemExit("blend cell not found for " + name)

    header = f"""# RSNA Knee - {name}

Variant of the v16/v19 body with ONE pre-declared blend change: {note}.

Everything else is byte-identical to the v16 body (cells [3..19]) plus the
v19 dkim54 leg. Fail-soft: any failure keeps the upstream submission.
"""
    cells = [{"cell_type": "markdown", "id": "v21v-header",
              "metadata": {}, "source": header.splitlines(keepends=True)}]
    for i, c in enumerate(body):
        cc = {"cell_type": c["cell_type"], "metadata": {}, "id": f"body-{i:02d}",
              "source": list(c["source"])}
        if cc["cell_type"] == "code":
            cc["execution_count"] = None
            cc["outputs"] = []
        cells.append(cc)

    datasets = list(V16_DATASETS)
    if body_kind == "v19":
        v19 = load_module(V19_BUILDER, "b19")
        # the v16 snapshot the leg blends into (without it the blend aborts and
        # the submission stays the v16 one - the v19 run showed that)
        backup = {"cell_type": "code", "id": "v21v-v16-backup", "metadata": {},
                  "execution_count": None, "outputs": [],
                  "source": ["# v21 variant: snapshot the v16 output before the dkim leg "
                             "touches anything (never raises).\n",
                             "import hashlib as _hl_backup\n",
                             "try:\n",
                             "    shutil.copyfile(str(WORK / 'submission.csv'), str(WORK / 'v16_sub.csv'))\n",
                             "    _hvb = _hl_backup.sha256()\n",
                             "    with open(str(WORK / 'v16_sub.csv'), 'rb') as _fvb:\n",
                             "        for _ch in iter(lambda: _fvb.read(1 << 20), b''):\n",
                             "            _hvb.update(_ch)\n",
                             "    print('v16 snapshot sha256:', _hvb.hexdigest(), flush=True)\n",
                             "except Exception as _evb:\n",
                             "    print('v16 snapshot FAILED:', repr(_evb), flush=True)\n"]}
        cells.append(backup)
        for cid, src in (("leg-locate", v19.CELL_LEG_LOCATE), ("leg-run", v19.CELL_LEG_RUN),
                         ("leg-blend", v19.CELL_LEG_BLEND), ("leg-receipt", v19.CELL_LEG_RECEIPT)):
            if dkim_w is not None and cid == "leg-blend":
                src = src.replace("W = 0.15 if LEG_N >= 5 else 0.08",
                                  "W = %s if LEG_N >= 5 else 0.08" % dkim_w)
                assert ("W = %s if LEG_N >= 5" % dkim_w) in src, "dkim weight not patched"
            ast.parse(src, filename=cid)
            cells.append({"cell_type": "code", "id": cid, "metadata": {},
                          "execution_count": None, "outputs": [],
                          "source": src.splitlines(keepends=True)})
        datasets += DKIM_DATASETS

    return {"cells": cells,
            "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python",
                                        "name": "python3"},
                         "language_info": {"name": "python", "version": "3.11"}},
            "nbformat": 4, "nbformat_minor": 5}


def main():
    for name, (body_kind, reader_scale, dkim_w, note) in VARIANTS.items():
        nb = build_variant(name, body_kind, reader_scale, dkim_w, note)
        text = json.dumps(nb, indent=1)
        json.loads(text)
        out_dir = OUT_ROOT / name
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / f"{name}.ipynb").write_text(text, encoding="utf-8")
        meta = {"id": f"basselashraf/rsna-knee-{name}",
                "title": f"RSNA Knee {name}", "code_file": f"{name}.ipynb",
                "language": "python", "kernel_type": "notebook", "is_private": True,
                "enable_gpu": True, "enable_internet": False,
                "dataset_sources": datasets_of(name), "competition_sources": [COMPETITION],
                "kernel_sources": [], "model_sources": []}
        (out_dir / "kernel-metadata.json").write_text(json.dumps(meta, indent=1) + "\n",
                                                      encoding="utf-8")
        # guards
        assert "OWN_W_PER_TARGET" in text
        if name in RAPTOR_SLOT:
            assert ("W_RAPTOR, W_OURS = %r" % RAPTOR_SLOT[name]) in text, name
        if name in A224:
            assert ("A224 = %r" % A224[name]) in text, name
        if body_kind == "v19":
            assert "DKIM_LEG_TIER" in text
        assert "v16_sub.csv" in text or body_kind == "v16"
        print("wrote", out_dir)


def datasets_of(name: str):
    body_kind = VARIANTS[name][0]
    ds = list(V16_DATASETS)
    if body_kind == "v19":
        ds += DKIM_DATASETS
    return ds


if __name__ == "__main__":
    main()
