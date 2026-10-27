"""Build `user_nb/rsna-knee-abnormality-detection-v19.ipynb`.

v19 = v16 (goodpjw's 0.958 notebook, cells [3..19] verbatim) + the dkim54/
rsna-knee-models-v11i leg appended after the reader stage, at FULL quality
(per-model flip TTA restored), run as TWO study-shard processes (one per T4),
blended per finding at a weight fixed before any v19 run (w = 0.15 for a
>=5-checkpoint leg, 0.08 for a 2-checkpoint slice).

Why this shape (single variable vs v18)
---------------------------------------

The probe (kernel `basselashraf/rsna-knee-dkim54-v11i-probe`, notebook v3)
scored 0.947 standalone with the member **as published** (10 checkpoints x
flip TTA) and the pre-declared weight guide maps 0.945-0.953 to "w = 0.15 or
skip".  v18 tested that member degraded (per-model TTA disabled - the cost of
fitting the full 10-checkpoint set) at w = 0.15 and scored 0.957 (-0.001).
v19 restores the TTA the probe measured and pays for it with the checkpoint
tier (5 instead of 10; est ~2.0 h, the same budget envelope as v18's tta-off
10).  This is the last untested cell of the dkim matrix: full-quality member
at a visible pre-declared weight.

* the trimmed spec keeps each model's published `"tta": true` (asserted),
* the leg runs on BOTH GPUs: two `infer_lib.predict` processes, one per T4
  (`CUDA_VISIBLE_DEVICES` honoured because infer_lib uses dev="cuda"), each
  scoring half the studies through a fake half-root (subset `test.csv` /
  `test_series.csv` + symlinked `test_series` - the pattern validated by the
  v16 calibration kernel), then merged back in test.csv order,
* the leg counts wall clock from v16's NB_T0, picks the largest tier
  (5 -> 2) whose estimate fits 9 h - 20 min reserve (tta-on marginal 2.07
  s/study/checkpoint, sharded by 2), and has an absolute kill deadline
  10 min before the wall,
* tier 5 = the first five spec entries = run `v2128r3` folds f0-f4 (prefix
  order, as v17/v18 pre-declared; tier 2 = f0-f1 of the same run),
* on any failure (budget, code, data, shard, merge, blend) it **leaves the
  v16 submission byte-identical** (`V19_FALLBACK_KEEP_V16`).  A skipped leg
  makes v19 score exactly like v16; it can never do worse.

The leg records per-finding Spearman rho between v16 and dkim outputs on the
scored studies (`v19_blend_receipt.json`).

Probe code SHA-256 pins (re-checked in the leg):
    code/dataset.py  36e960434083605082c54b1996c54d51e36a6ccf6a72ecfac324aec1238637f1
    code/infer_lib.py 76dd275d1161025db5d4d6b67ae439587db3e7afc594388f856abe174ec4efac
    code/model.py    94e4421a500a2742f5380bbd2d5d66ce748ffc90aab24d107fa91b33f3e6ed8a
    models.json      2719714fef4aecdba7455bd5895069a524f9c3a91a3751c756a1688b3c4f1e3e

Usage:  python notebooks/v19/build_v19.py
Writes: user_nb/rsna-knee-abnormality-detection-v19.ipynb
        notebooks/v19/v19_kernel/rsna-knee-abnormality-detection-v19.ipynb
        notebooks/v19/v19_kernel/kernel-metadata.json
"""
from __future__ import annotations

import ast
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
V16_NB = REPO / "user_nb" / "rsna-knee-abnormality-detection-v16.ipynb"
OUT_NB = REPO / "user_nb" / "rsna-knee-abnormality-detection-v19.ipynb"
OUT_DIR = REPO / "notebooks" / "v19" / "v19_kernel"
OUT_COPY = OUT_DIR / "rsna-knee-abnormality-detection-v19.ipynb"
OUT_META = OUT_DIR / "kernel-metadata.json"

KERNEL_ID = "basselashraf/rsna-knee-v19-v16-dkim54-tta-leg"   # Kaggle slugifies the title; keep id = resolved slug
COMPETITION = "rsna-knee-abnormality-detection"
DATASET_SOURCES = [
    "goodpjw2008/rsna-knee-2-5d-convnext-reader",
    "heliosli/rsna-knee-compact-r7cnxfsss",
    "rostigor/rsna-knee-r6-fe055406",
    "nartaa/rsna-knee-publication-swa-weights-20261007",
    "heliosli/rsna-orthofoundation-fold0-epoch7-20261004",
    "dkim54/rsna-knee-models-v11i",
    "basselashraf/rsna-knee-pylibjpeg-wheels-cp313",
]

HEADER = """# RSNA Knee - v19: v16 (LB 0.958) + dkim54 v11i leg at full quality (flip TTA, 2-GPU shard, fail-soft)

**This notebook is goodpjw2008's 0.958 notebook (v16 copy, cells [3..19]
verbatim) plus one appended stage**: the independent `dkim54/rsna-knee-models-v11i`
family (convnext_small v2128 @384, 5-fold x 2 runs), which scored **0.947
standalone** in its probe - a decorrelated member outside the Raptor/compact
line. The community has measured that re-weighting the existing public
members does not move this plateau (goodpjw2008: S6 next to R6 -0.001;
gold-58 per-finding weight tuning -0.001/-0.003), so this stage adds a model
the ensemble does not contain instead.

**v19 vs v18 - one variable: the member's flip TTA.** v18 tested the full
10-checkpoint set with per-model TTA disabled (the cost of fitting) at
w = 0.15: scored 0.957 (-0.001). The probe's pre-declared weight guide maps
0.945-0.953 to "w = 0.15 or skip", but the probe measured the member **with**
flip TTA; v19 restores it and pays with the checkpoint tier - 5 checkpoints
(run `v2128r3` folds f0-f4) instead of 10, est ~2.0 h (600 s + n_test * 2.07
* n / 2), the same budget envelope as v18's tta-off ten. This is the last
untested cell of the dkim matrix: full-quality member at a visible
pre-declared weight. It changes the member only - every Part A stage is
byte-identical to v16.

Pre-declared blend (same guide as v17/v18): per finding,
`final = (1 - w) * rank(v16) + w * rank(dkim)`, w = 0.15 for a
>=5-checkpoint leg, 0.08 for a 2-checkpoint slice. **Budget-gated and
fail-soft**: the leg counts wall clock from v16's NB_T0, picks the largest
tier whose estimate fits 9 h - 20 min reserve (5 -> 2), has an absolute kill
deadline, and on any failure (budget, code, data, shard, merge, blend) leaves
the v16 submission byte-identical - a skipped leg makes v19 score exactly
like v16, never worse. `v19_blend_receipt.json` records the decisions and
per-finding Spearman rho(v16, dkim) on the scored studies.

Part A stages: nartaa (384 / 224 CoAtNet), heliosli (compact R6 worker),
goodpjw (reader) - unchanged, embedded verbatim. dkim54's
`infer_lib.py`/`dataset.py`/`model.py` run unmodified (sha256-pinned).
"""

HEADER_LEG = """---
# Part C - dkim54 v11i leg (v19: flip TTA restored, 5-checkpoint tier)

Locate the pinned model set, verify SHA-256 provenance, pick the checkpoint
tier by wall-clock estimate (tta ON, 2-GPU shard), then run the family
unmodified over two half-study shards and merge. Fail-soft throughout: any
problem skips the leg and keeps the v16 submission.
"""

# ---------------------------------------------------------------------------
# Part C cells
# ---------------------------------------------------------------------------

CELL_LEG_LOCATE = '''# Part C1: locate + pin the dkim set, pick the sharded checkpoint tier from the wall clock.
# v19: the leg runs on BOTH GPUs (two study-shard processes) with per-model flip TTA ON
# (the probe's own configuration); the budget uses the tta-on marginal rate / 2.
# Fail-soft: any exception here sets LEG_READY = False and the leg is skipped below.
import csv, glob, hashlib, json, os, subprocess, sys, time
from pathlib import Path

def _sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()

# Budget math (pure; analysis/test_v19_leg.py exercises v19_tier directly).
MARGINAL_S = 2.07       # tta-on (v19), per study per checkpoint, ONE GPU, with margin (tta-off measured ~1.04)
PAR = 2                 # study-shard processes (one per T4)
STARTUP_S, RESERVE_S, WALL_S = 600.0, 20 * 60.0, 9 * 3600.0

def v19_est(n_test, n):
    return STARTUP_S + n_test * MARGINAL_S * n / PAR

def v19_tier(n_test, elapsed):
    for n in (5, 2):
        if elapsed + v19_est(n_test, n) + RESERVE_S <= WALL_S:
            return n
    return 0

LEG_DEADLINE_T = float(globals().get('NB_T0', time.time())) + WALL_S - 600.0
LEG_READY, LEG_N, n_test, TRIM, CODE, DKIM = False, 0, 0, None, None, None
try:
    COMP = str(DATA)   # competition root already resolved by the v16 stage cells
    n_test = sum(1 for _ in csv.DictReader(open(os.path.join(COMP, 'test.csv'), encoding='utf-8')))
    elapsed = time.time() - NB_T0   # NB_T0 = v16's Raptor-stage start (the same 9 h budget)
    print(f'v19 dkim leg: n_test={n_test} v16_elapsed={elapsed / 3600:.2f} h', flush=True)

    dkim_cands = ['/kaggle/input/rsna-knee-models-v11i',
                  '/kaggle/input/datasets/dkim54/rsna-knee-models-v11i']
    for d in sorted(glob.glob('/kaggle/input/*/')):
        dkim_cands.append(d.rstrip('/'))
    dkim_cands = [d for d in dkim_cands
                  if os.path.exists(os.path.join(d, 'models.json'))
                  and glob.glob(os.path.join(d, 'v2128r3b_f0.pt'))]
    print('dkim candidate dirs:', dkim_cands, flush=True)
    assert dkim_cands, 'dkim models.json + v2128r3b weights not found (leg will be skipped)'
    DKIM = dkim_cands[0]
    spec = json.load(open(os.path.join(DKIM, 'models.json'), encoding='utf-8'))
    assert 'v2128r3' in str(spec.get('note')), 'unexpected models.json - not the v11i bundle'

    CODE = os.path.join(DKIM, 'code')
    if not os.path.exists(os.path.join(CODE, 'infer_lib.py')):
        CODE = DKIM
    assert os.path.exists(os.path.join(CODE, 'infer_lib.py')), 'infer_lib.py not found'

    _pin_code = {'dataset.py': '36e960434083605082c54b1996c54d51e36a6ccf6a72ecfac324aec1238637f1',
                 'infer_lib.py': '76dd275d1161025db5d4d6b67ae439587db3e7afc594388f856abe174ec4efac',
                 'model.py': '94e4421a500a2742f5380bbd2d5d66ce748ffc90aab24d107fa91b33f3e6ed8a'}
    _pin_root = {'models.json': '2719714fef4aecdba7455bd5895069a524f9c3a91a3751c756a1688b3c4f1e3e'}
    for name, want in _pin_code.items():
        got = _sha(os.path.join(CODE, name))
        assert got == want, f'{name} drifted: {got} != {want}'
    for name, want in _pin_root.items():
        got = _sha(os.path.join(DKIM, name))
        assert got == want, f'{name} drifted: {got} != {want}'
    print('pinned code+spec sha256: all ok', flush=True)

    # Offline pylibjpeg wheels (ours cp313 + dkim's shipped set): optional, the comp
    # DICOMs are Explicit VR LE (uncompressed). A failed install must not stop the leg.
    wheel_dirs = [wd for wd in ('/kaggle/input/rsna-knee-pylibjpeg-wheels-cp313',
                                '/kaggle/input/datasets/basselashraf/rsna-knee-pylibjpeg-wheels-cp313',
                                os.path.join(DKIM, 'wheels')) if os.path.isdir(wd)]
    if wheel_dirs:
        find_links = []
        for wd in wheel_dirs:
            find_links += ['--find-links', wd]
        r = subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', '--no-index'] + find_links
                           + ['pylibjpeg', 'pylibjpeg-libjpeg', 'pylibjpeg-openjpeg'],
                           capture_output=True, text=True)
        print('wheels install rc:', r.returncode, flush=True)

    LEG_N = v19_tier(n_test, elapsed)
    if LEG_N == 0:
        print(f'DKIM_LEG_SKIP_BUDGET elapsed={elapsed / 3600:.2f} h, even 2 ckpt est '
              f'{v19_est(n_test, 2) / 3600:.2f} h does not fit', flush=True)
    else:
        models = [dict(m) for m in spec['models'][:LEG_N]]
        assert all(bool(m.get('tta')) for m in models), 'v19 runs the member with flip TTA on'
        TRIM = Path('/kaggle/working/dkim_trim')
        if TRIM.exists():
            for p in TRIM.iterdir():
                p.unlink()
        TRIM.mkdir(parents=True, exist_ok=True)
        for m in models:
            src = os.path.join(DKIM, m['weights'])
            dst = TRIM / os.path.basename(m['weights'])
            if not dst.exists():
                os.symlink(src, dst)
        tspec = dict(spec, models=models)
        (TRIM / 'models.json').write_text(json.dumps(tspec, indent=1), encoding='utf-8')
        LEG_READY = True
        print(f'DKIM_LEG_TIER n_models={LEG_N} tta=on par={PAR} est={v19_est(n_test, LEG_N) / 3600:.2f} h '
              f'elapsed={elapsed / 3600:.2f} h reserve={RESERVE_S / 60:.0f} min', flush=True)
except Exception as _e:
    LEG_READY = False
    print('DKIM_LEG_LOCATE_FAILED (leg will be skipped):', repr(_e), flush=True)
'''

CELL_LEG_RUN = '''# Part C2: two-GPU study-shard run of dkim54's infer_lib.predict (tta on, trimmed set). Fail-soft.
import numpy as np
import pandas as pd

DKIM_LABELS = ['ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus', 'Medial OA', 'Lateral OA',
               'PF OA', 'Effusion', 'Synovitis', "Baker's", 'Contusion', 'Fracture']
DKIM_LEG_OK, LEG_SECONDS = False, 0.0

def _write_half_root(root, uids_half, comp):
    """Fake competition root with only `uids_half` studies (test_series symlinked to the real one)."""
    import shutil as _sh
    root = Path(root)
    if root.exists():
        _sh.rmtree(root)
    root.mkdir(parents=True)
    n_studies = 0
    for name in ('test.csv', 'test_series.csv'):
        with open(os.path.join(comp, name), encoding='utf-8', newline='') as f:
            rd = csv.DictReader(f)
            hdr, rows = rd.fieldnames, [r for r in rd if r['StudyInstanceUID'] in uids_half]
        if name == 'test.csv':
            n_studies = len(rows)
        with open(root / name, 'w', encoding='utf-8', newline='') as f:
            wr = csv.DictWriter(f, fieldnames=hdr)
            wr.writeheader()
            wr.writerows(rows)
    try:
        os.symlink(os.path.join(comp, 'test_series'), root / 'test_series')
    except OSError:
        _sh.copytree(os.path.join(comp, 'test_series'), root / 'test_series')
    ss = os.path.join(comp, 'sample_submission.csv')
    if os.path.exists(ss):
        try:
            os.symlink(ss, root / 'sample_submission.csv')
        except OSError:
            _sh.copyfile(ss, root / 'sample_submission.csv')
    return n_studies

def _leg_merge(half_csvs, uids, out_csv):
    """Concatenate the shards, align to the test.csv order, write the merged file; returns row count."""
    m = pd.concat([pd.read_csv(p, dtype={'StudyInstanceUID': str}) for p in half_csvs], ignore_index=True)
    m = m.drop_duplicates('StudyInstanceUID', keep='first')
    assert m['StudyInstanceUID'].is_unique, 'duplicate UIDs after the sharded run'
    missing = [u for u in uids if u not in set(m['StudyInstanceUID'])]
    assert not missing, f'{len(missing)} studies missing after the sharded run'
    m = m.set_index('StudyInstanceUID').loc[uids].reset_index()
    m.to_csv(out_csv, index=False)
    return len(m)

if not LEG_READY:
    print('DKIM_LEG_SKIPPED before run', flush=True)
else:
    procs = []
    _t = time.time()
    try:
        uids = [r['StudyInstanceUID']
                for r in csv.DictReader(open(os.path.join(COMP, 'test.csv'), encoding='utf-8'))]
        assert len(uids) == n_test and len(set(uids)) == n_test, 'test.csv UIDs differ from the tier stage'
        halves = [set(uids[0::2]), set(uids[1::2])]
        outs = []
        for gpu in (0, 1):
            hdir = os.path.join(str(WORK), f'dkim_half{gpu}')
            n_half = _write_half_root(hdir, halves[gpu], COMP)
            out_csv = os.path.join(str(WORK), f'dkim_preds_half{gpu}.csv')
            outs.append(out_csv)
            snippet = ('import sys\\n'
                       f'sys.path.insert(0, {CODE!r})\\n'
                       'import infer_lib\\n'
                       f'infer_lib.predict(comp_dir={hdir!r}, models_dir={str(TRIM)!r}, '
                       f'out_csv={out_csv!r}, split="test", workers=2, bs=2)\\n')
            env = dict(os.environ)
            env['CUDA_VISIBLE_DEVICES'] = str(gpu)
            log = open(os.path.join(str(WORK), f'dkim_worker{gpu}.log'), 'w', encoding='utf-8')
            procs.append((subprocess.Popen([sys.executable, '-c', snippet], env=env,
                                           stdout=log, stderr=subprocess.STDOUT), log, gpu, n_half))
        while any(p.poll() is None for p, _l, _g, _n in procs):
            if time.time() > LEG_DEADLINE_T:
                for p, _l, _g, _n in procs:
                    p.kill()
                raise RuntimeError('leg exceeded its wall-clock deadline; workers killed')
            time.sleep(5)
        for p, log, gpu, n_half in procs:
            log.close()
            assert p.returncode == 0, f'dkim worker {gpu} exited rc={p.returncode}'
            print(f'dkim worker {gpu}: {n_half} studies rc=0', flush=True)
        merged = _leg_merge(outs, uids, os.path.join(str(WORK), 'dkim_sub.csv'))
        assert merged == n_test, f'merged {merged} of {n_test} studies'
        _cols = set(pd.read_csv(os.path.join(str(WORK), 'dkim_sub.csv'),
                                dtype={'StudyInstanceUID': str}, nrows=3).columns)
        _missing = [c for c in DKIM_LABELS if c not in _cols]
        assert not _missing, f'dkim submission missing columns: {_missing}'
        LEG_SECONDS = time.time() - _t
        DKIM_LEG_OK = True
        print(f'DKIM_LEG_DONE n_models={LEG_N} tta=on par=2 seconds={LEG_SECONDS:.1f} '
              f'({LEG_SECONDS / max(n_test, 1):.2f} s/study)', flush=True)
    except Exception as _e:
        DKIM_LEG_OK = False
        for p, log, gpu, n_half in procs:
            try:
                if p.poll() is None:
                    p.kill()
                log.close()
                lines = Path(str(WORK) / f'dkim_worker{gpu}.log').read_text(encoding='utf-8',
                                                                            errors='replace').splitlines()
                print(f'--- dkim worker {gpu} log tail ---', flush=True)
                for ln in lines[-15:]:
                    print(ln, flush=True)
            except Exception:
                pass
        print('DKIM_LEG_FAILED (v16 submission kept):', repr(_e), flush=True)
'''

CELL_LEG_BLEND = '''# Part C3: per-finding rank blend into submission.csv; on any error restore the v16 file.
# w is fixed before any v19 run: 0.15 for a >=5-checkpoint leg, 0.08 for a 2-checkpoint slice
# (sqrt(2/5) scaling - a 2-fold slice carries less independent signal).
import numpy as np
import pandas as pd

SUB = WORK / 'submission.csv'
V16B = WORK / 'v16_sub.csv'

if not DKIM_LEG_OK:
    print('V19_FALLBACK_KEEP_V16 (dkim leg skipped/failed; submission.csv untouched)', flush=True)
    if SUB.is_file():
        shutil.copyfile(SUB, V16B)
else:
    W = 0.15 if LEG_N >= 5 else 0.08
    try:
        assert V16B.is_file(), 'v16 backup missing'
        v16 = pd.read_csv(V16B, dtype={'StudyInstanceUID': str})
        dk = pd.read_csv(WORK / 'dkim_sub.csv', dtype={'StudyInstanceUID': str})
        labels = list(v16.columns[1:])
        missing_cols = [c for c in labels if c not in dk.columns]
        assert not missing_cols, f'dkim submission missing columns: {missing_cols}'
        v16 = v16.set_index('StudyInstanceUID')
        dk = dk.set_index('StudyInstanceUID')
        ids = v16.index.tolist()
        missing = [u for u in ids if u not in dk.index]
        assert not missing, f'{len(missing)} scored studies missing from the dkim leg'
        dk = dk.loc[ids, labels]
        v16v = v16[labels].values.astype(np.float64)
        dkv = dk.values.astype(np.float64)
        assert np.isfinite(v16v).all() and np.isfinite(dkv).all(), 'non-finite values'

        def _pct(a):
            return pd.Series(a).rank(method='average', pct=True).values

        out = np.zeros_like(v16v)
        rho = {}
        for j, c in enumerate(labels):
            out[:, j] = (1.0 - W) * _pct(v16v[:, j]) + W * _pct(dkv[:, j])
            if len(ids) >= 3:
                rho[c] = round(float(pd.Series(v16v[:, j]).corr(pd.Series(dkv[:, j]), method='spearman')), 4)
        final = pd.DataFrame(out.astype(np.float32), columns=labels)
        final.insert(0, 'StudyInstanceUID', ids)
        final.to_csv(SUB, index=False)
        print('V19_BLEND_DONE', json.dumps({'n_studies': len(ids), 'n_models': LEG_N, 'w': W,
                                            'leg_seconds': round(LEG_SECONDS, 1)}), flush=True)
        print('V19 rho(v16, dkim) per finding:', json.dumps(rho), flush=True)
    except Exception as _e:
        try:
            shutil.copyfile(V16B, SUB)
        except Exception as _e2:
            print('V19_RESTORE_FAILED', repr(_e2), flush=True)
        print('V19_BLEND_FAILED (v16 submission kept):', repr(_e), flush=True)

try:   # receipt is bookkeeping only - never restores over a good blend
    _rho = globals().get('rho')
    _ids = globals().get('ids')
    if DKIM_LEG_OK and _rho is not None and _ids is not None:
        receipt = {'stage': 'v19-dkim-leg', 'n_studies': len(_ids), 'n_models': LEG_N, 'w': W,
                   'leg_seconds': round(LEG_SECONDS, 1), 'elapsed_h_at_blend': round((time.time() - NB_T0) / 3600, 2),
                   'spearman_v16_vs_dkim': _rho, 'submission_sha256': _sha(str(SUB))}
        (WORK / 'v19_blend_receipt.json').write_text(json.dumps(receipt, indent=1))
except Exception as _e3:
    print('V19_RECEIPT_FAILED', repr(_e3), flush=True)
'''

CELL_LEG_RECEIPT = '''# Part C4: end-of-run summary (never raises).
import hashlib as _hl
import json as _js
import time as _tm
from pathlib import Path as _Path

_W = globals().get('WORK', _Path('/kaggle/working'))

def _shaF(p):
    try:
        h = _hl.sha256()
        with open(p, 'rb') as f:
            for chunk in iter(lambda: f.read(1 << 20), b''):
                h.update(chunk)
        return h.hexdigest()
    except Exception:
        return None

try:
    _summary = {'n_test': int(globals().get('n_test', 0) or 0),
                'leg_ready': bool(globals().get('LEG_READY', False)),
                'leg_n_models': int(globals().get('LEG_N', 0) or 0),
                'leg_ok': bool(globals().get('DKIM_LEG_OK', False)),
                'leg_seconds': round(float(globals().get('LEG_SECONDS', 0.0) or 0.0), 1),
                'elapsed_h': round((_tm.time() - globals().get('NB_T0', _tm.time())) / 3600, 2),
                'v16_submission_sha256': _shaF(str(_W / 'v16_sub.csv')),
                'final_submission_sha256': _shaF(str(_W / 'submission.csv'))}
    print('V19_SUMMARY', _js.dumps(_summary), flush=True)
except Exception as _e:
    print('V19_SUMMARY_FAILED', repr(_e), flush=True)
print('V19_DONE', flush=True)
'''


def main() -> int:
    v16 = json.loads(V16_NB.read_text(encoding="utf-8"))
    v16_cells = v16["cells"]

    # sanity: cells [3..19] are the v16 body we embed verbatim
    assert len(v16_cells) == 20, f"v16 cell count changed: {len(v16_cells)}"
    body = v16_cells[3:20]
    assert "%%writefile /tmp/raptor_run.py" in "".join(body[1]["source"]), "unexpected v16 cell layout"

    cells = [{"cell_type": "markdown", "metadata": {}, "id": "v19-header",
              "source": HEADER.splitlines(keepends=True)}]
    for i, c in enumerate(body):
        cc = {"cell_type": c["cell_type"], "metadata": {},
              "id": f"v16-body-{i:02d}", "source": list(c["source"])}
        if cc["cell_type"] == "code":
            cc["execution_count"] = None
            cc["outputs"] = []
        cells.append(cc)

    cells.append({"cell_type": "markdown", "metadata": {}, "id": "v19-leg-header",
                  "source": HEADER_LEG.splitlines(keepends=True)})
    for cid, src in (("v19-leg-locate", CELL_LEG_LOCATE),
                     ("v19-leg-run", CELL_LEG_RUN),
                     ("v19-leg-blend", CELL_LEG_BLEND),
                     ("v19-leg-receipt", CELL_LEG_RECEIPT)):
        ast.parse(src, filename=cid)
        cells.append({"cell_type": "code", "metadata": {}, "id": cid,
                      "execution_count": None, "outputs": [],
                      "source": src.splitlines(keepends=True)})

    # backup cell between the v16 body and the leg: submission.csv -> v16_sub.csv (never raises)
    backup = {"cell_type": "code", "metadata": {}, "id": "v19-v16-backup",
              "execution_count": None, "outputs": [],
              "source": ["# v19: snapshot the v16 output before the dkim leg touches anything (never raises).\n",
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
    leg_start = 1 + len(body)  # header + body, then the v16 backup, then the leg cells
    cells = cells[:leg_start] + [backup] + cells[leg_start:]
    ast.parse("".join(backup["source"]), filename="v19-v16-backup")

    nb = {"cells": cells,
          "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                       "language_info": {"name": "python", "version": "3.11"}},
          "nbformat": 4, "nbformat_minor": 5}
    text = json.dumps(nb, indent=1)
    json.loads(text)

    for marker in ("DKIM_LEG_TIER", "V19_FALLBACK_KEEP_V16", "V19_BLEND_DONE", "V19_DONE",
                   "v19_tier", "_write_half_root", "_leg_merge", "v16_sub.csv", "sha256"):
        assert marker in text, f"{marker} missing from assembled notebook"

    OUT_NB.parent.mkdir(parents=True, exist_ok=True)
    OUT_NB.write_text(text, encoding="utf-8")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_COPY.write_text(text, encoding="utf-8")
    meta = {"id": KERNEL_ID,
            "title": "RSNA Knee v19 (v16 + dkim54 TTA leg)",
            "code_file": OUT_COPY.name,
            "language": "python",
            "kernel_type": "notebook",
            "is_private": True,
            "enable_gpu": True,
            "enable_internet": False,
            "dataset_sources": DATASET_SOURCES,
            "competition_sources": [COMPETITION],
            "kernel_sources": [],
            "model_sources": []}
    OUT_META.write_text(json.dumps(meta, indent=1) + "\n", encoding="utf-8")

    print(f"wrote {OUT_NB}")
    print(f"wrote {OUT_COPY} + {OUT_META}")
    print(f"cells: {len(cells)} (header + v16 [3..19] verbatim + backup + 4 leg cells)")
    print(f"push:  kaggle kernels push -p {OUT_DIR.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
