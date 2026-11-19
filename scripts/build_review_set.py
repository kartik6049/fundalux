"""Build the clinician review set from graded photographs.

    python scripts/build_review_set.py --images DIR --labels labels.csv [--n 60] [--name NAME] [--site SITE]
    python scripts/build_review_set.py --idrid ~/.cache/netraxai/idrid --n 60      # the IDRiD test set

Photographs are spread evenly over reference grades, scored once by the models in
models/, and written to review/ under neutral ids. Replaces any existing set
(its answers move to review.old/).
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fundalux import datasets as D  # noqa: E402
from fundalux.pipeline import Fundalux  # noqa: E402
from fundalux.review import build_review_set, stratified_sample  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--images", help="folder of photographs")
    p.add_argument("--labels", help="CSV with image and grade columns")
    p.add_argument("--idrid", help="IDRiD download; uses its 103 test photographs and expert grades")
    p.add_argument("--n", type=int, default=60)
    p.add_argument("--name")
    p.add_argument("--site", help="site whose threshold the results use by default")
    p.add_argument("--models", default=str(ROOT / "models"))
    p.add_argument("--out", default=str(ROOT / "review"))
    a = p.parse_args()

    if a.idrid:
        idx = D.index_idrid(a.idrid)
        idx = idx[idx.image_id.str.startswith("test_")]
        items = [{"path": r.path, "reference_grade": int(r.grade)} for r in idx.itertuples()]
        name, site = a.name or "IDRiD test photographs (expert grades)", a.site or "IDRiD (Nanded, India)"
    else:
        table = pd.read_csv(a.labels) if a.labels else None
        refs = {}
        if table is not None:
            cols = {c.lower(): c for c in table.columns}
            refs = {Path(str(k)).stem.lower(): int(v) for k, v in zip(table[cols["image"]], table[cols["grade"]])}
        files = [f for f in sorted(Path(a.images).rglob("*")) if f.suffix.lower() in D.IMAGE_EXT]
        items = [{"path": str(f), "reference_grade": refs.get(f.stem.lower())} for f in files]
        name, site = a.name or Path(a.images).name, a.site

    chosen = stratified_sample(items, a.n)
    print(f"{len(chosen)} photographs; reference grades {sorted(pd.Series([c['reference_grade'] for c in chosen]).value_counts(dropna=False).to_dict().items(), key=str)}")
    bundle = Fundalux.load(a.models)
    out = Path(a.out)
    tmp = out.with_name(out.name + ".new")
    shutil.rmtree(tmp, ignore_errors=True)
    photos = [(Path(c["path"]).name, Path(c["path"]).read_bytes()) for c in chosen]
    references = {Path(c["path"]).stem.lower(): c["reference_grade"] for c in chosen if c["reference_grade"] is not None}
    build_review_set(bundle, photos, tmp, name, references, site, progress=lambda d, t: print(f"  scored {d}/{t}", end="\r", flush=True))
    if out.exists():
        shutil.rmtree(out.with_name(out.name + ".old"), ignore_errors=True)
        out.rename(out.with_name(out.name + ".old"))
    tmp.rename(out)
    print(f"\nWrote {out} ({sum(f.stat().st_size for f in out.rglob('*') if f.is_file()) / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
