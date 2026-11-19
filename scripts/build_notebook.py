"""Build Fundalux_Kaggle_Training.ipynb from scripts/notebook_src.py and fundalux/.

    python scripts/build_notebook.py          # write the notebook
    python scripts/build_notebook.py --check  # exit 1 if the notebook is out of date

The package source is embedded as %%writefile cells so the notebook is a single
self-contained file to import into Kaggle.
"""

from __future__ import annotations

import sys
from pathlib import Path

import nbformat

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "scripts" / "notebook_src.py"
NOTEBOOK = ROOT / "Fundalux_Kaggle_Training.ipynb"
# Modules the notebook (and the exported bundle's self-test) needs; the app-only ones stay out.
PACKAGE_FILES = ["__init__.py", "preprocess.py", "degrade.py", "quality.py", "metrics.py", "models.py", "gradcam.py", "data.py", "datasets.py", "synthetic.py", "pipeline.py", "sites.py", "inference.py", "heatmap.py", "export.py"]


def package_cells() -> list:
    cells = []
    for name in PACKAGE_FILES:
        body = (ROOT / "fundalux" / name).read_text().rstrip("\n")
        cells.append(nbformat.v4.new_code_cell(f"%%writefile fundalux/{name}\n{body}"))
    return cells


def build() -> nbformat.NotebookNode:
    cells = []
    chunks = SOURCE.read_text().split("\n# %%")
    for chunk in chunks[1:]:  # chunks[0] is the file header comment
        header, _, body = chunk.partition("\n")
        body = body.strip("\n")
        kind = header.strip()
        if kind == "[package]":
            cells.extend(package_cells())
        elif kind == "[markdown]":
            lines = [line[2:] if line.startswith("# ") else line.lstrip("#") for line in body.splitlines()]
            cells.append(nbformat.v4.new_markdown_cell("\n".join(lines)))
        else:
            cells.append(nbformat.v4.new_code_cell(body))
    for i, cell in enumerate(cells):
        cell["id"] = f"cell-{i:03d}"  # stable ids keep the build reproducible
    nb = nbformat.v4.new_notebook(cells=cells)
    nb.metadata.update(
        {
            "accelerator": "GPU",
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
        }
    )
    nbformat.validate(nb)
    return nb


def main() -> None:
    text = nbformat.writes(build()) + "\n"
    if "--check" in sys.argv:
        if not NOTEBOOK.exists() or NOTEBOOK.read_text() != text:
            sys.exit(f"{NOTEBOOK.name} is out of date: run python scripts/build_notebook.py")
        print(f"{NOTEBOOK.name} is up to date")
        return
    NOTEBOOK.write_text(text)
    print(f"Wrote {NOTEBOOK} ({len(text) / 1024:.0f} kB)")


if __name__ == "__main__":
    main()
