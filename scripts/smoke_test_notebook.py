"""Execute the training notebook end to end on tiny synthetic data (CPU, a few minutes).

    python scripts/smoke_test_notebook.py [work_dir]

This checks that every cell runs and that the exported bundle loads; it says
nothing about model quality. The outputs land in <work_dir>/fundalux_outputs.
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    work = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(tempfile.mkdtemp(prefix="fundalux_smoke_"))
    work.mkdir(parents=True, exist_ok=True)
    os.environ["FUNDALUX_SMOKE"] = "1"
    nb = nbformat.read(ROOT / "Fundalux_Kaggle_Training.ipynb", as_version=4)
    client = NotebookClient(nb, timeout=3600, kernel_name="python3", resources={"metadata": {"path": str(work)}})
    try:
        client.execute()
    finally:
        nbformat.write(nb, work / "executed.ipynb")
    print("Notebook ran end to end. Outputs:", work / "fundalux_outputs")


if __name__ == "__main__":
    main()
