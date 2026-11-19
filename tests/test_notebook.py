import ast
import importlib.util
import re
from pathlib import Path

import nbformat

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "Fundalux_Kaggle_Training.ipynb"


def _builder():
    spec = importlib.util.spec_from_file_location("build_notebook", ROOT / "scripts" / "build_notebook.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_notebook_is_up_to_date():
    """The .ipynb embeds fundalux/: it must be rebuilt whenever the package or the notebook source changes."""
    expected = nbformat.writes(_builder().build()) + "\n"
    assert NOTEBOOK.read_text() == expected, "run: python scripts/build_notebook.py"


def test_notebook_is_self_contained_and_valid_python():
    nb = nbformat.read(NOTEBOOK, as_version=4)
    code = [c.source for c in nb.cells if c.cell_type == "code"]
    assert all(not c.get("outputs") for c in nb.cells if c.cell_type == "code")
    written = {re.match(r"%%writefile fundalux/(\w+)\.py", c).group(1) for c in code if c.startswith("%%writefile")}
    imported = set()
    for src in code:
        body = src.split("\n", 1)[1] if src.startswith("%%writefile") else src
        for node in ast.walk(ast.parse(body)):  # also proves every cell parses
            if isinstance(node, ast.ImportFrom) and node.module and (node.module.startswith("fundalux.") or (node.level == 1 and node.module)):
                imported.add(node.module.split(".")[-1])
            if isinstance(node, ast.ImportFrom) and node.module == "fundalux":
                imported.update(a.name for a in node.names if a.name[0].islower())
    assert imported <= written, f"notebook uses modules it does not write: {imported - written}"
