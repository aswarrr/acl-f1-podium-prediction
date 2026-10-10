"""Execute the self-contained notebook in a clean local kernel and save outputs."""
from pathlib import Path
import sys

import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]
path = ROOT / "f1_podium_milestone1.ipynb"
kernel = sys.argv[1] if len(sys.argv) > 1 else "f1-acl"
notebook = nbformat.read(path, as_version=4)
NotebookClient(notebook, timeout=2400, kernel_name=kernel,
               resources={"metadata": {"path": str(ROOT)}}).execute()
nbformat.write(notebook, path)
print(f"Executed {sum(c.cell_type == 'code' for c in notebook.cells)} code cells using {kernel}: {path}")
