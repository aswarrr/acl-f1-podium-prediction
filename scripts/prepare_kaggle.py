"""Stage a clean notebook and Kaggle metadata for a Save & Run All upload."""

import argparse
import json
from pathlib import Path

import nbformat


ROOT = Path(__file__).resolve().parents[1]
SLUG = "formula-1-podium-prediction-milestone-1"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("username", help="Your Kaggle username (not an API token)")
    args = parser.parse_args()

    stage = ROOT / "kaggle_upload"
    stage.mkdir(exist_ok=True)
    source = ROOT / "f1_podium_milestone1.ipynb"
    notebook = nbformat.read(source, as_version=4)
    for cell in notebook.cells:
        if cell.cell_type == "code":
            cell.outputs = []
            cell.execution_count = None
    nbformat.write(notebook, stage / source.name)

    metadata = {
        "id": f"{args.username}/{SLUG}",
        "title": "Formula 1 Podium Prediction Milestone 1",
        "code_file": source.name,
        "language": "python",
        "kernel_type": "notebook",
        "is_private": "false",
        "enable_gpu": "false",
        "enable_internet": "true",
        "dataset_sources": ["rohanrao/formula-1-world-championship-1950-2020"],
        "competition_sources": [],
        "kernel_sources": [],
        "model_sources": [],
    }
    (stage / "kernel-metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Staged {stage}\nKaggle URL: https://www.kaggle.com/code/{args.username}/{SLUG}")


if __name__ == "__main__":
    main()
