# Formula 1 Podium Prediction — ACL Milestone 1

This repository contains a new, self-contained, run-all notebook and its analytical PDF report. It predicts whether a **race starter** finishes in the top three using only information available after qualifying and before the main race. The raw 1950–2024 relational dataset is audited and used for the three data-engineering questions; modeling uses 2003–2024 to avoid the severe pre-2003 qualifying gap.

The notebook now follows the course lab sequence: **Lab 1** loads, inspects, cleans, and plots DataFrames; **Lab 2** fits and evaluates classifiers; **Lab 3** adds a one-hidden-layer neural network; and **Lab 4** explains predictions with SHAP and LIME. Numbered, smaller cells show intermediate tables and explain the F1-specific historical joins. The model settings and reported results are unchanged.

## Files

- `f1_podium_milestone1.ipynb`: submission notebook; all analysis, training, explanations and inference run top to bottom.
- `f1_podium_milestone1.py`: paired Jupytext source for review and editing. The notebook contains the same code and does not import this script.
- `outputs/figures/` and `outputs/tables/`: figures, audit outputs, model comparisons and experiment evidence from the run.
- `report/report.md`, `report/report.pdf`: source text and submission PDF. `report/build_report.py` recreates both from notebook outputs.
- `requirements.txt`: local dependencies.

## Data and reproducibility

Use [Formula 1 World Championship (1950–2024)](https://www.kaggle.com/datasets/rohanrao/formula-1-world-championship-1950-2020), which provides these 14 CSVs: `circuits`, `constructors`, `constructor_results`, `constructor_standings`, `drivers`, `driver_standings`, `races`, `results`, `qualifying`, `lap_times`, `pit_stops`, `sprint_results`, `seasons`, and `status`.

On Kaggle, attach that dataset as a notebook input, enable Internet if `shap` or `lime` must be installed, then use **Save & Run All**. The notebook finds the attached folder automatically. No path cell needs editing. You can keep the Kaggle notebook private while exploring the results.

To stage a clean copy for Kaggle's API, run `python scripts/prepare_kaggle.py YOUR_KAGGLE_USERNAME`. Add `--slug EXISTING_NOTEBOOK_SLUG` to update an existing Kaggle notebook instead of creating a new one. This writes an output-free notebook and `kernel-metadata.json` to the ignored `kaggle_upload/` folder. With Kaggle API credentials configured locally and the Kaggle CLI installed, run `kaggle kernels push -p kaggle_upload` (without `--no-run`), then check that the Kaggle run succeeds and its output files appear. The staged metadata attaches the source dataset and enables Internet. See [Kaggle's kernel CLI documentation](https://github.com/Kaggle/kaggle-cli/blob/main/docs/kernels.md).

For a local run, place the 14 CSVs in `data/raw/` or set `F1_DATA_DIR` to the folder containing them. The raw CSVs are ignored by Git and are never rewritten. On Windows with `uv`:

```powershell
uv venv .venv
uv pip install --python .venv\Scripts\python.exe -r requirements.txt
.venv\Scripts\python.exe -m ipykernel install --user --name f1-acl --display-name "F1 ACL"
```

Select the `F1 ACL` kernel and **Run All** on `f1_podium_milestone1.ipynb`. To regenerate the PDF after a successful run:

```powershell
.venv\Scripts\python.exe report\build_report.py
```

The Python source can also be run directly. `jupytext --to ipynb f1_podium_milestone1.py` rebuilds the notebook source, but the Kaggle notebook should be executed and saved with outputs after the last edit.

## Decision log

- **Grain:** one `(raceId, driverId)` race start. Remove confirmed non-starters and the Indianapolis 500; resolve historical shared-drive duplicates by started car, then best finishing row. Some early shared drives can credit more than three distinct drivers with `positionOrder <= 3` in one race; this is an outcome-data quirk, not duplicate modeling rows.
- **Leakage:** current-race grid and qualifying are allowed; form is shifted to earlier starts/races; standings come from the latest strictly earlier round. Current-race outcome columns are excluded from the feature list. Earlier held-out race outcomes may be historical inputs for later races, matching what would be known then, without fitting on test labels.
- **Split:** 2003–2019 train, 2020–2021 validation, 2022–2024 test. Hyperparameters, preprocessing order, family and F1 threshold use validation only.
- **Metrics:** PR-AUC (average precision) is primary, F1 secondary; ROC-AUC, precision and recall complete the five reported metrics. The notebook shows train, validation and unseen test performance for each family.
- **Status mapping:** only explicitly named component/subsystem failures count as mechanical; ambiguous labels remain in the race-start denominator and are shown separately.
- **Interpretation:** home-race estimates and model explanations are associations/model behavior, not causal effects.

## Original milestone submission guidance

The milestone brief calls for an accessible Kaggle notebook link, an accessible GitHub repository link, and the PDF report through its specified form by **October 18, 2026, 11:59 PM**. This repository does not submit anything; you can run and inspect the notebook privately.
