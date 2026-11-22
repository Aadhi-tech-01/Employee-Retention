# PeoplePulse AI — Employee Retention Prediction

A runnable educational full-stack project built with Python, Flask, pandas and scikit-learn. It includes a pre-provided synthetic employee dataset, model training, a web dashboard, individual predictions, retention conversation suggestions and CSV batch scoring.

## Quick start (Windows)

1. Extract the ZIP file.
2. Open the extracted `Employee_Retention_AI_Project` folder.
3. Double-click `run_windows.bat`.
4. Wait for dependencies to install and the model to train the first time.
5. Open **http://127.0.0.1:5000** in your browser.

Requirements: Python 3.10 or newer and an internet connection for the first dependency installation. No API key, database account or manual code edits are required.

## macOS / Linux

From this folder, run:

```bash
chmod +x run_mac_linux.sh
./run_mac_linux.sh
```

Then open http://127.0.0.1:5000.

## What is included

- `data/employee_retention_ml_dataset_1000.csv`: 1,000-row synthetic ML dataset.
- `data/employee_retention_source.xlsx`: original workbook with source sheets.
- `train_model.py`: reproducible train/test split, preprocessing pipeline and Random Forest training.
- `app.py`: Flask web application.
- `templates/`: dashboard, individual prediction, result and batch-upload pages.
- `static/style.css`: responsive styling.
- `models/`: model and metadata are generated automatically on first launch.
- `outputs/`: test metrics and held-out predictions are generated during training.

## Target definition

`Attrition = 1` means the employee left. `Attrition = 0` means the employee stayed. The model excludes employee ID, the duplicate target proxy `Employee_Stayed`, and raw `Joining_Date`. Numeric missing values are imputed with training-set medians and categorical missing values with training-set modes inside the pipeline.

## Manual commands

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
python train_model.py
python app.py
```

## Important limitations and responsible use

The bundled data is synthetic. Metrics are a demonstration on a held-out split of that synthetic dataset; they do not prove real-world accuracy, fairness or generalization. Risk bands are illustrative. The application is a learning and decision-support prototype, not an employment decision system. Never use an automated score as the sole basis for hiring, promotion, discipline, termination or other consequential employment decisions. Validate on representative, lawfully collected data, perform fairness/privacy reviews, and involve qualified human reviewers before any real-world use.
