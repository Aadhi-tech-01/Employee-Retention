from pathlib import Path
import json
import warnings
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier

warnings.filterwarnings("ignore")
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "employee_retention_ml_dataset_1000.csv"
MODEL_DIR = BASE_DIR / "models"
OUTPUT_DIR = BASE_DIR / "outputs"
MODEL_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

TARGET = "Attrition"
DROP_COLUMNS = ["Employee_ID", "Employee_Stayed", TARGET, "Joining_Date",
                "Exit_Date", "Termination_Date", "Exit_Reason", "Resignation_Date"]

def load_data():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)
    if TARGET not in df.columns:
        if "Employee_Stayed" in df.columns:
            df[TARGET] = (pd.to_numeric(df["Employee_Stayed"], errors="coerce") == 0).astype(int)
        else:
            raise ValueError("Dataset must contain Attrition or Employee_Stayed.")
    df[TARGET] = pd.to_numeric(df[TARGET], errors="coerce")
    df = df.dropna(subset=[TARGET]).copy()
    if not set(df[TARGET].unique()).issubset({0, 1}):
        raise ValueError("Attrition target must be binary: 0=Stayed, 1=Left.")
    df[TARGET] = df[TARGET].astype(int)
    # Feature engineering using only information available in the dataset.
    if {"Current_Salary_LPA", "Joining_Salary_LPA"}.issubset(df.columns):
        growth = df["Current_Salary_LPA"] - df["Joining_Salary_LPA"]
        df["Salary_Growth_Absolute"] = growth
        df["Salary_Growth_Percent"] = (
            growth / df["Joining_Salary_LPA"].replace(0, np.nan)
        ) * 100
    satisfaction = [c for c in [
        "Avg_Job_Satisfaction", "Avg_Workload_Satisfaction",
        "Avg_Manager_Support", "Avg_Career_Growth_Satisfaction"
    ] if c in df.columns]
    if satisfaction:
        df["Composite_Satisfaction"] = df[satisfaction].mean(axis=1)
    if {"Avg_Skill_Gap_Score", "Total_Training_Hours"}.issubset(df.columns):
        df["Skill_Gap_Per_Training"] = df["Avg_Skill_Gap_Score"] / (df["Total_Training_Hours"] + 1)
    return df

def train():
    df = load_data()
    feature_cols = [c for c in df.columns if c not in DROP_COLUMNS]
    X = df[feature_cols].replace([np.inf, -np.inf], np.nan)
    y = df[TARGET]
    if y.nunique() < 2:
        raise ValueError("Training requires both Attrition classes (0 and 1).")
    cat_cols = X.select_dtypes(include=["object", "string", "category"]).columns.tolist()
    num_cols = [c for c in X.columns if c not in cat_cols]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    try:
        encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        encoder = OneHotEncoder(handle_unknown="ignore", sparse=False)

    preprocessor = ColumnTransformer([
        ("numeric", Pipeline([("imputer", SimpleImputer(strategy="median"))]), num_cols),
        ("categorical", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", encoder)
        ]), cat_cols)
    ])
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", RandomForestClassifier(
            n_estimators=350, max_depth=8, min_samples_leaf=3,
            class_weight="balanced", random_state=42, n_jobs=-1
        ))
    ])
    pipeline.fit(X_train, y_train)
    probabilities = pipeline.predict_proba(X_test)[:, 1]
    predictions = (probabilities >= 0.5).astype(int)
    metrics = {
        "accuracy": float(accuracy_score(y_test, predictions)),
        "precision": float(precision_score(y_test, predictions, zero_division=0)),
        "recall": float(recall_score(y_test, predictions, zero_division=0)),
        "f1": float(f1_score(y_test, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, probabilities))
    }
    joblib.dump(pipeline, MODEL_DIR / "employee_attrition_pipeline.joblib")
    # Baselines allow the web form to supply a small set of fields safely.
    defaults = {}
    for col in feature_cols:
        if col in cat_cols:
            mode = X_train[col].mode(dropna=True)
            defaults[col] = str(mode.iloc[0]) if len(mode) else "Unknown"
        else:
            med = pd.to_numeric(X_train[col], errors="coerce").median()
            defaults[col] = float(med) if pd.notna(med) else 0.0

    metadata = {
        "model": "Random Forest Classifier",
        "target": TARGET,
        "target_mapping": {"0": "Stayed", "1": "Left"},
        "threshold": 0.5,
        "metrics": metrics,
        "feature_columns": feature_cols,
        "categorical_columns": cat_cols,
        "numerical_columns": num_cols,
        "defaults": defaults,
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "dataset_rows": int(len(df))
    }
    (MODEL_DIR / "model_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    pd.DataFrame([metrics]).to_csv(OUTPUT_DIR / "test_metrics.csv", index=False)
    pd.DataFrame({
        "actual": y_test.to_numpy(),
        "predicted": predictions,
        "attrition_probability": probabilities
    }).to_csv(OUTPUT_DIR / "test_predictions.csv", index=False)
    (OUTPUT_DIR / "classification_report.txt").write_text(
        classification_report(y_test, predictions, target_names=["Stayed", "Left"], zero_division=0),
        encoding="utf-8"
    )
    print("Training completed.")
    print(f"Rows: {len(df)} | Features: {len(feature_cols)}")
    print("Metrics:", json.dumps(metrics, indent=2))
    print(f"Saved model: {MODEL_DIR / 'employee_attrition_pipeline.joblib'}")
    return metadata

if __name__ == "__main__":
    train()
