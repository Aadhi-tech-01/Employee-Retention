from pathlib import Path
import json
import io
import joblib
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, redirect, url_for, flash, send_file
from train_model import train, BASE_DIR, MODEL_DIR, OUTPUT_DIR, load_data

app = Flask(__name__)
app.secret_key = "employee-retention-local-demo-key"
MODEL_PATH = MODEL_DIR / "employee_attrition_pipeline.joblib"
META_PATH = MODEL_DIR / "model_metadata.json"

def get_model():
    if not MODEL_PATH.exists() or not META_PATH.exists():
        train()
    model = joblib.load(MODEL_PATH)
    metadata = json.loads(META_PATH.read_text(encoding="utf-8"))
    return model, metadata

def add_engineered_features(record):
    if "Current_Salary_LPA" in record and "Joining_Salary_LPA" in record:
        growth = float(record["Current_Salary_LPA"]) - float(record["Joining_Salary_LPA"])
        record["Salary_Growth_Absolute"] = growth
        joining = float(record["Joining_Salary_LPA"])
        record["Salary_Growth_Percent"] = (growth / joining * 100) if joining else np.nan
    sat = [record.get(c) for c in [
        "Avg_Job_Satisfaction", "Avg_Workload_Satisfaction",
        "Avg_Manager_Support", "Avg_Career_Growth_Satisfaction"
    ] if record.get(c) is not None]
    if sat:
        record["Composite_Satisfaction"] = float(np.mean(sat))
    if record.get("Avg_Skill_Gap_Score") is not None and record.get("Total_Training_Hours") is not None:
        record["Skill_Gap_Per_Training"] = float(record["Avg_Skill_Gap_Score"]) / (float(record["Total_Training_Hours"]) + 1)
    return record

def recommendations(record):
    tips = []
    if float(record.get("Avg_Job_Satisfaction", 5) or 5) < 3:
        tips.append(("Job satisfaction", "Schedule a confidential one-to-one and agree on 2–3 changes that would improve the employee's day-to-day experience."))
    if float(record.get("Avg_Workload_Satisfaction", 5) or 5) < 3:
        tips.append(("Workload", "Review workload, deadlines and staffing; rebalance tasks and agree on sustainable priorities."))
    if float(record.get("Avg_Manager_Support", 5) or 5) < 3:
        tips.append(("Manager support", "Arrange regular manager check-ins and clarify expectations, feedback and support needs."))
    if float(record.get("Avg_Career_Growth_Satisfaction", 5) or 5) < 3:
        tips.append(("Career growth", "Create a development plan with skills training, mentoring and transparent promotion milestones."))
    if float(record.get("Avg_Skill_Gap_Score", 0) or 0) > 5:
        tips.append(("Skill development", "Offer role-relevant training and protected learning time to close the identified skill gaps."))
    if float(record.get("Years_Since_Last_Promotion", 0) or 0) > 3:
        tips.append(("Career progression", "Review career progression and discuss a realistic growth path with measurable milestones."))
    if float(record.get("Avg_Task_Completion_Rate", 100) or 100) < 75:
        tips.append(("Performance support", "Explore blockers, tools and workload before setting a practical performance improvement plan."))
    if not tips:
        tips.append(("Maintain engagement", "Continue regular check-ins, recognition and development conversations; validate the employee's preferences directly."))
    return tips

@app.route("/")
def index():
    model, meta = get_model()
    df = load_data()
    metrics = meta["metrics"]
    counts = df["Attrition"].value_counts().to_dict()
    departments = []
    if "Department" in df.columns:
        dept = df.groupby("Department")["Attrition"].mean().sort_values(ascending=False)
        departments = [{"name": str(k), "rate": round(float(v)*100, 1)} for k, v in dept.items()]
    return render_template("index.html", metrics=metrics, total=len(df),
                           left=int(counts.get(1, 0)), stayed=int(counts.get(0, 0)),
                           attrition_rate=round(float(df["Attrition"].mean())*100, 1),
                           departments=departments, model_name=meta["model"])

@app.route("/predict", methods=["GET", "POST"])
def predict():
    model, meta = get_model()
    fields = [
        # Choices below match the exact categories in the bundled training dataset.
        ("Department", "select", ["Finance", "HR", "IT", "Operations", "Sales"]),
        ("Job_Level", "select", ["Junior", "Lead", "Mid", "Senior"]),
        ("Employment_Type", "select", ["Contract", "Full-time"]),
        ("Work_Location", "select", ["Bengaluru", "Chennai", "Coimbatore", "Hyderabad", "Remote"]),
        ("Previous_Experience_Years", "number", None),
        ("Joining_Salary_LPA", "number", None),
        ("Current_Salary_LPA", "number", None),
        ("Avg_Task_Completion_Rate", "number", None),
        ("Avg_Quality_Score", "number", None),
        ("Avg_Deadline_Adherence", "number", None),
        ("Avg_Productivity_Score", "number", None),
        ("Avg_Error_Rate", "number", None),
        ("Total_Training_Hours", "number", None),
        ("Avg_Skill_Gap_Score", "number", None),
        ("Avg_Job_Satisfaction", "number", None),
        ("Avg_Workload_Satisfaction", "number", None),
        ("Avg_Manager_Support", "number", None),
        ("Avg_Career_Growth_Satisfaction", "number", None),
        ("Years_Since_Last_Promotion", "number", None),
        ("Promotion_Count", "number", None),
        ("Recognition_Count", "number", None),
        # This dataset stores Performance_Trend as a numeric score, not text labels.
        ("Performance_Trend", "number", None)
    ]
    if request.method == "POST":
        record = dict(meta["defaults"])
        for name, kind, options in fields:
            raw = request.form.get(name)
            if raw is not None and raw != "":
                if kind == "number":
                    record[name] = float(raw)
                else:
                    record[name] = raw
        record = add_engineered_features(record)
        row = pd.DataFrame([{c: record.get(c, np.nan) for c in meta["feature_columns"]}])
        probability = float(model.predict_proba(row)[0, 1])
        risk = "High" if probability >= 0.70 else ("Medium" if probability >= 0.40 else "Low")
        tips = recommendations(record)
        return render_template("prediction.html", probability=round(probability*100, 1),
                               risk=risk, tips=tips, record=record)
    defaults = meta["defaults"]
    return render_template("predict.html", fields=fields, defaults=defaults)

@app.route("/bulk", methods=["GET", "POST"])
def bulk():
    if request.method == "POST":
        uploaded = request.files.get("file")
        if not uploaded or not uploaded.filename.lower().endswith(".csv"):
            flash("Please upload a CSV file.", "error")
            return redirect(url_for("bulk"))
        try:
            model, meta = get_model()
            data = pd.read_csv(uploaded)
            # Add absent model fields using training-set defaults.
            for col in meta["feature_columns"]:
                if col not in data.columns:
                    data[col] = meta["defaults"].get(col, np.nan)
            X = data[meta["feature_columns"]].copy()
            # Recreate the same engineered columns when absent.
            for idx in X.index:
                rec = add_engineered_features(X.loc[idx].to_dict())
                for col in ["Salary_Growth_Absolute", "Salary_Growth_Percent", "Composite_Satisfaction", "Skill_Gap_Per_Training"]:
                    if col in X.columns and col not in data.columns:
                        X.loc[idx, col] = rec.get(col, np.nan)
            probs = model.predict_proba(X)[:, 1]
            result = data.copy()
            result["Attrition_Probability"] = (probs * 100).round(1)
            result["Predicted_Attrition"] = np.where(probs >= 0.5, "Likely to Leave", "Likely to Stay")
            result["Risk_Level"] = np.where(probs >= 0.7, "High", np.where(probs >= 0.4, "Medium", "Low"))
            out = OUTPUT_DIR / "bulk_predictions.csv"
            result.to_csv(out, index=False)
            return send_file(out, as_attachment=True, download_name="employee_attrition_predictions.csv")
        except Exception as exc:
            flash(f"Could not process file: {exc}", "error")
            return redirect(url_for("bulk"))
    return render_template("bulk.html")

@app.route("/health")
def health():
    return {"status": "ok", "project": "Employee Retention AI"}

if __name__ == "__main__":
    get_model()
    app.run(host="127.0.0.1", port=5000, debug=False)
