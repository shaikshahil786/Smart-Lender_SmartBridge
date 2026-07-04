"""
app.py
=======
Smart Lender - Flask Web Application (Enhanced)
-------------------------------------------------
Routes:
  /           → Loan application form
  /predict    → POST: run single prediction
  /batch      → GET/POST: bulk CSV upload & scoring
  /history    → Session-based application history
  /about      → Model methodology & performance
  /reports    → Generated charts & analysis report
  /api/predict → JSON prediction API
  /api/emi    → JSON EMI calculator API
  /health     → Service health check

Run with:
    python app.py
"""

import os
import io
import json
import datetime

import numpy as np
import pandas as pd
import joblib
from flask import (
    Flask, render_template, request, jsonify,
    session, redirect, url_for, send_file, Response,
)

# --------------------------------------------------------------------------
# App setup
# --------------------------------------------------------------------------
app = Flask(__name__)
app.secret_key = os.environ.get("SMART_LENDER_SECRET", "smartlender-dev-secret-2024")

BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "model")

MODEL_PATH              = os.path.join(MODEL_DIR, "best_model.pkl")
SCALER_PATH             = os.path.join(MODEL_DIR, "scaler.pkl")
ENCODERS_PATH           = os.path.join(MODEL_DIR, "feature_encoders.pkl")
TARGET_ENCODER_PATH     = os.path.join(MODEL_DIR, "target_encoder.pkl")
FEATURE_COLUMNS_PATH    = os.path.join(MODEL_DIR, "feature_columns.json")
MODEL_COMPARISON_PATH   = os.path.join(MODEL_DIR, "model_comparison.json")
FEATURE_IMPORTANCES_PATH= os.path.join(MODEL_DIR, "feature_importances.json")

CATEGORICAL_COLS = [
    "Gender", "Married", "Dependents", "Education", "Self_Employed", "Property_Area",
]

FEATURE_LABELS = {
    "Gender": "Gender",
    "Married": "Marital Status",
    "Dependents": "No. of Dependents",
    "Education": "Education Level",
    "Self_Employed": "Self-Employed",
    "ApplicantIncome": "Applicant Income",
    "CoapplicantIncome": "Co-applicant Income",
    "LoanAmount": "Loan Amount",
    "Loan_Amount_Term": "Loan Term",
    "Credit_History": "Credit History",
    "Property_Area": "Property Area",
    "TotalIncome": "Total Household Income",
    "LoanAmount_log": "Loan Amount (log)",
    "TotalIncome_log": "Total Income (log)",
}

HISTORY_LIMIT = 10


# --------------------------------------------------------------------------
# Load artifacts
# --------------------------------------------------------------------------
def _load_artifacts():
    missing = [
        p for p in [MODEL_PATH, SCALER_PATH, ENCODERS_PATH, TARGET_ENCODER_PATH, FEATURE_COLUMNS_PATH]
        if not os.path.exists(p)
    ]
    if missing:
        raise FileNotFoundError(
            "Model artifacts not found:\n  " + "\n  ".join(missing) +
            "\n\nRun `python train_model.py` first."
        )

    model           = joblib.load(MODEL_PATH)
    scaler          = joblib.load(SCALER_PATH)
    feature_encoders= joblib.load(ENCODERS_PATH)
    target_encoder  = joblib.load(TARGET_ENCODER_PATH)

    with open(FEATURE_COLUMNS_PATH) as f:
        feature_columns = json.load(f)

    model_comparison = {}
    if os.path.exists(MODEL_COMPARISON_PATH):
        with open(MODEL_COMPARISON_PATH) as f:
            model_comparison = json.load(f)

    feature_importances = []
    if os.path.exists(FEATURE_IMPORTANCES_PATH):
        with open(FEATURE_IMPORTANCES_PATH) as f:
            feature_importances = json.load(f)

    return model, scaler, feature_encoders, target_encoder, feature_columns, model_comparison, feature_importances


try:
    (MODEL, SCALER, FEATURE_ENCODERS, TARGET_ENCODER,
     FEATURE_COLUMNS, MODEL_COMPARISON, FEATURE_IMPORTANCES) = _load_artifacts()
    MODEL_LOAD_ERROR = None
except FileNotFoundError as e:
    MODEL = SCALER = FEATURE_ENCODERS = TARGET_ENCODER = FEATURE_COLUMNS = None
    MODEL_COMPARISON = {}
    FEATURE_IMPORTANCES = []
    MODEL_LOAD_ERROR = str(e)
    print(f"[WARNING] {MODEL_LOAD_ERROR}")


# --------------------------------------------------------------------------
# Preprocessing helpers
# --------------------------------------------------------------------------
def build_feature_row(form_data: dict) -> dict:
    gender          = form_data["gender"]
    married         = form_data["married"]
    dependents      = form_data["dependents"]
    education       = form_data["education"]
    self_employed   = form_data["self_employed"]
    applicant_income   = float(form_data["applicant_income"])
    coapplicant_income = float(form_data.get("coapplicant_income", 0) or 0)
    loan_amount     = float(form_data["loan_amount"])
    loan_amount_term= float(form_data["loan_amount_term"])
    credit_history  = float(form_data["credit_history"])
    property_area   = form_data["property_area"]

    dependents_numeric = dependents.replace("3+", "3")
    total_income    = applicant_income + coapplicant_income
    loan_amount_log = np.log1p(loan_amount)
    total_income_log= np.log1p(total_income)

    return {
        "Gender": gender,
        "Married": married,
        "Dependents": int(dependents_numeric),
        "Education": education,
        "Self_Employed": self_employed,
        "ApplicantIncome": applicant_income,
        "CoapplicantIncome": coapplicant_income,
        "LoanAmount": loan_amount,
        "Loan_Amount_Term": loan_amount_term,
        "Credit_History": credit_history,
        "Property_Area": property_area,
        "TotalIncome": total_income,
        "LoanAmount_log": loan_amount_log,
        "TotalIncome_log": total_income_log,
    }


def encode_row(row: dict) -> pd.DataFrame:
    encoded_row = dict(row)
    for col in CATEGORICAL_COLS:
        le = FEATURE_ENCODERS[col]
        value = str(encoded_row[col])
        if value not in le.classes_:
            value = le.classes_[0]
        encoded_row[col] = le.transform([value])[0]

    ordered = {col: [encoded_row[col]] for col in FEATURE_COLUMNS}
    return pd.DataFrame(ordered, columns=FEATURE_COLUMNS)


def compute_feature_contributions(X_scaled) -> list:
    """Return per-feature contribution scores using feature importances + scaled values."""
    if not FEATURE_IMPORTANCES or MODEL is None:
        return []

    contributions = []
    fi_dict = {item["feature"]: item["importance"] for item in FEATURE_IMPORTANCES}

    for col in FEATURE_COLUMNS:
        importance = fi_dict.get(col, 0)
        # Positive contribution if feature value is above 0 (post-scaling mean=0)
        # This gives a directional sense for visualisation
        val = float(X_scaled[0][FEATURE_COLUMNS.index(col)])
        direction = "positive" if val >= 0 else "negative"
        contributions.append({
            "feature": col,
            "label": FEATURE_LABELS.get(col, col),
            "importance": round(importance, 2),
            "direction": direction,
        })

    # Sort by importance descending
    contributions.sort(key=lambda x: x["importance"], reverse=True)
    return contributions[:8]  # Top 8 features


def predict_loan_status(form_data: dict):
    row     = build_feature_row(form_data)
    X       = encode_row(row)
    X_scaled= SCALER.transform(X)

    pred_encoded = MODEL.predict(X_scaled)[0]
    pred_label   = TARGET_ENCODER.inverse_transform([pred_encoded])[0]

    probability = None
    if hasattr(MODEL, "predict_proba"):
        proba = MODEL.predict_proba(X_scaled)[0]
        probability = float(np.max(proba))

    contributions = compute_feature_contributions(X_scaled)

    return pred_label, probability, row, contributions


# --------------------------------------------------------------------------
# Batch prediction helper
# --------------------------------------------------------------------------
def predict_batch(df_input: pd.DataFrame) -> pd.DataFrame:
    """Run predictions on a DataFrame of applicant rows."""
    results = []
    for _, row_raw in df_input.iterrows():
        try:
            fd = {
                "gender":            str(row_raw.get("Gender", "Male")),
                "married":           str(row_raw.get("Married", "No")),
                "dependents":        str(row_raw.get("Dependents", "0")),
                "education":         str(row_raw.get("Education", "Graduate")),
                "self_employed":     str(row_raw.get("Self_Employed", "No")),
                "applicant_income":  float(row_raw.get("ApplicantIncome", 0)),
                "coapplicant_income":float(row_raw.get("CoapplicantIncome", 0)),
                "loan_amount":       float(row_raw.get("LoanAmount", 100)),
                "loan_amount_term":  float(row_raw.get("Loan_Amount_Term", 360)),
                "credit_history":    float(row_raw.get("Credit_History", 1)),
                "property_area":     str(row_raw.get("Property_Area", "Urban")),
            }
            label, prob, _, _ = predict_loan_status(fd)
            results.append({
                "Predicted_Status": label,
                "Approved": "Yes" if label == "Y" else "No",
                "Confidence_Pct": round(prob * 100, 2) if prob is not None else None,
            })
        except Exception as e:
            results.append({"Predicted_Status": "ERROR", "Approved": "ERROR", "Confidence_Pct": None})

    result_df = df_input.copy()
    result_df["Predicted_Status"] = [r["Predicted_Status"] for r in results]
    result_df["Approved"]         = [r["Approved"]         for r in results]
    result_df["Confidence_Pct"]   = [r["Confidence_Pct"]   for r in results]
    return result_df


# --------------------------------------------------------------------------
# Session history helpers
# --------------------------------------------------------------------------
def _add_to_history(form_data, approved, probability):
    if "history" not in session:
        session["history"] = []

    entry = {
        "ts": datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
        "approved": approved,
        "probability": round(probability * 100, 1) if probability else None,
        "gender": form_data.get("gender"),
        "education": form_data.get("education"),
        "applicant_income": form_data.get("applicant_income"),
        "loan_amount": form_data.get("loan_amount"),
        "property_area": form_data.get("property_area"),
        "credit_history": form_data.get("credit_history"),
    }
    history = session["history"]
    history.insert(0, entry)
    session["history"] = history[:HISTORY_LIMIT]
    session.modified = True


def _get_session_stats():
    history = session.get("history", [])
    if not history:
        return {"total": 0, "approval_rate": 0, "avg_confidence": 0}

    total    = len(history)
    approved = sum(1 for h in history if h["approved"])
    confs    = [h["probability"] for h in history if h["probability"] is not None]
    return {
        "total": total,
        "approval_rate": round(approved / total * 100, 1),
        "avg_confidence": round(sum(confs) / len(confs), 1) if confs else 0,
    }


# --------------------------------------------------------------------------
# Routes
# --------------------------------------------------------------------------
@app.route("/")
def home():
    stats = _get_session_stats()
    recent = session.get("history", [])[:3]
    return render_template(
        "index.html",
        model_error=MODEL_LOAD_ERROR,
        model_comparison=MODEL_COMPARISON,
        stats=stats,
        recent=recent,
    )


@app.route("/predict", methods=["POST"])
def predict():
    if MODEL is None:
        return render_template("result.html",
                               error="Model not loaded. Run `python train_model.py` first.")
    try:
        form_data = {
            "gender":            request.form["gender"],
            "married":           request.form["married"],
            "dependents":        request.form["dependents"],
            "education":         request.form["education"],
            "self_employed":     request.form["self_employed"],
            "applicant_income":  request.form["applicant_income"],
            "coapplicant_income":request.form.get("coapplicant_income", "0"),
            "loan_amount":       request.form["loan_amount"],
            "loan_amount_term":  request.form["loan_amount_term"],
            "credit_history":    request.form["credit_history"],
            "property_area":     request.form["property_area"],
        }

        pred_label, probability, row, contributions = predict_loan_status(form_data)
        approved = pred_label == "Y"

        _add_to_history(form_data, approved, probability)

        # Compute helpful metrics for the result page
        loan_amount_k   = float(form_data["loan_amount"])
        term_months     = float(form_data["loan_amount_term"])
        total_income    = float(form_data["applicant_income"]) + float(form_data.get("coapplicant_income") or 0)
        monthly_emi     = _calc_emi(loan_amount_k * 1000, 8.5, term_months)
        dti             = round((monthly_emi / total_income * 100), 1) if total_income > 0 else None

        return render_template(
            "result.html",
            approved=approved,
            probability=round(probability * 100, 2) if probability is not None else None,
            form_data=form_data,
            contributions=contributions,
            monthly_emi=round(monthly_emi, 2),
            dti=dti,
            error=None,
        )

    except (KeyError, ValueError) as e:
        return render_template("result.html",
                               error=f"Invalid input: {e}. Please go back and check all fields.")


@app.route("/batch", methods=["GET", "POST"])
def batch():
    if request.method == "GET":
        return render_template("batch.html", model_error=MODEL_LOAD_ERROR)

    if MODEL is None:
        return render_template("batch.html", error="Model not loaded.", model_error=MODEL_LOAD_ERROR)

    file = request.files.get("csv_file")
    if not file or not file.filename.endswith(".csv"):
        return render_template("batch.html", error="Please upload a valid .csv file.", model_error=MODEL_LOAD_ERROR)

    try:
        df_input = pd.read_csv(file)
        df_result = predict_batch(df_input)

        preview_html = df_input.head(5).to_html(
            classes="batch-preview-table", index=False, border=0
        )

        # Store result in session for download
        csv_bytes = df_result.to_csv(index=False).encode("utf-8")
        session["batch_result_csv"] = csv_bytes.decode("utf-8")
        session.modified = True

        total     = len(df_result)
        approved  = int((df_result["Approved"] == "Yes").sum())
        avg_conf  = round(df_result["Confidence_Pct"].mean(), 1) if "Confidence_Pct" in df_result else None

        return render_template(
            "batch.html",
            preview_html=preview_html,
            total=total,
            approved=approved,
            declined=total - approved,
            avg_conf=avg_conf,
            model_error=MODEL_LOAD_ERROR,
            download_ready=True,
        )
    except Exception as e:
        return render_template("batch.html",
                               error=f"Processing error: {e}",
                               model_error=MODEL_LOAD_ERROR)


@app.route("/batch/download")
def batch_download():
    csv_data = session.get("batch_result_csv")
    if not csv_data:
        return redirect(url_for("batch"))
    return Response(
        csv_data,
        mimetype="text/csv",
        headers={"Content-Disposition": "attachment; filename=smartlender_batch_results.csv"},
    )


@app.route("/history")
def history():
    entries = session.get("history", [])
    stats   = _get_session_stats()
    return render_template("history.html", history=entries, stats=stats)


@app.route("/history/clear", methods=["POST"])
def history_clear():
    session.pop("history", None)
    return redirect(url_for("history"))


@app.route("/about")
def about():
    return render_template("about.html",
                           model_comparison=MODEL_COMPARISON,
                           feature_importances=FEATURE_IMPORTANCES)


@app.route("/reports")
def reports():
    reports_dir = os.path.join(BASE_DIR, "static", "reports")
    report_files = []
    chart_order = [
        ("model_comparison.png",    "Model Comparison", "Test accuracy, CV accuracy, and ROC-AUC scores across all trained models."),
        ("feature_importances.png", "Feature Importances", "Relative contribution of each feature to the best model's predictions."),
        ("roc_curves.png",          "ROC Curves", "Receiver Operating Characteristic curves with AUC values for all models."),
        ("confusion_matrices.png",  "Confusion Matrices", "True/false positive/negative breakdown for each model on the test set."),
        ("calibration_curves.png",  "Calibration Curves", "How well each model's predicted probabilities match observed outcomes."),
        ("income_vs_loan.png",      "Income vs Loan Amount", "Scatter plot of income against loan amount, coloured by approval outcome."),
        ("approval_breakdown.png",  "Approval Rate by Category", "Approval rate segmented by applicant demographic and financial categories."),
        ("credit_history_impact.png","Credit History Impact", "Donut charts showing outcome split for applicants with and without credit history."),
    ]
    for fname, title, description in chart_order:
        path = os.path.join(reports_dir, fname)
        if os.path.exists(path):
            report_files.append({
                "filename": fname,
                "url": url_for("static", filename=f"reports/{fname}"),
                "title": title,
                "description": description,
            })

    return render_template("reports.html",
                           report_files=report_files,
                           model_comparison=MODEL_COMPARISON)


# --------------------------------------------------------------------------
# API routes
# --------------------------------------------------------------------------
def _calc_emi(principal: float, annual_rate_pct: float, term_months: float) -> float:
    """Standard EMI formula. Rate in percent per annum, term in months."""
    if term_months <= 0 or principal <= 0:
        return 0.0
    r = annual_rate_pct / 100 / 12
    if r == 0:
        return principal / term_months
    return principal * r * ((1 + r) ** term_months) / (((1 + r) ** term_months) - 1)


@app.route("/api/emi", methods=["POST"])
def api_emi():
    data = request.get_json(force=True)
    try:
        principal     = float(data.get("loan_amount_k", 0)) * 1000
        annual_rate   = float(data.get("annual_rate_pct", 8.5))
        term_months   = float(data.get("term_months", 360))
        monthly_income= float(data.get("monthly_income", 1))

        emi         = _calc_emi(principal, annual_rate, term_months)
        total_pay   = emi * term_months
        total_int   = total_pay - principal
        dti         = (emi / monthly_income * 100) if monthly_income > 0 else None

        return jsonify({
            "emi":         round(emi, 2),
            "total_pay":   round(total_pay, 2),
            "total_int":   round(total_int, 2),
            "dti_pct":     round(dti, 1) if dti else None,
        })
    except (KeyError, ValueError, ZeroDivisionError) as e:
        return jsonify({"error": str(e)}), 400


@app.route("/api/predict", methods=["POST"])
def api_predict():
    if MODEL is None:
        return jsonify({"error": MODEL_LOAD_ERROR}), 503

    try:
        data = request.get_json(force=True)
        form_data = {
            "gender":            data["gender"],
            "married":           data["married"],
            "dependents":        str(data["dependents"]),
            "education":         data["education"],
            "self_employed":     data["self_employed"],
            "applicant_income":  data["applicant_income"],
            "coapplicant_income":data.get("coapplicant_income", 0),
            "loan_amount":       data["loan_amount"],
            "loan_amount_term":  data["loan_amount_term"],
            "credit_history":    data["credit_history"],
            "property_area":     data["property_area"],
        }

        pred_label, probability, _, contributions = predict_loan_status(form_data)

        return jsonify({
            "loan_status":    pred_label,
            "approved":       pred_label == "Y",
            "confidence":     round(probability, 4) if probability is not None else None,
            "contributions":  contributions,
        })

    except (KeyError, ValueError) as e:
        return jsonify({"error": f"Invalid input: {e}"}), 400
    except Exception as e:
        return jsonify({"error": f"Unexpected error: {e}"}), 500


@app.route("/health")
def health():
    return jsonify({
        "status": "ok",
        "model_loaded": MODEL is not None,
        "feature_importances_loaded": bool(FEATURE_IMPORTANCES),
    })


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
