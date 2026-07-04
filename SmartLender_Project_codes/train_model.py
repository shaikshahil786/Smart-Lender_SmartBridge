"""
train_model.py
================
Smart Lender - Loan Eligibility Prediction
--------------------------------------------
This script:
  1. Loads the loan applicant dataset (data/train.csv)
  2. Cleans and preprocesses the data (missing values, encoding, feature engineering)
  3. Trains FIVE classification models:
        - Logistic Regression (baseline)
        - Decision Tree
        - Random Forest
        - K-Nearest Neighbors (KNN)
        - XGBoost
  4. Evaluates all models on:
        - Train / test accuracy
        - 5-fold cross-validation accuracy
        - ROC-AUC score
  5. Selects the best-performing model (by test accuracy) and saves it,
     along with the fitted preprocessing objects (scaler, encoders),
     so the Flask app can load them for real-time prediction.
  6. Exports feature_importances.json for the Risk Breakdown panel.
  7. Saves a detailed report with confusion matrices and metrics.

Run with:
    python train_model.py
"""

import os
import json
import warnings

import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")  # headless backend - no display needed
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    ConfusionMatrixDisplay,
)
from sklearn.calibration import calibration_curve

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    warnings.warn(
        "xgboost is not installed. Run `pip install xgboost` to enable the "
        "XGBoost model. Falling back to GradientBoostingClassifier."
    )
    from sklearn.ensemble import GradientBoostingClassifier

warnings.filterwarnings("ignore")

# --------------------------------------------------------------------------
# Paths / constants
# --------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "train.csv")
MODEL_DIR = os.path.join(BASE_DIR, "model")
REPORTS_DIR = os.path.join(BASE_DIR, "static", "reports")
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)

RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5

TARGET_COL = "Loan_Status"
ID_COL = "Loan_ID"

CATEGORICAL_COLS = [
    "Gender",
    "Married",
    "Dependents",
    "Education",
    "Self_Employed",
    "Property_Area",
]

NUMERIC_COLS = [
    "ApplicantIncome",
    "CoapplicantIncome",
    "LoanAmount",
    "Loan_Amount_Term",
    "Credit_History",
]

FEATURE_COLUMNS = [
    "Gender",
    "Married",
    "Dependents",
    "Education",
    "Self_Employed",
    "ApplicantIncome",
    "CoapplicantIncome",
    "LoanAmount",
    "Loan_Amount_Term",
    "Credit_History",
    "Property_Area",
    "TotalIncome",
    "LoanAmount_log",
    "TotalIncome_log",
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

# --------------------------------------------------------------------------
# Colour palette matching the CSS design system
# --------------------------------------------------------------------------
CLR_GREEN  = "#0f3d2e"
CLR_GOLD   = "#b8924a"
CLR_BRICK  = "#9b3b33"
CLR_PARCH  = "#f7f3e9"
CLR_INK    = "#1a1a17"
CLR_SOFT   = "#4a4a42"
CLR_LINE   = "#d8cfb4"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.facecolor": CLR_PARCH,
    "figure.facecolor": CLR_PARCH,
    "axes.edgecolor": CLR_LINE,
    "axes.labelcolor": CLR_INK,
    "xtick.color": CLR_SOFT,
    "ytick.color": CLR_SOFT,
    "text.color": CLR_INK,
    "grid.color": CLR_LINE,
    "grid.linewidth": 0.7,
})


# --------------------------------------------------------------------------
# Data loading & engineering
# --------------------------------------------------------------------------
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    print(f"Loaded dataset: {df.shape[0]} rows, {df.shape[1]} columns")
    return df


def clean_and_engineer(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Dependents"] = df["Dependents"].replace("3+", "3")

    for col in ["Gender", "Married", "Dependents", "Self_Employed"]:
        df[col] = df[col].fillna(df[col].mode()[0])

    df["LoanAmount"] = df["LoanAmount"].fillna(df["LoanAmount"].median())
    df["Loan_Amount_Term"] = df["Loan_Amount_Term"].fillna(df["Loan_Amount_Term"].mode()[0])
    df["Credit_History"] = df["Credit_History"].fillna(df["Credit_History"].mode()[0])

    df["Dependents"] = df["Dependents"].astype(int)

    df["TotalIncome"] = df["ApplicantIncome"] + df["CoapplicantIncome"]
    df["LoanAmount_log"] = np.log1p(df["LoanAmount"])
    df["TotalIncome_log"] = np.log1p(df["TotalIncome"])

    return df


def encode_features(df: pd.DataFrame, encoders: dict = None, fit: bool = True):
    df = df.copy()
    if encoders is None:
        encoders = {}

    for col in CATEGORICAL_COLS:
        if fit:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            encoders[col] = le
        else:
            le = encoders[col]
            df[col] = le.transform(df[col].astype(str))

    return df, encoders


# --------------------------------------------------------------------------
# Model building
# --------------------------------------------------------------------------
def build_models():
    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000, random_state=RANDOM_STATE, C=1.0
        ),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=5, random_state=RANDOM_STATE
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, max_depth=8, random_state=RANDOM_STATE
        ),
        "KNN": KNeighborsClassifier(n_neighbors=9),
    }

    if XGBOOST_AVAILABLE:
        models["XGBoost"] = XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.08,
            subsample=0.9,
            colsample_bytree=0.9,
            eval_metric="logloss",
            random_state=RANDOM_STATE,
        )
    else:
        models["XGBoost (fallback: GradientBoosting)"] = GradientBoostingClassifier(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.08,
            random_state=RANDOM_STATE,
        )

    return models


# --------------------------------------------------------------------------
# Training & evaluation
# --------------------------------------------------------------------------
def train_and_evaluate(models, X_train, X_test, y_train, y_test, scaler, X_full, y_full):
    results = {}
    X_train_s = scaler.transform(X_train)
    X_test_s  = scaler.transform(X_test)
    X_full_s  = scaler.transform(X_full)

    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    for name, model in models.items():
        model.fit(X_train_s, y_train)

        train_pred = model.predict(X_train_s)
        test_pred  = model.predict(X_test_s)

        train_acc = accuracy_score(y_train, train_pred)
        test_acc  = accuracy_score(y_test,  test_pred)

        cv_scores = cross_val_score(model, X_full_s, y_full, cv=cv, scoring="accuracy")

        roc_auc = None
        roc_data = None
        if hasattr(model, "predict_proba"):
            y_proba = model.predict_proba(X_test_s)[:, 1]
            roc_auc = roc_auc_score(y_test, y_proba)
            fpr, tpr, _ = roc_curve(y_test, y_proba)
            roc_data = {"fpr": fpr.tolist(), "tpr": tpr.tolist()}

        results[name] = {
            "model": model,
            "train_accuracy": train_acc,
            "test_accuracy": test_acc,
            "cv_mean": float(cv_scores.mean()),
            "cv_std": float(cv_scores.std()),
            "roc_auc": float(roc_auc) if roc_auc is not None else None,
            "roc_data": roc_data,
            "classification_report": classification_report(y_test, test_pred, output_dict=True),
            "confusion_matrix": confusion_matrix(y_test, test_pred).tolist(),
            "test_pred": test_pred,
            "y_test": y_test,
        }

        print(f"\n{'=' * 60}")
        print(f"Model: {name}")
        print(f"  Train Accuracy : {train_acc * 100:.2f}%")
        print(f"  Test  Accuracy : {test_acc  * 100:.2f}%")
        print(f"  CV Accuracy    : {cv_scores.mean() * 100:.2f}% ± {cv_scores.std() * 100:.2f}%")
        if roc_auc:
            print(f"  ROC-AUC        : {roc_auc:.4f}")
        print(classification_report(y_test, test_pred))

    return results


def select_best_model(results: dict):
    best_name = max(results, key=lambda n: results[n]["test_accuracy"])
    return best_name, results[best_name]


# --------------------------------------------------------------------------
# Feature importances
# --------------------------------------------------------------------------
def extract_feature_importances(model, feature_columns, feature_labels):
    importances = None

    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    elif hasattr(model, "coef_"):
        importances = np.abs(model.coef_[0])

    if importances is None:
        return {}

    total = importances.sum()
    if total == 0:
        return {}

    normed = (importances / total * 100).tolist()
    result = []
    for col, score in zip(feature_columns, normed):
        result.append({
            "feature": col,
            "label": feature_labels.get(col, col),
            "importance": round(score, 3),
        })
    result.sort(key=lambda x: x["importance"], reverse=True)
    return result


# --------------------------------------------------------------------------
# Chart / report generation
# --------------------------------------------------------------------------
def _fig_save(fig, filename):
    path = os.path.join(REPORTS_DIR, filename)
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  Saved: {path}")
    return path


def plot_model_comparison(results):
    names  = [n for n in results]
    test_accs = [results[n]["test_accuracy"] * 100 for n in names]
    cv_means  = [results[n]["cv_mean"] * 100 for n in names]
    cv_stds   = [results[n]["cv_std"] * 100 for n in names]
    aucs      = [results[n]["roc_auc"] * 100 if results[n]["roc_auc"] else 0 for n in names]

    x = np.arange(len(names))
    w = 0.28

    fig, ax = plt.subplots(figsize=(11, 5.5))
    fig.patch.set_facecolor(CLR_PARCH)
    ax.set_facecolor(CLR_PARCH)

    bars1 = ax.bar(x - w, test_accs, w, label="Test Accuracy (%)", color=CLR_GREEN,   alpha=0.9, zorder=3)
    bars2 = ax.bar(x,      cv_means,  w, label="CV Accuracy (%)",   color=CLR_GOLD,    alpha=0.9, zorder=3, yerr=cv_stds, capsize=4, error_kw={"ecolor": CLR_INK, "linewidth": 1.2})
    bars3 = ax.bar(x + w,  aucs,      w, label="ROC-AUC × 100",     color=CLR_BRICK,   alpha=0.85, zorder=3)

    for bar in [*bars1, *bars2, *bars3]:
        h = bar.get_height()
        if h > 0:
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.4, f"{h:.1f}",
                    ha="center", va="bottom", fontsize=7.5, color=CLR_INK, fontweight="600")

    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=12, ha="right", fontsize=9)
    ax.set_ylim(60, 105)
    ax.set_ylabel("Score", fontsize=10)
    ax.set_title("Model Comparison — Test Accuracy · CV Accuracy · ROC-AUC", fontsize=12, fontweight="700", pad=14)
    ax.legend(fontsize=9, framealpha=0.6, edgecolor=CLR_LINE)
    ax.yaxis.grid(True, zorder=0)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)

    return _fig_save(fig, "model_comparison.png")


def plot_confusion_matrices(results):
    n = len(results)
    ncols = min(3, n)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.5 * ncols, 4.5 * nrows))
    fig.patch.set_facecolor(CLR_PARCH)
    axes = np.array(axes).flatten()

    cmap = plt.cm.colors.LinearSegmentedColormap.from_list(
        "vault", [CLR_PARCH, CLR_GREEN]
    )

    for idx, (name, res) in enumerate(results.items()):
        ax = axes[idx]
        ax.set_facecolor(CLR_PARCH)
        cm = np.array(res["confusion_matrix"])
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Declined", "Approved"])
        disp.plot(ax=ax, colorbar=False, cmap=cmap)
        ax.set_title(name, fontsize=10, fontweight="700", color=CLR_INK, pad=8)
        ax.tick_params(labelsize=8)

    for i in range(len(results), len(axes)):
        axes[i].set_visible(False)

    fig.suptitle("Confusion Matrices — All Models", fontsize=13, fontweight="700", y=1.01)
    fig.tight_layout()
    return _fig_save(fig, "confusion_matrices.png")


def plot_roc_curves(results):
    fig, ax = plt.subplots(figsize=(7.5, 6))
    fig.patch.set_facecolor(CLR_PARCH)
    ax.set_facecolor(CLR_PARCH)

    colours = [CLR_GREEN, CLR_GOLD, CLR_BRICK, "#3a6ea5", "#7a4f8a"]
    for (name, res), colour in zip(results.items(), colours):
        if res["roc_data"] is None:
            continue
        fpr = res["roc_data"]["fpr"]
        tpr = res["roc_data"]["tpr"]
        auc = res["roc_auc"]
        ax.plot(fpr, tpr, lw=2, color=colour, label=f"{name}  (AUC={auc:.3f})")

    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5, label="Random classifier")
    ax.set_xlabel("False Positive Rate", fontsize=10)
    ax.set_ylabel("True Positive Rate", fontsize=10)
    ax.set_title("ROC Curves — All Models", fontsize=12, fontweight="700", pad=12)
    ax.legend(fontsize=9, framealpha=0.7, edgecolor=CLR_LINE, loc="lower right")
    ax.yaxis.grid(True, zorder=0)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)

    return _fig_save(fig, "roc_curves.png")


def plot_feature_importances(importances_list, best_name):
    if not importances_list:
        return None

    labels = [x["label"] for x in importances_list]
    scores = [x["importance"] for x in importances_list]

    colours = [CLR_GREEN if s > np.median(scores) else CLR_GOLD for s in scores]

    fig, ax = plt.subplots(figsize=(8, max(4, len(labels) * 0.5 + 1.5)))
    fig.patch.set_facecolor(CLR_PARCH)
    ax.set_facecolor(CLR_PARCH)

    y = np.arange(len(labels))
    bars = ax.barh(y, scores, color=colours, alpha=0.88, zorder=3, height=0.65)

    for bar, score in zip(bars, scores):
        ax.text(bar.get_width() + 0.2, bar.get_y() + bar.get_height() / 2,
                f"{score:.1f}%", va="center", fontsize=8.5, color=CLR_INK)

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9.5)
    ax.invert_yaxis()
    ax.set_xlabel("Relative Importance (%)", fontsize=10)
    ax.set_title(f"Feature Importances — {best_name}", fontsize=12, fontweight="700", pad=12)
    ax.xaxis.grid(True, zorder=0)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)

    patch_hi = mpatches.Patch(color=CLR_GREEN, label="Above median", alpha=0.88)
    patch_lo = mpatches.Patch(color=CLR_GOLD,  label="Below median", alpha=0.88)
    ax.legend(handles=[patch_hi, patch_lo], fontsize=8.5, framealpha=0.6, edgecolor=CLR_LINE)

    fig.tight_layout()
    return _fig_save(fig, "feature_importances.png")


def plot_calibration(results, X_test_s, y_test):
    fig, ax = plt.subplots(figsize=(7.5, 6))
    fig.patch.set_facecolor(CLR_PARCH)
    ax.set_facecolor(CLR_PARCH)

    colours = [CLR_GREEN, CLR_GOLD, CLR_BRICK, "#3a6ea5", "#7a4f8a"]
    for (name, res), colour in zip(results.items(), colours):
        model = res["model"]
        if not hasattr(model, "predict_proba"):
            continue
        prob_pos = model.predict_proba(X_test_s)[:, 1]
        fraction_of_pos, mean_predicted = calibration_curve(y_test, prob_pos, n_bins=8)
        ax.plot(mean_predicted, fraction_of_pos, marker="o", lw=2, color=colour, ms=5, label=name)

    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5, label="Perfect calibration")
    ax.set_xlabel("Mean Predicted Probability", fontsize=10)
    ax.set_ylabel("Fraction of Positives", fontsize=10)
    ax.set_title("Calibration Curves — All Models", fontsize=12, fontweight="700", pad=12)
    ax.legend(fontsize=9, framealpha=0.7, edgecolor=CLR_LINE)
    ax.yaxis.grid(True)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)

    return _fig_save(fig, "calibration_curves.png")


def plot_income_vs_loan(df_raw):
    """Exploratory chart: approved vs declined in income-loan space."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.patch.set_facecolor(CLR_PARCH)

    approved  = df_raw[df_raw["Loan_Status"] == "Y"]
    declined  = df_raw[df_raw["Loan_Status"] == "N"]

    for ax, xcol, xlabel in zip(
        axes,
        ["ApplicantIncome", "TotalIncome"],
        ["Applicant Income", "Total Household Income"],
    ):
        ax.set_facecolor(CLR_PARCH)
        ax.scatter(approved[xcol], approved["LoanAmount"],
                   alpha=0.45, s=22, color=CLR_GREEN, label="Approved", zorder=3)
        ax.scatter(declined[xcol], declined["LoanAmount"],
                   alpha=0.45, s=22, color=CLR_BRICK, label="Declined", marker="x", zorder=3)
        ax.set_xlabel(xlabel, fontsize=10)
        ax.set_ylabel("Loan Amount ($k)", fontsize=10)
        ax.set_title(f"{xlabel} vs Loan Amount", fontsize=11, fontweight="700")
        ax.legend(fontsize=9)
        ax.spines[["top", "right"]].set_visible(False)
        ax.yaxis.grid(True, zorder=0)
        ax.set_axisbelow(True)

    fig.suptitle("Income vs Loan Amount — Approved vs Declined", fontsize=13, fontweight="700")
    fig.tight_layout()
    return _fig_save(fig, "income_vs_loan.png")


def plot_approval_breakdown(df_raw):
    """Bar charts of approval rate by categorical variables."""
    cats = ["Gender", "Married", "Education", "Self_Employed", "Property_Area", "Credit_History"]
    n = len(cats)
    ncols = 3
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4.5 * nrows))
    fig.patch.set_facecolor(CLR_PARCH)
    axes = np.array(axes).flatten()

    for idx, col in enumerate(cats):
        ax = axes[idx]
        ax.set_facecolor(CLR_PARCH)
        grp = df_raw.groupby(col)["Loan_Status"].apply(
            lambda s: (s == "Y").mean() * 100
        ).reset_index()
        grp.columns = [col, "approval_rate"]
        xvals = grp[col].astype(str)
        bars = ax.bar(xvals, grp["approval_rate"], color=CLR_GREEN, alpha=0.88, zorder=3)
        for bar, v in zip(bars, grp["approval_rate"]):
            ax.text(bar.get_x() + bar.get_width() / 2, v + 0.5,
                    f"{v:.0f}%", ha="center", va="bottom", fontsize=8.5, color=CLR_INK)
        ax.set_title(f"Approval Rate by {col}", fontsize=10, fontweight="700")
        ax.set_ylabel("Approval Rate (%)", fontsize=9)
        ax.set_ylim(0, 105)
        ax.spines[["top", "right"]].set_visible(False)
        ax.yaxis.grid(True, zorder=0)
        ax.set_axisbelow(True)
        ax.tick_params(axis="x", labelrotation=10, labelsize=8.5)

    for i in range(n, len(axes)):
        axes[i].set_visible(False)

    fig.suptitle("Approval Rate by Applicant Category", fontsize=13, fontweight="700", y=1.01)
    fig.tight_layout()
    return _fig_save(fig, "approval_breakdown.png")


def plot_credit_history_impact(df_raw):
    """Pie / donut comparing credit history 0 vs 1 outcomes."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    fig.patch.set_facecolor(CLR_PARCH)

    for ax, ch_val, title in zip(axes, [0, 1], ["No Credit History", "Has Credit History"]):
        ax.set_facecolor(CLR_PARCH)
        subset = df_raw[df_raw["Credit_History"] == ch_val]["Loan_Status"].value_counts()
        approved_n = subset.get("Y", 0)
        declined_n = subset.get("N", 0)
        total = approved_n + declined_n
        wedges, texts, autotexts = ax.pie(
            [approved_n, declined_n],
            labels=["Approved", "Declined"],
            colors=[CLR_GREEN, CLR_BRICK],
            autopct="%1.1f%%",
            startangle=90,
            pctdistance=0.75,
            wedgeprops={"edgecolor": CLR_PARCH, "linewidth": 2},
        )
        # Draw a white circle in centre = donut
        centre_circle = plt.Circle((0, 0), 0.55, fc=CLR_PARCH)
        ax.add_patch(centre_circle)
        ax.set_title(f"{title}\n(n={total})", fontsize=11, fontweight="700")
        for at in autotexts:
            at.set_fontsize(9)

    fig.suptitle("Impact of Credit History on Loan Approval", fontsize=13, fontweight="700")
    fig.tight_layout()
    return _fig_save(fig, "credit_history_impact.png")


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main():
    print("\n" + "=" * 60)
    print("  SMART LENDER — Model Training Pipeline")
    print("=" * 60 + "\n")

    # 1. Load & engineer --------------------------------------------------
    df_raw = load_data(DATA_PATH)
    df     = clean_and_engineer(df_raw)

    # EDA charts (raw data, before encoding)
    print("\n[Charts] Generating exploratory data charts...")
    plot_income_vs_loan(df)
    plot_approval_breakdown(df)
    plot_credit_history_impact(df)

    # 2. Encode target & features -----------------------------------------
    target_encoder = LabelEncoder()
    y = target_encoder.fit_transform(df[TARGET_COL])

    df_encoded, feature_encoders = encode_features(df, fit=True)
    X = df_encoded[FEATURE_COLUMNS]

    # 3. Split & scale -----------------------------------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    scaler = StandardScaler()
    scaler.fit(X_train)

    X_test_s = scaler.transform(X_test)

    # 4. Train & evaluate --------------------------------------------------
    print("\n[Training] Fitting models and computing metrics...\n")
    models  = build_models()
    results = train_and_evaluate(models, X_train, X_test, y_train, y_test, scaler, X, y)

    # 5. Best model --------------------------------------------------------
    best_name, best_result = select_best_model(results)
    best_model = best_result["model"]

    print(f"\n{'#' * 60}")
    print(f"  BEST MODEL : {best_name}")
    print(f"  Train Acc  : {best_result['train_accuracy'] * 100:.2f}%")
    print(f"  Test  Acc  : {best_result['test_accuracy'] * 100:.2f}%")
    print(f"  CV Acc     : {best_result['cv_mean'] * 100:.2f}% ± {best_result['cv_std'] * 100:.2f}%")
    if best_result["roc_auc"]:
        print(f"  ROC-AUC    : {best_result['roc_auc']:.4f}")
    print(f"{'#' * 60}\n")

    # 6. Feature importances -----------------------------------------------
    fi_list = extract_feature_importances(best_model, FEATURE_COLUMNS, FEATURE_LABELS)

    # 7. Generate charts ---------------------------------------------------
    print("[Charts] Generating model performance charts...")
    plot_model_comparison(results)
    plot_confusion_matrices(results)
    plot_roc_curves(results)
    plot_feature_importances(fi_list, best_name)
    plot_calibration(results, X_test_s, y_test)

    # 8. Persist model artefacts -------------------------------------------
    joblib.dump(best_model,      os.path.join(MODEL_DIR, "best_model.pkl"))
    joblib.dump(scaler,          os.path.join(MODEL_DIR, "scaler.pkl"))
    joblib.dump(feature_encoders,os.path.join(MODEL_DIR, "feature_encoders.pkl"))
    joblib.dump(target_encoder,  os.path.join(MODEL_DIR, "target_encoder.pkl"))

    with open(os.path.join(MODEL_DIR, "feature_columns.json"), "w") as f:
        json.dump(FEATURE_COLUMNS, f, indent=2)

    with open(os.path.join(MODEL_DIR, "feature_importances.json"), "w") as f:
        json.dump(fi_list, f, indent=2)

    # Model comparison summary (used by Flask app)
    metrics_summary = {
        name: {
            "train_accuracy": round(res["train_accuracy"] * 100, 2),
            "test_accuracy":  round(res["test_accuracy"]  * 100, 2),
            "cv_mean":        round(res["cv_mean"]        * 100, 2),
            "cv_std":         round(res["cv_std"]         * 100, 2),
            "roc_auc":        round(res["roc_auc"],              4) if res["roc_auc"] else None,
        }
        for name, res in results.items()
    }
    metrics_summary["best_model"] = best_name

    with open(os.path.join(MODEL_DIR, "model_comparison.json"), "w") as f:
        json.dump(metrics_summary, f, indent=2)

    print("\n[Done] All artefacts saved:")
    for fname in ["best_model.pkl","scaler.pkl","feature_encoders.pkl",
                  "target_encoder.pkl","feature_columns.json",
                  "feature_importances.json","model_comparison.json"]:
        print(f"  model/{fname}")
    print("\n[Done] Report charts saved to static/reports/")


if __name__ == "__main__":
    main()
