"""
tests/test_model_performance.py
================================
Model performance / regression tests that verify the trained model
meets minimum quality thresholds on the real dataset.

These tests:
  - Are skipped automatically if model artifacts are missing
  - Serve as quality gates: CI fails if accuracy drops below threshold
  - Verify the feature_importances.json file is well-formed
  - Check that the scaler and encoders work on real data
"""

import os
import sys
import json
import pytest
import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import train_model as tm

MODEL_DIR  = os.path.join(ROOT, "model")
DATA_PATH  = os.path.join(ROOT, "data", "train.csv")

MODEL_PKL      = os.path.join(MODEL_DIR, "best_model.pkl")
SCALER_PKL     = os.path.join(MODEL_DIR, "scaler.pkl")
ENCODERS_PKL   = os.path.join(MODEL_DIR, "feature_encoders.pkl")
TARGET_ENC_PKL = os.path.join(MODEL_DIR, "target_encoder.pkl")
FI_JSON        = os.path.join(MODEL_DIR, "feature_importances.json")
COMPARISON_JSON= os.path.join(MODEL_DIR, "model_comparison.json")
FEATURES_JSON  = os.path.join(MODEL_DIR, "feature_columns.json")

# Skip all tests in this module if artifacts are absent
ARTIFACTS_OK = all(os.path.exists(p) for p in [
    MODEL_PKL, SCALER_PKL, ENCODERS_PKL, TARGET_ENC_PKL
]) and os.path.exists(DATA_PATH)

pytestmark = pytest.mark.skipif(
    not ARTIFACTS_OK,
    reason="Model artifacts or data/train.csv not found — run python train_model.py first"
)

# Minimum acceptable thresholds
MIN_TEST_ACCURACY = 0.75   # 75% minimum — well above random
MIN_ROC_AUC       = 0.70   # 70% AUC minimum


# ---------------------------------------------------------------------------
# Fixtures for this module
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def artifacts():
    model          = joblib.load(MODEL_PKL)
    scaler         = joblib.load(SCALER_PKL)
    encoders       = joblib.load(ENCODERS_PKL)
    target_encoder = joblib.load(TARGET_ENC_PKL)
    with open(FEATURES_JSON) as f:
        feature_columns = json.load(f)
    return model, scaler, encoders, target_encoder, feature_columns


@pytest.fixture(scope="module")
def prepared_test_data():
    """Load + clean + encode the real dataset and return the test split."""
    df_raw = tm.load_data(DATA_PATH)
    df     = tm.clean_and_engineer(df_raw)

    target_encoder = LabelEncoder()
    y = target_encoder.fit_transform(df["Loan_Status"])

    df_encoded, encoders = tm.encode_features(df, fit=True)
    X = df_encoded[tm.FEATURE_COLUMNS]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    scaler.fit(X_train)
    X_test_s = scaler.transform(X_test)

    return X_test_s, y_test


# ---------------------------------------------------------------------------
# Artifact existence tests
# ---------------------------------------------------------------------------
class TestArtifactFiles:

    def test_model_pkl_exists(self):
        assert os.path.exists(MODEL_PKL)

    def test_scaler_pkl_exists(self):
        assert os.path.exists(SCALER_PKL)

    def test_encoders_pkl_exists(self):
        assert os.path.exists(ENCODERS_PKL)

    def test_target_encoder_pkl_exists(self):
        assert os.path.exists(TARGET_ENC_PKL)

    def test_feature_columns_json_exists(self):
        assert os.path.exists(FEATURES_JSON)

    def test_model_comparison_json_exists(self):
        assert os.path.exists(COMPARISON_JSON)

    def test_feature_importances_json_exists(self):
        assert os.path.exists(FI_JSON)

    def test_model_pkl_non_empty(self):
        assert os.path.getsize(MODEL_PKL) > 500  # At least 500 bytes


# ---------------------------------------------------------------------------
# Model loading tests
# ---------------------------------------------------------------------------
class TestModelLoading:

    def test_model_is_loadable(self, artifacts):
        model, *_ = artifacts
        assert model is not None

    def test_model_has_predict(self, artifacts):
        model, *_ = artifacts
        assert hasattr(model, "predict")

    def test_model_has_predict_proba(self, artifacts):
        model, *_ = artifacts
        assert hasattr(model, "predict_proba")

    def test_scaler_is_loadable(self, artifacts):
        _, scaler, *_ = artifacts
        assert scaler is not None

    def test_encoders_dict_has_all_categorical_cols(self, artifacts):
        _, _, encoders, *_ = artifacts
        for col in tm.CATEGORICAL_COLS:
            assert col in encoders, f"Missing encoder for {col}"

    def test_feature_columns_json_valid(self, artifacts):
        *_, feature_columns = artifacts
        assert isinstance(feature_columns, list)
        assert len(feature_columns) == 14  # 14 feature columns expected


# ---------------------------------------------------------------------------
# Accuracy / performance tests
# ---------------------------------------------------------------------------
class TestModelAccuracy:

    def test_test_accuracy_above_threshold(self, artifacts, prepared_test_data):
        model, *_, feature_columns = artifacts
        X_test_s, y_test = prepared_test_data
        preds = model.predict(X_test_s)
        acc = accuracy_score(y_test, preds)
        assert acc >= MIN_TEST_ACCURACY, (
            f"Test accuracy {acc:.3f} is below minimum threshold {MIN_TEST_ACCURACY}"
        )

    def test_roc_auc_above_threshold(self, artifacts, prepared_test_data):
        model, *_, feature_columns = artifacts
        X_test_s, y_test = prepared_test_data
        if not hasattr(model, "predict_proba"):
            pytest.skip("Model has no predict_proba — skipping AUC test")
        proba = model.predict_proba(X_test_s)[:, 1]
        auc = roc_auc_score(y_test, proba)
        assert auc >= MIN_ROC_AUC, (
            f"ROC-AUC {auc:.3f} is below minimum threshold {MIN_ROC_AUC}"
        )

    def test_predictions_are_only_0_and_1(self, artifacts, prepared_test_data):
        model, *_ = artifacts
        X_test_s, _ = prepared_test_data
        preds = model.predict(X_test_s)
        assert set(preds).issubset({0, 1})

    def test_probabilities_between_0_and_1(self, artifacts, prepared_test_data):
        model, *_ = artifacts
        X_test_s, _ = prepared_test_data
        if not hasattr(model, "predict_proba"):
            pytest.skip("No predict_proba")
        proba = model.predict_proba(X_test_s)
        assert np.all(proba >= 0.0)
        assert np.all(proba <= 1.0)

    def test_probabilities_sum_to_1(self, artifacts, prepared_test_data):
        model, *_ = artifacts
        X_test_s, _ = prepared_test_data
        if not hasattr(model, "predict_proba"):
            pytest.skip("No predict_proba")
        proba = model.predict_proba(X_test_s)
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-5)

    def test_not_predicting_all_same_class(self, artifacts, prepared_test_data):
        model, *_ = artifacts
        X_test_s, _ = prepared_test_data
        preds = model.predict(X_test_s)
        # A decent model should predict both classes
        assert len(set(preds)) > 1, "Model is predicting only one class!"


# ---------------------------------------------------------------------------
# Feature importances JSON tests
# ---------------------------------------------------------------------------
class TestFeatureImportancesJson:

    def test_fi_json_is_valid_list(self):
        with open(FI_JSON) as f:
            fi = json.load(f)
        assert isinstance(fi, list)

    def test_fi_json_non_empty(self):
        with open(FI_JSON) as f:
            fi = json.load(f)
        assert len(fi) > 0

    def test_fi_each_item_has_feature_key(self):
        with open(FI_JSON) as f:
            fi = json.load(f)
        for item in fi:
            assert "feature" in item

    def test_fi_each_item_has_label_key(self):
        with open(FI_JSON) as f:
            fi = json.load(f)
        for item in fi:
            assert "label" in item

    def test_fi_each_item_has_importance_key(self):
        with open(FI_JSON) as f:
            fi = json.load(f)
        for item in fi:
            assert "importance" in item

    def test_fi_importances_are_positive(self):
        with open(FI_JSON) as f:
            fi = json.load(f)
        for item in fi:
            assert item["importance"] >= 0

    def test_fi_all_feature_names_valid(self):
        with open(FI_JSON) as f:
            fi = json.load(f)
        valid_features = set(tm.FEATURE_COLUMNS)
        for item in fi:
            assert item["feature"] in valid_features, f"Unknown feature: {item['feature']}"


# ---------------------------------------------------------------------------
# Model comparison JSON tests
# ---------------------------------------------------------------------------
class TestModelComparisonJson:

    def test_comparison_json_is_dict(self):
        with open(COMPARISON_JSON) as f:
            data = json.load(f)
        assert isinstance(data, dict)

    def test_comparison_json_has_best_model_key(self):
        with open(COMPARISON_JSON) as f:
            data = json.load(f)
        assert "best_model" in data

    def test_each_model_has_test_accuracy(self):
        with open(COMPARISON_JSON) as f:
            data = json.load(f)
        for key, val in data.items():
            if key == "best_model":
                continue
            assert "test_accuracy" in val, f"Missing test_accuracy for {key}"

    def test_each_model_has_cv_mean(self):
        with open(COMPARISON_JSON) as f:
            data = json.load(f)
        for key, val in data.items():
            if key == "best_model":
                continue
            assert "cv_mean" in val, f"Missing cv_mean for {key}"

    def test_accuracies_in_valid_range(self):
        with open(COMPARISON_JSON) as f:
            data = json.load(f)
        for key, val in data.items():
            if key == "best_model":
                continue
            acc = val["test_accuracy"]
            assert 0 <= acc <= 100, f"{key} accuracy {acc} out of range"

    def test_best_model_in_keys(self):
        with open(COMPARISON_JSON) as f:
            data = json.load(f)
        best = data["best_model"]
        assert best in data, f"best_model '{best}' not a key in comparison dict"
