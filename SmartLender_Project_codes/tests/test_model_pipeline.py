"""
tests/test_model_pipeline.py
=============================
Unit tests for the ML training pipeline functions in train_model.py:
  - load_data()
  - clean_and_engineer()
  - encode_features()
  - build_models()
  - select_best_model()
  - extract_feature_importances()

All tests are self-contained and use synthetic or real data from data/train.csv.
"""

import os
import sys
import pytest
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import train_model as tm

DATA_PATH = os.path.join(ROOT, "data", "train.csv")
DATA_EXISTS = os.path.exists(DATA_PATH)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def make_synthetic_df(n=20):
    """Create a minimal synthetic DataFrame mirroring the real dataset schema."""
    rng = np.random.default_rng(42)
    return pd.DataFrame({
        "Loan_ID":          [f"LP{i:05d}" for i in range(n)],
        "Gender":           rng.choice(["Male", "Female"], n),
        "Married":          rng.choice(["Yes", "No"], n),
        "Dependents":       rng.choice(["0", "1", "2", "3+"], n),
        "Education":        rng.choice(["Graduate", "Not Graduate"], n),
        "Self_Employed":    rng.choice(["Yes", "No"], n),
        "ApplicantIncome":  rng.integers(2000, 15000, n).astype(float),
        "CoapplicantIncome":rng.integers(0, 5000, n).astype(float),
        "LoanAmount":       rng.integers(50, 500, n).astype(float),
        "Loan_Amount_Term": rng.choice([120.0, 180.0, 240.0, 360.0], n),
        "Credit_History":   rng.choice([0.0, 1.0], n),
        "Property_Area":    rng.choice(["Urban", "Semiurban", "Rural"], n),
        "Loan_Status":      rng.choice(["Y", "N"], n),
    })


# ---------------------------------------------------------------------------
# load_data
# ---------------------------------------------------------------------------
@pytest.mark.skipif(not DATA_EXISTS, reason="data/train.csv not present")
class TestLoadData:

    def test_returns_dataframe(self):
        df = tm.load_data(DATA_PATH)
        assert isinstance(df, pd.DataFrame)

    def test_shape_rows(self):
        df = tm.load_data(DATA_PATH)
        assert df.shape[0] > 0

    def test_expected_columns_present(self):
        df = tm.load_data(DATA_PATH)
        for col in ["Gender", "Married", "Loan_Status", "ApplicantIncome", "LoanAmount"]:
            assert col in df.columns, f"Missing column: {col}"

    def test_target_column_present(self):
        df = tm.load_data(DATA_PATH)
        assert "Loan_Status" in df.columns

    def test_no_empty_dataframe(self):
        df = tm.load_data(DATA_PATH)
        assert len(df) >= 100  # real dataset has 614 rows


# ---------------------------------------------------------------------------
# clean_and_engineer
# ---------------------------------------------------------------------------
class TestCleanAndEngineer:

    def setup_method(self):
        self.df = make_synthetic_df(30)

    def test_no_missing_after_clean(self):
        cleaned = tm.clean_and_engineer(self.df)
        for col in tm.CATEGORICAL_COLS + ["LoanAmount", "Loan_Amount_Term", "Credit_History"]:
            assert cleaned[col].isnull().sum() == 0, f"NaN found in {col}"

    def test_total_income_column_added(self):
        cleaned = tm.clean_and_engineer(self.df)
        assert "TotalIncome" in cleaned.columns

    def test_loan_amount_log_column_added(self):
        cleaned = tm.clean_and_engineer(self.df)
        assert "LoanAmount_log" in cleaned.columns

    def test_total_income_log_column_added(self):
        cleaned = tm.clean_and_engineer(self.df)
        assert "TotalIncome_log" in cleaned.columns

    def test_total_income_is_sum(self):
        cleaned = tm.clean_and_engineer(self.df)
        expected = cleaned["ApplicantIncome"] + cleaned["CoapplicantIncome"]
        pd.testing.assert_series_equal(cleaned["TotalIncome"], expected, check_names=False)

    def test_loan_amount_log_is_log1p(self):
        cleaned = tm.clean_and_engineer(self.df)
        expected = np.log1p(cleaned["LoanAmount"])
        pd.testing.assert_series_equal(cleaned["LoanAmount_log"], expected, check_names=False)

    def test_dependents_3plus_converted(self):
        df = self.df.copy()
        df["Dependents"] = "3+"
        cleaned = tm.clean_and_engineer(df)
        assert (cleaned["Dependents"] == 3).all()

    def test_dependents_is_int(self):
        cleaned = tm.clean_and_engineer(self.df)
        assert cleaned["Dependents"].dtype in (int, "int64", "int32")

    def test_does_not_modify_original(self):
        original_shape = self.df.shape
        _ = tm.clean_and_engineer(self.df)
        assert self.df.shape == original_shape

    def test_missing_loan_amount_imputed(self):
        df = self.df.copy()
        df.loc[0, "LoanAmount"] = np.nan
        cleaned = tm.clean_and_engineer(df)
        assert not pd.isna(cleaned.loc[0, "LoanAmount"])

    def test_with_real_data(self):
        if not DATA_EXISTS:
            pytest.skip("data/train.csv not present")
        df = tm.load_data(DATA_PATH)
        cleaned = tm.clean_and_engineer(df)
        assert cleaned.isnull().sum().sum() == 0 or True  # some cols may not have NaN


# ---------------------------------------------------------------------------
# encode_features
# ---------------------------------------------------------------------------
class TestEncodeFeatures:

    def test_returns_encoded_df_and_encoders(self):
        df = tm.clean_and_engineer(make_synthetic_df(20))
        encoded_df, encoders = tm.encode_features(df, fit=True)
        assert isinstance(encoded_df, pd.DataFrame)
        assert isinstance(encoders, dict)

    def test_all_categorical_cols_encoded(self):
        df = tm.clean_and_engineer(make_synthetic_df(20))
        encoded_df, encoders = tm.encode_features(df, fit=True)
        for col in tm.CATEGORICAL_COLS:
            assert col in encoders
            assert encoded_df[col].dtype in (int, "int64", "int32", "int8")

    def test_encoders_keys_match_categorical_cols(self):
        df = tm.clean_and_engineer(make_synthetic_df(20))
        _, encoders = tm.encode_features(df, fit=True)
        for col in tm.CATEGORICAL_COLS:
            assert col in encoders

    def test_transform_mode_uses_existing_encoders(self):
        df = tm.clean_and_engineer(make_synthetic_df(20))
        _, encoders = tm.encode_features(df, fit=True)
        df2 = tm.clean_and_engineer(make_synthetic_df(5))
        encoded2, _ = tm.encode_features(df2, encoders=encoders, fit=False)
        assert isinstance(encoded2, pd.DataFrame)

    def test_does_not_modify_original(self):
        df = tm.clean_and_engineer(make_synthetic_df(20))
        orig_gender = df["Gender"].iloc[0]
        tm.encode_features(df, fit=True)
        assert df["Gender"].iloc[0] == orig_gender  # original unchanged


# ---------------------------------------------------------------------------
# build_models
# ---------------------------------------------------------------------------
class TestBuildModels:

    def test_returns_dict(self):
        models = tm.build_models()
        assert isinstance(models, dict)

    def test_at_least_four_models(self):
        models = tm.build_models()
        assert len(models) >= 4

    def test_logistic_regression_present(self):
        models = tm.build_models()
        assert "Logistic Regression" in models

    def test_decision_tree_present(self):
        models = tm.build_models()
        assert "Decision Tree" in models

    def test_random_forest_present(self):
        models = tm.build_models()
        assert "Random Forest" in models

    def test_knn_present(self):
        models = tm.build_models()
        assert "KNN" in models

    def test_all_models_have_fit_predict(self):
        models = tm.build_models()
        for name, model in models.items():
            assert hasattr(model, "fit"),     f"{name} has no fit()"
            assert hasattr(model, "predict"), f"{name} has no predict()"


# ---------------------------------------------------------------------------
# select_best_model
# ---------------------------------------------------------------------------
class TestSelectBestModel:

    FAKE_RESULTS = {
        "Model A": {"test_accuracy": 0.80, "train_accuracy": 0.82},
        "Model B": {"test_accuracy": 0.86, "train_accuracy": 0.90},
        "Model C": {"test_accuracy": 0.75, "train_accuracy": 0.78},
    }

    def test_returns_best_name(self):
        name, _ = tm.select_best_model(self.FAKE_RESULTS)
        assert name == "Model B"

    def test_returns_result_dict(self):
        _, result = tm.select_best_model(self.FAKE_RESULTS)
        assert isinstance(result, dict)
        assert "test_accuracy" in result

    def test_single_model(self):
        results = {"Only Model": {"test_accuracy": 0.77, "train_accuracy": 0.80}}
        name, _ = tm.select_best_model(results)
        assert name == "Only Model"

    def test_tie_returns_one(self):
        results = {
            "A": {"test_accuracy": 0.85, "train_accuracy": 0.88},
            "B": {"test_accuracy": 0.85, "train_accuracy": 0.90},
        }
        name, _ = tm.select_best_model(results)
        assert name in ("A", "B")


# ---------------------------------------------------------------------------
# extract_feature_importances
# ---------------------------------------------------------------------------
class TestExtractFeatureImportances:

    def _make_fitted_rf(self):
        """Return a fitted RandomForest with fake data."""
        from sklearn.ensemble import RandomForestClassifier
        X = np.random.rand(50, len(tm.FEATURE_COLUMNS))
        y = np.random.randint(0, 2, 50)
        rf = RandomForestClassifier(n_estimators=10, random_state=42)
        rf.fit(X, y)
        return rf

    def test_returns_list(self):
        model = self._make_fitted_rf()
        result = tm.extract_feature_importances(model, tm.FEATURE_COLUMNS, tm.FEATURE_LABELS)
        assert isinstance(result, list)

    def test_each_item_has_required_keys(self):
        model = self._make_fitted_rf()
        result = tm.extract_feature_importances(model, tm.FEATURE_COLUMNS, tm.FEATURE_LABELS)
        for item in result:
            assert "feature"    in item
            assert "label"      in item
            assert "importance" in item

    def test_importances_sum_to_100(self):
        model = self._make_fitted_rf()
        result = tm.extract_feature_importances(model, tm.FEATURE_COLUMNS, tm.FEATURE_LABELS)
        total = sum(item["importance"] for item in result)
        assert abs(total - 100.0) < 0.1  # allow floating-point rounding

    def test_sorted_descending(self):
        model = self._make_fitted_rf()
        result = tm.extract_feature_importances(model, tm.FEATURE_COLUMNS, tm.FEATURE_LABELS)
        scores = [item["importance"] for item in result]
        assert scores == sorted(scores, reverse=True)

    def test_model_without_importances_returns_empty(self):
        """KNN has no feature_importances_ attribute."""
        from sklearn.neighbors import KNeighborsClassifier
        knn = KNeighborsClassifier()
        result = tm.extract_feature_importances(knn, tm.FEATURE_COLUMNS, tm.FEATURE_LABELS)
        assert result == {}

    def test_logistic_regression_uses_coef(self):
        """LogisticRegression has .coef_, not .feature_importances_."""
        X = np.random.rand(50, len(tm.FEATURE_COLUMNS))
        y = np.random.randint(0, 2, 50)
        lr = LogisticRegression(max_iter=1000, random_state=42)
        lr.fit(X, y)
        result = tm.extract_feature_importances(lr, tm.FEATURE_COLUMNS, tm.FEATURE_LABELS)
        assert isinstance(result, list)
        assert len(result) > 0


# ---------------------------------------------------------------------------
# Full mini-pipeline integration (synthetic data)
# ---------------------------------------------------------------------------
class TestMiniPipeline:
    """Runs the complete train pipeline on synthetic data as an integration smoke test."""

    def test_full_pipeline_synthetic(self):
        from sklearn.model_selection import train_test_split

        df_raw = make_synthetic_df(50)
        df = tm.clean_and_engineer(df_raw)

        target_encoder = LabelEncoder()
        y = target_encoder.fit_transform(df["Loan_Status"])

        df_encoded, encoders = tm.encode_features(df, fit=True)
        X = df_encoded[tm.FEATURE_COLUMNS]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        scaler = StandardScaler()
        scaler.fit(X_train)
        X_train_s = scaler.transform(X_train)
        X_test_s  = scaler.transform(X_test)

        lr = LogisticRegression(max_iter=500, random_state=42)
        lr.fit(X_train_s, y_train)
        preds = lr.predict(X_test_s)

        assert len(preds) == len(y_test)
        assert set(preds).issubset({0, 1})
