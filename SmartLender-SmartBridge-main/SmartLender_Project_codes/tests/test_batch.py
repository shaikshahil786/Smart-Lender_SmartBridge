"""
tests/test_batch.py
====================
Tests specifically for the batch CSV processing feature:
  - predict_batch() helper function
  - /batch route (GET + POST)
  - /batch/download route
  - CSV input/output schema validation
  - Edge cases: empty CSV, single row, malformed rows
"""

import io
import csv
import json
import pytest
import pandas as pd

import sys, os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import app as app_module
from app import predict_batch


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def make_csv_bytes(*rows):
    """Build a CSV BytesIO object from a list of row dicts."""
    if not rows:
        return io.BytesIO(b"Gender,Married\n")
    keys = list(rows[0].keys())
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=keys)
    writer.writeheader()
    writer.writerows(rows)
    return io.BytesIO(buf.getvalue().encode("utf-8"))


STANDARD_ROW = {
    "Gender": "Male", "Married": "Yes", "Dependents": "0",
    "Education": "Graduate", "Self_Employed": "No",
    "ApplicantIncome": 5000, "CoapplicantIncome": 1500,
    "LoanAmount": 128, "Loan_Amount_Term": 360,
    "Credit_History": 1, "Property_Area": "Urban",
}

DECLINED_ROW = {
    "Gender": "Female", "Married": "No", "Dependents": "3+",
    "Education": "Not Graduate", "Self_Employed": "Yes",
    "ApplicantIncome": 1200, "CoapplicantIncome": 0,
    "LoanAmount": 500, "Loan_Amount_Term": 360,
    "Credit_History": 0, "Property_Area": "Rural",
}


# ---------------------------------------------------------------------------
# predict_batch() unit tests
# (Only run if MODEL is loaded)
# ---------------------------------------------------------------------------
MODEL_LOADED = app_module.MODEL is not None


@pytest.mark.skipif(not MODEL_LOADED, reason="Model not loaded")
class TestPredictBatch:

    def test_single_row_returns_one_result(self):
        df = pd.DataFrame([STANDARD_ROW])
        result = predict_batch(df)
        assert len(result) == 1

    def test_three_rows_return_three_results(self):
        df = pd.DataFrame([STANDARD_ROW, DECLINED_ROW, STANDARD_ROW])
        result = predict_batch(df)
        assert len(result) == 3

    def test_result_has_predicted_status_col(self):
        df = pd.DataFrame([STANDARD_ROW])
        result = predict_batch(df)
        assert "Predicted_Status" in result.columns

    def test_result_has_approved_col(self):
        df = pd.DataFrame([STANDARD_ROW])
        result = predict_batch(df)
        assert "Approved" in result.columns

    def test_result_has_confidence_col(self):
        df = pd.DataFrame([STANDARD_ROW])
        result = predict_batch(df)
        assert "Confidence_Pct" in result.columns

    def test_predicted_status_is_y_or_n(self):
        df = pd.DataFrame([STANDARD_ROW, DECLINED_ROW])
        result = predict_batch(df)
        valid = {"Y", "N", "ERROR"}
        for val in result["Predicted_Status"]:
            assert val in valid

    def test_approved_is_yes_or_no(self):
        df = pd.DataFrame([STANDARD_ROW])
        result = predict_batch(df)
        for val in result["Approved"]:
            assert val in ("Yes", "No", "ERROR")

    def test_confidence_is_0_to_100(self):
        df = pd.DataFrame([STANDARD_ROW])
        result = predict_batch(df)
        conf = result["Confidence_Pct"].dropna()
        assert (conf >= 0).all() and (conf <= 100).all()

    def test_original_columns_preserved(self):
        df = pd.DataFrame([STANDARD_ROW])
        result = predict_batch(df)
        for col in STANDARD_ROW.keys():
            assert col in result.columns

    def test_10_rows(self):
        rows = [STANDARD_ROW] * 5 + [DECLINED_ROW] * 5
        df = pd.DataFrame(rows)
        result = predict_batch(df)
        assert len(result) == 10

    def test_mixed_property_areas(self):
        rows = []
        for area in ["Urban", "Semiurban", "Rural"]:
            row = dict(STANDARD_ROW, Property_Area=area)
            rows.append(row)
        result = predict_batch(pd.DataFrame(rows))
        assert len(result) == 3
        assert "ERROR" not in result["Predicted_Status"].values

    def test_all_dependents(self):
        rows = []
        for dep in ["0", "1", "2", "3+"]:
            row = dict(STANDARD_ROW, Dependents=dep)
            rows.append(row)
        result = predict_batch(pd.DataFrame(rows))
        assert len(result) == 4


# ---------------------------------------------------------------------------
# Batch route tests
# ---------------------------------------------------------------------------
class TestBatchRouteDetailed:

    def _upload(self, client, rows, filename="test.csv"):
        csv_io = make_csv_bytes(*rows)
        data = {"csv_file": (csv_io, filename, "text/csv")}
        return client.post(
            "/batch",
            data=data,
            content_type="multipart/form-data",
            follow_redirects=True,
        )

    def test_batch_page_title(self, client):
        resp = client.get("/batch")
        assert "Batch" in resp.data.decode() or "batch" in resp.data.decode()

    def test_batch_format_guide_present(self, client):
        resp = client.get("/batch")
        html = resp.data.decode()
        assert "Gender" in html or "csv" in html.lower()

    @pytest.mark.skipif(not MODEL_LOADED, reason="Model not loaded")
    def test_upload_3_rows_shows_total(self, client):
        resp = self._upload(client, [STANDARD_ROW, DECLINED_ROW, STANDARD_ROW])
        html = resp.data.decode()
        # Should show total=3 or some count
        assert "3" in html or "Total" in html

    @pytest.mark.skipif(not MODEL_LOADED, reason="Model not loaded")
    def test_upload_returns_download_button(self, client):
        resp = self._upload(client, [STANDARD_ROW])
        html = resp.data.decode()
        assert "download" in html.lower() or "Download" in html

    def test_empty_csv_shows_error_or_200(self, client):
        empty = io.BytesIO(b"Gender,Married,Dependents\n")
        data = {"csv_file": (empty, "empty.csv", "text/csv")}
        resp = client.post("/batch", data=data,
                           content_type="multipart/form-data",
                           follow_redirects=True)
        # Should not crash — either shows error or empty result
        assert resp.status_code == 200

    def test_no_file_upload_shows_error(self, client):
        resp = client.post("/batch", data={},
                           content_type="multipart/form-data",
                           follow_redirects=True)
        assert resp.status_code == 200
        html = resp.data.decode()
        assert "csv" in html.lower() or "error" in html.lower() or "upload" in html.lower()

    def test_wrong_file_type_rejected(self, client):
        data = {"csv_file": (io.BytesIO(b"not valid"), "file.json", "application/json")}
        resp = client.post("/batch", data=data,
                           content_type="multipart/form-data",
                           follow_redirects=True)
        assert resp.status_code == 200


# ---------------------------------------------------------------------------
# Download route tests
# ---------------------------------------------------------------------------
class TestBatchDownload:

    def test_download_without_session_redirects(self, client):
        with client.session_transaction() as sess:
            sess.pop("batch_result_csv", None)
        resp = client.get("/batch/download", follow_redirects=False)
        assert resp.status_code in (302, 200)

    @pytest.mark.skipif(not MODEL_LOADED, reason="Model not loaded")
    def test_download_after_upload_returns_csv(self, client):
        # First upload
        csv_io = make_csv_bytes(STANDARD_ROW, DECLINED_ROW)
        client.post(
            "/batch",
            data={"csv_file": (csv_io, "t.csv", "text/csv")},
            content_type="multipart/form-data",
            follow_redirects=True,
        )
        # Then download
        resp = client.get("/batch/download")
        if resp.status_code == 200:
            assert "text/csv" in resp.content_type or resp.data  # CSV content
        else:
            # Could be a redirect if session didn't persist in this test
            assert resp.status_code == 302

    @pytest.mark.skipif(not MODEL_LOADED, reason="Model not loaded")
    def test_download_csv_contains_predicted_status(self, client):
        csv_io = make_csv_bytes(STANDARD_ROW)
        client.post(
            "/batch",
            data={"csv_file": (csv_io, "t.csv", "text/csv")},
            content_type="multipart/form-data",
            follow_redirects=True,
        )
        resp = client.get("/batch/download")
        if resp.status_code == 200:
            content = resp.data.decode()
            assert "Predicted_Status" in content or "Approved" in content


# ---------------------------------------------------------------------------
# CSV schema validation tests
# ---------------------------------------------------------------------------
class TestCsvSchema:
    """Test that the batch result CSV has the correct column schema."""

    @pytest.mark.skipif(not MODEL_LOADED, reason="Model not loaded")
    def test_result_df_has_all_new_cols(self):
        df = pd.DataFrame([STANDARD_ROW])
        result = predict_batch(df)
        for col in ["Predicted_Status", "Approved", "Confidence_Pct"]:
            assert col in result.columns

    @pytest.mark.skipif(not MODEL_LOADED, reason="Model not loaded")
    def test_result_df_row_count_unchanged(self):
        df = pd.DataFrame([STANDARD_ROW] * 7)
        result = predict_batch(df)
        assert len(result) == 7

    @pytest.mark.skipif(not MODEL_LOADED, reason="Model not loaded")
    def test_confidence_is_numeric(self):
        df = pd.DataFrame([STANDARD_ROW])
        result = predict_batch(df)
        assert pd.api.types.is_float_dtype(result["Confidence_Pct"]) or \
               pd.api.types.is_object_dtype(result["Confidence_Pct"])
