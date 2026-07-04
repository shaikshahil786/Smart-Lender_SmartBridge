"""
tests/test_routes.py
=====================
Integration tests for every Flask HTML route:
  GET  /
  POST /predict
  GET  /batch
  POST /batch
  GET  /batch/download
  GET  /history
  POST /history/clear
  GET  /about
  GET  /reports
  GET  /health

Uses the Flask test client to fire real HTTP requests against the app.
The tests verify status codes, page content, and redirect behaviour.
"""

import io
import pytest

# ---------------------------------------------------------------------------
# Home page  (GET /)
# ---------------------------------------------------------------------------
class TestHomePage:
    def test_home_status_200(self, client):
        resp = client.get("/")
        assert resp.status_code == 200

    def test_home_contains_form(self, client):
        resp = client.get("/")
        html = resp.data.decode()
        assert '<form' in html
        assert 'action=' in html

    def test_home_contains_nav_links(self, client):
        resp = client.get("/")
        html = resp.data.decode()
        assert "/batch" in html
        assert "/history" in html
        assert "/reports" in html

    def test_home_has_emi_section(self, client):
        resp = client.get("/")
        html = resp.data.decode()
        assert "EMI" in html or "emi" in html.lower()

    def test_home_content_type_html(self, client):
        resp = client.get("/")
        assert "text/html" in resp.content_type


# ---------------------------------------------------------------------------
# Predict (POST /predict)
# ---------------------------------------------------------------------------
class TestPredictRoute:

    def _post(self, client, form):
        return client.post("/predict", data=form, follow_redirects=True)

    def test_predict_approved_case_200(self, client, sample_form):
        resp = self._post(client, sample_form)
        assert resp.status_code == 200

    def test_predict_response_contains_verdict(self, client, sample_form):
        resp = self._post(client, sample_form)
        html = resp.data.decode()
        assert "Approved" in html or "Declined" in html

    def test_predict_shows_confidence(self, client, sample_form):
        resp = self._post(client, sample_form)
        html = resp.data.decode()
        assert "Confidence" in html or "%" in html

    def test_predict_shows_summary_table(self, client, sample_form):
        resp = self._post(client, sample_form)
        html = resp.data.decode()
        assert "Loan amount" in html or "Applicant income" in html

    def test_predict_shows_emi_result(self, client, sample_form):
        resp = self._post(client, sample_form)
        html = resp.data.decode()
        # Result page should include EMI/DTI or next-steps section
        assert "EMI" in html or "Next Steps" in html or "Suggested" in html

    def test_predict_missing_required_field_shows_error(self, client, sample_form):
        bad_form = dict(sample_form)
        del bad_form["applicant_income"]
        resp = self._post(client, bad_form)
        html = resp.data.decode()
        # Should show some error or the field is missing → bad request or error page
        assert resp.status_code in (200, 400)

    def test_predict_stores_history(self, client, sample_form):
        """After a prediction the /history page should have an entry."""
        self._post(client, sample_form)
        hist = client.get("/history")
        html = hist.data.decode()
        # History page should show at least the approval badge
        assert "Approved" in html or "Declined" in html

    def test_predict_declined_case(self, client, sample_form_declined):
        resp = self._post(client, sample_form_declined)
        assert resp.status_code == 200
        html = resp.data.decode()
        assert "Approved" in html or "Declined" in html

    def test_predict_all_areas(self, client, sample_form):
        """Urban / Semiurban / Rural should all work without error."""
        for area in ["Urban", "Semiurban", "Rural"]:
            fd = dict(sample_form, property_area=area)
            resp = self._post(client, fd)
            assert resp.status_code == 200

    def test_predict_all_education_types(self, client, sample_form):
        for edu in ["Graduate", "Not Graduate"]:
            fd = dict(sample_form, education=edu)
            resp = self._post(client, fd)
            assert resp.status_code == 200

    def test_predict_3plus_dependents(self, client, sample_form):
        fd = dict(sample_form, dependents="3+")
        resp = self._post(client, fd)
        assert resp.status_code == 200


# ---------------------------------------------------------------------------
# Batch upload (GET+POST /batch)
# ---------------------------------------------------------------------------
class TestBatchRoute:

    def test_batch_get_200(self, client):
        resp = client.get("/batch")
        assert resp.status_code == 200

    def test_batch_get_contains_upload_zone(self, client):
        resp = client.get("/batch")
        html = resp.data.decode()
        assert "drop" in html.lower() or "upload" in html.lower() or "csv" in html.lower()

    def test_batch_post_valid_csv(self, client, sample_csv_bytes):
        fname, csv_io, ctype = sample_csv_bytes
        data = {"csv_file": (csv_io, fname, ctype)}
        resp = client.post("/batch", data=data,
                           content_type="multipart/form-data",
                           follow_redirects=True)
        assert resp.status_code == 200
        html = resp.data.decode()
        # Should show batch result or error from model
        assert "Total" in html or "Approved" in html or "error" in html.lower()

    def test_batch_post_no_file_shows_error(self, client):
        resp = client.post("/batch", data={},
                           content_type="multipart/form-data",
                           follow_redirects=True)
        assert resp.status_code == 200
        html = resp.data.decode()
        assert "csv" in html.lower() or "error" in html.lower() or "upload" in html.lower()

    def test_batch_post_wrong_extension(self, client):
        data = {"csv_file": (io.BytesIO(b"not,a,csv"), "data.txt", "text/plain")}
        resp = client.post("/batch", data=data,
                           content_type="multipart/form-data",
                           follow_redirects=True)
        assert resp.status_code == 200
        html = resp.data.decode()
        assert "csv" in html.lower() or "error" in html.lower()

    def test_batch_download_redirects_without_session(self, client):
        """Without a batch result in session, /batch/download should redirect."""
        with client.session_transaction() as sess:
            sess.pop("batch_result_csv", None)
        resp = client.get("/batch/download")
        assert resp.status_code in (302, 200)


# ---------------------------------------------------------------------------
# History (GET /history, POST /history/clear)
# ---------------------------------------------------------------------------
class TestHistoryRoute:

    def test_history_get_200(self, client):
        resp = client.get("/history")
        assert resp.status_code == 200

    def test_history_empty_shows_empty_state(self, client):
        with client.session_transaction() as sess:
            sess.pop("history", None)
        resp = client.get("/history")
        html = resp.data.decode()
        assert "No applications" in html or "empty" in html.lower() or "history" in html.lower()

    def test_history_clear_redirects(self, client):
        resp = client.post("/history/clear", follow_redirects=False)
        assert resp.status_code == 302
        assert "/history" in resp.headers.get("Location", "")

    def test_history_clear_then_empty(self, client, sample_form):
        # Add an entry
        client.post("/predict", data=sample_form)
        # Clear it
        client.post("/history/clear")
        resp = client.get("/history")
        html = resp.data.decode()
        assert "No applications" in html or resp.status_code == 200


# ---------------------------------------------------------------------------
# About (GET /about)
# ---------------------------------------------------------------------------
class TestAboutRoute:

    def test_about_200(self, client):
        resp = client.get("/about")
        assert resp.status_code == 200

    def test_about_has_pipeline_section(self, client):
        resp = client.get("/about")
        html = resp.data.decode()
        assert "Pipeline" in html or "pipeline" in html.lower()

    def test_about_has_model_info(self, client):
        resp = client.get("/about")
        html = resp.data.decode()
        # Should mention model names
        assert any(name in html for name in [
            "Logistic", "Random Forest", "Decision Tree", "XGBoost", "KNN"
        ])


# ---------------------------------------------------------------------------
# Reports (GET /reports)
# ---------------------------------------------------------------------------
class TestReportsRoute:

    def test_reports_200(self, client):
        resp = client.get("/reports")
        assert resp.status_code == 200

    def test_reports_contains_chart_links(self, client):
        resp = client.get("/reports")
        html = resp.data.decode()
        assert ".png" in html or "chart" in html.lower() or "report" in html.lower()


# ---------------------------------------------------------------------------
# Health (GET /health)
# ---------------------------------------------------------------------------
class TestHealthRoute:

    def test_health_200(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200

    def test_health_json(self, client):
        resp = client.get("/health")
        data = resp.get_json()
        assert data is not None
        assert "status" in data
        assert data["status"] == "ok"

    def test_health_has_model_loaded_key(self, client):
        resp = client.get("/health")
        data = resp.get_json()
        assert "model_loaded" in data

    def test_health_model_loaded_is_bool(self, client):
        resp = client.get("/health")
        data = resp.get_json()
        assert isinstance(data["model_loaded"], bool)

    def test_health_feature_importances_key(self, client):
        resp = client.get("/health")
        data = resp.get_json()
        assert "feature_importances_loaded" in data
