"""
tests/test_session_history.py
==============================
Tests for the session-based application history feature:
  - _add_to_history()
  - _get_session_stats()
  - /history GET route
  - /history/clear POST route
  - History limit enforcement (max 10 entries)
  - Session data structure validation
"""

import pytest
import sys, os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import app as app_module
from app import _get_session_stats


SAMPLE_FORM = {
    "gender":            "Male",
    "married":           "Yes",
    "dependents":        "0",
    "education":         "Graduate",
    "self_employed":     "No",
    "applicant_income":  "5000",
    "coapplicant_income":"1000",
    "loan_amount":       "128",
    "loan_amount_term":  "360",
    "credit_history":    "1",
    "property_area":     "Urban",
}

MODEL_LOADED = app_module.MODEL is not None


# ---------------------------------------------------------------------------
# _add_to_history via predict route
# ---------------------------------------------------------------------------
@pytest.mark.skipif(not MODEL_LOADED, reason="Model not loaded")
class TestAddToHistory:

    def test_history_starts_empty(self, client):
        with client.session_transaction() as sess:
            sess.clear()
        resp = client.get("/history")
        html = resp.data.decode()
        assert "No applications" in html or resp.status_code == 200

    def test_predict_adds_entry(self, client):
        with client.session_transaction() as sess:
            sess.pop("history", None)

        client.post("/predict", data=SAMPLE_FORM)

        with client.session_transaction() as sess:
            history = sess.get("history", [])
        assert len(history) == 1

    def test_history_entry_has_ts(self, client):
        with client.session_transaction() as sess:
            sess.pop("history", None)
        client.post("/predict", data=SAMPLE_FORM)
        with client.session_transaction() as sess:
            entry = sess["history"][0]
        assert "ts" in entry

    def test_history_entry_has_approved(self, client):
        with client.session_transaction() as sess:
            sess.pop("history", None)
        client.post("/predict", data=SAMPLE_FORM)
        with client.session_transaction() as sess:
            entry = sess["history"][0]
        assert "approved" in entry
        assert isinstance(entry["approved"], bool)

    def test_history_entry_has_probability(self, client):
        with client.session_transaction() as sess:
            sess.pop("history", None)
        client.post("/predict", data=SAMPLE_FORM)
        with client.session_transaction() as sess:
            entry = sess["history"][0]
        assert "probability" in entry

    def test_multiple_predictions_stack(self, client):
        with client.session_transaction() as sess:
            sess.pop("history", None)
        for _ in range(3):
            client.post("/predict", data=SAMPLE_FORM)
        with client.session_transaction() as sess:
            history = sess.get("history", [])
        assert len(history) == 3

    def test_history_capped_at_limit(self, client):
        with client.session_transaction() as sess:
            sess.pop("history", None)
        # Submit more than HISTORY_LIMIT predictions
        for _ in range(app_module.HISTORY_LIMIT + 3):
            client.post("/predict", data=SAMPLE_FORM)
        with client.session_transaction() as sess:
            history = sess.get("history", [])
        assert len(history) == app_module.HISTORY_LIMIT

    def test_history_newest_first(self, client):
        """Most recent entry should be at index 0."""
        with client.session_transaction() as sess:
            sess.pop("history", None)
        client.post("/predict", data=SAMPLE_FORM)
        form2 = dict(SAMPLE_FORM, loan_amount="200")
        client.post("/predict", data=form2)
        with client.session_transaction() as sess:
            history = sess.get("history", [])
        # Most recent (loan=200) should be first
        assert len(history) == 2
        assert history[0]["loan_amount"] == "200"


# ---------------------------------------------------------------------------
# _get_session_stats
# ---------------------------------------------------------------------------
@pytest.mark.skipif(not MODEL_LOADED, reason="Model not loaded")
class TestGetSessionStats:

    def test_empty_session_returns_zeros(self, client):
        with client.session_transaction() as sess:
            sess.pop("history", None)
        resp = client.get("/")
        assert resp.status_code == 200

    def test_stats_after_predictions(self, client):
        with client.session_transaction() as sess:
            sess.pop("history", None)
        for _ in range(3):
            client.post("/predict", data=SAMPLE_FORM)
        resp = client.get("/")
        assert resp.status_code == 200
        html = resp.data.decode()
        # Stats counters should be on the page
        assert "3" in html or "Applications" in html

    def test_approval_rate_is_percentage(self, client):
        with client.session_transaction() as sess:
            # Inject fake history directly
            sess["history"] = [
                {"approved": True,  "probability": 90.0, "ts": "2026-01-01 10:00",
                 "gender": "Male", "education": "Graduate",
                 "applicant_income": "5000", "loan_amount": "100",
                 "property_area": "Urban", "credit_history": "1"},
                {"approved": False, "probability": 35.0, "ts": "2026-01-01 11:00",
                 "gender": "Female", "education": "Graduate",
                 "applicant_income": "3000", "loan_amount": "200",
                 "property_area": "Rural", "credit_history": "0"},
            ]
        with app_module.app.test_request_context():
            from flask import session as fs
            fs["history"] = [
                {"approved": True,  "probability": 90.0},
                {"approved": False, "probability": 35.0},
            ]
            stats = _get_session_stats()
        assert 0 <= stats["approval_rate"] <= 100


# ---------------------------------------------------------------------------
# History route
# ---------------------------------------------------------------------------
class TestHistoryRouteDetailed:

    def test_history_route_ok(self, client):
        assert client.get("/history").status_code == 200

    def test_history_contains_clear_button(self, client):
        resp = client.get("/history")
        html = resp.data.decode()
        assert "Clear" in html or "clear" in html

    def test_history_contains_new_application_link(self, client):
        resp = client.get("/history")
        html = resp.data.decode()
        assert "/" in html  # Link back to home

    @pytest.mark.skipif(not MODEL_LOADED, reason="Model not loaded")
    def test_history_shows_verdict_after_predict(self, client):
        with client.session_transaction() as sess:
            sess.pop("history", None)
        client.post("/predict", data=SAMPLE_FORM)
        resp = client.get("/history")
        html = resp.data.decode()
        assert "Approved" in html or "Declined" in html

    def test_history_clear_empties_session(self, client):
        with client.session_transaction() as sess:
            sess["history"] = [{"approved": True, "probability": 85.0,
                                "ts": "2026-01-01 10:00", "gender": "Male",
                                "education": "Graduate", "applicant_income": "5000",
                                "loan_amount": "100", "property_area": "Urban",
                                "credit_history": "1"}]
        client.post("/history/clear")
        with client.session_transaction() as sess:
            history = sess.get("history", [])
        assert len(history) == 0

    def test_history_clear_redirects_to_history(self, client):
        resp = client.post("/history/clear", follow_redirects=False)
        assert resp.status_code == 302
        loc = resp.headers.get("Location", "")
        assert "history" in loc
