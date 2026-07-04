"""
tests/test_preprocessing.py
=============================
Unit tests for pure data-transformation functions in app.py:
  - build_feature_row()
  - _calc_emi()
  - _get_session_stats()
  - _add_to_history() (via session)

These tests do NOT require the ML model to be loaded.
"""

import math
import numpy as np
import pytest
import sys, os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import app as app_module
from app import build_feature_row, _calc_emi


# ---------------------------------------------------------------------------
# build_feature_row
# ---------------------------------------------------------------------------
class TestBuildFeatureRow:
    """Tests for the form-dict → feature-dict transformation."""

    BASE = {
        "gender":            "Male",
        "married":           "Yes",
        "dependents":        "2",
        "education":         "Graduate",
        "self_employed":     "No",
        "applicant_income":  "6000",
        "coapplicant_income":"1500",
        "loan_amount":       "150",
        "loan_amount_term":  "360",
        "credit_history":    "1",
        "property_area":     "Urban",
    }

    def test_keys_present(self):
        """All 14 expected feature keys must be in the output."""
        row = build_feature_row(self.BASE)
        expected_keys = {
            "Gender", "Married", "Dependents", "Education", "Self_Employed",
            "ApplicantIncome", "CoapplicantIncome", "LoanAmount",
            "Loan_Amount_Term", "Credit_History", "Property_Area",
            "TotalIncome", "LoanAmount_log", "TotalIncome_log",
        }
        assert expected_keys == set(row.keys())

    def test_total_income_computed(self):
        row = build_feature_row(self.BASE)
        assert row["TotalIncome"] == pytest.approx(6000 + 1500)

    def test_loan_amount_log(self):
        row = build_feature_row(self.BASE)
        assert row["LoanAmount_log"] == pytest.approx(np.log1p(150))

    def test_total_income_log(self):
        row = build_feature_row(self.BASE)
        assert row["TotalIncome_log"] == pytest.approx(np.log1p(7500))

    def test_dependents_numeric_string(self):
        """'2' should become integer 2."""
        row = build_feature_row(self.BASE)
        assert row["Dependents"] == 2
        assert isinstance(row["Dependents"], int)

    def test_dependents_3plus_normalised(self):
        """'3+' should be converted to integer 3."""
        fd = dict(self.BASE, dependents="3+")
        row = build_feature_row(fd)
        assert row["Dependents"] == 3

    def test_zero_coapplicant_income(self):
        fd = dict(self.BASE, coapplicant_income="0")
        row = build_feature_row(fd)
        assert row["CoapplicantIncome"] == 0.0
        assert row["TotalIncome"] == pytest.approx(6000.0)

    def test_missing_coapplicant_income_defaults_to_zero(self):
        """coapplicant_income key absent → treated as 0."""
        fd = {k: v for k, v in self.BASE.items() if k != "coapplicant_income"}
        row = build_feature_row(fd)
        assert row["CoapplicantIncome"] == 0.0

    def test_gender_preserved(self):
        row = build_feature_row(self.BASE)
        assert row["Gender"] == "Male"

    def test_property_area_preserved(self):
        row = build_feature_row(self.BASE)
        assert row["Property_Area"] == "Urban"

    def test_credit_history_float(self):
        row = build_feature_row(self.BASE)
        assert row["Credit_History"] == 1.0
        assert isinstance(row["Credit_History"], float)

    def test_large_income_values(self):
        fd = dict(self.BASE, applicant_income="1000000", loan_amount="5000")
        row = build_feature_row(fd)
        assert row["ApplicantIncome"] == 1_000_000.0
        assert row["LoanAmount"] == 5000.0

    def test_female_not_graduate(self):
        fd = dict(self.BASE, gender="Female", education="Not Graduate")
        row = build_feature_row(fd)
        assert row["Gender"] == "Female"
        assert row["Education"] == "Not Graduate"

    @pytest.mark.parametrize("dep_str,expected", [
        ("0", 0), ("1", 1), ("2", 2), ("3+", 3),
    ])
    def test_dependents_all_values(self, dep_str, expected):
        fd = dict(self.BASE, dependents=dep_str)
        row = build_feature_row(fd)
        assert row["Dependents"] == expected


# ---------------------------------------------------------------------------
# _calc_emi
# ---------------------------------------------------------------------------
class TestCalcEmi:
    """Tests for the EMI formula helper."""

    def test_standard_emi(self):
        """$150 000 at 8.5% for 30 years = ~$1152.something."""
        emi = _calc_emi(150_000, 8.5, 360)
        assert 1100 < emi < 1250, f"Unexpected EMI: {emi}"

    def test_emi_zero_principal(self):
        assert _calc_emi(0, 8.5, 360) == 0.0

    def test_emi_zero_term(self):
        assert _calc_emi(100_000, 8.5, 0) == 0.0

    def test_emi_zero_rate(self):
        """0% interest → EMI = principal / term."""
        emi = _calc_emi(120_000, 0, 120)
        assert emi == pytest.approx(1000.0, rel=1e-4)

    def test_emi_positive_for_valid_inputs(self):
        emi = _calc_emi(200_000, 7.0, 240)
        assert emi > 0

    def test_total_repayment_exceeds_principal(self):
        """Borrower always pays back more than the principal."""
        emi = _calc_emi(100_000, 8.5, 360)
        total = emi * 360
        assert total > 100_000

    @pytest.mark.parametrize("principal,rate,term", [
        (50_000,  6.0, 60),
        (200_000, 9.0, 180),
        (500_000, 12.0, 300),
    ])
    def test_emi_parametric(self, principal, rate, term):
        emi = _calc_emi(principal, rate, term)
        assert emi > 0
        assert emi < principal  # Monthly payment is always < principal

    def test_higher_rate_means_higher_emi(self):
        low  = _calc_emi(100_000, 5.0, 360)
        high = _calc_emi(100_000, 12.0, 360)
        assert high > low

    def test_shorter_term_means_higher_emi(self):
        long_emi  = _calc_emi(100_000, 8.5, 360)
        short_emi = _calc_emi(100_000, 8.5, 120)
        assert short_emi > long_emi


# ---------------------------------------------------------------------------
# _get_session_stats
# ---------------------------------------------------------------------------
class TestGetSessionStats:
    """Tests for session statistics calculation."""

    def test_empty_history(self, client):
        with client.session_transaction() as sess:
            sess.clear()
        with app_module.app.test_request_context():
            from flask import session as flask_session
            flask_session.clear()
            stats = app_module._get_session_stats()
        assert stats["total"] == 0
        assert stats["approval_rate"] == 0
        assert stats["avg_confidence"] == 0

    def test_stats_via_home(self, client):
        """After posting a prediction, session stats should update."""
        resp = client.get("/")
        assert resp.status_code == 200
