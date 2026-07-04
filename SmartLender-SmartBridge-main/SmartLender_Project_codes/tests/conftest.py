"""
tests/conftest.py
==================
Shared pytest fixtures used across all test modules.

Fixtures
--------
flask_app        : configured Flask app in test mode
client           : Flask test client (with session support)
sample_form      : a valid form-data dict for a single prediction
sample_csv_bytes : in-memory CSV bytes for batch upload tests
"""

import io
import os
import sys
import pytest

# Make the project root importable
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import app as flask_app_module


# ---------------------------------------------------------------------------
# Flask app fixture
# ---------------------------------------------------------------------------
@pytest.fixture(scope="session")
def flask_app():
    """Return the Flask application configured for testing."""
    flask_app_module.app.config.update({
        "TESTING": True,
        "SECRET_KEY": "test-secret-key-12345",
        # Prevent WTF CSRF if ever added
        "WTF_CSRF_ENABLED": False,
    })
    yield flask_app_module.app


@pytest.fixture()
def client(flask_app):
    """Flask test client with session handling."""
    with flask_app.test_client() as c:
        with flask_app.app_context():
            yield c


# ---------------------------------------------------------------------------
# Common test data
# ---------------------------------------------------------------------------
@pytest.fixture()
def sample_form():
    """A valid loan application form-data dict."""
    return {
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


@pytest.fixture()
def sample_form_declined():
    """A form-data dict likely to be declined (bad credit, very low income)."""
    return {
        "gender":            "Female",
        "married":           "No",
        "dependents":        "3+",
        "education":         "Not Graduate",
        "self_employed":     "Yes",
        "applicant_income":  "1200",
        "coapplicant_income":"0",
        "loan_amount":       "500",
        "loan_amount_term":  "360",
        "credit_history":    "0",
        "property_area":     "Rural",
    }


VALID_CSV_CONTENT = """\
Gender,Married,Dependents,Education,Self_Employed,ApplicantIncome,CoapplicantIncome,LoanAmount,Loan_Amount_Term,Credit_History,Property_Area
Male,Yes,0,Graduate,No,5000,1500,128,360,1,Urban
Female,No,1,Graduate,No,3200,0,80,240,1,Semiurban
Male,Yes,2,Not Graduate,Yes,2800,1200,100,360,0,Rural
"""

@pytest.fixture()
def sample_csv_bytes():
    """Returns (filename, BytesIO, content_type) tuple for batch upload."""
    return ("applicants.csv", io.BytesIO(VALID_CSV_CONTENT.encode("utf-8")), "text/csv")


@pytest.fixture()
def sample_api_payload():
    """Valid JSON payload for /api/predict."""
    return {
        "gender":            "Male",
        "married":           "Yes",
        "dependents":        "0",
        "education":         "Graduate",
        "self_employed":     "No",
        "applicant_income":  5000,
        "coapplicant_income":1000,
        "loan_amount":       128,
        "loan_amount_term":  360,
        "credit_history":    1,
        "property_area":     "Urban",
    }
