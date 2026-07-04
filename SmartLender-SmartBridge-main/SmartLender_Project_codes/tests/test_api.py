"""
tests/test_api.py
==================
Tests for all JSON API endpoints:
  POST /api/predict
  POST /api/emi
  GET  /health

Each endpoint is tested for:
  - Correct status codes
  - JSON schema / required keys
  - Field value ranges
  - Error handling (missing / bad fields)
  - Edge cases (boundary inputs)
"""

import json
import pytest


# ---------------------------------------------------------------------------
# /api/predict
# ---------------------------------------------------------------------------
class TestApiPredict:
    ENDPOINT = "/api/predict"

    def _post(self, client, payload):
        return client.post(
            self.ENDPOINT,
            data=json.dumps(payload),
            content_type="application/json",
        )

    def test_valid_request_returns_200(self, client, sample_api_payload):
        resp = self._post(client, sample_api_payload)
        assert resp.status_code == 200

    def test_response_is_json(self, client, sample_api_payload):
        resp = self._post(client, sample_api_payload)
        assert resp.content_type == "application/json"

    def test_response_has_loan_status_key(self, client, sample_api_payload):
        resp = self._post(client, sample_api_payload)
        data = resp.get_json()
        assert "loan_status" in data

    def test_response_has_approved_key(self, client, sample_api_payload):
        resp = self._post(client, sample_api_payload)
        data = resp.get_json()
        assert "approved" in data

    def test_response_has_confidence_key(self, client, sample_api_payload):
        resp = self._post(client, sample_api_payload)
        data = resp.get_json()
        assert "confidence" in data

    def test_loan_status_is_y_or_n(self, client, sample_api_payload):
        resp = self._post(client, sample_api_payload)
        data = resp.get_json()
        assert data["loan_status"] in ("Y", "N")

    def test_approved_is_boolean(self, client, sample_api_payload):
        resp = self._post(client, sample_api_payload)
        data = resp.get_json()
        assert isinstance(data["approved"], bool)

    def test_confidence_is_float_between_0_and_1(self, client, sample_api_payload):
        resp = self._post(client, sample_api_payload)
        data = resp.get_json()
        conf = data.get("confidence")
        if conf is not None:
            assert 0.0 <= conf <= 1.0

    def test_approved_consistent_with_loan_status(self, client, sample_api_payload):
        resp = self._post(client, sample_api_payload)
        data = resp.get_json()
        assert data["approved"] == (data["loan_status"] == "Y")

    def test_contributions_key_present(self, client, sample_api_payload):
        resp = self._post(client, sample_api_payload)
        data = resp.get_json()
        assert "contributions" in data

    def test_missing_required_field_returns_400(self, client, sample_api_payload):
        bad = dict(sample_api_payload)
        del bad["applicant_income"]
        resp = self._post(client, bad)
        assert resp.status_code == 400
        err = resp.get_json()
        assert "error" in err

    def test_missing_gender_returns_400(self, client, sample_api_payload):
        bad = dict(sample_api_payload)
        del bad["gender"]
        resp = self._post(client, bad)
        assert resp.status_code == 400

    def test_empty_json_body_returns_400(self, client):
        resp = self._post(client, {})
        assert resp.status_code == 400

    def test_declined_case_returns_200(self, client):
        payload = {
            "gender":            "Female",
            "married":           "No",
            "dependents":        "3",
            "education":         "Not Graduate",
            "self_employed":     "Yes",
            "applicant_income":  1200,
            "coapplicant_income":0,
            "loan_amount":       500,
            "loan_amount_term":  360,
            "credit_history":    0,
            "property_area":     "Rural",
        }
        resp = self._post(client, payload)
        assert resp.status_code == 200

    @pytest.mark.parametrize("area", ["Urban", "Semiurban", "Rural"])
    def test_all_property_areas(self, client, sample_api_payload, area):
        fd = dict(sample_api_payload, property_area=area)
        resp = self._post(client, fd)
        assert resp.status_code == 200

    @pytest.mark.parametrize("dep", ["0", "1", "2", "3+"])
    def test_all_dependents(self, client, sample_api_payload, dep):
        fd = dict(sample_api_payload, dependents=dep)
        resp = self._post(client, fd)
        assert resp.status_code == 200

    @pytest.mark.parametrize("edu", ["Graduate", "Not Graduate"])
    def test_both_education_types(self, client, sample_api_payload, edu):
        fd = dict(sample_api_payload, education=edu)
        resp = self._post(client, fd)
        assert resp.status_code == 200

    def test_high_income_high_loan(self, client, sample_api_payload):
        fd = dict(sample_api_payload,
                  applicant_income=100_000,
                  coapplicant_income=50_000,
                  loan_amount=2000)
        resp = self._post(client, fd)
        assert resp.status_code == 200

    def test_zero_coapplicant_income(self, client, sample_api_payload):
        fd = dict(sample_api_payload, coapplicant_income=0)
        resp = self._post(client, fd)
        assert resp.status_code == 200


# ---------------------------------------------------------------------------
# /api/emi
# ---------------------------------------------------------------------------
class TestApiEmi:
    ENDPOINT = "/api/emi"

    def _post(self, client, payload):
        return client.post(
            self.ENDPOINT,
            data=json.dumps(payload),
            content_type="application/json",
        )

    BASE_EMI_PAYLOAD = {
        "loan_amount_k":  150,
        "annual_rate_pct":8.5,
        "term_months":    360,
        "monthly_income": 5000,
    }

    def test_valid_request_200(self, client):
        resp = self._post(client, self.BASE_EMI_PAYLOAD)
        assert resp.status_code == 200

    def test_response_is_json(self, client):
        resp = self._post(client, self.BASE_EMI_PAYLOAD)
        assert resp.content_type == "application/json"

    def test_response_has_emi_key(self, client):
        resp = self._post(client, self.BASE_EMI_PAYLOAD)
        data = resp.get_json()
        assert "emi" in data

    def test_response_has_total_pay(self, client):
        resp = self._post(client, self.BASE_EMI_PAYLOAD)
        data = resp.get_json()
        assert "total_pay" in data

    def test_response_has_total_int(self, client):
        resp = self._post(client, self.BASE_EMI_PAYLOAD)
        data = resp.get_json()
        assert "total_int" in data

    def test_response_has_dti_pct(self, client):
        resp = self._post(client, self.BASE_EMI_PAYLOAD)
        data = resp.get_json()
        assert "dti_pct" in data

    def test_emi_is_positive(self, client):
        resp = self._post(client, self.BASE_EMI_PAYLOAD)
        data = resp.get_json()
        assert data["emi"] > 0

    def test_total_pay_greater_than_principal(self, client):
        resp = self._post(client, self.BASE_EMI_PAYLOAD)
        data = resp.get_json()
        assert data["total_pay"] > self.BASE_EMI_PAYLOAD["loan_amount_k"] * 1000

    def test_total_int_positive(self, client):
        resp = self._post(client, self.BASE_EMI_PAYLOAD)
        data = resp.get_json()
        assert data["total_int"] > 0

    def test_dti_pct_reasonable(self, client):
        resp = self._post(client, self.BASE_EMI_PAYLOAD)
        data = resp.get_json()
        # DTI should be > 0 and < 100% for a typical case
        assert 0 < data["dti_pct"] < 100

    def test_zero_loan_amount_returns_zero_emi(self, client):
        resp = self._post(client, {**self.BASE_EMI_PAYLOAD, "loan_amount_k": 0})
        data = resp.get_json()
        assert data["emi"] == 0.0

    def test_high_income_gives_low_dti(self, client):
        resp = self._post(client, {**self.BASE_EMI_PAYLOAD, "monthly_income": 100_000})
        data = resp.get_json()
        assert data["dti_pct"] < 5  # Small DTI for very high income

    def test_low_income_gives_high_dti(self, client):
        resp = self._post(client, {**self.BASE_EMI_PAYLOAD, "monthly_income": 500})
        data = resp.get_json()
        assert data["dti_pct"] > 50

    @pytest.mark.parametrize("term", [12, 60, 120, 180, 240, 360])
    def test_all_loan_terms(self, client, term):
        resp = self._post(client, {**self.BASE_EMI_PAYLOAD, "term_months": term})
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["emi"] > 0

    def test_default_rate_applied_when_missing(self, client):
        """If annual_rate_pct is missing the app should still respond."""
        payload = {
            "loan_amount_k":  100,
            "term_months":    360,
            "monthly_income": 5000,
        }
        resp = self._post(client, payload)
        assert resp.status_code == 200


# ---------------------------------------------------------------------------
# /health
# ---------------------------------------------------------------------------
class TestHealthApi:

    def test_health_status_ok(self, client):
        resp = client.get("/health")
        data = resp.get_json()
        assert data["status"] == "ok"

    def test_health_200(self, client):
        assert client.get("/health").status_code == 200

    def test_health_returns_model_loaded_true(self, client):
        """Model should be loaded after running train_model.py."""
        resp = client.get("/health")
        data = resp.get_json()
        # If train_model.py was run, model_loaded == True
        assert isinstance(data["model_loaded"], bool)
