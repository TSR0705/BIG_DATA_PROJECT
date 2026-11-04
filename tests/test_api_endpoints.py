"""Tests for CreditLens FastAPI endpoints (dashboard data layer)."""

import pytest
from starlette.testclient import TestClient

from api.app.main import app


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def test_api_root_endpoint(client):
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "online"
    assert "CreditLens" in data["project"]


def test_api_health_endpoint(client):
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["models_loaded"] is True
    assert data["calibrator_loaded"] is True


def test_api_project_overview_matches_artifacts(client):
    """Overview numbers must equal the committed Phase 6 split report and Phase 11 validation metrics."""
    import json
    import pandas as pd
    from api.app.core.config import SPLIT_REPORT_PATH, SUBGROUP_ANALYSIS_PATH

    response = client.get("/api/project/overview")
    assert response.status_code == 200
    data = response.json()

    rep = json.load(open(SPLIT_REPORT_PATH, encoding="utf-8"))
    total = rep["splits"]["total"]
    assert data["total_applicants"] == int(total["rows"])
    assert data["total_defaults"] == int(total["positive_count"])

    overall = pd.read_csv(SUBGROUP_ANALYSIS_PATH)
    overall = overall[overall["subgroup_dimension"] == "Overall"].iloc[0]
    assert data["model_roc_auc"] == round(float(overall["roc_auc"]), 4)
    assert data["model_ece"] == round(float(overall["ece"]), 4)

    assert len(data["kpis"]) == 3
    assert len(data["top_risk_factors"]) == 5
    assert data["top_risk_factors"][0]["feature"] == "APP_EXT_SOURCES_MEAN"
    assert len(data["recent_predictions"]) == 5
    assert data["default_rate_profile"]["dimension"] == "age"
    assert len(data["default_rate_profile"]["bands"]) >= 5
    assert all(0 <= b["default_rate"] <= 100 for b in data["default_rate_profile"]["bands"])
    assert set(data["data_sources"]) >= {"counts", "validation_metrics", "risk_factors", "recent_predictions"}


@pytest.mark.parametrize("dimension", ["age", "credit", "income"])
def test_api_default_rate_profile(client, dimension):
    response = client.get(f"/api/project/default-rate-profile?dimension={dimension}")
    assert response.status_code == 200
    data = response.json()
    assert data["dimension"] == dimension
    assert sum(b["applicants"] for b in data["bands"]) + data["applicants_outside_bands"] == 307511


def test_api_default_rate_profile_rejects_unknown_dimension(client):
    assert client.get("/api/project/default-rate-profile?dimension=year").status_code == 422


def test_api_recent_predictions_are_real_engine_outputs(client):
    response = client.get("/api/predictions/recent?limit=5")
    assert response.status_code == 200
    data = response.json()
    assert 1 <= len(data) <= 5
    for p in data:
        assert p["applicant_id"].startswith("APP-")
        assert 0.0 <= p["probability"] <= 1.0
        assert p["risk_tier"] in {"Low Risk", "Medium Risk", "High Risk"}
        assert p["scored_at"]


def test_api_applicant_search_and_score(client):
    from api.app.services.applicant_store import test_split_ids

    sk_id = test_split_ids()[0]
    search = client.get(f"/api/applicants/search?q={sk_id}")
    assert search.status_code == 200
    assert any(h["sk_id_curr"] == sk_id for h in search.json()["results"])

    scored = client.get(f"/api/applicants/{sk_id}/score")
    assert scored.status_code == 200
    body = scored.json()
    assert body["sk_id_curr"] == sk_id
    assert 0.0 <= body["default_probability"] <= 1.0
    assert body["decision"] in {"APPROVE", "REJECT"}

    assert client.get("/api/applicants/1/score").status_code == 404


def test_api_model_metadata_endpoint(client):
    response = client.get("/api/model/metadata")
    assert response.status_code == 200
    data = response.json()
    assert data["total_features"] == 256
    assert data["categorical_features_count"] == 16
    assert data["numeric_features_count"] == 240
    assert data["thresholds"]["primary_cost_optimal_1_5"] == 0.1600


def test_api_research_metrics_endpoint(client):
    response = client.get("/api/research/metrics")
    assert response.status_code == 200
    data = response.json()
    assert data["project"] == "CreditLens"
    assert "validation_performance" in data
    assert "test_evaluation" in data
    assert len(data["benchmarks"]) >= 6
    assert len(data["subgroups"]) >= 4


def test_api_predict_risk_endpoint(client):
    payload = {
        "age": 32,
        "annual_income": 850000.0,
        "employment_type": "Salaried",
        "loan_amount": 1200000.0,
        "loan_tenure_months": 60,
        "existing_obligations": 18000.0,
    }
    response = client.post("/api/applicants/predict-risk", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["default_percentage"] == 14.2
    assert data["risk_tier"] == "Medium Risk"
    assert data["model_used"] == "Illustrative Heuristic Estimate"
    assert data["inference_type"] == "ILLUSTRATIVE_HEURISTIC"
    assert data["explanation_type"] == "ILLUSTRATIVE_HEURISTIC_FACTORS"
    assert len(data["risk_factors"]) == 5
    assert data["risk_factors"][0]["name"] == "Loan Amount"


def test_api_example_applicant_endpoint(client):
    response = client.get("/api/applicants/example")
    assert response.status_code == 200
    data = response.json()
    assert data["age"] == 32
    assert data["annual_income"] == 850000.0
    assert data["employment_type"] == "Salaried"



@pytest.mark.parametrize(
    "invalid_field, invalid_val",
    [
        ("age", 15),
        ("age", 105),
        ("age", -1),
        ("annual_income", 0.0),
        ("annual_income", -5000.0),
        ("loan_amount", 0.0),
        ("loan_amount", -100000.0),
        ("loan_tenure_months", 0),
        ("loan_tenure_months", -12),
        ("loan_tenure_months", 400),
        ("existing_obligations", -100.0),
        ("annual_income", "invalid-non-numeric"),
        ("loan_amount", "NaN"),
    ],
)
def test_api_predict_risk_strict_validation(client, invalid_field, invalid_val):
    """Priority 4: Strict Pydantic boundary validation must return HTTP 422 on invalid inputs."""
    valid_payload = {
        "age": 32,
        "annual_income": 850000.0,
        "employment_type": "Salaried",
        "loan_amount": 1200000.0,
        "loan_tenure_months": 60,
        "existing_obligations": 18000.0,
    }
    valid_payload[invalid_field] = invalid_val
    response = client.post("/api/applicants/predict-risk", json=valid_payload)
    assert response.status_code == 422, f"Expected 422 for {invalid_field}={invalid_val}, got {response.status_code}"


def test_api_predict_risk_discloses_inference_type(client):
    """Priority 1: Six-field applicant form must disclose illustrative heuristic estimate vs genuine model."""
    payload = {
        "age": 32,
        "annual_income": 850000.0,
        "employment_type": "Salaried",
        "loan_amount": 1200000.0,
        "loan_tenure_months": 60,
        "existing_obligations": 18000.0,
    }
    response = client.post("/api/applicants/predict-risk", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["inference_type"] == "ILLUSTRATIVE_HEURISTIC"
    assert data["explanation_type"] == "ILLUSTRATIVE_HEURISTIC_FACTORS"
    assert data["disclaimer"] is not None
    assert "heuristic" in data["disclaimer"].lower()
    assert "not the research-validated" in data["disclaimer"].lower()


def test_api_known_applicant_is_genuine_model(client):
    """Priority 1: Scoring a known applicant runs the genuine 256-feature model pipeline with TreeSHAP."""
    from api.app.services.applicant_store import test_split_ids

    sk_id = test_split_ids()[0]
    response = client.get(f"/api/applicants/{sk_id}/score")
    assert response.status_code == 200
    data = response.json()
    assert data["inference_type"] == "GENUINE_MODEL"
    assert data["explanation_type"] == "GENUINE_TREESHAP"
    assert "256 Features" in data["model_used"]
    assert len(data["risk_factors"]) == 5
