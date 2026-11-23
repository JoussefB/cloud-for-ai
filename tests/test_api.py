"""Tests voor het FastAPI endpoint en inputvalidatie (les 9 / deel 3 vereisten).

Test of:
1. Health endpoint en root informatie leveren.
2. Geldige payloads een predictie en alarm teruggeven.
3. Ongeldige payloads een nette 422 JSON fout geven (geen stacktrace!).
4. Fysische grenzen (negatieve slijtage, extreme temperaturen) netjes afgevangen worden.
"""

import pytest
from fastapi.testclient import TestClient

from src.app import app


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as test_client:
        yield test_client


def test_root_endpoint(client):
    """Controleer of de root endpoint metadata teruggeeft."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "project" in data
    assert data["decision_threshold"] == 0.15


def test_health_endpoint(client):
    """Controleer of de healthcheck aangeeft dat het model geladen is."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["model_loaded"] is True


def test_predict_normale_machine(client):
    """Een machine met normale parameters mag geen alarm triggeren."""
    payload = {
        "Type": "L",
        "Air temperature [K]": 298.1,
        "Process temperature [K]": 308.6,
        "Rotational speed [rpm]": 1551,
        "Torque [Nm]": 42.8,
        "Tool wear [min]": 0,
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "failure_probability" in data
    assert 0.0 <= data["failure_probability"] <= 1.0
    assert data["threshold"] == 0.15
    assert data["alarm"] is False
    assert data["status"] == "NORMAL_OPERATION"


def test_predict_defecte_machine_triggert_alarm(client):
    """Een machine met zware overbelasting (hoog koppel + maximale slijtage) moet alarm slaan."""
    payload = {
        "Type": "L",
        "Air temperature [K]": 304.0,
        "Process temperature [K]": 313.0,
        "Rotational speed [rpm]": 1200,
        "Torque [Nm]": 75.0,  # Zeer hoog koppel
        "Tool wear [min]": 240,  # Kritieke gereedschapsslijtage
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["alarm"] is True
    assert data["failure_probability"] >= 0.15
    assert data["status"] == "WARNING_FAILURE_PREDICTED"
    assert "Spoedinspectie vereist" in data["action"]


def test_validatie_onbekend_type_geeft_422_zonder_stacktrace(client):
    """Een ongeldig type (bv. 'Z' ipv L/M/H) moet een nette 422 fout geven, geen 500 error."""
    payload = {
        "Type": "Z",  # Ongeldig
        "Air temperature [K]": 298.1,
        "Process temperature [K]": 308.6,
        "Rotational speed [rpm]": 1551,
        "Torque [Nm]": 42.8,
        "Tool wear [min]": 0,
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 422
    data = response.json()
    assert data["error"] == "Validatiefout in invoergegevens"
    assert "details" in data
    # Bevat geen Python stacktrace
    assert "Traceback" not in response.text


def test_validatie_negatieve_slijtage_geeft_422(client):
    """Negatieve tool wear is fysisch onmogelijk en moet netjes geweigerd worden."""
    payload = {
        "Type": "L",
        "Air temperature [K]": 298.1,
        "Process temperature [K]": 308.6,
        "Rotational speed [rpm]": 1551,
        "Torque [Nm]": 42.8,
        "Tool wear [min]": -15,  # Onmogelijk
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 422
    data = response.json()
    assert data["error"] == "Validatiefout in invoergegevens"


def test_validatie_ontbrekend_veld_geeft_422(client):
    """Als een verplicht veld ontbreekt (bv. Torque), moet de API 422 geven."""
    payload = {
        "Type": "M",
        "Air temperature [K]": 298.1,
        "Process temperature [K]": 308.6,
        "Rotational speed [rpm]": 1551,
        # Torque ontbreekt
        "Tool wear [min]": 10,
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_batch_prediction(client):
    """Test of meerdere machines tegelijk in batch verwerkt kunnen worden."""
    batch_payload = [
        {
            "Type": "L",
            "Air temperature [K]": 298.1,
            "Process temperature [K]": 308.6,
            "Rotational speed [rpm]": 1551,
            "Torque [Nm]": 42.8,
            "Tool wear [min]": 0,
        },
        {
            "Type": "H",
            "Air temperature [K]": 304.0,
            "Process temperature [K]": 313.0,
            "Rotational speed [rpm]": 1200,
            "Torque [Nm]": 75.0,
            "Tool wear [min]": 240,
        },
    ]
    response = client.post("/predict/batch", json=batch_payload)
    assert response.status_code == 200
    data = response.json()
    assert data["count"] == 2
    assert len(data["predictions"]) == 2
