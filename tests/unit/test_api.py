from fastapi.testclient import TestClient

import src.api as api

# ============================================================
# Tests unitarios para la API
# Cubren endpoint health, version y consulta
# No requieren .env ni archivos externos
# Ejecutar:
#   python -m pytest tests/unit/test_api.py
# ============================================================

client = TestClient(api.app)


def test_health_endpoint():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_version_endpoint():
    response = client.get("/version")

    assert response.status_code == 200
    payload = response.json()
    assert payload["service"] == "rag-transito"
    assert payload["version"] == "1.0.0"


def test_consulta_endpoint_maps_response_fields(monkeypatch):
    def fake_run_consulta(question: str, verbose: bool = False):
        return {
            "respuesta": "La multa por no usar casco es de 15 SMLDV.",
            "status": "éxito",
            "modelo_usado": "gemini-2.0-flash",
            "tokens_prompt": 120,
            "query_original": question,
            "error": None,
        }

    monkeypatch.setattr(api, "run_consulta", fake_run_consulta)

    response = client.post(
        "/consulta",
        json={"question": "¿Cuál es la multa por no usar casco?", "verbose": True},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload == {
        "answer": "La multa por no usar casco es de 15 SMLDV.",
        "status": "éxito",
        "model": "gemini-2.0-flash",
        "prompt_tokens": 120,
        "original_question": "¿Cuál es la multa por no usar casco?",
        "error": None,
    }


def test_consulta_endpoint_requires_question():
    response = client.post("/consulta", json={"verbose": False})

    assert response.status_code == 422


def test_consulta_endpoint_rejects_short_question():
    response = client.post("/consulta", json={"question": "hi", "verbose": False})

    assert response.status_code == 422


def test_consulta_endpoint_sets_error_status(monkeypatch):
    def fake_run_consulta(question: str, verbose: bool = False):
        return {
            "respuesta": "",
            "status": "éxito",
            "modelo_usado": None,
            "tokens_prompt": 0,
            "query_original": question,
            "error": "No se encontró contexto suficiente.",
        }

    monkeypatch.setattr(api, "run_consulta", fake_run_consulta)

    response = client.post(
        "/consulta",
        json={"question": "¿Qué dice la norma sobre X?", "verbose": False},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "error"
    assert payload["error"] == "No se encontró contexto suficiente."
