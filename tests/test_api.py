from fastapi.testclient import TestClient

from api.main import app

client = TestClient(app)


def test_health_check() -> None:
    """Verifica que /health responda 200 con la forma esperada, sin asumir
    que el modelo esté cargado (depende de MLFLOW_TRACKING_URI/MODEL_URI,
    que pueden no estar configurados en el entorno de pruebas)."""
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] in {"ok", "degraded"}
    assert "model_loaded" in body
    assert "features_loaded" in body
