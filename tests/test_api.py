from fastapi.testclient import TestClient

from api.main import app

client = TestClient(app)


def test_health_check() -> None:
    """Verifica que el endpoint /health responda status 200 y JSON correcto."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "version": "0.1.0"}
