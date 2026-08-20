from typing import Any

from fastapi import FastAPI
from pydantic import BaseModel, Field

app = FastAPI(
    title="ML Prediction Service",
    description="API REST de inferencia de modelo (Microproyecto MLOps)",
    version="0.1.0",
)


class HealthResponse(BaseModel):
    status: str = Field(
        ...,
        description="Estado del servicio",
        json_schema_extra={"example": "ok"},
    )
    version: str = Field(
        ...,
        description="Versión de la API",
        json_schema_extra={"example": "0.1.0"},
    )


@app.get("/health", response_model=HealthResponse, tags=["Health"])
def health_check() -> dict[str, Any]:
    """Endpoint de salud para verificaciones del contenedor y orquestadores."""
    return {"status": "ok", "version": "0.1.0"}
