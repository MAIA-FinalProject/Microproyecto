"""
api/main.py

Endpoints para el microproyecto NeoRisk (predicción de alteración del
neurodesarrollo en prematuros a partir del modelo baseline de #18).

- GET  /health   -> verifica que el modelo y las variables esperadas estén cargados
- POST /predict  -> recibe el payload clínico del dashboard (#17) y retorna
                     probabilidad, puntaje 0-100, categoría y recomendación

Contrato con dashboard/app.py (#17) — confirmado leyendo el código real:
- El dashboard envía el payload PLANO (sin envolver en {"features": {...}}):
    {"DM": "yes"/"no", "preeclampsia": "yes"/"no",
     "PregnancyAge": float, "BirthWeight": float, "apgar5": int}
- El dashboard lee de la respuesta, en modo live, EXACTAMENTE estas claves:
    score (float 0-100), categoria (str), recomendacion (str)
  (nombres en español, no "risk_category"/"recommendation").

DECISIÓN DE EQUIPO (Martín, sobre el alcance para la entrega):
- El dashboard solo pide 5 variables al usuario: DM, preeclampsia,
  PregnancyAge, BirthWeight, apgar5 (coinciden 1 a 1 con los nombres del
  modelo, no hace falta traducirlas).
- Las otras 32 columnas que el modelo espera quedan FIJAS en DEFAULT_VALUES,
  calculadas de data/features/features.csv (moda para binarias, mediana para
  continuas) con scripts/compute_defaults.py. Esto es una simplificación
  consciente para la entrega, no la predicción más precisa posible — si el
  equipo define otros valores, solo hay que reemplazar el diccionario.
- "DM" y "preeclampsia" llegan como texto "yes"/"no"; se normalizan acá a 1/0
  (BINARY_STRING_MAP).

Contrato real confirmado leyendo src/models/train.py de Martín (#18):
- El pipeline NO se loguea con mlflow.sklearn.log_model(); se guarda con
  joblib.dump() y se sube como artefacto genérico vía mlflow.log_artifact(),
  con nombre "{nombre_del_modelo}.joblib" (ej. "logistic_regression.joblib")
  suelto en la raíz de artifacts del run. Por eso esta API busca el .joblib
  dentro del run en vez de usar mlflow.sklearn.load_model(runs:/.../model).
- Columnas confirmadas que el pipeline espera (de CONTINUOUS_VARS +
  IMPUTE_VARS en train.py): PregnancyAge, BirthWeight, apgar1, apgar5,
  duration.hopitalization, duration.O2, RoundHeadAtBirth, congenital.anomaly
  — más el resto de columnas de data/features/features.csv (dummies/binarias)
  que no se ven en ese fragmento. La lista completa y definitiva la da
  GET /health -> expected_features, una vez el modelo cargue bien.
- El dashboard (#17) hoy solo manda 5 de esas columnas (DM, preeclampsia,
  PregnancyAge, BirthWeight, apgar5) — probablemente faltan variables por
  agregar al formulario de Streamlit, no solo nombres por mapear.
"""

import logging
import os
import re
from typing import Any

import joblib
import mlflow
import mlflow.artifacts
import pandas as pd
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from mlflow.tracking import MlflowClient
from pydantic import BaseModel

load_dotenv()  # lee el .env de la raíz (MLFLOW_TRACKING_URI, MODEL_URI, etc.)

logger = logging.getLogger("neorisk.api")
logging.basicConfig(level=logging.INFO)

MLFLOW_TRACKING_URI = os.getenv(
    "MLFLOW_TRACKING_URI"
)  # URI del server en EC2, desde .env
MODEL_URI = os.getenv(
    "MODEL_URI"
)  # ej: "runs:/<run_id>/model" o "models:/neorisk/Production"

# Las 5 variables que el dashboard SÍ pide al usuario (#17). El resto queda
# fija en DEFAULT_VALUES.
FREE_FIELDS = ["DM", "preeclampsia", "PregnancyAge", "BirthWeight", "apgar5"]

# Calculado de data/features/features.csv (moda para binarias, mediana para
# continuas) con compute_defaults.py — ver decisión de equipo arriba.
DEFAULT_VALUES: dict[str, Any] = {
    "Sex": 0,
    "hypothyroid": 0,
    "PROM": 0,
    "IUGR": 0,
    "pregnancycomplication": 1,
    "pneumothorax": 0,
    "NEC": 0,
    "sepsis": 0,
    "PDA": 0,
    "icter": 1,
    "meningitis": 0,
    "IVH": 0,
    "seizure": 0,
    "BPD": 0,
    "apgar1": 8.0,
    "B.C": 0,
    "drug.mother": 0,
    "duration.hopitalization": 14.0,
    "ehya.badve.tavallod": 0,
    "RoundHeadAtBirth": 30.0,
    "notaggressive.ventilation": 1,
    "csf.culture": 0,
    "congenital.anomaly": 0,
    "duration.O2": 1.0,
    "SepsisnegativeCulture": 0,
    "mother.sonogarphy.result": 0,
    "intervencion_respiratoria_agresiva": 0,
    "laborType_NVD": 0,
    "laborType_cs": 1,
    "type.of.ressucitation_advanced": 0,
    "type.of.ressucitation_no_resuscitation": 1,
    "type.of.ressucitation_ppv": 0,
}

# Valores de texto que el dashboard manda para campos binarios ("yes"/"no")
BINARY_STRING_MAP: dict[str, int] = {"yes": 1, "no": 0}

# (límite_superior_exclusivo, categoría, recomendación) — igual a dashboard/app.py
RISK_BANDS: tuple[tuple[float, str, str], ...] = (
    (
        25.0,
        "Bajo",
        "Riesgo bajo de alteración del neurodesarrollo. Continuar con el "
        "protocolo estándar de seguimiento neonatal y tamizaje rutinario.",
    ),
    (
        50.0,
        "Moderado",
        "Riesgo moderado. Se sugiere intensificar la monitorización en "
        "UCIN y programar evaluación neuroconductual antes del egreso "
        "hospitalario.",
    ),
    (
        75.0,
        "Alto",
        "Riesgo alto. Se recomienda valoración prioritaria por neuropediatría, "
        "evaluación ecográfica cerebral y seguimiento en neurodesarrollo.",
    ),
)
CRITICAL_LABEL = "Crítico"
CRITICAL_RECOMMENDATION = (
    "Riesgo crítico. Alerta clínica prioritaria: activar protocolo de "
    "intervención temprana, panel multidisciplinario y neuroimagen."
)

app = FastAPI(title="NeoRisk API", version="0.1.0")

# Solo para desarrollo local: permite probar la API desde un formulario en el
# navegador (ej. un widget de prueba). En producción, restringe esto al
# origen real del dashboard.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_model = None
_expected_features: list[str] | None = None
_load_error: str | None = None


def _extract_run_id(model_uri: str) -> str:
    """Acepta tanto 'runs:/<run_id>/lo-que-sea' como un run_id plano."""
    match = re.match(r"runs:/([^/]+)", model_uri)
    if match:
        return str(match.group(1))
    return model_uri


def _find_joblib_artifact(run_id: str) -> str:
    """Busca el .joblib logueado por train.py entre los artifacts del run
    (train.py lo sube suelto en la raíz, con nombre '{nombre_modelo}.joblib')."""
    client = MlflowClient()
    artifacts = client.list_artifacts(run_id)
    joblib_files = [a.path for a in artifacts if a.path.endswith(".joblib")]
    if not joblib_files:
        raise FileNotFoundError(
            f"No se encontró ningún archivo .joblib en los artifacts del run {run_id}."
        )
    if len(joblib_files) > 1:
        logger.warning(
            "Más de un .joblib en el run %s (%s); usando el primero.",
            run_id,
            joblib_files,
        )
    return str(joblib_files[0])


@app.on_event("startup")
def load_model() -> None:
    global _model, _expected_features, _load_error

    if not MLFLOW_TRACKING_URI:
        _load_error = (
            "MLFLOW_TRACKING_URI no está configurada (debería venir del .env, ver #11)."
        )
        logger.error(_load_error)
        return
    if not MODEL_URI:
        _load_error = "MODEL_URI no está configurado (variable de entorno)."
        logger.error(_load_error)
        return

    try:
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        run_id = _extract_run_id(MODEL_URI)
        artifact_path = _find_joblib_artifact(run_id)
        local_path = mlflow.artifacts.download_artifacts(
            run_id=run_id, artifact_path=artifact_path
        )
        _model = joblib.load(local_path)

        feature_names = getattr(_model, "feature_names_in_", None)
        _expected_features = list(feature_names) if feature_names is not None else None

        if _expected_features is None:
            logger.warning(
                "El modelo no expone feature_names_in_; /predict no podrá "
                "validar el payload por nombre de columna."
            )

        logger.info(
            "Modelo cargado desde %s (features: %s)", MODEL_URI, _expected_features
        )

    except Exception as exc:
        _load_error = f"No se pudo cargar el modelo desde {MODEL_URI}: {exc}"
        logger.exception(_load_error)


class PredictRequest(BaseModel):
    DM: str
    preeclampsia: str
    PregnancyAge: float
    BirthWeight: float
    apgar5: int


class PredictResponse(BaseModel):
    probability: float
    score: float
    categoria: str
    recomendacion: str


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    features_loaded: bool
    expected_features: list[str] | None = None
    detail: str | None = None


def _translate_payload(raw: dict[str, Any]) -> dict[str, Any]:
    """Normaliza 'yes'/'no' a 1/0 en los campos libres que manda el dashboard."""
    translated: dict[str, Any] = {}
    for key, value in raw.items():
        if isinstance(value, str) and value.lower() in BINARY_STRING_MAP:
            value = BINARY_STRING_MAP[value.lower()]
        translated[key] = value
    return translated


def _risk_band(probability: float) -> tuple[float, str, str]:
    score = round(probability * 100, 1)
    score = min(max(score, 0.0), 100.0)
    for upper_bound, label, recommendation in RISK_BANDS:
        if score < upper_bound:
            return score, label, recommendation
    return score, CRITICAL_LABEL, CRITICAL_RECOMMENDATION


def _positive_class_probability(model: Any, row_df: pd.DataFrame) -> float:
    proba = model.predict_proba(row_df)[0]
    classes = list(getattr(model, "classes_", [0, 1]))
    try:
        idx = classes.index(1)
    except ValueError:
        idx = -1  # fallback: asume que la última columna es la clase positiva
    return float(proba[idx])


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    model_loaded = _model is not None
    features_loaded = bool(_expected_features)
    ok = model_loaded and features_loaded
    return HealthResponse(
        status="ok" if ok else "degraded",
        model_loaded=model_loaded,
        features_loaded=features_loaded,
        expected_features=_expected_features,
        detail=_load_error,
    )


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest) -> PredictResponse:
    if _model is None:
        raise HTTPException(
            status_code=503, detail=_load_error or "Modelo no disponible."
        )

    free_fields = _translate_payload(request.model_dump())
    full_payload = {**DEFAULT_VALUES, **free_fields}

    if _expected_features is not None:
        received = set(full_payload.keys())
        expected = set(_expected_features)
        missing = expected - received
        extra = received - expected
        if missing or extra:
            raise HTTPException(
                status_code=422,
                detail={
                    "message": (
                        "El payload combinado (campos libres + DEFAULT_VALUES) no "
                        "coincide con las variables del modelo. Revisa DEFAULT_VALUES "
                        "en api/main.py contra GET /health -> expected_features."
                    ),
                    "faltantes": sorted(missing),
                    "sobrantes": sorted(extra),
                },
            )
        ordered_row = {name: full_payload[name] for name in _expected_features}
    else:
        ordered_row = full_payload

    try:
        row_df = pd.DataFrame([ordered_row])
        probability = _positive_class_probability(_model, row_df)
    except Exception as exc:
        raise HTTPException(
            status_code=400, detail=f"Error al predecir: {exc}"
        ) from exc

    score, categoria, recomendacion = _risk_band(probability)
    return PredictResponse(
        probability=probability,
        score=score,
        categoria=categoria,
        recomendacion=recomendacion,
    )
