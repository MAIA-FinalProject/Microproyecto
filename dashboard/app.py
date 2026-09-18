"""dashboard/app.py.

Interfaz clínica interactiva para soporte a la toma de decisiones en UCIN
(Unidad de Cuidados Intensivos Neonatales) - Proyecto NeuroRisk.
"""

import os
from datetime import datetime
from typing import Any

import pandas as pd
import plotly.express as px
import requests
import streamlit as st

st.set_page_config(
    page_title="NeuroRisk - UCIN Decision Support",
    layout="wide",
    initial_sidebar_state="expanded",
)

DEFAULT_API_ENDPOINT: str = os.getenv("API_URL", "http://api:8000/predict")

FEATURE_LABELS: dict[str, str] = {
    "duration.O2": "Duración Oxigenoterapia (días)",
    "duration.hopitalization": "Duración Hospitalización (días)",
    "BirthWeight": "Peso al Nacer (g)",
    "PregnancyAge": "Edad Gestacional (sem)",
    "apgar5": "Puntaje APGAR 5 min",
    "DM": "Diabetes Materna (DM)",
    "preeclampsia": "Preeclampsia Materna",
}


def categorize_risk_score(score: float) -> str:
    """Clasifica el puntaje de riesgo continuo (0-100) en 4 bandas clínicas."""
    if score < 25.0:
        return "Bajo"
    if score < 50.0:
        return "Moderado"
    if score < 75.0:
        return "Alto"
    return "Crítico"


def get_clinical_recommendation(category: str) -> str:
    """Retorna la directriz clínica estandarizada según el nivel de riesgo."""
    recommendations: dict[str, str] = {
        "Bajo": (
            "Riesgo bajo de alteración del neurodesarrollo. Continuar con el "
            "protocolo estándar de seguimiento neonatal y tamizaje rutinario."
        ),
        "Moderado": (
            "Riesgo moderado. Se sugiere intensificar la monitorización en UCIN "
            "y programar evaluación neuroconductual antes del egreso hospitalario."
        ),
        "Alto": (
            "Riesgo alto. Se recomienda valoración prioritaria por neuropediatría, "
            "evaluación ecográfica cerebral y seguimiento continuo."
        ),
        "Crítico": (
            "Riesgo crítico. Alerta clínica prioritaria: activar protocolo de "
            "intervención temprana, panel multidisciplinario y neuroimagen."
        ),
    }
    return recommendations.get(category, "Seguimiento clínico estándar.")


def resolve_api_endpoint(target_url: str) -> str:
    """Normaliza la URL asegurando que apunte al endpoint /predict."""
    clean_url = target_url.strip().rstrip("/")
    if not clean_url.endswith("/predict"):
        return f"{clean_url}/predict"
    return clean_url


def call_predict_api(
    payload: dict[str, Any], endpoint_url: str, timeout: float = 10.0
) -> dict[str, Any] | None:
    """Realiza una petición HTTP POST a la API con manejo robusto de errores."""
    resolved_endpoint = resolve_api_endpoint(endpoint_url)
    try:
        response = requests.post(
            resolved_endpoint,
            json=payload,
            timeout=timeout,
            headers={"Content-Type": "application/json"},
        )

        if response.status_code == 200:
            data: dict[str, Any] = response.json()
            return data

        if response.status_code == 422:
            st.error(
                "Error de validación en los parámetros enviados (HTTP 422): "
                f"{response.text}"
            )
        elif response.status_code == 503:
            st.error(
                "El servicio de inferencia no tiene un modelo cargado "
                "en este momento (HTTP 503)."
            )
        else:
            st.error(
                f"Error devuelto por la API (HTTP {response.status_code}): "
                f"{response.text}"
            )
        return None

    except requests.exceptions.Timeout:
        st.error(
            "Tiempo de espera agotado al conectar con el servicio de inferencia. "
            "Por favor, reintente en unos momentos."
        )
        return None
    except requests.exceptions.ConnectionError:
        st.error(
            "No fue posible establecer conexión con el backend en: "
            f"`{resolved_endpoint}`. Verifique que FastAPI esté activo."
        )
        return None
    except requests.exceptions.RequestException as exc:
        st.error(f"Error de comunicación con la API de inferencia: {exc}")
        return None
    except Exception as exc:
        st.error(f"Error inesperado al procesar la respuesta: {exc}")
        return None


def check_api_health(endpoint_url: str, timeout: float = 3.0) -> bool:
    """Verifica si el servicio de FastAPI responde en su endpoint /health."""
    base_url = endpoint_url.strip().rstrip("/")
    if base_url.endswith("/predict"):
        base_url = base_url[:-8]
    health_url = f"{base_url}/health"

    try:
        res = requests.get(health_url, timeout=timeout)
        return res.status_code == 200
    except Exception:
        return False


def render_interpretability_chart(contributions: dict[str, float]) -> None:
    """Renderiza un gráfico de barras horizontal con las contribuciones
    de cada variable clínica evaluada."""
    if not contributions:
        return

    rows: list[dict[str, Any]] = []
    for raw_name, val in contributions.items():
        clean_name = FEATURE_LABELS.get(raw_name, raw_name)
        tipo = "Factor de Riesgo" if val > 0 else "Factor Protector"
        rows.append(
            {
                "Variable": clean_name,
                "Contribución": round(float(val), 4),
                "Tipo": tipo,
            }
        )

    df_contrib = pd.DataFrame(rows).sort_values(by="Contribución", ascending=True)

    st.markdown("---")
    st.markdown("#### Factores Clave del Paciente (Interpretabilidad Local)")
    st.caption(
        "Aporte relativo de cada variable evaluada en la predicción del paciente. "
        "Las barras en rojo indican factores que incrementan el riesgo, "
        "mientras que las barras verdes representan factores protectores."
    )

    fig = px.bar(
        df_contrib,
        x="Contribución",
        y="Variable",
        color="Tipo",
        orientation="h",
        color_discrete_map={
            "Factor de Riesgo": "#ff4b4b",
            "Factor Protector": "#00cc96",
        },
    )
    fig.update_layout(
        margin=dict(l=150, r=20, t=20, b=40),
        xaxis_title=None,
        yaxis_title=None,
        legend_title_text="Efecto Clínico",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )

    st.plotly_chart(fig, use_container_width=True)


def init_session_state() -> None:
    if "patient_history" not in st.session_state:
        st.session_state.patient_history = []
    if "last_result" not in st.session_state:
        st.session_state.last_result = None
    if "last_payload" not in st.session_state:
        st.session_state.last_payload = None
    if "api_url" not in st.session_state:
        st.session_state.api_url = DEFAULT_API_ENDPOINT


def reset_current_evaluation() -> None:
    st.session_state.last_result = None
    st.session_state.last_payload = None


def clear_patient_history() -> None:
    st.session_state.patient_history = []
    reset_current_evaluation()


init_session_state()

with st.sidebar:
    st.header("NeuroRisk UCIN")
    st.caption("Sistema de Soporte para Decisión Clínica Neonatal")

    st.markdown("---")
    st.subheader("Configuración del Backend")

    st.session_state.api_url = st.text_input(
        "URL de Inferencia (FastAPI):",
        value=st.session_state.api_url,
        help="Dirección del servicio REST donde corre la API.",
    )

    if st.button("Probar Conexión con API", use_container_width=True):
        with st.spinner("Comprobando conectividad..."):
            if check_api_health(st.session_state.api_url):
                st.success("Conexión exitosa con el servicio.")
            else:
                st.warning("No se obtuvo respuesta satisfactoria en /health.")

    st.markdown("---")
    st.subheader("Resumen de Sesión")
    total_evaluados = len(st.session_state.patient_history)
    st.caption(f"Pacientes en triaje activo: **{total_evaluados}**")

    if st.button("Limpiar Evaluación Actual", use_container_width=True):
        reset_current_evaluation()
        st.rerun()

    if total_evaluados > 0:
        if st.button("Reiniciar Panel de Triaje", use_container_width=True):
            clear_patient_history()
            st.rerun()

st.title("NeuroRisk - UCIN Decision Support")
st.markdown(
    "Herramienta clínica para la estimación temprana y priorización del riesgo de "
    "**alteración del neurodesarrollo** en recién nacidos prematuros al alta."
)

tab_calc, tab_triage = st.tabs(
    ["Calculadora de Riesgo", "Panel de Triaje de Pacientes"]
)

with tab_calc:
    st.subheader("Ingreso de Parámetros Clínicos al Alta")
    st.caption(
        "Diligencie las variables maternas y neonatales registradas durante la "
        "estancia hospitalaria para estimar el riesgo de alteración."
    )

    with st.form("clinical_evaluation_form"):
        col_maternal, col_neonatal = st.columns(2)

        with col_maternal:
            st.markdown("#### Factores Maternos")
            dm_input = st.selectbox(
                "Diabetes Mellitus Materna (DM):",
                options=["no", "yes"],
                index=0,
                format_func=lambda x: "Sí" if x == "yes" else "No",
                help="Presencia diagnosticada de diabetes materna pre o gestacional.",
            )
            preeclampsia_input = st.selectbox(
                "Preeclampsia durante la gestación:",
                options=["no", "yes"],
                index=0,
                format_func=lambda x: "Sí" if x == "yes" else "No",
                help="Trastorno hipertensivo inducido por el embarazo.",
            )

        with col_neonatal:
            st.markdown("#### Factores Neonatales")

            col_neo_1, col_neo_2 = st.columns(2)
            with col_neo_1:
                ga_input = st.number_input(
                    "Edad Gestacional (semanas):",
                    min_value=24.0,
                    max_value=38.0,
                    value=32.0,
                    step=0.5,
                    help="Semanas completas de gestación al nacimiento.",
                )
                bw_input = st.number_input(
                    "Peso al Nacer (gramos):",
                    min_value=500.0,
                    max_value=3500.0,
                    value=1500.0,
                    step=50.0,
                    help="Peso registrado al nacimiento en la balanza neonatal.",
                )
                apgar5_input = st.slider(
                    "Puntaje APGAR a los 5 minutos:",
                    min_value=0,
                    max_value=10,
                    value=8,
                    step=1,
                    help="Evaluación de vitalidad neonatal a los 5 minutos.",
                )

            with col_neo_2:
                hosp_days_input = st.number_input(
                    "Duración Hospitalización (días):",
                    min_value=0.0,
                    max_value=180.0,
                    value=14.0,
                    step=1.0,
                    help="Días acumulados de estancia hospitalaria en UCIN al alta.",
                )
                o2_days_input = st.number_input(
                    "Duración Oxigenoterapia (días):",
                    min_value=0.0,
                    max_value=120.0,
                    value=1.0,
                    step=0.5,
                    help="Días con soporte de oxígeno suplementario.",
                )

        submit_btn = st.form_submit_button(
            "Evaluar Paciente y Calcular Riesgo", use_container_width=True
        )

    if submit_btn:
        payload: dict[str, Any] = {
            "DM": dm_input,
            "preeclampsia": preeclampsia_input,
            "PregnancyAge": float(ga_input),
            "BirthWeight": float(bw_input),
            "apgar5": int(apgar5_input),
            "duration.hopitalization": float(hosp_days_input),
            "duration.O2": float(o2_days_input),
        }

        with st.spinner("Conectando con el servicio de inferencia clínica..."):
            api_result = call_predict_api(payload, st.session_state.api_url)

        if api_result is not None:
            score = float(api_result.get("score", 0.0))
            categoria = str(api_result.get("categoria", categorize_risk_score(score)))
            recomendacion = str(
                api_result.get("recomendacion", get_clinical_recommendation(categoria))
            )
            contributions = api_result.get("feature_contributions", {})

            st.session_state.last_result = {
                "score": score,
                "categoria": categoria,
                "recomendacion": recomendacion,
                "feature_contributions": contributions,
            }
            st.session_state.last_payload = payload

            nuevo_paciente: dict[str, Any] = {
                "ID Paciente": f"NEO-{len(st.session_state.patient_history) + 1:03d}",
                "Hora Registro": datetime.now().strftime("%H:%M:%S"),
                "Score Riesgo": score,
                "Categoría": categoria,
                "Edad Gestacional (sem)": float(ga_input),
                "Peso al Nacer (g)": float(bw_input),
                "APGAR 5 min": int(apgar5_input),
                "Hosp. (días)": float(hosp_days_input),
                "O2 (días)": float(o2_days_input),
                "DM Materna": "Sí" if dm_input == "yes" else "No",
                "Preeclampsia": "Sí" if preeclampsia_input == "yes" else "No",
                "Recomendación": recomendacion,
            }
            st.session_state.patient_history.append(nuevo_paciente)

    if st.session_state.last_result is not None:
        res = st.session_state.last_result
        score_val = float(res.get("score", 0.0))
        cat_val = str(res.get("categoria", "Desconocido"))
        recom_val = str(res.get("recomendacion", ""))
        contributions_val = res.get("feature_contributions")

        st.markdown("---")
        st.subheader("Resultado de la Evaluación")

        res_col1, res_col2 = st.columns([1, 2])

        with res_col1:
            st.metric(
                label="Puntaje de Riesgo Estimado",
                value=f"{score_val:.1f} / 100",
                delta=f"Nivel: {cat_val}",
                delta_color="inverse" if cat_val in ["Alto", "Crítico"] else "normal",
            )
            st.success("Inferencia procesada exitosamente por la API en vivo.")

        with res_col2:
            st.markdown("#### Recomendación de Acción Clínica")
            if cat_val == "Bajo":
                st.success(f"**Categoría: {cat_val}** — {recom_val}")
            elif cat_val == "Moderado":
                st.info(f"**Categoría: {cat_val}** — {recom_val}")
            elif cat_val == "Alto":
                st.warning(f"**Categoría: {cat_val}** — {recom_val}")
            else:
                st.error(f"**Categoría: {cat_val}** — {recom_val}")

        if isinstance(contributions_val, dict) and contributions_val:
            render_interpretability_chart(contributions_val)

with tab_triage:
    st.subheader("Panel de Triaje y Priorización de Pacientes")
    st.markdown(
        "Listado consolidado de neonatos evaluados durante la sesión clínica, "
        "**ordenados descendentemente por nivel de riesgo** para priorizar "
        "intervenciones y optimizar la asignación de recursos en UCIN."
    )

    if not st.session_state.patient_history:
        st.info(
            "Aún no hay pacientes evaluados en esta sesión. "
            "Diligencie el formulario en la pestaña **Calculadora de Riesgo** "
            "para agregar neonatos al triaje."
        )
    else:
        df_history = pd.DataFrame(st.session_state.patient_history)
        df_history = df_history.sort_values(
            by="Score Riesgo", ascending=False
        ).reset_index(drop=True)

        total_pts = len(df_history)
        criticos = int((df_history["Categoría"] == "Crítico").sum())
        altos = int((df_history["Categoría"] == "Alto").sum())
        moderados = int((df_history["Categoría"] == "Moderado").sum())
        bajos = int((df_history["Categoría"] == "Bajo").sum())
        avg_score = float(df_history["Score Riesgo"].mean())

        kpi1, kpi2, kpi3, kpi4 = st.columns(4)
        with kpi1:
            st.metric("Total en Triaje", f"{total_pts} neonatos")
        with kpi2:
            st.metric(
                "Prioridad Alta / Crítica",
                f"{criticos + altos}",
                delta=f"{((criticos + altos) / total_pts) * 100:.1f}% del total"
                if total_pts > 0
                else None,
                delta_color="inverse",
            )
        with kpi3:
            st.metric("Riesgo Promedio", f"{avg_score:.1f} / 100")
        with kpi4:
            st.metric("Bajo / Moderado", f"{bajos + moderados}")

        st.markdown("---")

        col_filtro, _ = st.columns([2, 1])
        with col_filtro:
            selected_categories = st.multiselect(
                "Filtrar por Categoría de Riesgo:",
                options=["Crítico", "Alto", "Moderado", "Bajo"],
                default=["Crítico", "Alto", "Moderado", "Bajo"],
            )

        filtered_df = df_history[df_history["Categoría"].isin(selected_categories)]

        st.dataframe(
            filtered_df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Score Riesgo": st.column_config.ProgressColumn(
                    "Score Riesgo (0-100)",
                    help="Puntaje continuo de riesgo estimado por el modelo",
                    format="%.1f",
                    min_value=0,
                    max_value=100,
                ),
            },
        )

        csv_data = filtered_df.to_csv(index=False).encode("utf-8")
        timestamp_export = datetime.now().strftime("%Y%m%d_%H%M%S")

        col_export, _ = st.columns([1, 3])
        with col_export:
            st.download_button(
                label="Exportar Triaje (CSV)",
                data=csv_data,
                file_name=f"triaje_neorisk_{timestamp_export}.csv",
                mime="text/csv",
                use_container_width=True,
            )
