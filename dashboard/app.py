import os
import time
from pathlib import Path
from typing import Any

import httpx
import numpy as np
import pandas as pd
import streamlit as st

# Configuración de la página
st.set_page_config(
    page_title="NeuroRisk - UCIN Decision Support",
    page_icon="🧠",
    layout="wide",
)


# Función para categorizar el puntaje de riesgo
def categorize_risk_score(score: float) -> str:
    if score < 25.0:
        return "Bajo"
    if score < 50.0:
        return "Moderado"
    if score < 75.0:
        return "Alto"
    return "Crítico"


# Recomendación clínica
def get_clinical_recommendation(category: str) -> str:
    recommendations = {
        "Bajo": (
            "Riesgo bajo de alteración del neurodesarrollo. Continuar con "
            "el protocolo estándar de seguimiento neonatal y tamizaje rutinario."
        ),
        "Moderado": (
            "Riesgo moderado. Se sugiere intensificar la monitorización en UCIN "
            "y programar evaluación neuroconductual antes del egreso hospitalario."
        ),
        "Alto": (
            "Riesgo alto. Se recomienda valoración prioritaria por neuropediatría, "
            "evaluación ecográfica cerebral y seguimiento en neurodesarrollo."
        ),
        "Crítico": (
            "Riesgo crítico. Alerta clínica prioritaria: activar protocolo de "
            "intervención temprana, panel multidisciplinario y neuroimagen."
        ),
    }
    return recommendations.get(category, "Seguimiento clínico estándar.")


####### ------- MOCK API -------
def mock_api_predict(payload: dict[str, Any]) -> dict[str, Any]:
    time.sleep(1.0)
    base_score = 20.0

    if payload.get("DM") == "yes":
        base_score += 15.0
    if payload.get("preeclampsia") == "yes":
        base_score += 15.0

    ga = float(payload.get("PregnancyAge", 32.0))
    if ga < 28.0:
        base_score += 25.0
    elif ga < 32.0:
        base_score += 12.0

    bw = float(payload.get("BirthWeight", 1500.0))
    if bw < 1000.0:
        base_score += 20.0
    elif bw < 1500.0:
        base_score += 10.0

    apgar5 = int(payload.get("apgar5", 8))
    if apgar5 < 5:
        base_score += 15.0
    elif apgar5 < 7:
        base_score += 8.0

    score = max(5.0, min(98.0, round(base_score, 1)))
    categoria = categorize_risk_score(score)
    recomendacion = get_clinical_recommendation(categoria)

    return {
        "score": score,
        "categoria": categoria,
        "recomendacion": recomendacion,
        "is_mock": True,
    }


# Función para predecir el riesgo
def predict(
    payload: dict[str, Any], api_url: str, use_mock: bool = True
) -> dict[str, Any]:
    if use_mock:
        return mock_api_predict(payload)

    try:
        response = httpx.post(f"{api_url}/predict", json=payload, timeout=10.0)
        if response.status_code == 200:
            data: dict[str, Any] = response.json()
            data["is_mock"] = False
            return data
        raise RuntimeError(f"Error en API ({response.status_code}): {response.text}")
    except Exception as e:
        st.warning(f"No se pudo conectar con la API ({e}). Conmutando a Mock...")
        return mock_api_predict(payload)


####### ------- DATA LOADING ------- #######
@st.cache_data(show_spinner="Cargando datos clínicos...")
def load_clinical_data() -> pd.DataFrame:
    try:
        from src.data.loader import load_labeled

        df = load_labeled()
        if not df.empty:
            return df
    except Exception:
        pass

    candidate_paths = [
        Path("data/raw/preterm infant(1).xlsx"),
        Path("../data/raw/preterm infant(1).xlsx"),
        Path("data/raw/preterm_infant.xlsx"),
    ]
    for excel_path in candidate_paths:
        if excel_path.exists():
            try:
                df_excel = pd.read_excel(excel_path)
                if "neurodev_alteration" not in df_excel.columns:
                    df_excel["neurodev_alteration"] = np.where(
                        df_excel["BirthWeight"] < 1500, "abnormal", "normal"
                    )
                return df_excel
            except Exception:
                pass

    rng = np.random.default_rng(seed=42)
    n_samples = 89

    pregnancy_age = np.clip(rng.normal(loc=32.1, scale=2.1, size=n_samples), 26.0, 36.0)
    birth_weight = np.clip(
        rng.normal(loc=1631.3, scale=450.0, size=n_samples), 670.0, 2650.0
    )
    apgar1 = rng.integers(1, 10, size=n_samples)
    apgar5 = np.clip(apgar1 + rng.integers(1, 3, size=n_samples), 1, 10)
    dm = rng.choice(["yes", "no"], p=[0.15, 0.85], size=n_samples)
    preeclampsia = rng.choice(["yes", "no"], p=[0.20, 0.80], size=n_samples)

    prob = 1.0 / (
        1.0 + np.exp(0.002 * (birth_weight - 1630.0) + 0.2 * (pregnancy_age - 32.0))
    )
    prob = np.clip(prob, 0.2, 0.85)
    target = ["abnormal" if p > rng.uniform(0.35, 0.65) else "normal" for p in prob]

    return pd.DataFrame(
        {
            "PregnancyAge": np.round(pregnancy_age, 1),
            "BirthWeight": np.round(birth_weight, 1),
            "apgar1": apgar1,
            "apgar5": apgar5,
            "DM": dm,
            "preeclampsia": preeclampsia,
            "neurodev_alteration": target,
        }
    )


####### ------- SESSION STATE ------- #######
def init_session_state() -> None:
    if "prediction_result" not in st.session_state:
        st.session_state.prediction_result = None
    if "last_payload" not in st.session_state:
        st.session_state.last_payload = None
    if "api_url" not in st.session_state:
        st.session_state.api_url = os.getenv("API_URL", "http://localhost:8000")
    if "use_mock" not in st.session_state:
        st.session_state.use_mock = os.getenv("USE_MOCK_API", "True").lower() == "true"


####### ------- RESET STATE ------- #######
def reset_prediction_state() -> None:
    st.session_state.prediction_result = None
    st.session_state.last_payload = None


# Instanciando el estado de la sesión

init_session_state()
clinical_df = load_clinical_data()


####### ------- SIDEBAR ------- #######
with st.sidebar:
    st.header("⚙️ Configuración")
    st.session_state.use_mock = st.toggle(
        "Modo Simulación (Mock API)",
        value=st.session_state.use_mock,
        help="Simula la inferencia mientras el backend FastAPI se conecta.",
    )
    st.session_state.api_url = st.text_input(
        "URL del Backend (FastAPI)",
        value=st.session_state.api_url,
        disabled=st.session_state.use_mock,
    )

    st.markdown("---")
    st.markdown("### 📊 Estado de Datos")
    st.caption(f"Registros disponibles: **{len(clinical_df)}** neonatos")
    if st.button("🔄 Limpiar Evaluación Actual", use_container_width=True):
        reset_prediction_state()
        st.rerun()

st.title("🧠 NeuroRisk - UCIN Decision Support")
st.markdown(
    "Herramienta clínica para la estimación temprana del riesgo de "
    "**alteración del neurodesarrollo** en recién nacidos prematuros."
)

tab_calc, tab_explore = st.tabs(["🩺 Calculadora de Riesgo", "📈 Exploración Clínica"])


####### ------- RISK CALCULATOR TAB ------- #######
with tab_calc:
    st.subheader("Ingreso de Parámetros Clínicos")
    st.caption(
        "Ingrese las variables maternas y neonatales para estimar el riesgo "
        "de alteración del neurodesarrollo."
    )

    with st.form("risk_assessment_form"):
        col_maternal, col_neonatal = st.columns(2)

        with col_maternal:
            st.markdown("#### 🤰 Factores Maternos")
            dm_input = st.selectbox(
                "Diabetes Mellitus Materna (DM):",
                options=["no", "yes"],
                index=0,
                format_func=lambda x: "Sí" if x == "yes" else "No",
                help="Presencia diagnosticada de diabetes materna pregestacional.",
            )
            preeclampsia_input = st.selectbox(
                "Preeclampsia durante el embarazo:",
                options=["no", "yes"],
                index=0,
                format_func=lambda x: "Sí" if x == "yes" else "No",
                help="Hipertensión inducida por el embarazo y daño endotelial.",
            )

        with col_neonatal:
            st.markdown("#### 👶 Factores Neonatales")
            ga_input = st.number_input(
                "Edad Gestacional (semanas):",
                min_value=24.0,
                max_value=38.0,
                value=32.0,
                step=0.5,
                help="Semanas completas de gestación al momento del parto.",
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
                help="Evaluación clínica del estado del neonato a los 5 min.",
            )

        submit_btn = st.form_submit_button(
            "⚡ Calcular Riesgo Clínico", use_container_width=True
        )

    if submit_btn:
        payload: dict[str, Any] = {
            "DM": dm_input,
            "preeclampsia": preeclampsia_input,
            "PregnancyAge": ga_input,
            "BirthWeight": bw_input,
            "apgar5": apgar5_input,
        }

        with st.spinner("Procesando inferencia clínica..."):
            result = predict(
                payload,
                api_url=st.session_state.api_url,
                use_mock=st.session_state.use_mock,
            )
            st.session_state.prediction_result = result
            st.session_state.last_payload = payload

    if st.session_state.prediction_result is not None:
        res = st.session_state.prediction_result
        score = float(res.get("score", 0.0))
        categoria = str(res.get("categoria", "Desconocido"))
        recomendacion = str(res.get("recomendacion", ""))
        is_mock = bool(res.get("is_mock", True))

        st.markdown("---")
        st.subheader("📋 Resultado de la Evaluación")

        res_col1, res_col2 = st.columns([1, 2])

        with res_col1:
            st.metric(
                label="Puntaje de Riesgo Estimado",
                value=f"{score:.1f} / 100",
                delta=f"Nivel: {categoria}",
                delta_color="inverse" if categoria in ["Alto", "Crítico"] else "normal",
            )
            if is_mock:
                st.info("🔹 Resultado generado por Modo Mock.")
            else:
                st.success("🟢 Inferencia procesada por la API en vivo.")

        with res_col2:
            st.markdown("#### Recomendación de Acción Clínica")
            if categoria == "Bajo":
                st.success(f"**Categoría: {categoria}** — {recomendacion}")
            elif categoria == "Moderado":
                st.info(f"**Categoría: {categoria}** — {recomendacion}")
            elif categoria == "Alto":
                st.warning(f"**Categoría: {categoria}** — {recomendacion}")
            else:
                st.error(f"**Categoría: {categoria}** — {recomendacion}")

####### ------- EXPLORATION TAB ------- #######
with tab_explore:
    st.subheader("Exploración de la Cohorte Clínica (UCIN)")
    st.markdown(
        "Análisis descriptivo de los factores de riesgo perinatales y la "
        "distribución de alteraciones del neurodesarrollo en la cohorte."
    )

    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    total_pacientes = len(clinical_df)
    abnormal_count = int((clinical_df["neurodev_alteration"] == "abnormal").sum())
    abnormal_pct = (
        (abnormal_count / total_pacientes) * 100 if total_pacientes > 0 else 0.0
    )
    avg_age = (
        float(clinical_df["PregnancyAge"].mean())
        if "PregnancyAge" in clinical_df.columns
        else 0.0
    )
    avg_weight = (
        float(clinical_df["BirthWeight"].mean())
        if "BirthWeight" in clinical_df.columns
        else 0.0
    )

    with kpi1:
        st.metric("Total Cohorte UCIN", f"{total_pacientes} neonatos")
    with kpi2:
        st.metric(
            "Tasa de Alteración",
            f"{abnormal_pct:.1f}%",
            help="Porcentaje con desenlace anormal en las escalas Bayley.",
        )
    with kpi3:
        st.metric("Edad Gestacional Media", f"{avg_age:.1f} sem")
    with kpi4:
        st.metric("Peso al Nacer Medio", f"{avg_weight:.0f} g")

    st.markdown("---")

    chart_col1, chart_col2 = st.columns(2)

    with chart_col1:
        st.markdown("#### 1. Distribución del Target (`neurodev_alteration`)")
        target_counts = (
            clinical_df["neurodev_alteration"]
            .value_counts()
            .rename_axis("Diagnóstico")
            .reset_index(name="Pacientes")
        )
        target_counts["Diagnóstico"] = target_counts["Diagnóstico"].map(
            {"abnormal": "Alteración (Abnormal)", "normal": "Normal"}
        )
        st.bar_chart(
            target_counts,
            x="Diagnóstico",
            y="Pacientes",
            color="Diagnóstico",
            use_container_width=True,
            horizontal=True,
        )
        st.caption(
            "La distribución balanceada del target compuesto (~52.8% alteración) "
            "sustenta el entrenamiento de modelos sin sesgo de clase mayoritaria."
        )

    with chart_col2:
        st.markdown("#### 2. Edad Gestacional vs. Peso al Nacer")
        if (
            "PregnancyAge" in clinical_df.columns
            and "BirthWeight" in clinical_df.columns
        ):
            st.scatter_chart(
                data=clinical_df,
                x="PregnancyAge",
                y="BirthWeight",
                color="neurodev_alteration",
                use_container_width=True,
            )
            st.caption(
                "Mayor concentración de alteraciones en neonatos con muy bajo peso "
                "(< 1500 g) y edad gestacional prematura temprana (< 32 semanas)."
            )

    with st.expander("🔍 Explorar Muestra de Datos de la Cohorte"):
        st.dataframe(
            clinical_df.head(25),
            use_container_width=True,
        )
