import httpx
import streamlit as st

st.set_page_config(
    page_title="MLOps Prediction Dashboard",
    page_icon="🤖",
    layout="wide",
)

st.title("🤖 MLOps Prediction Dashboard")
st.subheader("Plataforma de Diagnóstico e Inferencia")

api_url = st.sidebar.text_input("URL del Backend API", "http://localhost:8000")

st.markdown("### Estado del Sistema")
if st.button("Verificar Conexión con la API"):
    try:
        response = httpx.get(f"{api_url}/health", timeout=3.0)
        if response.status_code == 200:
            st.success(f"Conexión Exitosa: {response.json()}")
        else:
            st.error(f"Error HTTP {response.status_code}: {response.text}")
    except Exception as e:
        st.error(f"No se pudo conectar a la API: {e}")
