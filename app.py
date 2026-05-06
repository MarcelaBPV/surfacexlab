# =========================================================
# SurfaceXLab — Plataforma Científica Integrada
# Arquitetura Modular | Sample-Centric | Multimodal
# =========================================================

import streamlit as st
import logging
from datetime import datetime


# =========================================================
# IMPORTAÇÃO DOS MÓDULOS
# =========================================================
from raman_tab import render_raman_tab

from resistividade_tab import (
    render_resistividade_tab
)

from tensiometria_tab import (
    render_tensiometria_tab
)

from perfilometria_tab import (
    render_perfilometria_tab
)

from pca_tab import (
    render_pca_tab
)

# =========================================================
# CONFIGURAÇÃO DO APP
# =========================================================
APP_NAME = "SurfaceXLab"

st.set_page_config(
    page_title=APP_NAME,
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =========================================================
# LOGGING
# =========================================================
logging.basicConfig(
    filename="surfacexlab.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

logging.info("Aplicação iniciada")


# =========================================================
# ESTILO GLOBAL
# =========================================================
st.markdown("""
<style>

.block-container {
    padding-top: 1rem;
    padding-bottom: 1rem;
}

h1, h2, h3 {
    font-weight: 600;
}

[data-testid="stMetricValue"] {
    font-size: 22px;
}

[data-testid="stMetricLabel"] {
    font-size: 14px;
}

</style>
""", unsafe_allow_html=True)


# =========================================================
# INICIALIZAÇÃO DO SESSION STATE
# =========================================================
if "samples" not in st.session_state:
    st.session_state.samples = {}

if "logs" not in st.session_state:
    st.session_state.logs = []


# =========================================================
# FUNÇÕES AUXILIARES
# =========================================================
def create_sample(sample_id, material="", treatment=""):
    """
    Cria estrutura centralizada da amostra.
    """

    if sample_id not in st.session_state.samples:

        st.session_state.samples[sample_id] = {

            "metadata": {
                "sample_id": sample_id,
                "material": material,
                "treatment": treatment,
                "created_at": str(datetime.now())
            },

            "raman": {},

            "electrical": {},

            "tensiometry": {},

            "perfilometry": {}
        }

        logging.info(f"Amostra criada: {sample_id}")


def get_total_samples():
    return len(st.session_state.samples)


def count_completed_modules(module_name):
    count = 0

    for sample in st.session_state.samples.values():
        if sample[module_name]:
            count += 1

    return count


# =========================================================
# HEADER
# =========================================================
st.title("🔬 SurfaceXLab")

st.caption(
    "Plataforma integrada para caracterização multimodal "
    "de superfícies e interfaces"
)

st.divider()


# =========================================================
# SIDEBAR — GERENCIAMENTO DE AMOSTRAS
# =========================================================
with st.sidebar:

    st.header("🧪 Gerenciamento de Amostras")

    sample_id = st.text_input("ID da Amostra")

    material = st.text_input("Material")

    treatment = st.text_input("Tratamento")

    if st.button("➕ Criar Amostra"):

        if sample_id.strip():

            create_sample(
                sample_id=sample_id,
                material=material,
                treatment=treatment
            )

            st.success(f"Amostra '{sample_id}' criada.")

        else:
            st.warning("Informe um ID válido.")

    st.divider()

    st.subheader("📂 Amostras Registradas")

    if st.session_state.samples:

        for sample_name in st.session_state.samples.keys():
            st.write(f"• {sample_name}")

    else:
        st.info("Nenhuma amostra cadastrada.")


# =========================================================
# DASHBOARD GERAL
# =========================================================
st.subheader("📊 Visão Geral")

col1, col2, col3, col4, col5 = st.columns(5)

col1.metric(
    "🧪 Amostras",
    get_total_samples()
)

col2.metric(
    "🧬 Raman",
    count_completed_modules("raman")
)

col3.metric(
    "⚡ Elétrico",
    count_completed_modules("electrical")
)

col4.metric(
    "💧 Tensiometria",
    count_completed_modules("tensiometry")
)

col5.metric(
    "📏 Perfilometria",
    count_completed_modules("perfilometry")
)

st.divider()


# =========================================================
# ABAS PRINCIPAIS
# =========================================================
tabs = st.tabs([
    "🧬 Raman",
    "⚡ Resistividade",
    "💧 Tensiometria",
    "📏 Perfilometria",
    "🧠 Integração Multimodal"
])


# =========================================================
# ABA RAMAN
# =========================================================
with tabs[0]:

    try:

        render_raman_tab(st.session_state.samples)

    except Exception as e:

        logging.error(f"Erro módulo Raman: {str(e)}")

        st.error("Erro no módulo Raman.")
        st.exception(e)


# =========================================================
# ABA ELÉTRICA
# =========================================================
with tabs[1]:

    try:

        render_resistividade_tab(st.session_state.samples)

    except Exception as e:

        logging.error(f"Erro módulo elétrico: {str(e)}")

        st.error("Erro no módulo elétrico.")
        st.exception(e)


# =========================================================
# ABA TENSIOMETRIA
# =========================================================
with tabs[2]:

    try:

        render_tensiometria_tab(st.session_state.samples)

    except Exception as e:

        logging.error(f"Erro módulo tensiometria: {str(e)}")

        st.error("Erro no módulo de tensiometria.")
        st.exception(e)


# =========================================================
# ABA PERFILOMETRIA
# =========================================================
with tabs[3]:

    try:

        render_perfilometria_tab(st.session_state.samples)

    except Exception as e:

        logging.error(f"Erro módulo perfilometria: {str(e)}")

        st.error("Erro no módulo de perfilometria.")
        st.exception(e)


# =========================================================
# ABA MULTIMODAL
# =========================================================
with tabs[4]:

    try:

        render_pca_tab()

    except Exception as e:

        logging.error(f"Erro módulo multimodal: {str(e)}")

        st.error("Erro na integração multimodal.")
        st.exception(e)


# =========================================================
# RODAPÉ
# =========================================================
st.divider()

st.caption(
    "SurfaceXLab © Plataforma científica integrada para "
    "caracterização multimodal de superfícies"
)
