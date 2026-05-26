# =========================================================
# SurfaceXLab — Plataforma Científica Integrada
# Arquitetura Modular | Sample-Centric | Multimodal
# VERSÃO FINAL CORRIGIDA
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

# =========================================================
# PCA GERAL
# =========================================================

from pca_tab import (
    render_pca_tab
)

# =========================================================
# PCA WEG
# =========================================================

from pca_weg import (
    render_pca_weg
)


# =========================================================
# CONFIGURAÇÃO DO APP
# =========================================================

APP_NAME = "SurfaceXLab"

st.set_page_config(

    page_title=APP_NAME,

    page_icon="",

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
# SESSION STATE
# =========================================================

if "samples" not in st.session_state:

    st.session_state.samples = {}

if "logs" not in st.session_state:

    st.session_state.logs = []

if "raman_samples" not in st.session_state:

    st.session_state.raman_samples = {}

if "electrical_samples" not in st.session_state:

    st.session_state.electrical_samples = {}

if "tensiometria_samples" not in st.session_state:

    st.session_state.tensiometria_samples = {}

if "perfilometria_samples" not in st.session_state:

    st.session_state.perfilometria_samples = {}


# =========================================================
# FUNÇÕES AUXILIARES
# =========================================================

def create_sample(

    sample_id,

    material="",

    treatment=""
):

    if sample_id not in st.session_state.samples:

        st.session_state.samples[sample_id] = {

            "metadata": {

                "sample_id": sample_id,

                "material": material,

                "treatment": treatment,

                "created_at": str(datetime.now())
            }
        }

        logging.info(
            f"Amostra criada: {sample_id}"
        )


def get_total_samples():

    return len(
        st.session_state.samples
    )


def get_total_module(module_key):

    data = st.session_state.get(

        module_key,

        {}
    )

    return len(data)


# =========================================================
# HEADER
# =========================================================

st.title("*SurfaceXLab*")

st.caption(

    "Plataforma integrada para caracterização "
    "multimodal de superfícies e interfaces"
)

st.divider()


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.header("🧪 Gerenciamento de Amostras")

    sample_id = st.text_input(
        "ID da Amostra"
    )

    material = st.text_input(
        "Material"
    )

    treatment = st.text_input(
        "Tratamento"
    )

    if st.button("➕ Criar Amostra"):

        if sample_id.strip():

            create_sample(

                sample_id=sample_id,

                material=material,

                treatment=treatment
            )

            st.success(
                f"Amostra '{sample_id}' criada."
            )

        else:

            st.warning(
                "Informe um ID válido."
            )

    st.divider()

    st.subheader("📂 Amostras Registradas")

    if st.session_state.samples:

        for sample_name in st.session_state.samples.keys():

            st.write(f"• {sample_name}")

    else:

        st.info(
            "Nenhuma amostra cadastrada."
        )


# =========================================================
# DASHBOARD
# =========================================================

st.subheader("*Visão Geral*")

col1, col2, col3, col4, col5, col6 = st.columns(6)

col1.metric(

    "🧪 Amostras",

    get_total_samples()
)

col2.metric(

    "🧬 Raman",

    get_total_module(
        "raman_samples"
    )
)

col3.metric(

    "⚡ Elétrico",

    get_total_module(
        "electrical_samples"
    )
)

col4.metric(

    "💧 Tensiometria",

    get_total_module(
        "tensiometria_samples"
    )
)

col5.metric(

    "📏 Perfilometria",

    get_total_module(
        "perfilometria_samples"
    )
)

col6.metric(

    "📊 PCA",

    2
)

st.divider()


# =========================================================
# ABAS
# =========================================================

tabs = st.tabs([

    "1 Raman",

    "2 Resistividade",

    "3 Tensiometria",

    "4 Perfilometria",

    "5 PCA Multimodal",

    "6 PCA WEG"
])


# =========================================================
# ABA RAMAN
# =========================================================

with tabs[0]:

    try:

        render_raman_tab()

    except Exception as e:

        logging.error(
            f"Erro módulo Raman: {str(e)}"
        )

        st.error(
            "Erro no módulo Raman."
        )

        st.exception(e)


# =========================================================
# ABA ELÉTRICA
# =========================================================

with tabs[1]:

    try:

        render_resistividade_tab()

    except Exception as e:

        logging.error(
            f"Erro módulo elétrico: {str(e)}"
        )

        st.error(
            "Erro no módulo elétrico."
        )

        st.exception(e)


# =========================================================
# ABA TENSIOMETRIA
# =========================================================

with tabs[2]:

    try:

        render_tensiometria_tab()

    except Exception as e:

        logging.error(
            f"Erro módulo tensiometria: {str(e)}"
        )

        st.error(
            "Erro no módulo de tensiometria."
        )

        st.exception(e)


# =========================================================
# ABA PERFILOMETRIA
# =========================================================

with tabs[3]:

    try:

        render_perfilometria_tab()

    except Exception as e:

        logging.error(
            f"Erro módulo perfilometria: {str(e)}"
        )

        st.error(
            "Erro no módulo de perfilometria."
        )

        st.exception(e)


# =========================================================
# ABA PCA GERAL
# =========================================================

with tabs[4]:

    try:

        render_pca_tab()

    except Exception as e:

        logging.error(
            f"Erro módulo PCA geral: {str(e)}"
        )

        st.error(
            "Erro na integração multimodal."
        )

        st.exception(e)


# =========================================================
# ABA PCA WEG
# =========================================================

with tabs[5]:

    try:

        render_pca_weg()

    except Exception as e:

        logging.error(
            f"Erro módulo PCA WEG: {str(e)}"
        )

        st.error(
            "Erro no módulo PCA WEG."
        )

        st.exception(e)


# =========================================================
# RODAPÉ
# =========================================================

st.divider()

st.caption(

    "SurfaceXLab © Plataforma científica integrada "
    "para caracterização multimodal de superfícies"
)
