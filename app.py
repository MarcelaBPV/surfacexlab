# =========================================================
# SurfaceXLab — Plataforma Científica Integrada (BLINDADO)
# =========================================================

import streamlit as st


# =========================================================
# IMPORTAÇÃO DOS MÓDULOS
# =========================================================
from raman_tab import render_raman_tab
from resistividade_tab import render_resistividade_tab
from tensiometria_tab import render_tensiometria_tab
from perfilometria_tab import render_perfilometria_tab
from analise_completa_amostras_tab import render_analise_completa_amostras_tab


# =========================================================
# CONFIGURAÇÃO DA PÁGINA
# =========================================================
st.set_page_config(
    page_title="SurfaceXLab",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="collapsed"
)


# =========================================================
# ESTILO
# =========================================================
st.markdown("""
<style>
.block-container {
    padding-top: 1.2rem;
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
# HEADER
# =========================================================
st.markdown("# 🔬 SurfaceXLab")
st.caption("Plataforma integrada para caracterização avançada de superfícies")
st.divider()


# =========================================================
# SESSION STATE INIT
# =========================================================
if "raman_peaks" not in st.session_state:
    st.session_state.raman_peaks = {}

if "electrical_samples" not in st.session_state:
    st.session_state.electrical_samples = {}

if "tensiometry_samples" not in st.session_state:
    st.session_state.tensiometry_samples = {}

if "perfilometria_samples" not in st.session_state:
    st.session_state.perfilometria_samples = {}


# =========================================================
# DASHBOARD
# =========================================================
st.subheader("📊 Visão Geral dos Ensaios")

col1, col2, col3, col4 = st.columns(4)

col1.metric("🧬 Raman", len(st.session_state.raman_peaks))
col2.metric("⚡ Elétrico", len(st.session_state.electrical_samples))
col3.metric("💧 Tensiometria", len(st.session_state.tensiometry_samples))
col4.metric("📏 Perfilometria", len(st.session_state.perfilometria_samples))

st.divider()


# =========================================================
# ABAS
# =========================================================
tabs = st.tabs([
    "🧬 Raman",
    "⚡ Resistividade (4 Pontas)",
    "💧 Tensiometria",
    "📏 Perfilometria",
    "🧠 Análise Integrada"
])


# =========================================================
# ABA 1 — RAMAN (NÃO PODE QUEBRAR)
# =========================================================
with tabs[0]:
    try:
        render_raman_tab()
    except Exception as e:
        st.error("Erro no módulo Raman")
        st.exception(e)


# =========================================================
# ABA 2 — RESISTIVIDADE (AQUI ESTAVA QUEBRANDO TUDO)
# =========================================================
with tabs[1]:
    try:
        render_resistividade_tab()
    except Exception as e:
        st.error("Erro no módulo elétrico (isso NÃO trava mais o app)")
        st.exception(e)


# =========================================================
# ABA 3 — TENSIOMETRIA
# =========================================================
with tabs[2]:
    try:
        render_tensiometria_tab()
    except Exception as e:
        st.error("Erro no módulo de tensiometria")
        st.exception(e)


# =========================================================
# ABA 4 — PERFILOMETRIA
# =========================================================
with tabs[3]:
    try:
        render_perfilometria_tab()
    except Exception as e:
        st.error("Erro no módulo de perfilometria")
        st.exception(e)


# =========================================================
# ABA 5 — ANÁLISE INTEGRADA
# =========================================================
with tabs[4]:
    try:
        render_analise_completa_amostras_tab()
    except Exception as e:
        st.error("Erro na análise integrada")
        st.exception(e)


# =========================================================
# RODAPÉ
# =========================================================
st.divider()
st.caption("SurfaceXLab © Plataforma científica integrada para análise de superfícies")
