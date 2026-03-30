# =========================================================
# SurfaceXLab — Plataforma Científica Integrada
# =========================================================

import streamlit as st

# =========================================================
# IMPORTAÇÃO DOS MÓDULOS
# =========================================================
from raman_tab import render_raman_tab
from resistividade_tab import render_resistividade_tab
from tensiometria_tab import render_tensiometria_tab
from analise_completa_amostras_tab import render_analise_completa_amostras_tab


# =========================================================
# CONFIGURAÇÃO DA PÁGINA
# =========================================================
st.set_page_config(
    page_title="SurfaceXLab",
    page_icon="🔬",
    layout="wide"
)

# =========================================================
# ESTILO (visual mais limpo e profissional)
# =========================================================
st.markdown("""
<style>
.block-container {
    padding-top: 1.2rem;
}
h1, h2, h3 {
    font-weight: 600;
}
</style>
""", unsafe_allow_html=True)


# =========================================================
# HEADER
# =========================================================
st.markdown("# 🔬 SurfaceXLab")
st.caption("Plataforma integrada para caracterização de superfícies")
st.divider()


# =========================================================
# DASHBOARD INICIAL (VISÃO GERAL)
# =========================================================
col1, col2, col3 = st.columns(3)

raman_count = len(st.session_state.get("raman_peaks", {}))
electrical_count = len(st.session_state.get("electrical_samples", {}))
tensiometry_count = len(st.session_state.get("tensiometry_samples", {}))

col1.metric("🧬 Ensaios Raman", raman_count)
col2.metric("⚡ Ensaios Elétricos", electrical_count)
col3.metric("💧 Ensaios Tensiometria", tensiometry_count)

st.divider()


# =========================================================
# ABAS PRINCIPAIS
# =========================================================
tabs = st.tabs([
    "🧬 Raman",
    "⚡ Resistividade",
    "💧 Tensiometria",
    "🧠 Análise Integrada"
])


# =========================================================
# ABA 1 — RAMAN
# =========================================================
with tabs[0]:
    render_raman_tab()


# =========================================================
# ABA 2 — RESISTIVIDADE
# =========================================================
with tabs[1]:
    render_resistividade_tab()


# =========================================================
# ABA 3 — TENSIOMETRIA
# =========================================================
with tabs[2]:
    render_tensiometria_tab()


# =========================================================
# ABA 4 — ANÁLISE COMPLETA
# =========================================================
with tabs[3]:
    render_analise_completa_amostras_tab()


# =========================================================
# RODAPÉ
# =========================================================
st.divider()
st.caption("SurfaceXLab © Plataforma científica para análise de superfícies")
