# app.py
# -*- coding: utf-8 -*-

"""
SurfaceXLab
Plataforma integrada para caracterização e otimização de superfícies
"""

import streamlit as st
from supabase import create_client, Client

# =========================================================
# CONFIGURAÇÃO DA PÁGINA
# =========================================================
st.set_page_config(
    page_title="SurfaceXLab",
    page_icon="",
    layout="wide"
)

st.title("**SurfaceXLab — Plataforma Integrada**")

# =========================================================
# CONEXÃO COM SUPABASE
# =========================================================

@st.cache_resource
def init_supabase():
    url = st.secrets["SUPABASE_URL"]
    key = st.secrets["SUPABASE_ANON_KEY"]
    return create_client(url, key)

supabase: Client = init_supabase()

# =========================================================
# IMPORTAÇÃO DOS MÓDULOS
# =========================================================
from raman_tab import render_raman_tab
from electrical_tab import render_electrical_tab
from physical_tab import render_physical_tab
from optimizer_tab import render_optimizer_tab
import helpers

# =========================================================
# SIDEBAR — CADASTRO DE AMOSTRAS
# =========================================================
with st.sidebar:
    st.header("📦 Cadastro de Amostras")

    sample_code = st.text_input("Código da Amostra")
    material_type = st.text_input("Tipo de Material")
    substrate = st.text_input("Substrato")
    surface_treatment = st.text_input("Tratamento de Superfície")
    description = st.text_area("Descrição")

    if st.button("Salvar Amostra"):
        if not sample_code:
            st.warning("Código da amostra é obrigatório.")
        else:
            data = {
                "sample_code": sample_code,
                "material_type": material_type,
                "substrate": substrate,
                "surface_treatment": surface_treatment,
                "description": description
            }
            supabase.table("samples").insert(data).execute()
            st.success("✔ Amostra cadastrada com sucesso!")

    st.divider()
    st.caption("SurfaceXLab © Pesquisa & Engenharia")

# =========================================================
# ABAS (MÓDULOS)
# =========================================================
tabs = st.tabs([
    "1 Molecular (Raman)",
    "2 Elétrica",
    "3 Físico-Mecânica",
    "4 Otimizador (IA)"
])

with tabs[0]:
    render_raman_tab(supabase, helpers)

with tabs[1]:
    render_electrical_tab(supabase)

with tabs[2]:
    render_physical_tab(supabase)

with tabs[3]:
    render_optimizer_tab(supabase)
