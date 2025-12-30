# app.py
# -*- coding: utf-8 -*-

"""
SurfaceXLab
Plataforma integrada para caracterização e otimização de superfícies

Módulos:
- Raman (molecular)
- Resistividade elétrica
- Tensiometria / Físico-mecânica
- Otimizador (Machine Learning)

Frontend: Streamlit
Backend: Supabase (PostgreSQL)
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

st.title("*SurfaceXLab — Plataforma Integrada*")

# =========================================================
# CONEXÃO COM SUPABASE
# =========================================================
@st.cache_resource
def init_supabase() -> Client:
    url = st.secrets["SUPABASE_URL"]
    key = st.secrets["SUPABASE_ANON_KEY"]
    return create_client(url, key)

supabase = init_supabase()

# =========================================================
# IMPORTAÇÃO DOS MÓDULOS (NOMES REAIS DO REPO)
# =========================================================
from raman_tab import render_raman_tab
from resistividade_tab import render_resistividade_tab
from tensiometria_tab import render_tensiometria_tab
from ml_tab import render_ml_tab

# =========================================================
# SIDEBAR — CADASTRO DE AMOSTRAS (NÚCLEO DO SISTEMA)
# =========================================================
with st.sidebar:
    st.header("📦 Cadastro de Amostras")

    sample_code = st.text_input("Código da Amostra *")
    material_type = st.text_input("Tipo de Material")
    substrate = st.text_input("Substrato")
    surface_treatment = st.text_input("Tratamento de Superfície")
    description = st.text_area("Descrição")

    if st.button("Salvar Amostra"):
        if not sample_code:
            st.warning("O código da amostra é obrigatório.")
        else:
            data = {
                "sample_code": sample_code,
                "material_type": material_type,
                "substrate": substrate,
                "surface_treatment": surface_treatment,
                "description": description
            }

            res = supabase.table("samples").insert(data).execute()

            if res.data:
                st.success("✔ Amostra cadastrada com sucesso!")
            else:
                st.error("Erro ao salvar amostra.")

    st.divider()
    st.caption("SurfaceXLab © Pesquisa & Engenharia")

# =========================================================
# ABAS (MÓDULOS)
# =========================================================
tabs = st.tabs([
    "1 Molecular — Raman",
    "2 Elétrica — Resistividade",
    "3 Físico-Mecânica — Tensiometria",
    "4 Otimizador — IA"
])

with tabs[0]:
    render_raman_tab(supabase)

with tabs[1]:
    render_resistividade_tab(supabase)

with tabs[2]:
    render_tensiometria_tab(supabase)

with tabs[3]:
    render_ml_tab(supabase)
