# app.py
# -*- coding: utf-8 -*-
"""
SurfaceXLab - Plataforma genérica para análise de superfície de materiais

Abas:
1) Raman (cadastro de pacientes + import Google Forms + espectros Raman)
2) Tensiometria (ângulo de contato)
3) Resistividade (4 pontas)
4) Otimização ML (Machine Learning)
"""

import streamlit as st
from typing import Optional

# Supabase (opcional)
try:
    from supabase import create_client, Client
except Exception:
    create_client = None
    Client = None

# Abas (arquivos separados)
from raman_tab import render_raman_tab
from tensiometria_tab import render_tensiometria_tab
from resistividade_tab import render_resistividade_tab
from ml_tab import render_ml_tab

# ---------------------------------------------------
# CONFIG GERAL + BRANDING
# ---------------------------------------------------
st.set_page_config(
    page_title="SurfaceXLab — Advanced Material Surface Intelligence",
    layout="wide",
    page_icon="🧪",
)

col_logo, col_title = st.columns([1, 3])

with col_logo:
    try:
        st.image(
            "SurfaceXLab Lettermark Logo - High-Tech Aesthetic.png",
            use_column_width=True,
        )
    except Exception:
        st.write("SurfaceXLab Logo")

with col_title:
    st.markdown("# SurfaceXLab")
    st.markdown("#### *Advanced Material Surface Intelligence*")

st.markdown("---")

# ---------------------------------------------------
# CONEXÃO SUPABASE
# ---------------------------------------------------
def init_supabase() -> Optional["Client"]:
    if not hasattr(st, "secrets"):
        st.warning("st.secrets não encontrado. Configure SUPABASE_URL e SUPABASE_KEY.")
        return None

    url = st.secrets.get("SUPABASE_URL")
    key = st.secrets.get("SUPABASE_KEY")
    if not (url and key and create_client):
        st.info("Configure SUPABASE_URL e SUPABASE_KEY em st.secrets para habilitar o banco.")
        return None

    try:
        client = create_client(url, key)
        return client
    except Exception as e:
        st.error(f"Erro conectando ao Supabase: {e}")
        return None


supabase = init_supabase()

# ---------------------------------------------------
# ABAS PRINCIPAIS
# ---------------------------------------------------
tab_raman, tab_tensio, tab_resist, tab_ml = st.tabs(
    [
        "1️⃣ Raman",
        "2️⃣ Tensiometria",
        "3️⃣ Resistividade",
        "4️⃣ Otimização ML",
    ]
)

with tab_raman:
    render_raman_tab(supabase)

with tab_tensio:
    render_tensiometria_tab(supabase)

with tab_resist:
    render_resistividade_tab(supabase)

with tab_ml:
    render_ml_tab(supabase)

# ---------------------------------------------------
# FOOTER
# ---------------------------------------------------
st.markdown("---")
st.caption(
    """
SurfaceXLab — Advanced Material Surface Intelligence  
© 2025 — Desenvolvido por Marcela Veiga para pesquisa e desenvolvimento em caracterização de superfícies.
"""
)
