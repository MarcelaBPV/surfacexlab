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
    page_icon="",
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
# GLOBAL DARK-TECH THEME + HEADER
# ---------------------------------------------------
st.markdown(
    """
    <style>
        /* ====== LAYOUT GERAL ====== */
        .stApp {
            background-color: #050509;
            color: #f4f4f7;
        }

        /* Remove margens laterais muito claras */
        .block-container {
            padding-top: 0rem;
            padding-bottom: 3rem;
            max-width: 1200px;
        }

        /* ====== HEADER: FAIXA PRETA COM LOGO ====== */
        .header-container {
            background: linear-gradient(90deg, #000000 0%, #060612 60%, #110014 100%);
            padding: 22px 0px 12px 0px;
            width: 100%;
            margin: 0 0 10px 0;
            border-bottom: 1px solid #22222e;
            box-shadow: 0 8px 18px rgba(0, 0, 0, 0.65);
        }

        .header-content {
            display: flex;
            flex-direction: row;
            align-items: center;
            justify-content: flex-start;
            gap: 26px;
            width: 85%;
            margin: auto;
        }

        .header-title {
            color: #ffffff;
            font-size: 40px;
            font-weight: 800;
            letter-spacing: 0.04em;
            margin: 0;
            padding: 0;
        }

        .header-subtitle {
            color: #c3c3d9;
            font-size: 18px;
            margin-top: 4px;
        }

        .header-logo {
            border-radius: 18px;
            box-shadow: 0 0 25px rgba(255, 255, 255, 0.09);
        }

        /* ====== TÍTULOS / TEXTOS ====== */
        h1, h2, h3, h4, h5 {
            color: #f5f5ff !important;
        }

        p, label, span, .markdown-text-container {
            color: #dedeea;
        }

        /* ====== TABS (ABAS) ====== */
        .stTabs [role="tablist"] {
            background-color: #000000;
            padding-left: 40px;
            border-bottom: 1px solid #252536;
        }

        .stTabs [role="tablist"] button {
            background-color: transparent !important;
            color: #a0a0c0 !important;
            border-radius: 0px !important;
            padding: 12px 18px !important;
            font-weight: 500;
            border: none !important;
        }

        .stTabs [role="tablist"] button[aria-selected="true"] {
            border-bottom: 3px solid #ff3259 !important;
            color: #ffffff !important;
        }

        /* ====== CARDS / CONTAINERS ====== */
        .st-emotion-cache-16idsys, .st-emotion-cache-1r6slb0, .st-emotion-cache-1r6slb0 div {
            background-color: #11111a !important;
        }

        /* Dataframes / tabelas */
        .stDataFrame, .stTable {
            background-color: #0c0c14 !important;
        }
        .stDataFrame table, .stTable table {
            color: #e5e5f1 !important;
        }
        .stDataFrame thead tr, .stTable thead tr {
            background-color: #1a1a26 !important;
        }

        /* ====== INPUTS / SELECTS / SLIDERS ====== */
        input, textarea {
            background-color: #12121c !important;
            color: #f0f0ff !important;
            border-radius: 6px !important;
            border: 1px solid #303049 !important;
        }
        .stSelectbox div[data-baseweb="select"] > div {
            background-color: #12121c !important;
            color: #f0f0ff !important;
            border-radius: 6px !important;
            border: 1px solid #303049 !important;
        }

        /* Sliders */
        .stSlider > div > div > div > div {
            background: linear-gradient(90deg, #8b5bff, #ff3259) !important;
        }
        .stSlider > div > div > div > div > div[role="slider"] {
            background: #ffffff !important;
            box-shadow: 0 0 10px rgba(255, 50, 89, 0.8);
        }

        /* File uploader */
        [data-testid="stFileUploader"] section {
            background-color: #12121c !important;
            border-radius: 10px !important;
            border: 1px dashed #3c3c55 !important;
            color: #d0d0e2 !important;
        }

        /* ====== BOTÕES ====== */
        .stButton>button {
            background-image: linear-gradient(135deg, #8b5bff, #ff3259);
            color: white;
            border-radius: 999px;
            border: none;
            padding: 0.5rem 1.3rem;
            font-weight: 600;
            box-shadow: 0 0 12px rgba(255, 50, 89, 0.5);
        }
        .stButton>button:hover {
            filter: brightness(1.1);
            box-shadow: 0 0 18px rgba(255, 50, 89, 0.8);
        }

        /* ====== CHECKBOX / RADIO ====== */
        .stCheckbox label, .stRadio label {
            color: #dedeea !important;
        }

        /* ====== FOOTER ====== */
        footer, .st-footer {
            background: transparent !important;
        }
    </style>

    <div class="header-container">
        <div class="header-content">
            <img src="SurfaceXLab Lettermark Logo - High-Tech Aesthetic.png" width="140px" class="header-logo">

            <div>
                <div class="header-title">SurfaceXLab</div>
                <div class="header-subtitle">Advanced Material Surface Intelligence</div>
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


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
