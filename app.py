# app.py
# -*- coding: utf-8 -*-

"""
SurfaceXLab
Plataforma CRM para caracterização e otimização de superfícies
"""

import streamlit as st
from supabase import create_client, Client
from pathlib import Path
from PIL import Image


# =========================================================
# LOGO — CARREGAMENTO SEGURO
# =========================================================
BASE_DIR = Path(__file__).parent
LOGO_PATH = BASE_DIR / "assets" / "surfacexlab_logo.png"

logo_image = None
if LOGO_PATH.exists():
    try:
        logo_image = Image.open(LOGO_PATH)
    except Exception:
        logo_image = None


# =========================================================
# CONFIGURAÇÃO DA PÁGINA
# =========================================================
st.set_page_config(
    page_title="SurfaceXLab",
    page_icon=logo_image if logo_image else "🧪",
    layout="wide",
)

st.title("SurfaceXLab — Plataforma de Caracterização e Otimização de Superfícies")


# =========================================================
# CONEXÃO COM SUPABASE
# =========================================================
@st.cache_resource
def init_supabase() -> Client:
    return create_client(
        st.secrets["SUPABASE_URL"],
        st.secrets["SUPABASE_ANON_KEY"],
    )


supabase = init_supabase()


# =========================================================
# IMPORTAÇÃO SEGURA DE MÓDULOS
# =========================================================
def safe_import(module_name: str, func_name: str, optional: bool = False):
    """
    Importa funções de módulos de forma segura.
    Se optional=True, não interrompe a aplicação caso o módulo não exista.
    """
    try:
        module = __import__(module_name, fromlist=[func_name])
        if not hasattr(module, func_name):
            raise AttributeError
        return getattr(module, func_name)
    except Exception:
        if optional:
            return None
        st.error(f"❌ Função `{func_name}` não encontrada em `{module_name}.py`")
        st.stop()


# =========================================================
# IMPORTAÇÃO DOS MÓDULOS ANALÍTICOS
# =========================================================
render_raman_tab = safe_import("raman_tab", "render_raman_tab")

render_mapeamento_molecular_tab = safe_import(
    "mapeamento_molecular_tab",
    "render_mapeamento_molecular_tab",
    optional=True
)

render_resistividade_tab = safe_import(
    "resistividade_tab",
    "render_resistividade_tab",
    optional=True
)

render_tensiometria_tab = safe_import(
    "tensiometria_tab",
    "render_tensiometria_tab",
    optional=True
)

render_ml_tab = safe_import(
    "ml_tab",
    "render_ml_tab",
    optional=True
)

render_dashboard_tab = safe_import(
    "dashboard_tab",
    "render_dashboard_tab",
    optional=True
)


# =========================================================
# SIDEBAR — CRM / CADASTRO DE AMOSTRAS
# =========================================================
with st.sidebar:

    if logo_image:
        st.image(logo_image, use_container_width=True)
        st.divider()

    st.header("📦 Cadastro de Amostra")

    sample_code = st.text_input("Código da Amostra *", key="crm_sample_code")
    material_type = st.text_input("Tipo de Material", key="crm_material_type")
    substrate = st.text_input("Substrato", key="crm_substrate")
    surface_treatment = st.text_input(
        "Tratamento de Superfície",
        key="crm_surface_treatment"
    )
    description = st.text_area("Descrição", key="crm_description")

    if st.button("Salvar Amostra", key="crm_save_button"):

        if not sample_code:
            st.warning("⚠ Código da amostra é obrigatório.")
        else:
            try:
                supabase.table("samples").insert({
                    "sample_code": sample_code,
                    "material_type": material_type,
                    "substrate": substrate,
                    "surface_treatment": surface_treatment,
                    "description": description,
                }).execute()

                st.success("✔ Amostra cadastrada com sucesso!")

            except Exception as e:
                st.error(f"Erro ao salvar no banco: {e}")


# =========================================================
# ABAS PRINCIPAIS
# =========================================================
tabs = st.tabs([
    "🧬 Molecular — Raman",
    "🗺️ Mapeamento Molecular",
    "⚡ Elétrica — Resistividade",
    "💧 Físico-Mecânica — Tensiometria",
    "🤖 Otimizador — PCA + IA",
    "📊 Dashboard"
])


# ---------------------------------------------------------
# RAMAN PADRÃO
# ---------------------------------------------------------
with tabs[0]:
    render_raman_tab(supabase)


# ---------------------------------------------------------
# MAPEAMENTO MOLECULAR RAMAN
# ---------------------------------------------------------
with tabs[1]:

    if render_mapeamento_molecular_tab:
        render_mapeamento_molecular_tab(supabase)
    else:
        st.info("Módulo de mapeamento molecular ainda não implementado.")


# ---------------------------------------------------------
# RESISTIVIDADE
# ---------------------------------------------------------
with tabs[2]:

    if render_resistividade_tab:
        render_resistividade_tab(supabase)
    else:
        st.info("Módulo de resistividade ainda não implementado.")


# ---------------------------------------------------------
# TENSIOMETRIA
# ---------------------------------------------------------
with tabs[3]:

    if render_tensiometria_tab:
        render_tensiometria_tab(supabase)
    else:
        st.info("Módulo de tensiometria ainda não implementado.")


# ---------------------------------------------------------
# OTIMIZAÇÃO IA
# ---------------------------------------------------------
with tabs[4]:

    if render_ml_tab:
        render_ml_tab(supabase)
    else:
        st.info("Módulo de otimização inteligente ainda não implementado.")


# ---------------------------------------------------------
# DASHBOARD
# ---------------------------------------------------------
with tabs[5]:

    if render_dashboard_tab:
        render_dashboard_tab(supabase)
    else:
        st.info("Dashboard ainda não implementado.")
