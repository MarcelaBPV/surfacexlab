# app.py
# -*- coding: utf-8 -*-

"""
SurfaceXLab
Plataforma integrada para caracterização e otimização de superfícies

Módulos:
- Molecular (Raman)
- Elétrica (Resistividade)
- Físico-mecânica (Tensiometria)
- Otimizador (Machine Learning)

Frontend: Streamlit
Backend: Supabase (PostgreSQL)

© Pesquisa & Engenharia
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

st.title("SurfaceXLab — Plataforma Integrada")


# =========================================================
# CONEXÃO COM SUPABASE
# =========================================================
@st.cache_resource(show_spinner=False)
def init_supabase() -> Client:
    """
    Inicializa cliente Supabase de forma segura.
    """
    try:
        url = st.secrets["SUPABASE_URL"]
        key = st.secrets["SUPABASE_ANON_KEY"]
        return create_client(url, key)
    except Exception as e:
        st.error("❌ Erro ao conectar com o Supabase.")
        st.exception(e)
        st.stop()


supabase = init_supabase()


# =========================================================
# IMPORTAÇÃO SEGURA DE MÓDULOS
# =========================================================
def safe_import(module_name: str, func_name: str):
    """
    Importa funções de módulos de forma segura,
    evitando falhas silenciosas no Streamlit.
    """
    try:
        module = __import__(module_name, fromlist=[func_name])
    except Exception as e:
        st.error(f"❌ Erro ao importar `{module_name}.py`")
        st.exception(e)
        st.stop()

    if not hasattr(module, func_name):
        st.error(
            f"❌ Função `{func_name}` não encontrada em `{module_name}.py`.\n\n"
            "➡ Verifique o nome da função e se o arquivo correto foi enviado."
        )
        st.stop()

    return getattr(module, func_name)


render_raman_tab = safe_import("raman_tab", "render_raman_tab")
render_resistividade_tab = safe_import("resistividade_tab", "render_resistividade_tab")
render_tensiometria_tab = safe_import("tensiometria_tab", "render_tensiometria_tab")
render_ml_tab = safe_import("ml_tab", "render_ml_tab")


# =========================================================
# SIDEBAR — CADASTRO DE AMOSTRAS (NÚCLEO DO SISTEMA)
# =========================================================
with st.sidebar:

    if logo_image:
        st.image(logo_image, use_column_width=True)
        st.divider()

    st.header("Cadastro de Amostras")

    sample_code = st.text_input("Código da Amostra *")
    material_type = st.text_input("Tipo de Material")
    substrate = st.text_input("Substrato")
    surface_treatment = st.text_input("Tratamento de Superfície")
    description = st.text_area("Descrição")

    if st.button("Salvar Amostra"):
        if not sample_code:
            st.warning("⚠ O código da amostra é obrigatório.")
        else:
            try:
                payload = {
                    "sample_code": sample_code,
                    "material_type": material_type,
                    "substrate": substrate,
                    "surface_treatment": surface_treatment,
                    "description": description,
                }

                res = supabase.table("samples").insert(payload).execute()

                if res.data:
                    st.success("✔ Amostra cadastrada com sucesso!")
                else:
                    st.error("❌ Erro ao salvar amostra.")

            except Exception as e:
                st.error("❌ Falha ao inserir amostra no banco.")
                st.exception(e)

    st.divider()
    st.caption("SurfaceXLab © Pesquisa & Engenharia")


# =========================================================
# ABAS PRINCIPAIS (MÓDULOS)
# =========================================================
tabs = st.tabs([
    "1 Molecular — Raman",
    "2 Elétrica — Resistividade",
    "3 Físico-Mecânica — Tensiometria",
    "4 Otimizador — IA",
])

with tabs[0]:
    render_raman_tab(supabase)

with tabs[1]:
    render_resistividade_tab(supabase)

with tabs[2]:
    render_tensiometria_tab(supabase)

with tabs[3]:
    render_ml_tab(supabase)
