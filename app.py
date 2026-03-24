# =========================================================
# SurfaceXLab
# =========================================================

import streamlit as st
from supabase import create_client, Client
from pathlib import Path
from PIL import Image


# =========================================================
# CONFIG
# =========================================================

BASE_DIR = Path(__file__).parent
ASSETS_DIR = BASE_DIR / "assets"
LOGO_PATH = ASSETS_DIR / "surfacexlab_logo.png"


def load_logo():
    if LOGO_PATH.exists():
        try:
            return Image.open(LOGO_PATH)
        except:
            return None
    return None


logo_image = load_logo()


st.set_page_config(
    page_title="SurfaceXLab",
    page_icon=logo_image if logo_image else "X",
    layout="wide"
)

st.title("SurfaceXLab — Plataforma de Caracterização de Superfícies")


# =========================================================
# SUPABASE
# =========================================================

@st.cache_resource
def init_supabase() -> Client:
    return create_client(
        st.secrets["SUPABASE_URL"],
        st.secrets["SUPABASE_ANON_KEY"]
    )


supabase = init_supabase()


# =========================================================
# IMPORT SEGURO
# =========================================================

def safe_import(module, func, optional=False):

    try:
        mod = __import__(module, fromlist=[func])
        return getattr(mod, func)

    except:
        if optional:
            return None

        st.error(f"Erro ao carregar {module}")
        st.stop()


# =========================================================
# IMPORTS
# =========================================================

render_raman_tab = safe_import("raman_tab","render_raman_tab")
render_resistividade_tab = safe_import("resistividade_tab","render_resistividade_tab",True)
render_tensiometria_tab = safe_import("tensiometria_tab","render_tensiometria_tab",True)
render_mapeamento_molecular_tab = safe_import("mapeamento_molecular_tab","render_mapeamento_molecular_tab",True)

# 🔥 NOVO
render_analise_completa_amostras_tab = safe_import(
    "analise_completa_amostras_tab",
    "render_analise_completa_amostras_tab",
    True
)


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    if logo_image:
        st.image(logo_image)

    st.header("Cadastro de Amostra")

    sample_code = st.text_input("Código *")

    if st.button("Salvar"):

        if not sample_code:
            st.warning("Obrigatório")
        else:
            supabase.table("samples").insert({
                "sample_code": sample_code
            }).execute()

            st.success("Salvo!")


# =========================================================
# ABAS
# =========================================================

tabs = st.tabs([
    "Raman",
    "Resistividade",
    "Tensiometria",
    "Mapeamento",
    "🔥 Análise Completa"
])


with tabs[0]:
    render_raman_tab(supabase)

with tabs[1]:
    if render_resistividade_tab:
        render_resistividade_tab(supabase)

with tabs[2]:
    if render_tensiometria_tab:
        render_tensiometria_tab(supabase)

with tabs[3]:
    if render_mapeamento_molecular_tab:
        render_mapeamento_molecular_tab(supabase)

with tabs[4]:
    if render_analise_completa_amostras_tab:
        render_analise_completa_amostras_tab(supabase)
