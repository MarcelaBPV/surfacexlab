# =========================================================
# SurfaceXLab — APP CRM (FUNDO BRANCO ESTÁVEL)
# =========================================================

import streamlit as st
from supabase import create_client, Client

# =========================================================
# CONFIG
# =========================================================

st.set_page_config(
    page_title="SurfaceXLab",
    layout="wide",
    page_icon="🧪"
)

st.title("🧪 SurfaceXLab — Plataforma de Caracterização")

# =========================================================
# SUPABASE (COM PROTEÇÃO)
# =========================================================

@st.cache_resource
def init_supabase():
    try:
        return create_client(
            st.secrets["SUPABASE_URL"],
            st.secrets["SUPABASE_ANON_KEY"]
        )
    except:
        return None

supabase = init_supabase()

if supabase:
    st.success("✅ Supabase conectado")
else:
    st.warning("⚠ Supabase não conectado")

# =========================================================
# IMPORT SEGURO DOS MÓDULOS
# =========================================================

def safe_import(module, func):
    try:
        mod = import(module, fromlist=[func])
        return getattr(mod, func)
    except:
        return None

render_raman_tab = safe_import("raman_tab", "render_raman_tab")
render_resistividade_tab = safe_import("resistividade_tab", "render_resistividade_tab")
render_tensiometria_tab = safe_import("tensiometria_tab", "render_tensiometria_tab")
render_mapeamento_tab = safe_import("mapeamento_molecular_tab", "render_mapeamento_molecular_tab")

try:
    from analise_completa_amostras_tab import render_analise_completa_amostras_tab
except:
    render_analise_completa_amostras_tab = None

# =========================================================
# SIDEBAR (MENU)
# =========================================================

st.sidebar.title("Menu")

menu = st.sidebar.radio(
    "Navegação",
    [
        "Dashboard",
        "Raman",
        "Resistividade",
        "Tensiometria",
        "Mapeamento",
        "Análise Completa"
    ]
)

# =========================================================
# DASHBOARD
# =========================================================

if menu == "Dashboard":

    st.header("📊 Dashboard")

    col1, col2, col3 = st.columns(3)

    col1.metric("Amostras", "—")
    col2.metric("Ensaios", "—")
    col3.metric("Projetos", "—")

    st.info("Dashboard inicial — pode integrar com Supabase depois")

# =========================================================
# RAMAN
# =========================================================

elif menu == "Raman":

    st.header("🔬 Raman")

    if render_raman_tab:
        render_raman_tab(supabase)
    else:
        st.warning("Módulo Raman não disponível")

# =========================================================
# RESISTIVIDADE
# =========================================================

elif menu == "Resistividade":

    st.header("⚡ Resistividade")

    if render_resistividade_tab:
        render_resistividade_tab(supabase)
    else:
        st.warning("Módulo não disponível")

# =========================================================
# TENSIOMETRIA
# =========================================================

elif menu == "Tensiometria":

    st.header("💧 Tensiometria")

    if render_tensiometria_tab:
        render_tensiometria_tab(supabase)
    else:
        st.warning("Módulo não disponível")

# =========================================================
# MAPEAMENTO
# =========================================================

elif menu == "Mapeamento":

    st.header("🧬 Mapeamento")

    if render_mapeamento_tab:
        render_mapeamento_tab(supabase)
    else:
        st.warning("Módulo não disponível")

# =========================================================
# ANÁLISE COMPLETA
# =========================================================

elif menu == "Análise Completa":

    st.header("🧠 Análise Completa")

    if render_analise_completa_amostras_tab:
        render_analise_completa_amostras_tab(supabase)
    else:
        st.error("Erro ao carregar módulo")
