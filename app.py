import streamlit as st
from supabase import create_client

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

# SUPABASE

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

# IMPORTS SEGUROS

# =========================================================

def safe_import(module, func):
try:
mod = **import**(module, fromlist=[func])
return getattr(mod, func)
except:
return None

render_raman_tab = safe_import("raman_tab", "render_raman_tab")
render_resistividade_tab = safe_import("resistividade_tab", "render_resistividade_tab")
render_tensiometria_tab = safe_import("tensiometria_tab", "render_tensiometria_tab")
render_mapeamento_tab = safe_import("mapeamento_molecular_tab", "render_mapeamento_molecular_tab")

# 🔥 IMPORT DA ANÁLISE COMPLETA COM DEBUG

try:
from analise_completa_amostras_tab import render_analise_completa_amostras_tab
except Exception as e:
render_analise_completa_amostras_tab = None
st.error("❌ Erro ao importar análise completa")
st.exception(e)

# =========================================================

# ABAS NO TOPO (IGUAL ANTES)

# =========================================================

tabs = st.tabs([
"🔬 Raman",
"⚡ Resistividade",
"💧 Tensiometria",
"🧬 Mapeamento",
"🧠 Análise Completa"
])

# =========================================================

# RAMAN

# =========================================================

with tabs[0]:
if render_raman_tab:
render_raman_tab(supabase)
else:
st.info("Módulo não disponível")

# =========================================================

# RESISTIVIDADE

# =========================================================

with tabs[1]:
if render_resistividade_tab:
render_resistividade_tab(supabase)
else:
st.info("Módulo não disponível")

# =========================================================

# TENSIOMETRIA

# =========================================================

with tabs[2]:
if render_tensiometria_tab:
render_tensiometria_tab(supabase)
else:
st.info("Módulo não disponível")

# =========================================================

# MAPEAMENTO

# =========================================================

with tabs[3]:
if render_mapeamento_tab:
render_mapeamento_tab(supabase)
else:
st.info("Módulo não disponível")

# =========================================================

# ANÁLISE COMPLETA

# =========================================================

with tabs[4]:
if render_analise_completa_amostras_tab:
render_analise_completa_amostras_tab(supabase)
else:
st.error("Erro ao carregar módulo de análise completa")
