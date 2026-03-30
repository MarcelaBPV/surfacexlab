# =========================================================

# SurfaceXLab — APP FINAL COMPLETO

# =========================================================

import streamlit as st
from pathlib import Path
from PIL import Image
from supabase import create_client, Client

# =========================================================

# CONFIGURAÇÃO

# =========================================================

BASE_DIR = Path(**file**).parent
ASSETS_DIR = BASE_DIR / "assets"
LOGO_PATH = ASSETS_DIR / "surfacexlab_logo.png"

# =========================================================

# LOGO

# =========================================================

def load_logo():
if LOGO_PATH.exists():
try:
return Image.open(LOGO_PATH)
except:
return None
return None

logo_image = load_logo()

# =========================================================

# PAGE CONFIG

# =========================================================

st.set_page_config(
page_title="SurfaceXLab",
page_icon=logo_image if logo_image else "🧪",
layout="wide"
)

st.title("SurfaceXLab — Plataforma de Caracterização e Otimização de Superfícies")

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

# IMPORTAÇÃO DA ANÁLISE COMPLETA

# =========================================================

try:
from analise_completa_amostras_tab import render_analise_completa_amostras_tab
except Exception as e:
render_analise_completa_amostras_tab = None
st.error("❌ Erro ao carregar módulo de análise completa")
st.exception(e)

# =========================================================

# SIDEBAR — CADASTRO

# =========================================================

with st.sidebar:

```
if logo_image:
    st.image(logo_image, use_container_width=True)
    st.divider()

st.header("Cadastro de Amostra")

sample_code = st.text_input("Código da Amostra *")
material_type = st.text_input("Tipo de Material")
substrate = st.text_input("Substrato")
surface_treatment = st.text_input("Tratamento de Superfície")
description = st.text_area("Descrição")

if st.button("Salvar Amostra"):

    if not sample_code:
        st.warning("⚠ Código obrigatório")
    else:
        try:
            supabase.table("samples").insert({
                "sample_code": sample_code,
                "material_type": material_type,
                "substrate": substrate,
                "surface_treatment": surface_treatment,
                "description": description,
            }).execute()

            st.success("✔ Amostra salva com sucesso")

        except Exception as e:
            st.error("Erro ao salvar")
            st.exception(e)
```

# =========================================================

# ABAS PRINCIPAIS

# =========================================================

tabs = st.tabs([
"🔬 Análise Completa"
])

# =========================================================

# ANÁLISE COMPLETA

# =========================================================

with tabs[0]:

```
if render_analise_completa_amostras_tab is None:
    st.error("❌ Módulo de análise completa não carregado")
else:
    render_analise_completa_amostras_tab(supabase)
```
