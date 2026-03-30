# =========================================================
# SurfaceXLab
# =========================================================

import streamlit as st
from supabase import create_client, Client
from pathlib import Path
from PIL import Image

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
# CONFIG
# =========================================================

st.set_page_config(
page_title="SurfaceXLab",
page_icon=logo_image if logo_image else "🧪",
layout="wide"
)

st.title("SurfaceXLab — Plataforma de Caracterização de Superfícies")

# =========================================================
# SUPABASE - banco de dados
# =========================================================

@st.cache_resource
def init_supabase() -> Client:
return create_client(
st.secrets["SUPABASE_URL"],
st.secrets["SUPABASE_ANON_KEY"]
)

supabase = init_supabase()

# =========================================================
# SAFE IMPORT
# =========================================================

def safe_import(module_name, func_name, optional=False):

```
try:
    module = __import__(module_name, fromlist=[func_name])
    func = getattr(module, func_name)
    return func

except Exception as e:

    if optional:
        st.warning(f"⚠ Módulo opcional não carregado: {module_name}")
        return None

    st.error(f"❌ Erro ao carregar {module_name}.{func_name}")
    st.exception(e)
    st.stop()
```

# =========================================================
# IMPORTAÇÃO DOS MÓDULOS
# =========================================================

render_raman_tab = safe_import("raman_tab", "render_raman_tab")

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

render_mapeamento_molecular_tab = safe_import(
"mapeamento_molecular_tab",
"render_mapeamento_molecular_tab",
optional=True
)

# análise completa (com debug real)

try:
from analise_completa_amostras_tab import render_analise_completa_amostras_tab
except Exception as e:
render_analise_completa_amostras_tab = None
st.error("❌ Erro ao carregar análise completa")
st.exception(e)

# =========================================================
# BARRA LATERAL
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

            st.success("✔ Salvo com sucesso")

        except Exception as e:
            st.error("Erro ao salvar")
            st.exception(e)
```

# =========================================================
# ABAS
# =========================================================

tabs = st.tabs([
"1 Raman",
"2 Resistividade",
"3 Tensiometria",
"4 Mapeamento",
"5 Análise Completa"
])

# =========================================================
# RAMAN
# =========================================================

with tabs[0]:
if render_raman_tab:
render_raman_tab(supabase)

# =========================================================
# RESISTIVIDADE
# =========================================================

with tabs[1]:

```
if render_resistividade_tab:
    render_resistividade_tab(supabase)
else:
    st.info("Módulo não disponível")
```

# =========================================================
# TENSIOMETRIA
# =========================================================

with tabs[2]:

```
if render_tensiometria_tab:
    render_tensiometria_tab(supabase)
else:
    st.info("Módulo não disponível")
```

# =========================================================
# MAPEAMENTO
# =========================================================

with tabs[3]:

```
if render_mapeamento_molecular_tab:
    render_mapeamento_molecular_tab(supabase)
else:
    st.info("Módulo não disponível")
```

# =========================================================
# ANÁLISE COMPLETA
# =========================================================

with tabs[4]:

```
if render_analise_completa_amostras_tab is None:
    st.error("❌ Módulo não carregado")
else:
    render_analise_completa_amostras_tab(supabase)
```
