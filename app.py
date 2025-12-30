# app.py
# -*- coding: utf-8 -*-

"""
SurfaceXLab — Plataforma CRM para Caracterização de Superfícies
Módulos:
1. Análises Moleculares (Raman)
2. Análises Elétricas
3. Análises Físico-Mecânicas
4. Otimizador com IA

Uso científico e acadêmico.
"""

# =========================================================
# IMPORTS
# =========================================================
from typing import Optional
import io
import streamlit as st
import pandas as pd

# AG-Grid (opcional)
try:
    from st_aggrid import AgGrid, GridOptionsBuilder, DataReturnMode, GridUpdateMode
    AGGRID_AVAILABLE = True
except Exception:
    AGGRID_AVAILABLE = False

# Supabase (opcional)
try:
    from supabase import create_client, Client
except Exception:
    create_client = None
    Client = None

# Módulos da plataforma
from raman_tab import render_raman_tab
from resistividade_tab import render_resistividade_tab
from tensiometria_tab import render_tensiometria_tab
from ml_tab import render_ml_tab

# =========================================================
# CONFIGURAÇÃO DA PÁGINA
# =========================================================
st.set_page_config(
    page_title="SurfaceXLab — CRM Científico",
    layout="wide",
)

# =========================================================
# CSS — DARK CRM (científico / industrial)
# =========================================================
st.markdown("""
<style>
:root{
  --bg:#181A1F; --card:#202328; --card2:#24262c;
  --text:#F5F5F5; --muted:#A6A9AE; --accent:#9CA3AF;
  --border:rgba(255,255,255,0.05);
}
html, body, [class*="css"] {
  background-color: var(--bg);
  color: var(--text);
}
.sx-card{
  background: linear-gradient(180deg,var(--card),var(--card2));
  border-radius:12px;
  padding:14px;
  border:1px solid var(--border);
}
.sx-title{
  font-size:22px;
  font-weight:700;
}
.sx-sub{
  color:var(--muted);
  font-size:13px;
}
.sx-kpi{
  font-size:26px;
  font-weight:700;
}
.side-panel{
  background:linear-gradient(180deg,#1E2026,#181A1F);
  border-radius:12px;
  padding:12px;
  border:1px solid var(--border);
}
</style>
""", unsafe_allow_html=True)

# =========================================================
# HEADER
# =========================================================
col1, col2 = st.columns([1, 8])
with col1:
    st.markdown("🧪")
with col2:
    st.markdown("<div class='sx-title'>SurfaceXLab</div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='sx-sub'>Plataforma CRM para Análise Molecular, Elétrica e Físico-Mecânica de Superfícies</div>",
        unsafe_allow_html=True
    )

st.markdown("---")

# =========================================================
# SUPABASE INIT (OPCIONAL)
# =========================================================
def init_supabase() -> Optional["Client"]:
    if create_client is None:
        return None
    try:
        url = st.secrets["SUPABASE_URL"]
        key = st.secrets["SUPABASE_KEY"]
        return create_client(url, key)
    except Exception:
        return None

supabase = init_supabase()

if supabase is None:
    st.info("Supabase não configurado — operando em modo local (session state).")

# =========================================================
# HELPERS COMPARTILHADOS (CRM CORE)
# =========================================================
if "side_open" not in st.session_state:
    st.session_state.side_open = False
    st.session_state.side_df = None
    st.session_state.side_title = ""

def open_side(df: pd.DataFrame, title: str):
    st.session_state.side_open = True
    st.session_state.side_df = df
    st.session_state.side_title = title

def close_side():
    st.session_state.side_open = False
    st.session_state.side_df = None
    st.session_state.side_title = ""

def show_aggrid(df: pd.DataFrame, height: int = 300):
    if AGGRID_AVAILABLE:
        gb = GridOptionsBuilder.from_dataframe(df)
        gb.configure_pagination(paginationAutoPageSize=False, paginationPageSize=10)
        gb.configure_side_bar()
        grid_opts = gb.build()
        AgGrid(
            df,
            gridOptions=grid_opts,
            height=height,
            data_return_mode=DataReturnMode.FILTERED_AND_SORTED,
            update_mode=GridUpdateMode.NO_UPDATE,
        )
    else:
        st.dataframe(df, height=height, use_container_width=True)

def download_df(df: pd.DataFrame, name: str):
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button("⬇ CSV", csv, f"{name}.csv", "text/csv")

helpers = {
    "open_side": open_side,
    "close_side": close_side,
    "show_aggrid": show_aggrid,
    "download_df": download_df,
    "supabase": supabase,
}

# =========================================================
# KPI ROW (VISÃO GERAL)
# =========================================================
k1, k2, k3, k4 = st.columns(4)

with k1:
    st.markdown("<div class='sx-card'><div class='sx-sub'>Amostras</div>", unsafe_allow_html=True)
    total = len(supabase.table("samples").select("id").execute().data) if supabase else 0
    st.markdown(f"<div class='sx-kpi'>{total}</div></div>", unsafe_allow_html=True)

with k2:
    st.markdown("<div class='sx-card'><div class='sx-sub'>Análises Moleculares</div>", unsafe_allow_html=True)
    st.markdown("<div class='sx-kpi'>—</div></div>", unsafe_allow_html=True)

with k3:
    st.markdown("<div class='sx-card'><div class='sx-sub'>Análises Elétricas</div>", unsafe_allow_html=True)
    st.markdown("<div class='sx-kpi'>—</div></div>", unsafe_allow_html=True)

with k4:
    st.markdown("<div class='sx-card'><div class='sx-sub'>Modelos IA</div>", unsafe_allow_html=True)
    st.markdown("<div class='sx-kpi'>—</div></div>", unsafe_allow_html=True)

st.markdown("---")

# =========================================================
# ABAS — MÓDULOS DA PLATAFORMA
# =========================================================
tab_mol, tab_ele, tab_phys, tab_ml = st.tabs([
    "1 Análises Moleculares",
    "2 Análises Elétricas",
    "3 Análises Físico-Mecânicas",
    "4 Otimizador (IA)",
])

with tab_mol:
    render_raman_tab(supabase, helpers)

with tab_ele:
    render_resistividade_tab(supabase, helpers)

with tab_phys:
    render_tensiometria_tab(supabase, helpers)

with tab_ml:
    render_ml_tab(supabase, helpers)

# =========================================================
# PAINEL LATERAL (DRILL-DOWN CRM)
# =========================================================
if st.session_state.side_open:
    _, col = st.columns([3, 1])
    with col:
        st.markdown("<div class='side-panel'>", unsafe_allow_html=True)
        st.markdown(f"### {st.session_state.side_title}")
        if st.session_state.side_df is not None:
            show_aggrid(st.session_state.side_df, height=360)
            download_df(st.session_state.side_df, st.session_state.side_title)
        if st.button("Fechar painel"):
            close_side()
        st.markdown("</div>", unsafe_allow_html=True)

# =========================================================
# FOOTER
# =========================================================
st.markdown("---")
st.caption("SurfaceXLab • Plataforma CRM Científica • © 2025 • Marcela Veiga")
