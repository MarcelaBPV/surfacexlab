# app.py
# -*- coding: utf-8 -*-
"""
SurfaceXLab - CRM-style single-file launcher
Integra suas abas (raman_tab, tensiometria_tab, resistividade_tab, ml_tab)
e fornece layout CRM (navbar, KPI row, side panel, AG-Grid helper, exports).
"""

from typing import Optional
import io
import importlib
import streamlit as st
import pandas as pd
import numpy as np

# Try to import st-aggrid for nicer tables; fall back to st.dataframe
try:
    from st_aggrid import AgGrid, GridOptionsBuilder, DataReturnMode, GridUpdateMode
    AGGRID_AVAILABLE = True
except Exception:
    AGGRID_AVAILABLE = False

# Try supabase client — optional
try:
    from supabase import create_client, Client
except Exception:
    create_client = None
    Client = None

# Import the user's tab renderers (they must exist in repo)
# Each should expose a function with signature: render_xxx_tab(supabase, helpers)
# helpers is a dict with open_side, close_side, show_aggrid, download_df helpers, etc.
from raman_tab import render_raman_tab
from tensiometria_tab import render_tensiometria_tab
from resistividade_tab import render_resistividade_tab
from ml_tab import render_ml_tab

# -------------------- Streamlit page config --------------------
st.set_page_config(page_title="SurfaceXLab — CRM", layout="wide")
# Minimal branding header
col_logo, col_title = st.columns([1, 6])
with col_logo:
    # optional: add your logo file to repo and change path
    try:
        st.image("assets/logo.png", width=64)
    except Exception:
        st.markdown("<div style='font-weight:700'>SurfaceXLab</div>", unsafe_allow_html=True)
with col_title:
    st.markdown("<h2 style='margin:0;color:#F5F5F5'>SurfaceXLab</h2>", unsafe_allow_html=True)
    st.markdown("<div style='color:#B8BCC2;margin-top:0'>Advanced Material Surface Intelligence — CRM-style dashboard</div>", unsafe_allow_html=True)
st.markdown("---")

# -------------------- Theme CSS (Dark Silver CRM-like) --------------------
DARK_CSS = """
<style>
:root{
  --bg: #181A1F; --card: #202328; --card-2:#23262b;
  --muted:#A6A9AE; --text:#F5F5F5; --accent:#9CA3AF;
  --border: rgba(255,255,255,0.04);
  --shadow: 0 8px 24px rgba(0,0,0,0.6);
}
.reportview-container, .main, body { background-color: var(--bg) !important; color:var(--text); }
.sx-navbar { background: linear-gradient(90deg,#151619,#1b1d21); padding:10px 14px; border-radius:8px; display:flex; align-items:center; gap:12px; margin-bottom:12px; }
.sx-logo { font-weight:700; color:var(--text); font-size:15px; margin-right:6px; }
.sx-navlink { color:var(--muted); padding:6px 10px; border-radius:8px; text-decoration:none; }
.sx-navlink.active { color:var(--text); background: rgba(255,255,255,0.02); }
.sx-card{ background:linear-gradient(180deg,var(--card),var(--card-2)); border-radius:10px; padding:12px; border:1px solid var(--border); box-shadow:var(--shadow); margin-bottom:12px;}
.sx-card-title{font-weight:700;color:var(--text); font-size:14px;}
.sx-card-sub{color:var(--muted);font-size:12px;margin-top:6px;}
.sx-kpi{font-size:26px;font-weight:700;color:var(--text);}
.side-panel{ background: linear-gradient(180deg,#1B1D21,#16181B); border-radius:8px; padding:12px; border:1px solid rgba(255,255,255,0.03); }
.sx-smallbtn{ background:transparent; color:var(--muted); padding:6px 10px; border-radius:8px; border:1px solid rgba(255,255,255,0.03); }
</style>
"""
st.markdown(DARK_CSS, unsafe_allow_html=True)

# -------------------- Supabase init helper --------------------
def init_supabase() -> Optional["Client"]:
    """
    Try to initialize and return supabase client from st.secrets.
    If not configured, return None.
    """
    if create_client is None:
        return None
    url = st.secrets.get("SUPABASE_URL") if hasattr(st, "secrets") else None
    key = st.secrets.get("SUPABASE_KEY") if hasattr(st, "secrets") else None
    if not url or not key:
        return None
    try:
        client = create_client(url, key)
        return client
    except Exception:
        return None

supabase = init_supabase()
if supabase is None:
    st.sidebar.info("Supabase não configurado (opcional). Defina SUPABASE_URL e SUPABASE_KEY em st.secrets para ativar DB.")

# -------------------- AG-Grid helper --------------------
def show_aggrid(df: pd.DataFrame, height: int = 300):
    """Show df via AG-Grid if available, else st.dataframe."""
    if AGGRID_AVAILABLE:
        gb = GridOptionsBuilder.from_dataframe(df)
        gb.configure_pagination(paginationAutoPageSize=False, paginationPageSize=10)
        gb.configure_side_bar()
        gb.configure_selection(selection_mode="single", use_checkbox=False)
        grid_opts = gb.build()
        resp = AgGrid(df, gridOptions=grid_opts, height=height,
                      data_return_mode=DataReturnMode.FILTERED_AND_SORTED,
                      update_mode=GridUpdateMode.NO_UPDATE)
        return resp
    else:
        st.dataframe(df, height=height)
        return None

# -------------------- Side panel (global) --------------------
if "side_open" not in st.session_state:
    st.session_state.side_open = False
if "side_df" not in st.session_state:
    st.session_state.side_df = None
if "side_title" not in st.session_state:
    st.session_state.side_title = ""

def open_side(df: Optional[pd.DataFrame], title: str):
    st.session_state.side_open = True
    st.session_state.side_df = df
    st.session_state.side_title = title

def close_side():
    st.session_state.side_open = False
    st.session_state.side_df = None
    st.session_state.side_title = ""

def download_df_buttons(df: pd.DataFrame, filename_prefix: str):
    """Render download buttons for CSV and XLSX; returns nothing."""
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button("Exportar CSV", csv, file_name=f"{filename_prefix}.csv", mime="text/csv")
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="sheet1")
    buf.seek(0)
    st.download_button("Exportar XLSX", buf.read(), file_name=f"{filename_prefix}.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

# -------------------- Navbar + KPI row --------------------
def render_navbar(active: str = "dashboard"):
    st.markdown(f"""
    <div class="sx-navbar">
      <div class="sx-logo">SurfaceXLab</div>
      <div style="display:flex;gap:6px;">
        <a class="sx-navlink {'active' if active=='raman' else ''}">1️⃣ Raman</a>
        <a class="sx-navlink {'active' if active=='tensi' else ''}">2️⃣ Tensiometria</a>
        <a class="sx-navlink {'active' if active=='resist' else ''}">3️⃣ Resistividade</a>
        <a class="sx-navlink {'active' if active=='ml' else ''}">4️⃣ Otimização</a>
      </div>
      <div style="margin-left:auto;color:var(--muted)">Usuário • SurfaceXLab</div>
    </div>
    """, unsafe_allow_html=True)

render_navbar()

# KPI row (quick counts)
k1, k2, k3, k4, k5 = st.columns(5)
with k1:
    st.markdown("<div class='sx-card'><div class='sx-card-title'>Total Amostras</div>", unsafe_allow_html=True)
    try:
        if supabase:
            res = supabase.table("samples").select("id").execute()
            total_samples = len(res.data) if res and res.data else 0
        else:
            total_samples = 0
    except Exception:
        total_samples = 0
    st.markdown(f"<div class='sx-kpi'>{total_samples}</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)
with k2:
    st.markdown("<div class='sx-card'><div class='sx-card-title'>Ensaios Raman</div>", unsafe_allow_html=True)
    try:
        total_raman = len(supabase.table("measurements").select("id").eq("type","raman").execute().data) if supabase else 0
    except Exception:
        total_raman = 0
    st.markdown(f"<div class='sx-kpi'>{total_raman}</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)
with k3:
    st.markdown("<div class='sx-card'><div class='sx-card-title'>Ensaios 4 Pontas</div>", unsafe_allow_html=True)
    try:
        total_4p = len(supabase.table("measurements").select("id").eq("type","4_pontas").execute().data) if supabase else 0
    except Exception:
        total_4p = 0
    st.markdown(f"<div class='sx-kpi'>{total_4p}</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)
with k4:
    st.markdown("<div class='sx-card'><div class='sx-card-title'>Ensaios Tensiometria</div>", unsafe_allow_html=True)
    try:
        total_tensi = len(supabase.table("measurements").select("id").eq("type","tensiometria").execute().data) if supabase else 0
    except Exception:
        total_tensi = 0
    st.markdown(f"<div class='sx-kpi'>{total_tensi}</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)
with k5:
    st.markdown("<div class='sx-card'><div class='sx-card-title'>Modelos IA</div><div style='color:var(--muted);margin-top:6px'>—</div></div>", unsafe_allow_html=True)

st.markdown("---")

# -------------------- Tabs (call user's renderers) --------------------
# Build a helpers dict to pass to renderers so they can open side panel and use aggrid/download consistently.
helpers = {
    "open_side": open_side,
    "close_side": close_side,
    "show_aggrid": show_aggrid,
    "download_df_buttons": download_df_buttons,
    "supabase": supabase,
    "aggrid_available": AGGRID_AVAILABLE,
}

tab_raman, tab_tensi, tab_resist, tab_ml = st.tabs(["1️⃣ Raman", "2️⃣ Tensiometria", "3️⃣ Resistividade", "4️⃣ Otimização ML"])

with tab_raman:
    # the user's render_raman_tab should accept (supabase, helpers) or (helpers) — we try both signatures
    try:
        render_raman_tab(supabase, helpers)
    except TypeError:
        # fallback to render_raman_tab(helpers)
        render_raman_tab(helpers)

with tab_tensi:
    try:
        render_tensiometria_tab(supabase, helpers)
    except TypeError:
        render_tensiometria_tab(helpers)

with tab_resist:
    try:
        render_resistividade_tab(supabase, helpers)
    except TypeError:
        render_resistividade_tab(helpers)

with tab_ml:
    try:
        render_ml_tab(supabase, helpers)
    except TypeError:
        render_ml_tab(helpers)

# -------------------- Global side panel rendering --------------------
if st.session_state.side_open:
    left, right = st.columns([3,1])
    with left:
        st.markdown("")  # empty column to keep layout
    with right:
        st.markdown("<div class='side-panel'>", unsafe_allow_html=True)
        st.markdown(f"### {st.session_state.side_title}")
        if st.session_state.side_df is None or st.session_state.side_df.empty:
            st.write("_Tabela vazia_")
        else:
            if AGGRID_AVAILABLE:
                show_aggrid(st.session_state.side_df, height=380)
            else:
                st.dataframe(st.session_state.side_df, height=380)
            download_df_buttons(st.session_state.side_df, st.session_state.side_title)
        if st.button("Fechar painel lateral"):
            close_side()
        st.markdown("</div>", unsafe_allow_html=True)

st.markdown("---")
st.caption("SurfaceXLab — © 2025 — Desenvolvido por Marcela Veiga")
