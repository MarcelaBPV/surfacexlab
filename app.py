# app.py
# -*- coding: utf-8 -*-

"""
SurfaceXLab
Plataforma CRM para caracterização e otimização de superfícies

Fluxo:
1. Cadastro da amostra
2. Upload de dados experimentais (evento)
3. Processamento por módulo (Raman / Elétrica / Tensiometria)
4. ML e otimização

Frontend: Streamlit
Backend: Supabase (PostgreSQL)
"""

import streamlit as st
from supabase import create_client, Client
from pathlib import Path
from PIL import Image


# =========================================================
# LOGO (CARREGAMENTO SEGURO)
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

st.title("SurfaceXLab — Plataforma Integrada de Pesquisa")


# =========================================================
# CONEXÃO COM SUPABASE
# =========================================================
@st.cache_resource
def init_supabase() -> Client:
    try:
        return create_client(
            st.secrets["SUPABASE_URL"],
            st.secrets["SUPABASE_ANON_KEY"],
        )
    except Exception as e:
        st.error("❌ Erro ao conectar com o Supabase.")
        st.exception(e)
        st.stop()


supabase = init_supabase()


# =========================================================
# IMPORTAÇÃO SEGURA DE MÓDULOS
# =========================================================
def safe_import(module_name: str, func_name: str):
    try:
        module = __import__(module_name, fromlist=[func_name])
    except Exception as e:
        st.error(f"❌ Erro ao importar `{module_name}.py`")
        st.exception(e)
        st.stop()

    if not hasattr(module, func_name):
        st.error(f"❌ Função `{func_name}` não encontrada em `{module_name}.py`")
        st.stop()

    return getattr(module, func_name)


render_raman_tab = safe_import("raman_tab", "render_raman_tab")
render_resistividade_tab = safe_import("resistividade_tab", "render_resistividade_tab")
render_tensiometria_tab = safe_import("tensiometria_tab", "render_tensiometria_tab")
render_ml_tab = safe_import("ml_tab", "render_ml_tab")


# =========================================================
# SIDEBAR — CRM CORE
# =========================================================
with st.sidebar:

    if logo_image:
        st.image(logo_image, use_column_width=True)
        st.divider()

    # -----------------------------------------------------
    # 1️⃣ CADASTRO DA AMOSTRA
    # -----------------------------------------------------
    st.header("📦 Amostra")

    sample_code = st.text_input("Código da Amostra *")
    material_type = st.text_input("Tipo de Material")
    substrate = st.text_input("Substrato")
    surface_treatment = st.text_input("Tratamento de Superfície")
    description = st.text_area("Descrição")

    if st.button("Salvar Amostra"):
        if not sample_code:
            st.warning("⚠ Código da amostra é obrigatório.")
        else:
            try:
                res = supabase.table("samples").insert({
                    "sample_code": sample_code,
                    "material_type": material_type,
                    "substrate": substrate,
                    "surface_treatment": surface_treatment,
                    "description": description,
                }).execute()

                if res.data:
                    st.success("✔ Amostra cadastrada com sucesso!")
                else:
                    st.error("Erro ao salvar amostra.")
            except Exception as e:
                st.error("❌ Falha ao salvar amostra.")
                st.exception(e)

    # -----------------------------------------------------
    # 2️⃣ UPLOAD DE DADOS EXPERIMENTAIS (EVENTO CRM)
    # -----------------------------------------------------
    st.divider()
    st.header("📤 Entrada de Dados")

    samples_res = supabase.table("samples").select("id, sample_code").execute()
    samples = samples_res.data if samples_res.data else []

    if samples:
        sample_map = {s["sample_code"]: s["id"] for s in samples}

        selected_sample = st.selectbox(
            "Amostra associada",
            options=list(sample_map.keys()),
        )

        data_type = st.selectbox(
            "Tipo de dado",
            [
                "Molecular — Raman (sangue)",
                "Elétrico — Resistividade / Motores",
                "Físico-Mecânico — Tensiometria",
            ],
        )

        uploaded_file = st.file_uploader(
            "Arquivo experimental",
            type=["csv", "txt", "xlsx"],
        )

        if st.button("Registrar Upload"):
            if not uploaded_file:
                st.warning("Selecione um arquivo.")
            else:
                try:
                    # Cria experimento automaticamente
                    exp = supabase.table("experiments").insert({
                        "sample_id": sample_map[selected_sample],
                        "experiment_type": data_type.split("—")[0].strip(),
                        "notes": f"Upload inicial: {uploaded_file.name}",
                    }).execute()

                    experiment_id = exp.data[0]["id"]

                    # Registra upload (metadado)
                    supabase.table("raw_uploads").insert({
                        "experiment_id": experiment_id,
                        "filename": uploaded_file.name,
                        "file_type": data_type,
                    }).execute()

                    st.success("✔ Upload registrado com sucesso!")
                except Exception as e:
                    st.error("❌ Erro ao registrar upload.")
                    st.exception(e)
    else:
        st.info("Cadastre uma amostra para habilitar upload.")

    st.divider()
    st.caption("SurfaceXLab © Pesquisa & Engenharia")


# =========================================================
# ABAS — MÓDULOS DE ANÁLISE
# =========================================================
tabs = st.tabs([
    "🧬 Molecular — Raman",
    "⚡ Elétrica — Resistividade",
    "💧 Físico-Mecânica — Tensiometria",
    "🤖 Otimizador — IA",
])

with tabs[0]:
    render_raman_tab(supabase)

with tabs[1]:
    render_resistividade_tab(supabase)

with tabs[2]:
    render_tensiometria_tab(supabase)

with tabs[3]:
    render_ml_tab(supabase)
