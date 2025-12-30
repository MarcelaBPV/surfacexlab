# resistividade_tab.py
# -*- coding: utf-8 -*-

"""
Aba 2 — Análises Elétricas (Resistividade)
CRM científico:
Paciente → Amostra → Ensaio → Propriedades elétricas → Banco
"""

# =========================================================
# IMPORTS
# =========================================================
import streamlit as st
import pandas as pd
from typing import Dict
from datetime import datetime

from resistividade_processing import process_resistivity

# =========================================================
# SUPABASE HELPERS
# =========================================================
def create_sample(supabase, patient_id: str, name: str, description: str = ""):
    res = supabase.table("samples").insert({
        "patient_id": patient_id,
        "name": name,
        "description": description,
        "created_at": datetime.utcnow().isoformat()
    }).execute()
    return res.data[0]


def create_measurement(supabase, sample_id: str, raw_meta: Dict):
    res = supabase.table("measurements_raw").insert({
        "sample_id": sample_id,
        "module": "electrical",
        "raw_data": raw_meta,
        "taken_at": datetime.utcnow().isoformat()
    }).execute()
    return res.data[0]


# =========================================================
# RENDER DA ABA
# =========================================================
def render_resistividade_tab(supabase, helpers):

    st.subheader("⚡ Análises Elétricas — Resistividade (I × V)")

    st.markdown(
        """
Esta aba realiza a **análise elétrica de materiais** a partir de
curvas **corrente × tensão**, permitindo a determinação de:

- Resistência elétrica (R)  
- Resistividade elétrica (ρ)  
- Condutividade elétrica (σ)  
- Classificação física do material  

📌 Método compatível com **quatro pontas (Smits, 1958)**.
"""
    )

    # =====================================================
    # SESSION STATE
    # =====================================================
    if "resist_results" not in st.session_state:
        st.session_state.resist_results = None

    # =====================================================
    # BLOCO 1 — AMOSTRA (CRM)
    # =====================================================
    st.markdown("### 🧪 Amostra")

    sample_name = st.text_input("Identificação da amostra")
    description = st.text_area("Material, processo ou observações")

    # =====================================================
    # BLOCO 2 — CONFIGURAÇÃO EXPERIMENTAL
    # =====================================================
    st.markdown("### ⚙️ Configuração experimental")

    thickness_nm = st.number_input(
        "Espessura do filme (nm)",
        min_value=1.0,
        value=200.0,
        step=10.0,
    )

    geometry = st.selectbox(
        "Geometria do ensaio",
        ["four_point_film", "bulk"],
        index=0,
        help="Selecione 'four_point_film' para filmes finos.",
    )

    # =====================================================
    # BLOCO 3 — UPLOAD DO ARQUIVO I–V
    # =====================================================
    st.markdown("### 📤 Upload do arquivo elétrico")

    uploaded = st.file_uploader(
        "Arquivo I × V (.csv ou .txt)",
        type=["csv", "txt"],
    )

    if uploaded is None:
        st.info("Envie um arquivo I × V para iniciar a análise.")
        return

    # =====================================================
    # BLOCO 4 — PROCESSAMENTO
    # =====================================================
    try:
        results = process_resistivity(
            file_like=uploaded,
            thickness_m=thickness_nm * 1e-9,
            geometry=geometry,
        )
        st.session_state.resist_results = results
        st.success("Dados elétricos processados com sucesso.")

    except Exception as e:
        st.error(f"Erro ao processar o arquivo: {e}")
        return

    # =====================================================
    # BLOCO 5 — KPIs
    # =====================================================
    st.markdown("### 📊 Indicadores elétricos")

    r = results["R_ohm"]
    rho = results["rho_ohm_m"]
    sigma = results["sigma_S_m"]
    classe = results["classe"]
    R2 = results["fit"]["R2"]

    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("R (Ω)", f"{r:.3e}")
    k2.metric("ρ (Ω·m)", f"{rho:.3e}")
    k3.metric("σ (S/m)", f"{sigma:.3e}")
    k4.metric("R²", f"{R2:.4f}")
    k5.metric("Classe", classe)

    # =====================================================
    # BLOCO 6 — VISUALIZAÇÃO
    # =====================================================
    st.markdown("### 📈 Curva Corrente × Tensão")
    st.pyplot(results["figure"])

    st.markdown("### 📋 Dados experimentais")
    helpers["show_aggrid"](results["df"], height=260)

    if st.button("🔍 Abrir dados no painel lateral"):
        helpers["open_side"](results["df"], "Dados Elétricos (I × V)")

    # =====================================================
    # BLOCO 7 — SALVAR NO SUPABASE
    # =====================================================
    if supabase and st.button("💾 Salvar ensaio elétrico"):
        try:
            patient_id = st.session_state.get("selected_patient", {}).get("id")

            sample = create_sample(
                supabase,
                patient_id=patient_id,
                name=sample_name,
                description=description,
            )

            meas = create_measurement(
                supabase,
                sample_id=sample["id"],
                raw_meta={
                    "filename": uploaded.name,
                    "geometry": geometry,
                    "thickness_nm": thickness_nm,
                }
            )

            supabase.table("results_electrical").insert({
                "measurement_id": meas["id"],
                "resistance_ohm": r,
                "resistivity_ohm_m": rho,
                "conductivity_s_m": sigma,
                "r2": R2,
                "class_label": classe,
            }).execute()

            st.success("Ensaio elétrico salvo com sucesso.")

        except Exception as e:
            st.error(f"Erro ao salvar no Supabase: {e}")
