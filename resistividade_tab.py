# resistividade_tab.py
# -*- coding: utf-8 -*-

"""
Aba 2 — Análises Elétricas
Método de quatro pontas / V × I
CRM científico: Amostra → Ensaio → Propriedades elétricas
"""

# =========================================================
# IMPORTS
# =========================================================
import streamlit as st
import pandas as pd
from typing import Dict, List
from datetime import datetime

from resistividade import process_resistivity  # seu módulo de cálculo

# =========================================================
# SUPABASE HELPERS
# =========================================================
def create_sample(supabase, name: str, description: str = ""):
    res = supabase.table("samples").insert({
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


def save_electrical_results(supabase, measurement_id: str, result: Dict):
    supabase.table("results_electrical").insert({
        "measurement_id": measurement_id,
        "resistance": float(result["R"]),
        "resistivity": float(result["rho"]),
        "conductivity": float(result["sigma"]),
        "regression_coefficients": {
            "R2": float(result["R2"])
        },
        "stats": {
            "class": result["classe"],
            "mode": result["mode"],
            "thickness_m": float(result["thickness_m"])
        },
        "created_at": datetime.utcnow().isoformat()
    }).execute()

# =========================================================
# RENDER DA ABA
# =========================================================
def render_resistividade_tab(supabase, helpers):

    st.subheader("⚡ Análises Elétricas — Resistividade (V × I)")

    st.markdown(
        """
Envie um arquivo **CSV** contendo pares de valores de corrente e tensão  
(colunas equivalentes a `current_a` e `voltage_v`).

O sistema irá:
- Ajustar a curva V × I (regressão linear)
- Calcular **R**, **ρ** e **σ**
- Avaliar o **R²**
- Classificar o comportamento elétrico do material
        """
    )

    # =====================================================
    # BLOCO 1 — DADOS DA AMOSTRA (CRM)
    # =====================================================
    st.markdown("### 🧪 Amostra")

    sample_name = st.text_input("Identificação da amostra")
    description = st.text_area("Material / processo / observações")

    # =====================================================
    # BLOCO 2 — PARÂMETROS EXPERIMENTAIS
    # =====================================================
    st.markdown("### ⚙️ Parâmetros experimentais")

    col1, col2 = st.columns(2)
    with col1:
        thickness_nm = st.number_input(
            "Espessura do filme (nm)", min_value=1.0, value=200.0, step=10.0
        )
    with col2:
        mode = st.selectbox("Modelo de cálculo", ["filme", "bulk"], index=0)

    thickness_m = thickness_nm * 1e-9

    # =====================================================
    # BLOCO 3 — UPLOAD DOS DADOS
    # =====================================================
    st.markdown("### 📤 Upload dos dados V × I")

    uploaded = st.file_uploader(
        "Arquivo CSV (corrente × tensão)",
        type=["csv"]
    )

    if uploaded is None:
        st.info("Envie um arquivo CSV para iniciar a análise.")
        return

    # =====================================================
    # BLOCO 4 — PROCESSAMENTO
    # =====================================================
    try:
        df_preview = pd.read_csv(uploaded)
        st.markdown("### 🔍 Pré-visualização dos dados")
        st.dataframe(df_preview.head(), use_container_width=True)

        uploaded.seek(0)
        result = process_resistivity(
            uploaded,
            thickness_m=thickness_m,
            mode=mode
        )

    except Exception as e:
        st.error(f"Erro ao processar os dados: {e}")
        return

    df = result["df"]
    R = result["R"]
    rho = result["rho"]
    sigma = result["sigma"]
    classe = result["classe"]
    R2 = result["R2"]
    fig = result["figure"]

    # =====================================================
    # BLOCO 5 — KPIs
    # =====================================================
    st.markdown("### 📊 Indicadores principais")

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("R (Ω)", f"{R:.3e}")
    k2.metric("ρ (Ω·m)", f"{rho:.3e}")
    k3.metric("σ (S/m)", f"{sigma:.3e}")
    k4.metric("R²", f"{R2:.4f}")

    # =====================================================
    # BLOCO 6 — VISUALIZAÇÃO
    # =====================================================
    st.markdown("### 📈 Curva V × I")
    st.pyplot(fig)

    st.markdown("### 📋 Dados completos do ensaio")
    helpers["show_aggrid"](df, height=260)

    if st.button("🔍 Abrir tabela no painel lateral"):
        helpers["open_side"](df, "Dados V × I")

    st.download_button(
        "⬇️ Exportar dados (CSV)",
        df.to_csv(index=False).encode("utf-8"),
        file_name=f"{sample_name}_VxI.csv",
        mime="text/csv",
    )

    # =====================================================
    # BLOCO 7 — SALVAR NO SUPABASE
    # =====================================================
    if supabase and st.button("💾 Salvar ensaio elétrico"):
        try:
            sample = create_sample(
                supabase,
                name=sample_name,
                description=description
            )

            meas = create_measurement(
                supabase,
                sample_id=sample["id"],
                raw_meta={
                    "filename": uploaded.name,
                    "mode": mode
                }
            )

            save_electrical_results(
                supabase,
                meas["id"],
                {
                    "R": R,
                    "rho": rho,
                    "sigma": sigma,
                    "R2": R2,
                    "classe": classe,
                    "mode": mode,
                    "thickness_m": thickness_m
                }
            )

            st.success("Ensaio elétrico salvo com sucesso.")

        except Exception as e:
            st.error(f"Erro ao salvar no Supabase: {e}")
