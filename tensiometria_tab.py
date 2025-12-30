# tensiometria_tab.py
# -*- coding: utf-8 -*-

"""
Aba 3 — Análises Físico-Mecânicas
Tensiometria óptica, energia livre de superfície (OWRK) e molhabilidade
CRM científico: Amostra → Ensaio → Parâmetros interfaciais
"""

# =========================================================
# IMPORTS
# =========================================================
import streamlit as st
import pandas as pd
from typing import Dict
from datetime import datetime
from io import StringIO

from tensiometria import process_contact_angle  # seu módulo de cálculo

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
        "module": "physical_mechanical",
        "raw_data": raw_meta,
        "taken_at": datetime.utcnow().isoformat()
    }).execute()
    return res.data[0]


def save_physical_mechanical_results(supabase, measurement_id: str, result: Dict):
    supabase.table("results_physical_mechanical").insert({
        "measurement_id": measurement_id,
        "contact_angle_avg": float(result["theta_mean"]),
        "contact_angle_sd": float(result["theta_std"]),
        "surface_energy_total": float(result["surface_energy_total"]),
        "surface_energy_components": result["surface_energy_components"],
        "fit_r2": float(result["r2"]),
        "fit_errors": result["fit_errors"],
        "classification": result["classification"],
        "created_at": datetime.utcnow().isoformat()
    }).execute()

# =========================================================
# RENDER DA ABA
# =========================================================
def render_tensiometria_tab(supabase, helpers):

    st.subheader("🧪 Análises Físico-Mecânicas — Tensiometria Óptica")

    st.markdown(
        """
Envie um arquivo de **ângulo de contato** (TXT ou CSV) contendo colunas equivalentes a:

`Time` | `Theta(L)` | `Theta(R)` | `Mean`

O sistema irá:
- Ajustar a evolução temporal do ângulo de contato
- Calcular parâmetros estatísticos
- Aplicar o modelo **Owens–Wendt–Rabel–Kaelble (OWRK)**
- Classificar a superfície quanto à **molhabilidade**
        """
    )

    # =====================================================
    # BLOCO 1 — AMOSTRA (CRM)
    # =====================================================
    st.markdown("### 🧪 Amostra")

    sample_name = st.text_input("Identificação da amostra / superfície")
    description = st.text_area("Material, tratamento superficial ou observações")

    # =====================================================
    # BLOCO 2 — PARÂMETROS DO AJUSTE
    # =====================================================
    st.markdown("### ⚙️ Parâmetros do ajuste")

    fit_order = st.number_input(
        "Ordem do polinômio para ajuste temporal",
        min_value=1,
        max_value=6,
        value=3
    )

    # =====================================================
    # BLOCO 3 — UPLOAD DOS DADOS
    # =====================================================
    st.markdown("### 📤 Upload do log de ângulo de contato")

    uploaded = st.file_uploader(
        "Arquivo de tensiometria (txt ou csv)",
        type=["txt", "csv"]
    )

    if uploaded is None:
        st.info("Envie um arquivo de tensiometria para iniciar a análise.")
        return

    # =====================================================
    # BLOCO 4 — PROCESSAMENTO
    # =====================================================
    try:
        content = uploaded.read().decode("utf-8", errors="ignore")
        sio = StringIO(content)

        result = process_contact_angle(
            sio,
            fit_order=fit_order
        )

    except Exception as e:
        st.error(f"Erro ao processar o arquivo: {e}")
        return

    df = result["df"]
    fig = result["figure"]

    theta_mean = result["theta_mean"]
    theta_std = result["theta_std"]
    surface_energy_total = result["surface_energy_total"]
    surface_energy_components = result["surface_energy_components"]
    r2 = result["r2"]
    fit_errors = result["fit_errors"]
    classification = result["classification"]

    # =====================================================
    # BLOCO 5 — KPIs
    # =====================================================
    st.markdown("### 📊 Indicadores principais")

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Ângulo médio (°)", f"{theta_mean:.2f}")
    k2.metric("Desvio padrão (°)", f"{theta_std:.2f}")
    k3.metric("Energia superficial (mJ/m²)", f"{surface_energy_total:.2f}")
    k4.metric("Classificação", classification)

    # =====================================================
    # BLOCO 6 — VISUALIZAÇÃO
    # =====================================================
    st.markdown("### 📈 Ajuste temporal do ângulo de contato")
    st.pyplot(fig)

    st.markdown("### 📋 Dados experimentais")
    helpers["show_aggrid"](df, height=260)

    if st.button("🔍 Abrir tabela no painel lateral"):
        helpers["open_side"](df, "Dados de Tensiometria")

    st.download_button(
        "⬇️ Exportar dados (CSV)",
        df.to_csv(index=False).encode("utf-8"),
        file_name=f"{sample_name}_tensiometria.csv",
        mime="text/csv",
    )

    # =====================================================
    # BLOCO 7 — SALVAR NO SUPABASE
    # =====================================================
    if supabase and st.button("💾 Salvar ensaio físico-mecânico"):
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
                    "fit_order": fit_order
                }
            )

            save_physical_mechanical_results(
                supabase,
                meas["id"],
                {
                    "theta_mean": theta_mean,
                    "theta_std": theta_std,
                    "surface_energy_total": surface_energy_total,
                    "surface_energy_components": surface_energy_components,
                    "r2": r2,
                    "fit_errors": fit_errors,
                    "classification": classification
                }
            )

            st.success("Ensaio físico-mecânico salvo com sucesso.")

        except Exception as e:
            st.error(f"Erro ao salvar no Supabase: {e}")
