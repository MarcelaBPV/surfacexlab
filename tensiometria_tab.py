# tensiometria_tab.py
# -*- coding: utf-8 -*-

"""
Aba 3 — Análises Físico-Mecânicas (Tensiometria Óptica)
CRM científico:
Paciente → Amostra → Ensaio → Energia superficial → Banco
"""

# =========================================================
# IMPORTS
# =========================================================
import streamlit as st
import pandas as pd
from typing import Dict
from datetime import datetime

from tensiometria_processing import process_tensiometry

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
        "module": "physical_mechanical",
        "raw_data": raw_meta,
        "taken_at": datetime.utcnow().isoformat()
    }).execute()
    return res.data[0]


# =========================================================
# RENDER DA ABA
# =========================================================
def render_tensiometria_tab(supabase, helpers):

    st.subheader("Análises Físico-Mecânicas — Tensiometria Óptica")

    st.markdown(
        """
Esta aba realiza a **análise físico-mecânica de superfícies** por meio de
**medições do ângulo de contato**, permitindo o cálculo da:

- Energia livre de superfície total  
- Componentes **dispersiva** e **polar** (OWRK)  
- Classificação de **molhabilidade**  

⚠️ Uso científico — **não diagnóstico**.
"""
    )

    # =====================================================
    # SESSION STATE
    # =====================================================
    if "tensio_results" not in st.session_state:
        st.session_state.tensio_results = None

    # =====================================================
    # BLOCO 1 — AMOSTRA (CRM)
    # =====================================================
    st.markdown("### Amostras")

    sample_name = st.text_input("Identificação da amostra / superfície")
    description = st.text_area("Material, tratamento superficial ou observações")

    # =====================================================
    # BLOCO 2 — CONFIGURAÇÃO EXPERIMENTAL
    # =====================================================
    st.markdown("### Configuração experimental")

    liquid_name = st.selectbox(
        "Líquido padrão utilizado",
        ["water", "diiodomethane", "formamide"],
        index=0,
        help="Necessário para o cálculo OWRK",
    )

    # =====================================================
    # BLOCO 3 — UPLOAD DO LOG
    # =====================================================
    st.markdown("### Upload do arquivo do goniômetro")

    uploaded = st.file_uploader(
        "Arquivo de tensiometria (.LOG, .TXT ou .CSV)",
        type=["log", "txt", "csv"],
    )

    if uploaded is None:
        st.info("Envie um arquivo do goniômetro para iniciar a análise.")
        return

    # =====================================================
    # BLOCO 4 — PROCESSAMENTO
    # =====================================================
    try:
        results = process_tensiometry(
            file_like=uploaded,
            liquid_name=liquid_name,
        )
        st.session_state.tensio_results = results
        st.success("Dados processados com sucesso.")

    except Exception as e:
        st.error(f"Erro ao processar o arquivo: {e}")
        return

    # =====================================================
    # BLOCO 5 — KPIs
    # =====================================================
    stats = results["statistics"]
    owkr = results["owrk"]

    st.markdown("### 📊 Indicadores principais")

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Ângulo médio (°)", f"{stats['theta_mean_deg']:.2f}")
    k2.metric("Desvio padrão (°)", f"{stats['theta_std_deg']:.2f}")
    k3.metric("Energia superficial (mJ/m²)", f"{owkr['gamma_s_total']:.2f}")
    k4.metric("Molhabilidade", results["wettability"])

    # =====================================================
    # BLOCO 6 — VISUALIZAÇÃO
    # =====================================================
    st.markdown("### 📈 Evolução temporal do ângulo de contato")
    st.pyplot(results["figure"])

    st.markdown("### 📋 Dados experimentais tratados")
    helpers["show_aggrid"](results["df_clean"], height=260)

    if st.button("🔍 Abrir dados no painel lateral"):
        helpers["open_side"](results["df_clean"], "Dados de Tensiometria")

    # =====================================================
    # BLOCO 7 — RESULTADOS OWRK
    # =====================================================
    st.markdown("### Energia livre de superfície (OWRK)")

    df_energy = pd.DataFrame([{
        "Energia total (mJ/m²)": owkr["gamma_s_total"],
        "Componente dispersiva (mJ/m²)": owkr["gamma_s_d"],
        "Componente polar (mJ/m²)": owkr["gamma_s_p"],
        "Fração polar": owkr["polar_fraction"],
        "R² do ajuste": owkr["R2"],
    }])

    helpers["show_aggrid"](df_energy, height=140)

    # =====================================================
    # BLOCO 8 — SALVAR NO SUPABASE
    # =====================================================
    if supabase and st.button("💾 Salvar ensaio físico-mecânico"):
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
                    "liquid": liquid_name,
                }
            )

            supabase.table("results_physical_mechanical").insert({
                "measurement_id": meas["id"],
                "contact_angle_stats": stats,
                "surface_energy": owkr,
                "wettability": results["wettability"],
            }).execute()

            st.success("Ensaio físico-mecânico salvo com sucesso.")

        except Exception as e:
            st.error(f"Erro ao salvar no Supabase: {e}")
