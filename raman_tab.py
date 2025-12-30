# raman_tab.py
# -*- coding: utf-8 -*-

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from datetime import date
import uuid

# =========================================================
# HELPERS – BANCO
# =========================================================

def get_samples(supabase):
    res = (
        supabase
        .table("samples")
        .select("*")
        .order("created_at", desc=True)
        .execute()
    )
    return res.data if res.data else []


def create_experiment(supabase, sample_id, operator, equipment, notes):
    data = {
        "sample_id": sample_id,
        "experiment_type": "Raman",
        "operator": operator,
        "equipment": equipment,
        "notes": notes,
        "experiment_date": str(date.today())
    }
    res = supabase.table("experiments").insert(data).execute()
    return res.data[0]["id"]


def create_raman_measurement(
    supabase,
    experiment_id,
    laser_wavelength_nm,
    laser_power_mw,
    acquisition_time_s,
    baseline_method,
    normalization_method,
    r2_fit
):
    data = {
        "experiment_id": experiment_id,
        "laser_wavelength_nm": laser_wavelength_nm,
        "laser_power_mw": laser_power_mw,
        "acquisition_time_s": acquisition_time_s,
        "baseline_method": baseline_method,
        "normalization_method": normalization_method,
        "r2_fit": r2_fit
    }
    res = supabase.table("raman_measurements").insert(data).execute()
    return res.data[0]["id"]


def insert_raman_peaks(supabase, raman_measurement_id, peaks_df):
    if peaks_df.empty:
        return

    records = []
    for _, row in peaks_df.iterrows():
        records.append({
            "raman_measurement_id": raman_measurement_id,
            "peak_position_cm": float(row["position"]),
            "peak_intensity": float(row["intensity"]),
            "peak_fwhm": float(row.get("fwhm", 0)),
            "molecular_group": row.get("group", None)
        })

    supabase.table("raman_peaks").insert(records).execute()

# =========================================================
# UI – RAMAN TAB
# =========================================================

def render_raman_tab(supabase, helpers):
    st.header("🔬 Análises Moleculares — Espectroscopia Raman")

    # -----------------------------------------------------
    # 1. Seleção da amostra
    # -----------------------------------------------------
    samples = get_samples(supabase)

    if not samples:
        st.warning("Nenhuma amostra cadastrada.")
        return

    sample_map = {s["sample_code"]: s["id"] for s in samples}
    sample_code = st.selectbox(
        "Amostra",
        options=list(sample_map.keys())
    )
    sample_id = sample_map[sample_code]

    # -----------------------------------------------------
    # 2. Metadados do experimento
    # -----------------------------------------------------
    st.subheader("Metadados do Experimento")

    col1, col2 = st.columns(2)
    with col1:
        operator = st.text_input("Operador", "")
        equipment = st.text_input("Equipamento", "Raman Spectrometer")
    with col2:
        notes = st.text_area("Observações")

    # -----------------------------------------------------
    # 3. Parâmetros Raman
    # -----------------------------------------------------
    st.subheader("Parâmetros de Aquisição")

    col3, col4, col5 = st.columns(3)
    with col3:
        laser_wavelength_nm = st.number_input("Comprimento de onda do laser (nm)", value=785.0)
    with col4:
        laser_power_mw = st.number_input("Potência do laser (mW)", value=50.0)
    with col5:
        acquisition_time_s = st.number_input("Tempo de aquisição (s)", value=10.0)

    baseline_method = st.selectbox(
        "Método de correção de baseline",
        ["ALS", "Polynomial", "None"]
    )
    normalization_method = st.selectbox(
        "Método de normalização",
        ["Area", "Max", "None"]
    )

    r2_fit = st.number_input("R² do ajuste (opcional)", value=0.0)

    # -----------------------------------------------------
    # 4. Upload e processamento Raman
    # -----------------------------------------------------
    st.subheader("Upload do espectro Raman")

    uploaded_file = st.file_uploader(
        "Arquivo (.csv ou .txt)",
        type=["csv", "txt"]
    )

    if uploaded_file and st.button("Processar e Salvar"):
        df, peaks_df, fig = helpers.process_raman_file(uploaded_file)

        st.pyplot(fig)

        # -------------------------------------------------
        # 5. Salvar no banco
        # -------------------------------------------------
        experiment_id = create_experiment(
            supabase,
            sample_id,
            operator,
            equipment,
            notes
        )

        raman_measurement_id = create_raman_measurement(
            supabase,
            experiment_id,
            laser_wavelength_nm,
            laser_power_mw,
            acquisition_time_s,
            baseline_method,
            normalization_method,
            r2_fit
        )

        insert_raman_peaks(supabase, raman_measurement_id, peaks_df)

        st.success("✔ Análise Raman salva com sucesso!")

    # -----------------------------------------------------
    # 6. Histórico
    # -----------------------------------------------------
    st.subheader("Histórico de Análises Raman")

    history = (
        supabase
        .table("raman_measurements")
        .select("id, created_at, experiments(sample_id)")
        .execute()
    )

    if history.data:
        hist_df = pd.DataFrame(history.data)
        st.dataframe(hist_df)
