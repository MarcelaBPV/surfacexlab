# -*- coding: utf-8 -*-

import streamlit as st
import pandas as pd
from datetime import date

from raman_processing import process_raman_spectrum_with_groups


# =========================================================
# HELPERS — BANCO DE DADOS
# =========================================================

def get_samples(supabase):
    res = (
        supabase
        .table("samples")
        .select("id, sample_code")
        .order("created_at", desc=True)
        .execute()
    )
    return res.data if res.data else []


def create_experiment(supabase, sample_id, operator, equipment, notes):
    res = supabase.table("experiments").insert({
        "sample_id": sample_id,
        "experiment_type": "Raman",
        "operator": operator,
        "equipment": equipment,
        "notes": notes,
        "experiment_date": str(date.today())
    }).execute()
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
    res = supabase.table("raman_measurements").insert({
        "experiment_id": experiment_id,
        "laser_wavelength_nm": laser_wavelength_nm,
        "laser_power_mw": laser_power_mw,
        "acquisition_time_s": acquisition_time_s,
        "baseline_method": baseline_method,
        "normalization_method": normalization_method,
        "r2_fit": r2_fit
    }).execute()
    return res.data[0]["id"]


def insert_raman_peaks(supabase, raman_measurement_id, peaks_df):
    if peaks_df is None or peaks_df.empty:
        return

    records = []
    for _, row in peaks_df.iterrows():
        records.append({
            "raman_measurement_id": raman_measurement_id,
            "peak_position_cm": float(row["center_fit"]),
            "peak_intensity": float(row["intensity_norm"]),
            "peak_fwhm": float(row["fwhm"]),
            "molecular_group": None
        })

    if records:
        supabase.table("raman_peaks").insert(records).execute()


# =========================================================
# UI — ABA RAMAN
# =========================================================

def render_raman_tab(supabase):
    st.header("🔬 Análises Moleculares — Espectroscopia Raman")

    # -----------------------------------------------------
    # 1️⃣ Seleção da amostra
    # -----------------------------------------------------
    samples = get_samples(supabase)

    if not samples:
        st.warning("Nenhuma amostra cadastrada.")
        return

    sample_map = {s["sample_code"]: s["id"] for s in samples}
    sample_code = st.selectbox("Amostra", list(sample_map.keys()))
    sample_id = sample_map[sample_code]

    # -----------------------------------------------------
    # 2️⃣ Metadados do experimento
    # -----------------------------------------------------
    st.subheader("Metadados do Experimento")

    col1, col2 = st.columns(2)
    with col1:
        operator = st.text_input("Operador")
        equipment = st.text_input("Equipamento", "Raman Spectrometer")
    with col2:
        notes = st.text_area("Observações")

    # -----------------------------------------------------
    # 3️⃣ Parâmetros Raman
    # -----------------------------------------------------
    st.subheader("Parâmetros de Aquisição")

    col3, col4, col5 = st.columns(3)
    with col3:
        laser_wavelength_nm = st.number_input(
            "Comprimento de onda do laser (nm)", value=785.0
        )
    with col4:
        laser_power_mw = st.number_input(
            "Potência do laser (mW)", value=50.0
        )
    with col5:
        acquisition_time_s = st.number_input(
            "Tempo de aquisição (s)", value=10.0
        )

    baseline_method = st.selectbox(
        "Método de correção de baseline",
        ["ASLS", "None"]
    )

    normalization_method = st.selectbox(
        "Método de normalização",
        ["Máx.", "None"]
    )

    r2_fit = st.number_input(
        "R² do ajuste (opcional)", value=0.0
    )

    # -----------------------------------------------------
    # 4️⃣ Upload e processamento Raman
    # -----------------------------------------------------
    st.subheader("Upload do Espectro Raman")

    uploaded_file = st.file_uploader(
        "Arquivo do espectro",
        type=["csv", "txt", "xls", "xlsx"]
    )

    if uploaded_file and st.button("Processar e Salvar"):

        with st.spinner("Processando espectro Raman..."):
            result = process_raman_spectrum_with_groups(
                file_like=uploaded_file,
                peak_prominence=0.02
            )

        spectrum_df = result["spectrum_df"]
        peaks_df = result["peaks_df"]
        fig = result["figure"]

        # -------------------------------------------------
        # 4.1️⃣ Gráfico Raman
        # -------------------------------------------------
        st.subheader("Espectro Raman Processado")
        st.pyplot(fig)

        # -------------------------------------------------
        # 4.2️⃣ Tabela de picos
        # -------------------------------------------------
        st.subheader("Picos identificados")

        if not peaks_df.empty:
            st.dataframe(peaks_df)
        else:
            st.info("Nenhum pico detectado.")

        # -------------------------------------------------
        # 5️⃣ Salvar no banco
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

        insert_raman_peaks(
            supabase,
            raman_measurement_id,
            peaks_df
        )

        st.success("✔ Análise Raman processada e salva com sucesso!")

    # -----------------------------------------------------
    # 6️⃣ Histórico
    # -----------------------------------------------------
    st.subheader("Histórico de Análises Raman")

    history = (
        supabase
        .table("raman_measurements")
        .select("id, created_at, experiment_id")
        .order("created_at", desc=True)
        .execute()
    )

    if history.data:
        st.dataframe(pd.DataFrame(history.data))
    else:
        st.info("Nenhuma análise Raman registrada.")
