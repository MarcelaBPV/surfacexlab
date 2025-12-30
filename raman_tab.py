# raman_tab.py
# -*- coding: utf-8 -*-

"""
Aba 1 — Análises Moleculares (Raman)
CRM científico:
Paciente → Amostra → Espectro → Features → Banco
"""

# =========================================================
# IMPORTS
# =========================================================
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Optional
from datetime import datetime

from raman_processing import process_raman_pipeline

# =========================================================
# SUPABASE HELPERS
# =========================================================
def safe_insert(supabase, table: str, records: List[Dict]):
    if not supabase:
        return []
    if not records:
        return []
    batch = 500
    out = []
    for i in range(0, len(records), batch):
        res = supabase.table(table).insert(records[i:i+batch]).execute()
        out.extend(res.data or [])
    return out


def get_patients(supabase):
    if not supabase:
        return []
    res = supabase.table("patients").select("*").order("created_at").execute()
    return res.data or []


def create_patient(supabase, data: Dict):
    if not supabase:
        return None
    res = supabase.table("patients").insert(data).execute()
    return res.data[0]


def create_sample(supabase, patient_id: str, name: str):
    res = supabase.table("samples").insert({
        "patient_id": patient_id,
        "name": name,
        "created_at": datetime.utcnow().isoformat()
    }).execute()
    return res.data[0]


def create_measurement(supabase, sample_id: str, raw_meta: Dict):
    res = supabase.table("measurements_raw").insert({
        "sample_id": sample_id,
        "module": "molecular",
        "raw_data": raw_meta,
        "taken_at": datetime.utcnow().isoformat()
    }).execute()
    return res.data[0]


def save_raman_results(supabase, measurement_id: str, results: Dict):
    supabase.table("results_molecular").insert({
        "measurement_id": measurement_id,
        "peak_positions": results["peaks"],
        "intensities": results["intensities"],
        "fwhm": results["fwhm"],
        "molecular_groups": results["groups"],
        "created_at": datetime.utcnow().isoformat()
    }).execute()


# =========================================================
# RENDER DA ABA
# =========================================================
def render_raman_tab(supabase, helpers):

    st.subheader("🧬 Análises Moleculares — Espectroscopia Raman")

    # =====================================================
    # SESSION STATE LOCAL
    # =====================================================
    if "raman_results" not in st.session_state:
        st.session_state.raman_results = None
    if "selected_patient" not in st.session_state:
        st.session_state.selected_patient = None

    # =====================================================
    # BLOCO 1 — PACIENTES (CRM)
    # =====================================================
    st.markdown("### 👤 Pacientes")

    patients = get_patients(supabase) if supabase else []
    patient_names = ["— Novo paciente —"] + [
        f'{p["name"]} ({p.get("email","")})' for p in patients
    ]

    selected = st.selectbox("Selecionar paciente", patient_names)

    if selected == "— Novo paciente —":
        with st.expander("Cadastrar novo paciente", expanded=True):
            name = st.text_input("Nome")
            email = st.text_input("E-mail")
            genero = st.selectbox("Gênero", ["F", "M", "Outro"])
            fumante = st.selectbox("Fumante", ["não", "sim"])
            doenca = st.text_input("Doença declarada", value="controle")

            if st.button("➕ Cadastrar paciente"):
                if supabase:
                    patient = create_patient(supabase, {
                        "name": name,
                        "email": email,
                        "genero": genero,
                        "fumante": fumante,
                        "doenca": doenca,
                        "created_at": datetime.utcnow().isoformat()
                    })
                    st.session_state.selected_patient = patient
                    st.success("Paciente cadastrado.")
                else:
                    st.warning("Supabase não configurado.")
    else:
        idx = patient_names.index(selected) - 1
        st.session_state.selected_patient = patients[idx]

    # =====================================================
    # BLOCO 2 — PARÂMETROS RAMAN
    # =====================================================
    st.markdown("### ⚙️ Parâmetros Raman")

    with st.expander("Configuração do processamento", expanded=True):
        fit_model = st.selectbox("Modelo de ajuste", [None, "gauss", "lorentz", "voigt"])
        col1, col2, col3 = st.columns(3)
        with col1:
            peak_height = st.slider("Altura mínima", 0.0, 1.0, 0.03, 0.01)
        with col2:
            peak_prominence = st.slider("Proeminência", 0.0, 1.0, 0.03, 0.01)
        with col3:
            peak_distance = st.slider("Distância mínima", 1, 500, 5)

    # =====================================================
    # BLOCO 3 — UPLOAD & PROCESSAMENTO
    # =====================================================
    st.markdown("### 📤 Upload do espectro Raman")

    spectrum_file = st.file_uploader(
        "Arquivo Raman (.txt, .csv, .xlsx)",
        type=["txt", "csv", "xls", "xlsx"]
    )

    if st.button("▶ Processar espectro"):
        if spectrum_file is None:
            st.warning("Faça upload do espectro.")
        else:
            results = process_raman_pipeline(
                spectrum_file,
                peak_height=peak_height,
                peak_prominence=peak_prominence,
                peak_distance=peak_distance,
                fit_model=fit_model
            )
            st.session_state.raman_results = results
            st.success("Espectro processado com sucesso.")

    # =====================================================
    # BLOCO 4 — VISUALIZAÇÃO
    # =====================================================
    if st.session_state.raman_results:
        data = st.session_state.raman_results

        st.markdown("### 📈 Espectro processado")

        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(data["x_proc"], data["y_proc"], lw=1.4)
        ax.set_xlabel("Raman shift (cm⁻¹)")
        ax.set_ylabel("Intensidade normalizada (u.a.)")
        st.pyplot(fig)

        st.markdown("### 📊 Features Raman (ML-ready)")

        df_feat = pd.DataFrame([data["features"]])
        helpers["show_aggrid"](df_feat, height=180)

        if st.button("🔍 Abrir detalhes"):
            helpers["open_side"](df_feat, "Features Raman")

        # =================================================
        # BLOCO 5 — SALVAR NO BANCO
        # =================================================
        if supabase and st.session_state.selected_patient:
            if st.button("💾 Salvar ensaio Raman"):
                patient_id = st.session_state.selected_patient["id"]

                sample = create_sample(
                    supabase,
                    patient_id,
                    name=f"Raman {datetime.now().strftime('%Y-%m-%d %H:%M')}"
                )

                meas = create_measurement(
                    supabase,
                    sample["id"],
                    raw_meta={"filename": spectrum_file.name}
                )

                save_raman_results(supabase, meas["id"], data)

                st.success("Ensaio Raman salvo com sucesso.")
