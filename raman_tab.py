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
import matplotlib.pyplot as plt
from typing import Dict, List
from datetime import datetime

from raman_processing import process_raman_pipeline
from raman_features import extract_raman_features

# =========================================================
# SUPABASE HELPERS
# =========================================================
def safe_insert(supabase, table: str, records: List[Dict]):
    if not supabase or not records:
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


# =========================================================
# RENDER DA ABA
# =========================================================
def render_raman_tab(supabase, helpers):

    st.subheader("Análises Moleculares — Espectroscopia Raman")

    # =====================================================
    # SESSION STATE
    # =====================================================
    if "raman_results" not in st.session_state:
        st.session_state.raman_results = None
    if "raman_features" not in st.session_state:
        st.session_state.raman_features = None
    if "selected_patient" not in st.session_state:
        st.session_state.selected_patient = None

    # =====================================================
    # BLOCO 1 — PACIENTES
    # =====================================================
    st.markdown("### Pacientes")

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
    st.markdown("### Parâmetros Raman")

    with st.expander("Configuração do processamento", expanded=True):
        sg_window = st.slider("Janela Savitzky–Golay", 5, 31, 11, 2)
        sg_poly = st.slider("Ordem do polinômio", 2, 5, 3)
        asls_lambda = st.number_input("ASLS λ", value=1e5, format="%.1e")
        asls_p = st.slider("ASLS p", 0.001, 0.1, 0.01, 0.005)
        peak_prominence = st.slider("Proeminência mínima", 0.005, 0.2, 0.02, 0.005)

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
            spectrum_df, peaks_df, fig = process_raman_pipeline(
                spectrum_file,
                sg_window=sg_window,
                sg_poly=sg_poly,
                asls_lambda=asls_lambda,
                asls_p=asls_p,
                peak_prominence=peak_prominence
            )

            features = extract_raman_features(
                spectrum_df=spectrum_df,
                peaks_df=peaks_df
            )

            st.session_state.raman_results = {
                "spectrum_df": spectrum_df,
                "peaks_df": peaks_df,
                "fig": fig
            }
            st.session_state.raman_features = features

            st.success("Espectro processado e features extraídas com sucesso.")

    # =====================================================
    # BLOCO 4 — VISUALIZAÇÃO
    # =====================================================
    if st.session_state.raman_results:

        spectrum_df = st.session_state.raman_results["spectrum_df"]
        peaks_df = st.session_state.raman_results["peaks_df"]
        fig = st.session_state.raman_results["fig"]
        features = st.session_state.raman_features

        st.markdown("### 📈 Espectro Raman processado")
        st.pyplot(fig)

        # -----------------------------
        st.markdown("### Grupos moleculares identificados")
        df_groups = features["peaks_annotated"][
            ["peak_cm1", "molecular_group", "intensity_norm", "fwhm"]
        ].dropna(subset=["molecular_group"])

        helpers["show_aggrid"](df_groups, height=220)

        # -----------------------------
        st.markdown("### Áreas integradas por grupo molecular")
        df_areas = pd.DataFrame.from_dict(
            features["group_areas"], orient="index", columns=["Área integrada"]
        ).reset_index().rename(columns={"index": "Grupo molecular"})

        helpers["show_aggrid"](df_areas, height=200)

        # -----------------------------
        st.markdown("### Razões espectrais")
        df_ratios = pd.DataFrame.from_dict(
            features["peak_ratios"], orient="index", columns=["Razão"]
        ).reset_index().rename(columns={"index": "Razão espectral"})

        helpers["show_aggrid"](df_ratios, height=180)

        # -----------------------------
        st.markdown("### Fingerprint molecular (ML-ready)")
        df_fp = pd.DataFrame([features["fingerprint"]])
        helpers["show_aggrid"](df_fp, height=160)

        if st.button("Abrir fingerprint no painel lateral"):
            helpers["open_side"](df_fp, "Fingerprint Raman")

        # -----------------------------
        if features["exploratory_rules"]:
            st.markdown("### ⚠️ Inferências exploratórias (não diagnósticas)")
            for rule in features["exploratory_rules"]:
                st.info(f"**{rule['rule']}** — {rule['description']}")

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

                # resultados espectrais
                supabase.table("results_molecular").insert({
                    "measurement_id": meas["id"],
                    "peaks": peaks_df.to_dict(orient="records"),
                }).execute()

                # features moleculares
                supabase.table("results_molecular_features").insert({
                    "measurement_id": meas["id"],
                    "group_areas": features["group_areas"],
                    "peak_ratios": features["peak_ratios"],
                    "fingerprint": features["fingerprint"],
                    "exploratory_rules": features["exploratory_rules"],
                }).execute()

                st.success("Ensaio Raman salvo com sucesso.")
