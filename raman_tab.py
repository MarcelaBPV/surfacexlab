# raman_tab.py
# -*- coding: utf-8 -*-

"""
SurfaceXLab — Aba Raman (Streamlit)
"""

import streamlit as st

from raman_processing import process_raman_spectrum_with_groups
from raman_features import extract_raman_features


def render_raman_tab(supabase=None):
    st.header("🔬 Análises Moleculares — Espectroscopia Raman")

    # -----------------------------------------------------
    # Upload do espectro
    # -----------------------------------------------------
    uploaded_file = st.file_uploader(
        "Arquivo do espectro Raman",
        type=["csv", "txt", "xls", "xlsx"]
    )

    process_clicked = st.button("Processar espectro")

    if uploaded_file and process_clicked:

        with st.spinner("Processando espectro Raman..."):
            result = process_raman_spectrum_with_groups(
                uploaded_file
            )

        # -------------------------------------------------
        # Gráficos
        # -------------------------------------------------
        st.subheader("Espectros Raman")

        tab1, tab2, tab3 = st.tabs([
            "Bruto",
            "Baseline",
            "Processado"
        ])

        with tab1:
            st.pyplot(result["figures"]["raw"])

        with tab2:
            st.pyplot(result["figures"]["baseline"])

        with tab3:
            st.pyplot(result["figures"]["processed"])

        # -------------------------------------------------
        # Tabela de picos
        # -------------------------------------------------
        st.subheader("Picos identificados")

        if not result["peaks_df"].empty:
            st.dataframe(result["peaks_df"])
        else:
            st.info("Nenhum pico detectado.")

        # -------------------------------------------------
        # Features / Fingerprint
        # -------------------------------------------------
        st.subheader("Fingerprint espectral")

        features = extract_raman_features(
            spectrum_df=result["spectrum_df"],
            peaks_df=result["peaks_df"]
        )

        st.json(features["fingerprint"])

        # -------------------------------------------------
        # Regras exploratórias
        # -------------------------------------------------
        if features["exploratory_rules"]:
            st.subheader("Regras exploratórias ativadas")
            st.json(features["exploratory_rules"])
