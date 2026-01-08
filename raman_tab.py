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

    if not uploaded_file:
        st.info("Faça o upload de um arquivo espectral para iniciar.")
        return

    if not process_clicked:
        return

    # -----------------------------------------------------
    # Processamento
    # -----------------------------------------------------
    with st.spinner("Processando espectro Raman..."):
        try:
            result = process_raman_spectrum_with_groups(uploaded_file)
        except Exception as e:
            st.error("❌ Erro durante o processamento do espectro.")
            st.exception(e)
            return

    # -----------------------------------------------------
    # Validação do retorno
    # -----------------------------------------------------
    figures = result.get("figures", {})
    spectrum_df = result.get("spectrum_df")
    peaks_df = result.get("peaks_df")

    if spectrum_df is None or peaks_df is None:
        st.error("Resultado do processamento incompleto.")
        st.write("Chaves retornadas:", result.keys())
        return

    # -----------------------------------------------------
    # Gráficos
    # -----------------------------------------------------
    st.subheader("Espectros Raman")

    tab1, tab2, tab3 = st.tabs([
        "Bruto",
        "Baseline",
        "Processado"
    ])

    with tab1:
        if "raw" in figures:
            st.pyplot(figures["raw"])
        else:
            st.warning("Figura do espectro bruto não disponível.")

    with tab2:
        if "baseline" in figures:
            st.pyplot(figures["baseline"])
        else:
            st.warning("Figura de baseline não disponível.")

    with tab3:
        if "processed" in figures:
            st.pyplot(figures["processed"])
        else:
            st.warning("Figura do espectro processado não disponível.")

    # -----------------------------------------------------
    # Tabela de picos
    # -----------------------------------------------------
    st.subheader("Picos identificados")

    if not peaks_df.empty:
        st.dataframe(peaks_df)
    else:
        st.info("Nenhum pico detectado.")

    # -----------------------------------------------------
    # Features / Fingerprint
    # -----------------------------------------------------
    st.subheader("Fingerprint espectral")

    try:
        features = extract_raman_features(
            spectrum_df=spectrum_df,
            peaks_df=peaks_df
        )
        st.json(features["fingerprint"])
    except Exception as e:
        st.error("Erro na extração de características.")
        st.exception(e)
        return

    # -----------------------------------------------------
    # Regras exploratórias
    # -----------------------------------------------------
    if features.get("exploratory_rules"):
        st.subheader("Regras exploratórias ativadas")
        st.json(features["exploratory_rules"])
