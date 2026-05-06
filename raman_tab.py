# =========================================================
# raman_tab.py
# SurfaceXLab — Raman Module
# =========================================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# =========================================================
# IMPORTS
# =========================================================
from raman_processing import (

    process_raman_spectrum_with_groups,

    run_raman_pca
)

from raman_mapping import (
    render_mapeamento_molecular_tab
)


# =========================================================
# TAB PRINCIPAL
# =========================================================
def render_raman_tab():

    st.header("🧬 Análises Moleculares")

    st.markdown("""
    Processamento espectral Raman com:
    correção de linha de base,
    suavização Savitzky–Golay,
    ajuste Lorentziano,
    identificação molecular
    e PCA espectral.
    """)

    st.divider()

    # =====================================================
    # SUBABAS
    # =====================================================
    subtabs = st.tabs([

        "📐 Processamento Raman",

        "🗺️ Mapping Molecular",

        "📊 PCA Raman"
    ])

    # =====================================================
    # SUBABA 1
    # =====================================================
    with subtabs[0]:

        st.subheader("📐 Processamento Raman")

        files = st.file_uploader(

            "Upload espectros Raman",

            type=[

                "csv",
                "txt",
                "xlsx",
                "xls",
                "log"
            ],

            accept_multiple_files=True
        )

        if files:

            all_samples = []

            for file in files:

                st.divider()

                st.markdown(f"### 📄 {file.name}")

                try:

                    result = process_raman_spectrum_with_groups(
                        file
                    )

                except Exception as e:

                    st.error(
                        "Erro no processamento Raman"
                    )

                    st.exception(e)

                    continue

                # =========================================
                # FIGURAS
                # =========================================
                st.pyplot(
                    result["figures"]["spectrum"]
                )

                st.pyplot(
                    result["figures"]["deconvolution"]
                )

                # =========================================
                # PARÂMETROS
                # =========================================
                st.subheader("📊 Parâmetros Raman")

                summary_df = pd.DataFrame(

                    [result["summary"]]
                )

                st.dataframe(

                    summary_df,

                    use_container_width=True
                )

                # =========================================
                # PICOS
                # =========================================
                st.subheader("🧬 Picos Identificados")

                st.dataframe(

                    result["peaks_df"],

                    use_container_width=True
                )

                # =========================================
                # SESSION STATE
                # =========================================
                all_samples.append(

                    result["summary"]
                )

            # =============================================
            # SALVA DATASET
            # =============================================
            if all_samples:

                st.session_state[
                    "raman_samples"
                ] = pd.DataFrame(all_samples)

                st.success("""
                Dataset Raman salvo para
                integração multimodal.
                """)

    # =====================================================
    # SUBABA 2
    # =====================================================
    with subtabs[1]:

        render_mapeamento_molecular_tab()

    # =====================================================
    # SUBABA 3
    # =====================================================
    with subtabs[2]:

        st.subheader("📊 PCA Raman")

        raman_df = st.session_state.get(

            "raman_samples",

            pd.DataFrame()
        )

        if raman_df.empty:

            st.warning("""
            Carregue espectros Raman
            para executar o PCA.
            """)

            return

        try:

            pca_result = run_raman_pca(
                raman_df
            )

        except Exception as e:

            st.error("Erro no PCA Raman")

            st.exception(e)

            return

        # =============================================
        # FIGURA PCA
        # =============================================
        st.pyplot(
            pca_result["figure"]
        )

        # =============================================
        # SCORES
        # =============================================
        st.subheader("🧠 Scores PCA")

        st.dataframe(

            pca_result["scores"],

            use_container_width=True
        )

        # =============================================
        # LOADINGS
        # =============================================
        st.subheader("📈 Loadings")

        st.dataframe(

            pca_result["loadings"],

            use_container_width=True
        )

        # =============================================
        # VARIÂNCIA
        # =============================================
        st.subheader("📊 Variância Explicada")

        explained_df = pd.DataFrame({

            "Componente":

                ["PC1", "PC2"],

            "Variância (%)":

                pca_result["explained"]
        })

        st.dataframe(

            explained_df,

            use_container_width=True
        )

        st.success("""
        PCA Raman executado com sucesso.
        """)
