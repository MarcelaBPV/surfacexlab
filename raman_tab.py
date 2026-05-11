# =========================================================
# raman_tab.py
# SurfaceXLab — Raman Module
# VERSÃO FINAL ESTÁVEL
# =========================================================

import streamlit as st
import pandas as pd


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

    st.header("🧬 Raman — SurfaceXLab")

    st.markdown("""
    Plataforma integrada para processamento
    espectral Raman, deconvolução Pseudo-Voigt,
    identificação molecular e análise multimodal.
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

        st.subheader(
            "📐 Processamento Espectral"
        )

        st.markdown("""
        Upload de espectros Raman para:

        - correção de baseline;
        - suavização Savitzky-Golay;
        - deconvolução Pseudo-Voigt;
        - identificação molecular;
        - extração automática de parâmetros.
        """)

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

        # =================================================
        # SEM ARQUIVOS
        # =================================================
        if not files:

            st.info("""
            Faça upload dos espectros Raman.
            """)

        else:

            # =============================================
            # DATASET PCA
            # =============================================
            all_samples = []

            # =============================================
            # LOOP DOS ARQUIVOS
            # =============================================
            for file in files:

                st.divider()

                st.markdown(
                    f"## 📄 {file.name}"
                )

                # =========================================
                # PROCESSAMENTO
                # =========================================
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
                # ESPECTRO
                # =========================================
                st.subheader(
                    "📈 Espectro Raman"
                )

                st.pyplot(
                    result["figures"]["spectrum"]
                )

                # =========================================
                # DECONVOLUÇÃO
                # =========================================
                st.subheader(
                    "🧬 Deconvolução Pseudo-Voigt"
                )

                st.pyplot(
                    result["figures"]["deconvolution"]
                )

                # =========================================
                # SUMMARY
                # =========================================
                st.subheader(
                    "📊 Parâmetros Gerais"
                )

                summary_df = pd.DataFrame([

                    result["summary"]
                ])

                st.dataframe(

                    summary_df,

                    width="stretch"
                )

                # =========================================
                # PICOS
                # =========================================
                st.subheader(
                    "🧪 Tabela de Picos"
                )

                peaks_df = result["peaks_df"]

                st.dataframe(

                    peaks_df,

                    width="stretch"
                )

                # =========================================
                # DOWNLOAD
                # =========================================
                csv_peaks = peaks_df.to_csv(
                    index=False
                )

                st.download_button(

                    label="⬇ Download Peaks Table",

                    data=csv_peaks,

                    file_name=f"{file.name}_peaks.csv",

                    mime="text/csv"
                )

                # =========================================
                # PCA DATASET
                # =========================================
                all_samples.append(

                    result["summary"]
                )

            # =============================================
            # SESSION STATE
            # =============================================
            if len(all_samples) > 0:

                raman_df = pd.DataFrame(
                    all_samples
                )

                st.session_state[
                    "raman_samples"
                ] = raman_df

                st.success("""
                Dataset Raman salvo
                para PCA multimodal.
                """)

    # =====================================================
    # SUBABA 2
    # =====================================================
    with subtabs[1]:

        render_mapeamento_molecular_tab()

    # =====================================================
    # SUBABA 3 — PCA
    # =====================================================
    with subtabs[2]:

        st.subheader("📊 PCA Raman")

        st.markdown("""
        Análise multivariada dos parâmetros
        espectrais Raman.
        """)

        # =================================================
        # SESSION STATE
        # =================================================
        raman_df = st.session_state.get(
            "raman_samples"
        )

        # =================================================
        # SEM DADOS
        # =================================================
        if raman_df is None:

            st.warning("""
            Nenhum dataset Raman encontrado.
            """)

            return

        # =================================================
        # GARANTE DATAFRAME
        # =================================================
        if not isinstance(
            raman_df,
            pd.DataFrame
        ):

            raman_df = pd.DataFrame(
                raman_df
            )

        # =================================================
        # DATAFRAME VAZIO
        # =================================================
        if len(raman_df) == 0:

            st.warning("""
            Dataset Raman vazio.
            """)

            return

        # =================================================
        # PCA
        # =================================================
        try:

            pca_result = run_raman_pca(
                raman_df
            )

        except Exception as e:

            st.error(
                "Erro no PCA Raman"
            )

            st.exception(e)

            return

        # =================================================
        # FIGURA PCA
        # =================================================
        st.pyplot(
            pca_result["figure"]
        )

        # =================================================
        # SCORES
        # =================================================
        st.subheader(
            "🧠 Scores PCA"
        )

        st.dataframe(

            pca_result["scores"],

            width="stretch"
        )

        # =================================================
        # LOADINGS
        # =================================================
        st.subheader(
            "📈 Loadings"
        )

        st.dataframe(

            pca_result["loadings"],

            width="stretch"
        )

        # =================================================
        # VARIÂNCIA
        # =================================================
        st.subheader(
            "📊 Variância Explicada"
        )

        explained_df = pd.DataFrame({

            "Componente": [

                "PC1",
                "PC2"
            ],

            "Variância (%)":

                pca_result["explained"]
        })

        st.dataframe(

            explained_df,

            width="stretch"
        )

        # =================================================
        # EXPORT
        # =================================================
        csv_pca = raman_df.to_csv(
            index=False
        )

        st.download_button(

            label="⬇ Download Dataset PCA",

            data=csv_pca,

            file_name="raman_pca_dataset.csv",

            mime="text/csv"
        )

        st.success("""
        PCA Raman executado com sucesso.
        """)
