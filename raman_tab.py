# =========================================================
# raman_tab.py
# SurfaceXLab — Raman Scientific Analysis
# =========================================================

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from raman_processing import (
    process_raman_spectrum_with_groups,
    run_raman_pca
)

# =========================================================
# NOVO IMPORT
# =========================================================
from raman_mapping import (
    render_raman_mapping_tab
)

# =========================================================
# TAB RAMAN
# =========================================================
def render_raman_tab():

    st.header("🧪 Raman Scientific Analysis")

    st.markdown("""
    Plataforma científica para processamento Raman,
    deconvolução pseudo-Voigt, análise SERS,
    PCA e correlação físico-química.
    """)

    # =====================================================
    # SUB ABAS
    # =====================================================
    tab1, tab2, tab3 = st.tabs([

        "📈 Raman Spectrum",
        "🧠 PCA / Correlação",
        "🗺️ Raman Mapping"
    ])

    # =====================================================
    # =====================================================
    # TAB 1 — ESPECTRO NORMAL
    # =====================================================
    # =====================================================
    with tab1:

        # =================================================
        # SIDEBAR
        # =================================================
        st.sidebar.header("⚙ Configurações Raman")

        preset = st.sidebar.selectbox(

            "Faixa espectral",

            [
                "Biomédica",
                "Fingerprint",
                "CH Stretching",
                "Custom"
            ]
        )

        if preset == "Biomédica":

            shift_min = 1500
            shift_max = 1700

        elif preset == "Fingerprint":

            shift_min = 400
            shift_max = 1800

        elif preset == "CH Stretching":

            shift_min = 2800
            shift_max = 3100

        else:

            shift_min = 400
            shift_max = 3500

        shift_min = st.sidebar.number_input(

            "Shift mínimo",

            value=float(shift_min)
        )

        shift_max = st.sidebar.number_input(

            "Shift máximo",

            value=float(shift_max)
        )

        sg_window = st.sidebar.slider(

            "Savitzky Window",

            5,
            31,
            11,
            step=2
        )

        sg_poly = st.sidebar.slider(

            "Savitzky Polyorder",

            2,
            5,
            3
        )

        baseline_lambda = st.sidebar.number_input(

            "ALS Lambda",

            value=1e7,

            format="%.1e"
        )

        baseline_p = st.sidebar.slider(

            "ALS p",

            0.0001,
            0.01,
            0.001
        )

        prominence = st.sidebar.slider(

            "Peak Prominence",

            0.001,
            0.2,
            0.03
        )

        distance = st.sidebar.slider(

            "Peak Distance",

            1,
            100,
            20
        )

        files = st.file_uploader(

            "Upload espectros Raman",

            accept_multiple_files=True,

            type=[
                "txt",
                "csv",
                "xlsx",
                "xls"
            ]
        )

        if not files:

            st.info(
                "Faça upload dos espectros Raman."
            )

        else:

            all_features = []

            for file in files:

                st.divider()

                st.subheader(f"📈 {file.name}")

                try:

                    result = process_raman_spectrum_with_groups(

                        file_like=file,

                        shift_min=shift_min,
                        shift_max=shift_max,

                        sg_window=sg_window,
                        sg_poly=sg_poly,

                        baseline_lambda=baseline_lambda,
                        baseline_p=baseline_p,

                        prominence=prominence,
                        distance=distance
                    )

                    st.pyplot(
                        result["figures"]["deconvolution"]
                    )

                    st.pyplot(
                        result["figures"]["spectrum"]
                    )

                    summary = result["summary"]

                    col1, col2, col3 = st.columns(3)

                    col1.metric(
                        "R²",
                        summary["R²"]
                    )

                    col2.metric(
                        "N Peaks",
                        summary["N Peaks"]
                    )

                    col3.metric(
                        "Main Peak",
                        summary["Main Peak"]
                    )

                    st.subheader(
                        "📋 Peak Assignment"
                    )

                    st.dataframe(

                        result["peaks_df"],

                        use_container_width=True
                    )

                    features = result["features"]

                    all_features.append(features)

                except Exception as e:

                    st.error(
                        f"Erro no processamento de {file.name}"
                    )

                    st.exception(e)

            # =============================================
            # SESSION STATE
            # =============================================
            if len(all_features) > 0:

                features_df = pd.DataFrame(
                    all_features
                )

                st.session_state[
                    "raman_features"
                ] = features_df

    # =====================================================
    # =====================================================
    # TAB 2 — PCA
    # =====================================================
    # =====================================================
    with tab2:

        if "raman_features" not in st.session_state:

            st.info(
                "Faça upload de múltiplos espectros na aba Raman Spectrum."
            )

        else:

            features_df = st.session_state[
                "raman_features"
            ]

            numeric_df = features_df.select_dtypes(
                include="number"
            )

            st.subheader("🧠 PCA Raman")

            pca_result = run_raman_pca(
                numeric_df
            )

            st.pyplot(
                pca_result["figure"]
            )

            st.subheader(
                "📊 Correlação Físico-Química"
            )

            corr = numeric_df.corr()

            fig_corr, ax = plt.subplots(

                figsize=(10,8),

                dpi=300
            )

            im = ax.imshow(

                corr,

                cmap="coolwarm",

                aspect="auto"
            )

            ax.set_xticks(
                range(len(corr.columns))
            )

            ax.set_yticks(
                range(len(corr.columns))
            )

            ax.set_xticklabels(

                corr.columns,

                rotation=90,

                fontsize=7
            )

            ax.set_yticklabels(

                corr.columns,

                fontsize=7
            )

            fig_corr.colorbar(im)

            plt.tight_layout()

            st.pyplot(fig_corr)

            st.subheader(
                "⬇ Exportação"
            )

            csv = features_df.to_csv(
                index=False
            )

            st.download_button(

                "Exportar Features Raman",

                data=csv,

                file_name="raman_features.csv",

                mime="text/csv"
            )

    # =====================================================
    # =====================================================
    # TAB 3 — MAPEAMENTO
    # =====================================================
    # =====================================================
    with tab3:

        render_raman_mapping_tab()
