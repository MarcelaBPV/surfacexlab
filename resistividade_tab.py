# =========================================================
# pca_tab.py
# SurfaceXLab — PCA Multimodal
# Upload Manual + Integração Automática
# =========================================================

import streamlit as st
import pandas as pd

from pca_processing import run_pca_analysis


# =========================================================
# TAB PCA
# =========================================================
def render_pca_tab():

    st.header("📊 PCA Multimodal")

    st.markdown("""
    Integração multimodal de parâmetros
    espectroscópicos, elétricos,
    interfaciais e topográficos.
    """)

    # =====================================================
    # MODO
    # =====================================================
    mode = st.radio(

        "Modo de análise",

        [

            "📂 Upload Manual",
            "🔗 Integração Automática"

        ],

        horizontal=True
    )

    st.divider()

    # =====================================================
    # =====================================================
    # MODO 1 — UPLOAD MANUAL
    # =====================================================
    # =====================================================
    if mode == "📂 Upload Manual":

        with st.expander(
            "📋 Exemplo da matriz experimental"
        ):

            example_df = pd.DataFrame({

                "Amostra": [
                    "C",
                    "LD",
                    "MD",
                    "HD"
                ],

                "Rrms": [
                    0.3,
                    1.5,
                    2.4,
                    1.0
                ],

                "ID_IG": [
                    0.44,
                    0.55,
                    0.52,
                    0.60
                ],

                "I2D_IG": [
                    0.23,
                    0.37,
                    0.61,
                    0.71
                ],

                "Theta": [
                    119,
                    133,
                    140,
                    152
                ]

            })

            st.dataframe(
                example_df,
                use_container_width=True
            )

        # =================================================
        # UPLOAD
        # =================================================
        uploaded = st.file_uploader(

            "Upload matriz PCA (.csv ou .xlsx)",

            type=[
                "csv",
                "xlsx"
            ]
        )

        if uploaded is None:

            st.info(
                "Faça upload da matriz experimental"
            )

            return

        # =================================================
        # LEITURA
        # =================================================
        try:

            if uploaded.name.endswith(".csv"):

                df = pd.read_csv(uploaded)

            else:

                df = pd.read_excel(uploaded)

            st.subheader(
                "📋 Matriz Experimental"
            )

            st.dataframe(
                df,
                use_container_width=True
            )

            st.session_state[
                "pca_samples"
            ] = df

        except Exception as e:

            st.error(
                "Erro na leitura do arquivo"
            )

            st.exception(e)

            return

        # =================================================
        # PCA
        # =================================================
        run_pca(df)

    # =====================================================
    # =====================================================
    # MODO 2 — INTEGRAÇÃO AUTOMÁTICA
    # =====================================================
    # =====================================================
    else:

        st.subheader(
            "🔗 Integração automática dos módulos"
        )

        # =================================================
        # CHECKBOXES
        # =================================================
        use_raman = st.checkbox(
            "🧬 Raman",
            value=True
        )

        use_electrical = st.checkbox(
            "⚡ Resistividade",
            value=True
        )

        use_tensiometry = st.checkbox(
            "💧 Tensiometria",
            value=True
        )

        use_profilometry = st.checkbox(
            "📏 Perfilometria",
            value=True
        )

        # =================================================
        # GERAR MATRIZ
        # =================================================
        if st.button(
            "📊 Gerar PCA Multimodal"
        ):

            try:

                integrated_df = build_multimodal_dataframe(

                    use_raman,
                    use_electrical,
                    use_tensiometry,
                    use_profilometry

                )

                if integrated_df.empty:

                    st.warning(
                        "Nenhum dado disponível."
                    )

                    return

                st.subheader(
                    "📋 Matriz Integrada"
                )

                st.dataframe(
                    integrated_df,
                    use_container_width=True
                )

                st.session_state[
                    "pca_samples"
                ] = integrated_df

                # =========================================
                # PCA
                # =========================================
                run_pca(
                    integrated_df
                )

            except Exception as e:

                st.error(
                    "Erro na integração multimodal"
                )

                st.exception(e)


# =========================================================
# EXECUTA PCA
# =========================================================
def run_pca(df):

    try:

        result = run_pca_analysis(df)

        st.divider()

        st.subheader(
            "📈 PCA Scores + Loadings"
        )

        st.pyplot(
            result["fig"]
        )

        # =================================================
        # VARIÂNCIA
        # =================================================
        col1, col2 = st.columns(2)

        col1.metric(

            "PC1",

            f'{result["pc1"]:.1f}%'
        )

        col2.metric(

            "PC2",

            f'{result["pc2"]:.1f}%'
        )

        # =================================================
        # LOADINGS
        # =================================================
        st.subheader(
            "📌 Loadings"
        )

        st.dataframe(

            result["loadings"],

            use_container_width=True
        )

        # =================================================
        # DOWNLOAD FIGURA
        # =================================================
        with open(
            "pca_nanotubos.png",
            "rb"
        ) as f:

            st.download_button(

                "📥 Download Figura PCA",

                f,

                file_name="PCA_multimodal.png"
            )

    except Exception as e:

        st.error(
            "Erro no processamento PCA"
        )

        st.exception(e)


# =========================================================
# INTEGRAÇÃO AUTOMÁTICA
# =========================================================
def build_multimodal_dataframe(

    use_raman,
    use_electrical,
    use_tensiometry,
    use_profilometry

):

    data = {}

    # =====================================================
    # RAMAN
    # =====================================================
    if use_raman:

        raman_df = st.session_state.get(
            "raman_samples"
        )

        if isinstance(raman_df, pd.DataFrame):

            if "ID/IG" in raman_df.columns:

                data["ID_IG"] = raman_df[
                    "ID/IG"
                ]

    # =====================================================
    # ELÉTRICO
    # =====================================================
    if use_electrical:

        elec_df = st.session_state.get(
            "electrical_samples"
        )

        if isinstance(elec_df, pd.DataFrame):

            if "Resistividade" in elec_df.columns:

                data["Resistividade"] = elec_df[
                    "Resistividade"
                ]

    # =====================================================
    # TENSIOMETRIA
    # =====================================================
    if use_tensiometry:

        tens_df = st.session_state.get(
            "tensiometria_samples"
        )

        if isinstance(tens_df, pd.DataFrame):

            if "Theta final (°)" in tens_df.columns:

                data["Theta"] = tens_df[
                    "Theta final (°)"
                ]

    # =====================================================
    # PERFILOMETRIA
    # =====================================================
    if use_profilometry:

        prof_df = st.session_state.get(
            "perfilometria_samples"
        )

        if isinstance(prof_df, pd.DataFrame):

            if "Rq" in prof_df.columns:

                data["Rrms"] = prof_df[
                    "Rq"
                ]

    # =====================================================
    # DATAFRAME FINAL
    # =====================================================
    integrated_df = pd.DataFrame(data)

    integrated_df.insert(

        0,

        "Amostra",

        [

            f"S{i+1}"
            for i in range(
                len(integrated_df)
            )
        ]
    )

    return integrated_df
