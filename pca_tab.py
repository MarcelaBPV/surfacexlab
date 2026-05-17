# =========================================================
# pca_tab.py
# SurfaceXLab — PCA Multimodal
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
    espectroscópicos, elétricos, interfaciais
    e topográficos.
    """)

    # =====================================================
    # UPLOAD
    # =====================================================
    uploaded = st.file_uploader(
        "Upload matriz experimental (.csv ou .xlsx)",
        type=["csv", "xlsx"]
    )

    if uploaded is None:

        st.info("Faça upload da matriz PCA")
        return

    # =====================================================
    # LEITURA
    # =====================================================
    if uploaded.name.endswith(".csv"):

        df = pd.read_csv(uploaded)

    else:

        df = pd.read_excel(uploaded)

    st.subheader("📋 Matriz Experimental")

    st.dataframe(
        df,
        use_container_width=True
    )

    # =====================================================
    # PCA
    # =====================================================
    try:

        result = run_pca_analysis(df)

        st.divider()

        st.subheader("📈 PCA Scores + Loadings")

        st.pyplot(result["fig"])

        # =================================================
        # MÉTRICAS
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
        st.subheader("📌 Loadings")

        st.dataframe(
            result["loadings"],
            use_container_width=True
        )

        # =================================================
        # DOWNLOAD FIGURA
        # =================================================
        with open("pca_publicacao.png", "rb") as f:

            st.download_button(
                "📥 Download Figura PCA",
                f,
                file_name="PCA_publicacao.png"
            )

    except Exception as e:

        st.error("Erro no processamento PCA")

        st.exception(e)
