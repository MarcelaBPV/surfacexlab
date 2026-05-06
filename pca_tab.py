# =========================================================
# pca_tab.py
# SurfaceXLab — Integração Multimodal
# =========================================================

import streamlit as st
import pandas as pd

from pca_processing import (

    build_multimodal_dataframe,

    run_multimodal_pca,

    generate_pca_plot
)


# =========================================================
# TAB PRINCIPAL
# =========================================================
def render_pca_tab():

    st.header("🧠 Integração Multimodal e PCA")

    st.markdown("""
    Integração automática de parâmetros espectroscópicos,
    elétricos, interfaciais e topográficos para análise
    multivariada das superfícies investigadas.
    """)

    # =====================================================
    # SESSION STATE
    # =====================================================
    raman_df = st.session_state.get(
        "raman_samples"
    )

    electrical_df = st.session_state.get(
        "electrical_samples"
    )

    tensio_df = st.session_state.get(
        "tensiometria_samples"
    )

    perfil_df = st.session_state.get(
        "perfilometria_samples"
    )

    # =====================================================
    # VALIDAÇÃO
    # =====================================================
    if any(

        x is None

        for x in [

            raman_df,
            electrical_df,
            tensio_df,
            perfil_df
        ]
    ):

        st.warning(
            "Todos os módulos devem possuir dados."
        )

        return

    # =====================================================
    # DATAFRAME MULTIMODAL
    # =====================================================
    try:

        df = build_multimodal_dataframe(

            raman_df,
            electrical_df,
            tensio_df,
            perfil_df
        )

    except Exception as e:

        st.error(
            "Erro na integração multimodal"
        )

        st.exception(e)

        return

    # =====================================================
    # PCA
    # =====================================================
    result = run_multimodal_pca(df)

    # =====================================================
    # FIGURA
    # =====================================================
    fig = generate_pca_plot(

        result["scores"],
        result["loadings"],
        result["explained"]
    )

    st.pyplot(fig)

    # =====================================================
    # VARIÂNCIA
    # =====================================================
    st.subheader("📊 Variância Explicada")

    explained_df = pd.DataFrame({

        "Componente": ["PC1", "PC2"],

        "Variância (%)":
            result["explained"]
    })

    st.dataframe(
        explained_df,
        use_container_width=True
    )

    # =====================================================
    # LOADINGS
    # =====================================================
    st.subheader("📈 Importância das Variáveis")

    st.dataframe(

        result["loadings"],

        use_container_width=True
    )

    # =====================================================
    # DATASET FINAL
    # =====================================================
    st.subheader("🧬 Dataset Multimodal")

    st.dataframe(

        df,

        use_container_width=True
    )

    # =====================================================
    # EXPORT
    # =====================================================
    csv = df.to_csv()

    st.download_button(

        label="⬇ Exportar Dataset PCA",

        data=csv,

        file_name="multimodal_pca_dataset.csv",

        mime="text/csv"
    )
