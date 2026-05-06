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
    Integração automática dos parâmetros Raman,
    elétricos, tensiométricos e topográficos
    para análise multivariada das superfícies.
    """)

    st.divider()

    # =====================================================
    # RECUPERA DADOS DOS MÓDULOS
    # =====================================================
    raman_df = st.session_state.get(
        "raman_samples",
        pd.DataFrame()
    )

    electrical_df = st.session_state.get(
        "electrical_samples",
        pd.DataFrame()
    )

    tensio_df = st.session_state.get(
        "tensiometria_samples",
        pd.DataFrame()
    )

    perfil_df = st.session_state.get(
        "perfilometria_samples",
        pd.DataFrame()
    )

    # =====================================================
    # VALIDAÇÃO
    # =====================================================
    if (
        raman_df.empty or
        electrical_df.empty or
        tensio_df.empty or
        perfil_df.empty
    ):

        st.warning("""
        Todos os módulos precisam possuir dados
        processados para executar o PCA multimodal.
        """)

        return

    # =====================================================
    # CONSTRUÇÃO DATASET MULTIMODAL
    # =====================================================
    st.subheader("🧬 Construção do Dataset Multimodal")

    try:

        multimodal_df = build_multimodal_dataframe(

            raman_df,
            electrical_df,
            tensio_df,
            perfil_df
        )

    except Exception as e:

        st.error("Erro na integração dos módulos")
        st.exception(e)

        return

    st.success("Dataset multimodal construído com sucesso.")

    st.dataframe(
        multimodal_df,
        use_container_width=True
    )

    st.divider()

    # =====================================================
    # EXECUÇÃO PCA
    # =====================================================
    st.subheader("📊 PCA Multimodal")

    try:

        result = run_multimodal_pca(
            multimodal_df
        )

    except Exception as e:

        st.error("Erro na execução do PCA")
        st.exception(e)

        return

    # =====================================================
    # FIGURA PCA
    # =====================================================
    fig = generate_pca_plot(

        result["scores"],
        result["loadings"],
        result["explained"]
    )

    st.pyplot(fig)

    st.caption("""
    PCA multimodal integrando parâmetros
    estruturais, elétricos, interfaciais
    e topográficos.
    """)

    st.divider()

    # =====================================================
    # VARIÂNCIA EXPLICADA
    # =====================================================
    st.subheader("📈 Variância Explicada")

    explained_df = pd.DataFrame({

        "Componente":

            ["PC1", "PC2"],

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
    st.subheader("🧠 Importância das Variáveis")

    st.dataframe(

        result["loadings"],

        use_container_width=True
    )

    st.caption("""
    Os loadings indicam a contribuição
    físico-química de cada parâmetro
    experimental na separação das amostras.
    """)

    st.divider()

    # =====================================================
    # EXPORTAÇÃO
    # =====================================================
    st.subheader("⬇ Exportação")

    csv = multimodal_df.to_csv(
        index=True
    )

    st.download_button(

        label="⬇ Exportar Dataset Multimodal",

        data=csv,

        file_name="surfacexlab_multimodal_pca.csv",

        mime="text/csv"
    )

    st.success("""
    PCA multimodal concluído com sucesso.
    """)
