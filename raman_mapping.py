# =========================================================
# raman_mapping.py
# SurfaceXLab — Raman Molecular Mapping
# =========================================================

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# =========================================================
# TAB MAPEAMENTO
# =========================================================
def render_mapeamento_molecular_tab():

    st.subheader("🗺️ Mapping Molecular Raman")

    st.markdown("""
    Upload de espectros Raman obtidos ao longo
    da varredura espacial de amostras biológicas.
    """)

    uploaded_file = st.file_uploader(

        "Upload arquivo mapping Raman",

        type=["csv", "txt", "xlsx", "xls"]
    )

    if uploaded_file is None:

        st.info("""
        Faça upload do arquivo contendo
        os espectros do mapping.
        """)

        return

    # =====================================================
    # LEITURA
    # =====================================================
    try:

        if uploaded_file.name.endswith(

            (".xlsx", ".xls")
        ):

            df = pd.read_excel(
                uploaded_file
            )

        else:

            df = pd.read_csv(
                uploaded_file
            )

    except Exception as e:

        st.error(
            "Erro ao ler arquivo"
        )

        st.exception(e)

        return

    # =====================================================
    # VISUALIZAÇÃO
    # =====================================================
    st.subheader("📄 Dataset Mapping")

    st.dataframe(
        df,
        use_container_width=True
    )

    # =====================================================
    # SIMULA MAPA
    # =====================================================
    st.subheader("🧬 Intensidade Raman")

    try:

        numeric_df = df.select_dtypes(
            include=[np.number]
        )

        data = numeric_df.values

        fig, ax = plt.subplots(

            figsize=(6,5),
            dpi=300
        )

        im = ax.imshow(

            data,

            aspect="auto"
        )

        plt.colorbar(
            im,
            ax=ax,
            label="Intensidade"
        )

        ax.set_title(
            "Mapa Raman Molecular"
        )

        ax.set_xlabel(
            "Deslocamento Raman"
        )

        ax.set_ylabel(
            "Posição Espacial"
        )

        st.pyplot(fig)

    except Exception as e:

        st.error(
            "Erro no mapa Raman"
        )

        st.exception(e)
