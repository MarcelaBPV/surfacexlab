# =========================================================
# raman_mapping.py
# SurfaceXLab — Raman Molecular Mapping
# =========================================================

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


# =========================================================
# LEITOR UNIVERSAL
# =========================================================
def read_mapping_file(file_like):

    name = file_like.name.lower()

    # =====================================================
    # EXCEL
    # =====================================================
    if name.endswith((".xlsx", ".xls")):

        df = pd.read_excel(file_like)

    # =====================================================
    # CSV / TXT
    # =====================================================
    else:

        try:

            df = pd.read_csv(

                file_like,

                sep=None,

                engine="python",

                encoding="utf-8"
            )

        except:

            df = pd.read_csv(

                file_like,

                sep=None,

                engine="python",

                encoding="latin1"
            )

    return df


# =========================================================
# TAB MAPEAMENTO
# =========================================================
def render_mapeamento_molecular_tab():

    st.subheader("🗺️ Mapping Molecular Raman")

    st.markdown("""
    Processamento espacial de espectros Raman
    obtidos ao longo da superfície analisada.
    """)

    uploaded_file = st.file_uploader(

        "Upload arquivo Raman Mapping",

        type=[

            "csv",
            "txt",
            "xlsx",
            "xls"
        ]
    )

    # =====================================================
    # SEM ARQUIVO
    # =====================================================
    if uploaded_file is None:

        st.info("""
        Faça upload do arquivo contendo
        os espectros Raman do mapping.
        """)

        return

    # =====================================================
    # LEITURA
    # =====================================================
    try:

        df = read_mapping_file(
            uploaded_file
        )

    except Exception as e:

        st.error(
            "Erro ao ler arquivo"
        )

        st.exception(e)

        return

    # =====================================================
    # DATAFRAME
    # =====================================================
    st.subheader("📄 Dataset Importado")

    st.dataframe(

        df,

        width="stretch"
    )

    # =====================================================
    # DADOS NUMÉRICOS
    # =====================================================
    try:

        numeric_df = df.select_dtypes(
            include=[np.number]
        )

        # remove colunas vazias
        numeric_df = numeric_df.dropna(

            axis=1,

            how="all"
        )

        # remove linhas vazias
        numeric_df = numeric_df.dropna(

            axis=0,

            how="all"
        )

        if numeric_df.empty:

            st.warning("""
            O arquivo não possui
            dados numéricos válidos.
            """)

            return

        data = numeric_df.values

        # =================================================
        # GARANTE MATRIZ 2D
        # =================================================
        if data.ndim == 1:

            data = data.reshape(1, -1)

        # evita matriz degenerada
        if data.shape[0] < 2:

            data = np.vstack([

                data,
                data
            ])

        if data.shape[1] < 2:

            data = np.hstack([

                data,
                data
            ])

    except Exception as e:

        st.error(
            "Erro nos dados numéricos"
        )

        st.exception(e)

        return

    # =====================================================
    # MAPA RAMAN
    # =====================================================
    st.subheader("🧬 Raman Intensity Map")

    try:

        fig_map, ax_map = plt.subplots(

            figsize=(7,5),

            dpi=300
        )

        im = ax_map.imshow(

            data,

            aspect="auto",

            origin="lower",

            cmap="inferno"
        )

        cbar = plt.colorbar(

            im,

            ax=ax_map
        )

        cbar.set_label(
            "Raman Intensity"
        )

        ax_map.set_title(
            "Raman Intensity Map"
        )

        ax_map.set_xlabel(
            "Raman Shift"
        )

        ax_map.set_ylabel(
            "Spatial Position"
        )

        st.pyplot(fig_map)

    except Exception as e:

        st.error(
            "Erro na reconstrução do mapa Raman"
        )

        st.exception(e)

        return

    # =====================================================
    # ESPECTROS
    # =====================================================
    st.subheader("📈 Espectros Raman")

    try:

        fig_spec, ax_spec = plt.subplots(

            figsize=(7,5),

            dpi=300
        )

        max_spectra = min(
            18,
            data.shape[0]
        )

        x_axis = np.arange(
            data.shape[1]
        )

        for i in range(max_spectra):

            spectrum = data[i]

            spectrum_norm = (

                spectrum -
                np.min(spectrum)

            ) / (

                np.max(spectrum) -
                np.min(spectrum) + 1e-9
            )

            ax_spec.plot(

                x_axis,

                spectrum_norm,

                linewidth=1
            )

        ax_spec.set_title(
            "Raman Spectra Mapping"
        )

        ax_spec.set_xlabel(
            "Raman Shift"
        )

        ax_spec.set_ylabel(
            "Normalized Intensity"
        )

        ax_spec.grid(alpha=0.2)

        st.pyplot(fig_spec)

    except Exception as e:

        st.error(
            "Erro na plotagem dos espectros"
        )

        st.exception(e)

    # =====================================================
    # PCA MAPPING
    # =====================================================
    st.subheader("📊 PCA Mapping")

    try:

        X = StandardScaler().fit_transform(
            data
        )

        pca = PCA(
            n_components=2
        )

        scores = pca.fit_transform(X)

        explained = (
            pca.explained_variance_ratio_ * 100
        )

        fig_pca, ax_pca = plt.subplots(

            figsize=(6,5),

            dpi=300
        )

        ax_pca.scatter(

            scores[:,0],

            scores[:,1]
        )

        for i in range(scores.shape[0]):

            ax_pca.text(

                scores[i,0],

                scores[i,1],

                f"S{i+1}",

                fontsize=7
            )

        ax_pca.set_xlabel(
            f"PC1 ({explained[0]:.1f}%)"
        )

        ax_pca.set_ylabel(
            f"PC2 ({explained[1]:.1f}%)"
        )

        ax_pca.set_title(
            "PCA Raman Mapping"
        )

        ax_pca.grid(alpha=0.2)

        st.pyplot(fig_pca)

    except Exception as e:

        st.error(
            "Erro no PCA Mapping"
        )

        st.exception(e)

    # =====================================================
    # EXPORT
    # =====================================================
    st.subheader("⬇ Exportar Dados")

    csv = numeric_df.to_csv(
        index=False
    )

    st.download_button(

        label="Download Dataset Mapping",

        data=csv,

        file_name="raman_mapping.csv",

        mime="text/csv"
    )
