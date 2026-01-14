# raman_tab.py
# -*- coding: utf-8 -*-

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from raman_processing import process_raman_spectrum_with_groups


# =========================================================
# ABA RAMAN
# =========================================================
def render_raman_tab(supabase=None):

    st.header("🧬 Análises Moleculares — Espectroscopia Raman")

    st.markdown(
        """
        Este módulo realiza o **processamento completo de espectros Raman brutos**,
        incluindo correção de baseline, suavização, normalização, detecção automática
        de picos e análise multivariada por PCA.
        """
    )

    # =====================================================
    # Upload múltiplo (para PCA)
    # =====================================================
    uploaded_files = st.file_uploader(
        "Upload dos espectros Raman (.csv, .txt, .xls, .xlsx)",
        type=["csv", "txt", "xls", "xlsx"],
        accept_multiple_files=True,
        key="raman_upload"
    )

    if not uploaded_files:
        st.info("Envie um ou mais espectros Raman para iniciar.")
        return

    # =====================================================
    # Processamento individual
    # =====================================================
    spectra = []
    figures = []

    for file in uploaded_files:
        try:
            result = process_raman_spectrum_with_groups(
                file_like=file,
                peak_prominence=0.02
            )

            spectrum_df = result.get("spectrum_df")
            fig_dict = result.get("figures")

            if spectrum_df is not None:
                spectrum_df = spectrum_df.copy()
                spectrum_df["Amostra"] = file.name
                spectra.append(spectrum_df)

            if isinstance(fig_dict, dict) and "processed" in fig_dict:
                figures.append((file.name, fig_dict["processed"]))

        except Exception as e:
            st.error(f"Erro ao processar {file.name}")
            st.exception(e)

    # =====================================================
    # Visualização dos espectros processados
    # =====================================================
    st.subheader("Espectros Raman Processados")

    for name, fig in figures:
        st.markdown(f"**{name}**")
        st.pyplot(fig)

    if len(spectra) < 2:
        st.info("Carregue ao menos dois espectros para habilitar a PCA.")
        return

    # =====================================================
    # MATRIZ PARA PCA
    # =====================================================
    df_all = pd.concat(spectra, axis=0)

    pivot = df_all.pivot_table(
        index="Amostra",
        columns="raman_shift_cm1",
        values="intensity_norm"
    )

    pivot = pivot.fillna(0.0)

    st.subheader("Matriz espectral (entrada da PCA)")
    st.dataframe(pivot, use_container_width=True)

    # =====================================================
    # PCA RAMAN
    # =====================================================
    X = pivot.values
    labels = pivot.index.values

    X_scaled = StandardScaler().fit_transform(X)

    pca = PCA(n_components=2)
    scores = pca.fit_transform(X_scaled)
    loadings = pca.components_.T
    explained = pca.explained_variance_ratio_ * 100

    # =====================================================
    # BIPLOT PADRONIZADO
    # =====================================================
    st.subheader("PCA — Biplot Raman")

    fig, ax = plt.subplots(figsize=(6, 6), dpi=300)

    # Scores
    ax.scatter(
        scores[:, 0],
        scores[:, 1],
        s=80,
        color="#1f77b4",
        edgecolor="black"
    )

    for i, label in enumerate(labels):
        ax.text(
            scores[i, 0] + 0.03,
            scores[i, 1] + 0.03,
            label,
            fontsize=9
        )

    # Loadings (reduzidos visualmente)
    scale = np.max(np.abs(scores)) * 0.8
    step = max(1, loadings.shape[0] // 20)

    for i in range(0, loadings.shape[0], step):
        ax.arrow(
            0, 0,
            loadings[i, 0] * scale,
            loadings[i, 1] * scale,
            color="black",
            alpha=0.3,
            width=0.002,
            length_includes_head=True
        )

    ax.axhline(0, color="gray", lw=0.6)
    ax.axvline(0, color="gray", lw=0.6)

    ax.set_xlabel(f"PC1 ({explained[0]:.1f}%)")
    ax.set_ylabel(f"PC2 ({explained[1]:.1f}%)")
    ax.set_title("PCA — Espectroscopia Raman")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(alpha=0.3)

    st.pyplot(fig)

    # =====================================================
    # Variância explicada
    # =====================================================
    st.subheader("Variância explicada")

    st.dataframe(
        pd.DataFrame({
            "Componente": ["PC1", "PC2"],
            "Variância explicada (%)": explained.round(2)
        })
    )
