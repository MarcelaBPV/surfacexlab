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
        Este módulo executa o **processamento completo de espectros Raman**:
        - Visualização do espectro **bruto**
        - Correção de **linha de base**
        - Espectro **processado**
        - Identificação automática de **picos e grupos moleculares**
        - **PCA multivariada** baseada nos picos
        """
    )

    # =====================================================
    # Upload múltiplo
    # =====================================================
    uploaded_files = st.file_uploader(
        "Upload dos espectros Raman (.csv, .txt, .xls, .xlsx)",
        type=["csv", "txt", "xls", "xlsx"],
        accept_multiple_files=True
    )

    if not uploaded_files:
        st.info("Envie um ou mais espectros Raman para iniciar.")
        return

    spectra_for_pca = []

    # =====================================================
    # PROCESSAMENTO INDIVIDUAL
    # =====================================================
    for file in uploaded_files:

        st.markdown(f"## 📄 Amostra: `{file.name}`")

        try:
            result = process_raman_spectrum_with_groups(
                file_like=file,
                peak_prominence=0.02
            )
        except Exception as e:
            st.error(f"Erro ao processar {file.name}")
            st.exception(e)
            continue

        # =================================================
        # GRÁFICOS
        # =================================================
        figures = result.get("figures", {})

        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("**Bruto**")
            if "raw" in figures:
                st.pyplot(figures["raw"], use_container_width=True)

        with col2:
            st.markdown("**Bruto + baseline**")
            if "baseline" in figures:
                st.pyplot(figures["baseline"], use_container_width=True)

        with col3:
            st.markdown("**Processado**")
            if "processed" in figures:
                st.pyplot(figures["processed"], use_container_width=True)

        # =================================================
        # TABELA DE PICOS — ROBUSTA
        # =================================================
        st.subheader("Picos identificados")

        peaks_df = result.get("peaks_df")

        if peaks_df is None or peaks_df.empty:
            st.info("Nenhum pico Raman identificado.")
            continue

        peaks_df = peaks_df.copy()

        # ----------------------------
        # Mapeamento flexível de colunas
        # ----------------------------
        column_map = {}

        for col in peaks_df.columns:
            col_l = col.lower()

            if "peak" in col_l or "shift" in col_l:
                column_map[col] = "Raman shift (cm⁻¹)"
            elif "intensity" in col_l or "height" in col_l:
                column_map[col] = "Intensidade (norm.)"
            elif "group" in col_l or "assignment" in col_l:
                column_map[col] = "Grupo molecular"

        peaks_df = peaks_df.rename(columns=column_map)

        display_cols = [
            c for c in
            ["Raman shift (cm⁻¹)", "Intensidade (norm.)", "Grupo molecular"]
            if c in peaks_df.columns
        ]

        if not display_cols:
            st.warning("Picos detectados, mas colunas não reconhecidas.")
            st.dataframe(peaks_df, use_container_width=True)
        else:
            st.dataframe(
                peaks_df[display_cols],
                use_container_width=True
            )

        # =================================================
        # MATRIZ PARA PCA (INTENSIDADE DOS PICOS)
        # =================================================
        if (
            "Raman shift (cm⁻¹)" in peaks_df.columns
            and "Intensidade (norm.)" in peaks_df.columns
        ):
            tmp = peaks_df[
                ["Raman shift (cm⁻¹)", "Intensidade (norm.)"]
            ].copy()
            tmp["Amostra"] = file.name
            spectra_for_pca.append(tmp)

    # =====================================================
    # PCA RAMAN
    # =====================================================
    if len(spectra_for_pca) < 2:
        st.info("Carregue ao menos duas amostras para habilitar a PCA.")
        return

    df_all = pd.concat(spectra_for_pca)

    pivot = df_all.pivot_table(
        index="Amostra",
        columns="Raman shift (cm⁻¹)",
        values="Intensidade (norm.)",
        fill_value=0.0
    )

    st.subheader("Matriz espectral (entrada da PCA)")
    st.dataframe(pivot, use_container_width=True)

    # =====================================================
    # PCA
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
    st.subheader("PCA — Espectroscopia Raman")

    fig, ax = plt.subplots(figsize=(6, 6), dpi=300)

    ax.scatter(
        scores[:, 0],
        scores[:, 1],
        s=90,
        color="#1f77b4",
        edgecolor="black"
    )

    for i, label in enumerate(labels):
        ax.text(
            scores[i, 0] + 0.04,
            scores[i, 1] + 0.04,
            label,
            fontsize=9
        )

    scale = np.max(np.abs(scores)) * 0.8
    step = max(1, loadings.shape[0] // 25)

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
    ax.set_title("PCA — Raman")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(alpha=0.3)

    st.pyplot(fig)

    # =====================================================
    # Variância explicada
    # =====================================================
    st.subheader("Variância explicada")
    st.dataframe(pd.DataFrame({
        "Componente": ["PC1", "PC2"],
        "Variância explicada (%)": explained.round(2)
    }))
