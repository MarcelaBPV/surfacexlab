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
        Este módulo realiza o **processamento completo de espectros Raman**,
        incluindo:

        • Espectro bruto  
        • Correção de baseline  
        • Espectro processado  
        • Identificação automática de picos  
        • Associação com grupos moleculares  
        • **Análise multivariada por PCA**
        """
    )

    # =====================================================
    # Upload múltiplo
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

    # Containers
    all_peaks = []
    fingerprint_rows = []

    # =====================================================
    # PROCESSAMENTO INDIVIDUAL
    # =====================================================
    for file in uploaded_files:

        st.markdown(f"---\n### 📄 Amostra: `{file.name}`")

        try:
            result = process_raman_spectrum_with_groups(
                file_like=file,
                peak_prominence=0.02
            )
        except Exception as e:
            st.error("Erro ao processar o espectro Raman.")
            st.exception(e)
            continue

        # -------------------------------------------------
        # GRÁFICOS RAMAN (PADRÃO ARTIGO)
        # -------------------------------------------------
        figures = result.get("figures")

        if isinstance(figures, dict):
            for name, fig in figures.items():

                fig.set_size_inches(10, 5)
                fig.set_dpi(300)

                for ax in fig.axes:
                    ax.grid(alpha=0.25)
                    ax.tick_params(labelsize=11)
                    ax.set_xlabel(ax.get_xlabel(), fontsize=12)
                    ax.set_ylabel(ax.get_ylabel(), fontsize=12)

                st.markdown(f"**{name}**")
                st.pyplot(fig, use_container_width=True)

        # -------------------------------------------------
        # TABELA DE PICOS (ROBUSTA)
        # -------------------------------------------------
        peaks_df = result.get("peaks_df")

        st.subheader("Picos Raman Identificados")

        if isinstance(peaks_df, pd.DataFrame) and not peaks_df.empty:

            # Mapeamento flexível de colunas
            col_map = {
                "peak": ["peak_cm1", "raman_shift", "peak_position"],
                "intensity": ["intensity_norm", "intensity"],
                "group": ["molecular_group", "assignment", "group"]
            }

            resolved = {}

            for std, candidates in col_map.items():
                for c in candidates:
                    if c in peaks_df.columns:
                        resolved[std] = c
                        break

            if len(resolved) < 3:
                st.warning("Formato de picos não reconhecido.")
                st.dataframe(peaks_df)
            else:
                table = peaks_df[
                    [resolved["peak"], resolved["intensity"], resolved["group"]]
                ].copy()

                table.columns = [
                    "Pico Raman (cm⁻¹)",
                    "Intensidade normalizada",
                    "Grupo molecular"
                ]

                st.dataframe(table, use_container_width=True)

                # Guarda para PCA
                temp = table.copy()
                temp["Amostra"] = file.name
                all_peaks.append(temp)

                # Fingerprint simples (pico → intensidade)
                fp = (
                    temp
                    .groupby("Pico Raman (cm⁻¹)")["Intensidade normalizada"]
                    .mean()
                )
                fingerprint_rows.append(fp.rename(file.name))

        else:
            st.info("Nenhum pico Raman identificado.")

    # =====================================================
    # PCA RAMAN (SUBABA LÓGICA)
    # =====================================================
    if len(fingerprint_rows) < 2:
        st.info("Carregue ao menos duas amostras para habilitar a PCA Raman.")
        return

    st.markdown("---")
    st.header("📊 PCA — Espectroscopia Raman (baseada nos picos)")

    df_fp = pd.concat(fingerprint_rows, axis=1).T.fillna(0.0)

    st.subheader("Matriz de entrada da PCA")
    st.dataframe(df_fp, use_container_width=True)

    # =====================================================
    # PCA
    # =====================================================
    X = df_fp.values
    labels = df_fp.index.values

    X_scaled = StandardScaler().fit_transform(X)

    pca = PCA(n_components=2)
    scores = pca.fit_transform(X_scaled)
    loadings = pca.components_.T
    explained = pca.explained_variance_ratio_ * 100

    # =====================================================
    # BIPLOT PADRONIZADO
    # =====================================================
    fig, ax = plt.subplots(figsize=(6, 6), dpi=300)

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
    # VARIÂNCIA EXPLICADA
    # =====================================================
    st.subheader("Variância explicada")

    st.dataframe(
        pd.DataFrame({
            "Componente": ["PC1", "PC2"],
            "Variância explicada (%)": explained.round(2)
        })
    )
