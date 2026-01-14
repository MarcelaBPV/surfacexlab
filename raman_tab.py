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
        com identificação de picos, atribuição molecular e **análise PCA
        baseada exclusivamente nos picos detectados**.
        """
    )

    # =====================================================
    # SUBABAS
    # =====================================================
    subtabs = st.tabs([
        "🔬 Processamento Raman",
        "📊 PCA — Picos Raman"
    ])

    # =====================================================
    # SUBABA 1 — PROCESSAMENTO
    # =====================================================
    with subtabs[0]:

        uploaded_files = st.file_uploader(
            "Upload dos espectros Raman (.csv, .txt, .xls, .xlsx)",
            type=["csv", "txt", "xls", "xlsx"],
            accept_multiple_files=True,
            key="raman_upload"
        )

        if not uploaded_files:
            st.info("Envie um ou mais espectros Raman.")
            return

        # Armazenamento para PCA
        if "raman_peaks" not in st.session_state:
            st.session_state["raman_peaks"] = []

        st.session_state["raman_peaks"].clear()

        for file in uploaded_files:

            st.markdown(f"### 📄 Amostra: `{file.name}`")

            try:
                result = process_raman_spectrum_with_groups(
                    file_like=file,
                    peak_prominence=0.02
                )
            except Exception as e:
                st.error(f"Erro ao processar {file.name}")
                st.exception(e)
                continue

            figures = result.get("figures", {})
            peaks_df = result.get("peaks_df")

            # -----------------------------
            # GRÁFICOS
            # -----------------------------
            if isinstance(figures, dict):
                col1, col2, col3 = st.columns(3)

                with col1:
                    st.caption("Bruto")
                    st.pyplot(figures.get("raw"))

                with col2:
                    st.caption("Bruto + baseline")
                    st.pyplot(figures.get("baseline"))

                with col3:
                    st.caption("Processado")
                    st.pyplot(figures.get("processed"))

            # -----------------------------
            # TABELA DE PICOS
            # -----------------------------
            st.subheader("Picos identificados")

            if isinstance(peaks_df, pd.DataFrame) and not peaks_df.empty:

                table = peaks_df[[
                    "peak_cm1",
                    "intensity_norm",
                    "molecular_group"
                ]].copy()

                table.columns = [
                    "Pico (cm⁻¹)",
                    "Intensidade normalizada",
                    "Grupo molecular"
                ]

                table["Amostra"] = file.name

                st.dataframe(table, use_container_width=True)

                st.session_state["raman_peaks"].append(table)

            else:
                st.info("Nenhum pico identificado.")

    # =====================================================
    # SUBABA 2 — PCA RAMAN (PICOS)
    # =====================================================
    with subtabs[1]:

        st.subheader("📊 PCA — Intensidades dos Picos Raman")

        if "raman_peaks" not in st.session_state or len(st.session_state["raman_peaks"]) < 2:
            st.info("Carregue ao menos duas amostras na aba de processamento.")
            return

        df_peaks = pd.concat(st.session_state["raman_peaks"])

        # Pivot: amostra × pico
        pivot = df_peaks.pivot_table(
            index="Amostra",
            columns="Pico (cm⁻¹)",
            values="Intensidade normalizada"
        ).fillna(0.0)

        st.caption("Matriz PCA (amostras × picos)")
        st.dataframe(pivot, use_container_width=True)

        # PCA
        X = StandardScaler().fit_transform(pivot.values)
        labels = pivot.index.values

        pca = PCA(n_components=2)
        scores = pca.fit_transform(X)
        loadings = pca.components_.T
        explained = pca.explained_variance_ratio_ * 100

        # -----------------------------
        # BIPLOT PADRONIZADO
        # -----------------------------
        fig, ax = plt.subplots(figsize=(6, 6), dpi=300)

        ax.scatter(
            scores[:, 0],
            scores[:, 1],
            s=80,
            edgecolor="black"
        )

        for i, label in enumerate(labels):
            ax.text(
                scores[i, 0] + 0.03,
                scores[i, 1] + 0.03,
                label,
                fontsize=9
            )

        scale = np.max(np.abs(scores)) * 0.9

        for i, peak in enumerate(pivot.columns):
            ax.arrow(
                0, 0,
                loadings[i, 0] * scale,
                loadings[i, 1] * scale,
                color="red",
                alpha=0.4,
                head_width=0.08,
                length_includes_head=True
            )

        ax.axhline(0, color="gray", lw=0.6)
        ax.axvline(0, color="gray", lw=0.6)
        ax.set_xlabel(f"PC1 ({explained[0]:.1f}%)")
        ax.set_ylabel(f"PC2 ({explained[1]:.1f}%)")
        ax.set_title("PCA — Picos Raman")
        ax.grid(alpha=0.3)

        st.pyplot(fig)

        st.subheader("Variância explicada")
        st.dataframe(pd.DataFrame({
            "Componente": ["PC1", "PC2"],
            "Variância (%)": explained.round(2)
        }))
