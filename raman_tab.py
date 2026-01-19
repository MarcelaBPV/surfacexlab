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
        **Subaba 1**  
        Processamento completo do espectro Raman com identificação automática
        dos grupos moleculares (NR + CaP) via ajuste Lorentziano.

        **Subaba 2**  
        PCA multivariada baseada exclusivamente nos picos Raman validados.
        """
    )

    # =====================================================
    # SESSION STATE
    # =====================================================
    if "raman_peaks" not in st.session_state:
        st.session_state.raman_peaks = {}

    subtabs = st.tabs([
        "📐 Upload & Processamento",
        "📊 PCA — Raman"
    ])

    # =====================================================
    # SUBABA 1 — PROCESSAMENTO
    # =====================================================
    with subtabs[0]:

        uploaded_files = st.file_uploader(
            "Upload dos espectros Raman (.csv, .txt, .xls, .xlsx)",
            type=["csv", "txt", "xls", "xlsx"],
            accept_multiple_files=True
        )

        if uploaded_files:

            for file in uploaded_files:

                if file.name in st.session_state.raman_peaks:
                    st.warning(f"{file.name} já foi processado.")
                    continue

                st.markdown(f"---\n## 📄 Amostra: `{file.name}`")

                try:
                    result = process_raman_spectrum_with_groups(
                        file_like=file
                    )
                except Exception as e:
                    st.error("Erro ao processar o espectro Raman.")
                    st.exception(e)
                    continue

                figures = result.get("figures", {})
                spectrum_df = result.get("spectrum_df")
                peaks_df = result.get("peaks_df")

                # =================================================
                # GRÁFICOS BASE
                # =================================================
                col1, col2 = st.columns(2)

                with col1:
                    st.markdown("**Espectro Bruto**")
                    if "raw" in figures:
                        st.pyplot(figures["raw"], use_container_width=True)

                with col2:
                    st.markdown("**Baseline corrigido**")
                    if "baseline" in figures:
                        st.pyplot(figures["baseline"], use_container_width=True)

                # =================================================
                # GRÁFICO FINAL — SOMENTE PICOS QUÍMICOS
                # =================================================
                st.subheader("Espectro processado — Picos químicos identificados")

                if peaks_df is None or peaks_df.empty:
                    st.warning("Nenhum pico molecular válido identificado.")
                    continue

                # -------------------------------
                # Filtra apenas picos classificados
                # -------------------------------
                peaks_valid = peaks_df[
                    peaks_df["chemical_group"] != "Unassigned"
                ].copy()

                if peaks_valid.empty:
                    st.warning("Nenhum pico com grupo molecular atribuído.")
                    continue

                # -------------------------------
                # Plot científico final
                # -------------------------------
                fig, ax = plt.subplots(figsize=(10, 4), dpi=300)

                ax.plot(
                    spectrum_df["shift"],
                    spectrum_df["intensity_norm"],
                    lw=1.4,
                    label="Processado"
                )

                ax.scatter(
                    peaks_valid["center_fit"],
                    peaks_valid["intensity_norm"],
                    s=50,
                    zorder=5,
                    label="Picos Lorentzianos"
                )

                for _, row in peaks_valid.iterrows():
                    ax.axvline(
                        row["center_fit"],
                        ls="--",
                        lw=0.9,
                        alpha=0.6
                    )

                ax.set_xlabel("Raman shift (cm⁻¹)")
                ax.set_ylabel("Intensidade normalizada")
                ax.legend(frameon=False)
                ax.set_title("Picos Raman associados aos grupos moleculares")

                st.pyplot(fig, use_container_width=True)

                # =================================================
                # TABELA CIENTÍFICA — PICOS + GRUPOS
                # =================================================
                st.subheader("Tabela de picos Raman e atribuições moleculares")

                table_df = peaks_valid[[
                    "center_fit",
                    "amplitude",
                    "fwhm",
                    "chemical_group"
                ]].copy()

                table_df.columns = [
                    "Pico Raman (cm⁻¹)",
                    "Intensidade (norm.)",
                    "FWHM (cm⁻¹)",
                    "Grupo molecular"
                ]

                table_df = table_df.sort_values("Pico Raman (cm⁻¹)")

                st.dataframe(table_df, use_container_width=True)

                # =================================================
                # FINGERPRINT PARA PCA (SÓ GRUPOS QUÍMICOS)
                # =================================================
                fingerprint = (
                    table_df
                    .groupby("Grupo molecular")["Intensidade (norm.)"]
                    .mean()
                    .astype(float)
                )

                st.session_state.raman_peaks[file.name] = fingerprint

                st.success("✔ Processamento Raman concluído")

        # =====================================================
        # PREVIEW GERAL
        # =====================================================
        if st.session_state.raman_peaks:

            st.markdown("---")
            st.subheader("📋 Fingerprints Raman armazenados")

            preview = (
                pd.DataFrame(st.session_state.raman_peaks)
                .fillna(0.0)
            )

            st.dataframe(preview, use_container_width=True)

            df_ml = preview.T.reset_index()
            df_ml = df_ml.rename(columns={"index": "Amostra"})

            st.session_state.raman_fingerprint = df_ml

            if st.button("🗑 Limpar dados Raman"):
                st.session_state.raman_peaks = {}
                st.session_state.raman_fingerprint = None
                st.experimental_rerun()

    # =====================================================
    # SUBABA 2 — PCA
    # =====================================================
    with subtabs[1]:

        if len(st.session_state.raman_peaks) < 2:
            st.info("Carregue ao menos duas amostras Raman para PCA.")
            return

        df_fp = (
            pd.DataFrame(st.session_state.raman_peaks)
            .T
            .fillna(0.0)
        )

        st.subheader("Matriz fingerprint Raman (entrada PCA)")
        st.dataframe(df_fp, use_container_width=True)

        X = df_fp.values
        labels = df_fp.index.values

        X_scaled = StandardScaler().fit_transform(X)

        pca = PCA(n_components=2)
        scores = pca.fit_transform(X_scaled)
        loadings = pca.components_.T
        explained = pca.explained_variance_ratio_ * 100

        # =================================================
        # BIPLOT PCA
        # =================================================
        fig, ax = plt.subplots(figsize=(7, 7), dpi=300)

        ax.scatter(scores[:, 0], scores[:, 1], s=90, edgecolor="black")

        for i, label in enumerate(labels):
            ax.text(
                scores[i, 0] + 0.03,
                scores[i, 1] + 0.03,
                label,
                fontsize=9
            )

        scale = np.max(np.abs(scores)) * 0.8

        for i in range(loadings.shape[0]):
            ax.arrow(
                0, 0,
                loadings[i, 0] * scale,
                loadings[i, 1] * scale,
                alpha=0.3,
                width=0.002,
                length_includes_head=True
            )

        ax.axhline(0, lw=0.6)
        ax.axvline(0, lw=0.6)

        ax.set_xlabel(f"PC1 ({explained[0]:.1f}%)")
        ax.set_ylabel(f"PC2 ({explained[1]:.1f}%)")
        ax.set_title("PCA — Fingerprint Raman")
        ax.set_aspect("equal", adjustable="box")
        ax.grid(alpha=0.3)

        st.pyplot(fig)

        st.subheader("Variância explicada")

        st.dataframe(pd.DataFrame({
            "Componente": ["PC1", "PC2"],
            "Variância explicada (%)": explained.round(2)
        }))
