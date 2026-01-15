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
        Upload e processamento completo do espectro Raman  
        (bruto, baseline, processado, picos e grupos moleculares)

        **Subaba 2**  
        PCA multivariada baseada **exclusivamente nos picos Raman**
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
                        file_like=file,
                        peak_prominence=0.02
                    )
                except Exception as e:
                    st.error("Erro ao processar o espectro Raman.")
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
                # PICOS RAMAN
                # =================================================
                peaks_df = result.get("peaks_df")

                st.subheader("Picos Raman identificados")

                if peaks_df is None or peaks_df.empty:
                    st.info("Nenhum pico Raman identificado.")
                    continue

                peaks_df = peaks_df.copy()

                col_map = {}

                for c in peaks_df.columns:
                    cl = c.lower()
                    if "peak" in cl or "shift" in cl:
                        col_map[c] = "Pico Raman (cm⁻¹)"
                    elif "intensity" in cl or "height" in cl:
                        col_map[c] = "Intensidade (norm.)"
                    elif "group" in cl or "assignment" in cl:
                        col_map[c] = "Grupo molecular"

                peaks_df = peaks_df.rename(columns=col_map)

                required = [
                    "Pico Raman (cm⁻¹)",
                    "Intensidade (norm.)",
                    "Grupo molecular"
                ]

                if not all(c in peaks_df.columns for c in required):
                    st.warning("Formato de picos não reconhecido.")
                    st.dataframe(peaks_df, use_container_width=True)
                    continue

                display_df = peaks_df[required].copy()

                st.dataframe(display_df, use_container_width=True)

                # =================================================
                # FINGERPRINT PARA PCA + ML
                # =================================================
                fingerprint = (
                    display_df
                    .groupby("Pico Raman (cm⁻¹)")["Intensidade (norm.)"]
                    .mean()
                    .astype(float)
                )

                st.session_state.raman_peaks[file.name] = fingerprint

                st.success("✔ Amostra Raman processada com sucesso")

        # =====================================================
        # PREVIEW GLOBAL
        # =====================================================
        if st.session_state.raman_peaks:

            st.markdown("---")
            st.subheader("📋 Amostras Raman processadas")

            preview = (
                pd.DataFrame(st.session_state.raman_peaks)
                .fillna(0.0)
            )

            st.dataframe(preview, use_container_width=True)

            # 👉 EXPORTA PARA ML GLOBAL
            df_ml = preview.T.reset_index()
            df_ml = df_ml.rename(columns={"index": "Amostra"})

            st.session_state.raman_fingerprint = df_ml

            if st.button("🗑 Limpar amostras Raman"):
                st.session_state.raman_peaks = {}
                st.session_state.raman_fingerprint = None
                st.experimental_rerun()

    # =====================================================
    # SUBABA 2 — PCA RAMAN
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

        st.subheader("Matriz fingerprint (entrada da PCA)")
        st.dataframe(df_fp, use_container_width=True)

        X = df_fp.values
        labels = df_fp.index.values

        X_scaled = StandardScaler().fit_transform(X)

        pca = PCA(n_components=2)
        scores = pca.fit_transform(X_scaled)
        loadings = pca.components_.T
        explained = pca.explained_variance_ratio_ * 100

        # ---------------------------
        # BIPLOT
        # ---------------------------
        fig, ax = plt.subplots(figsize=(7, 7), dpi=300)

        ax.scatter(scores[:, 0], scores[:, 1], s=90, edgecolor="black")

        for i, label in enumerate(labels):
            ax.text(scores[i, 0] + 0.03, scores[i, 1] + 0.03, label, fontsize=9)

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
        ax.set_title("PCA — Espectroscopia Raman")
        ax.set_aspect("equal", adjustable="box")
        ax.grid(alpha=0.3)

        st.pyplot(fig)

        st.subheader("Variância explicada")

        st.dataframe(pd.DataFrame({
            "Componente": ["PC1", "PC2"],
            "Variância explicada (%)": explained.round(2)
        }))
