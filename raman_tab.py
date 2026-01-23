# raman_tab.py
# -*- coding: utf-8 -*-

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import r2_score

from scipy.ndimage import gaussian_filter1d

from raman_processing import process_raman_spectrum_with_groups


# =========================================================
# FUNÇÃO — PLOT PAPER STYLE + R² + RESIDUAL + PEAK MARKERS
# =========================================================

def plot_raman_paper_style(x, y_exp, peaks_df):

    def lorentz_plot(x, amp, cen, fwhm):
        gamma = 0.5 * fwhm
        return amp * ((gamma)**2 / ((x - cen)**2 + gamma**2))

    # Ordena fisicamente os picos
    peaks_df = peaks_df.sort_values("center_fit")

    peak_curves = []
    peak_sum = np.zeros_like(x)
    peak_centers = []

    # ==========================
    # Calcula curvas Lorentzianas
    # ==========================

    for _, row in peaks_df.iterrows():

        curve = lorentz_plot(
            x,
            row["amplitude"],
            row["center_fit"],
            row["fwhm"]
        )

        peak_curves.append(curve)
        peak_sum += curve
        peak_centers.append(row["center_fit"])

    # ==========================
    # Normaliza PeakSum
    # ==========================

    if peak_sum.max() > 0:
        peak_sum = peak_sum / peak_sum.max() * y_exp.max()

    # ==========================
    # Suavização VISUAL (plot only)
    # ==========================

    y_exp_s = gaussian_filter1d(y_exp, sigma=1.1)
    peak_sum_s = gaussian_filter1d(peak_sum, sigma=1.1)

    peak_curves_s = [
        gaussian_filter1d(curve, sigma=1.1)
        for curve in peak_curves
    ]

    residual = y_exp_s - peak_sum_s

    # ==========================
    # Métrica R²
    # ==========================

    r2 = r2_score(y_exp_s, peak_sum_s)

    # ==========================
    # FIGURA PAPER — 2 PAINÉIS
    # ==========================

    fig, (ax1, ax2) = plt.subplots(
        2, 1,
        figsize=(6.8, 5.4),
        dpi=300,
        gridspec_kw={"height_ratios": [3, 1]},
        sharex=True
    )

    # =====================================================
    # PAINEL SUPERIOR — FIT
    # =====================================================

    # Experimental
    ax1.plot(
        x,
        y_exp_s,
        color="black",
        linewidth=1.0,
        label="Experimental"
    )

    # Picos individuais
    colors = ["#1f77b4", "#9467bd", "#2ca02c", "#ff7f0e", "#8c564b"]

    for i, (curve, (_, row)) in enumerate(zip(peak_curves_s, peaks_df.iterrows())):

        ax1.plot(
            x,
            curve,
            linewidth=0.9,
            color=colors[i % len(colors)],
            alpha=0.85,
            label=row["chemical_group"]
        )

    # PeakSum
    ax1.plot(
        x,
        peak_sum_s,
        color="crimson",
        linewidth=1.6,
        label="PeakSum"
    )

    # ==========================
    # Marcação dos centros dos picos
    # ==========================

    for cen in peak_centers:

        idx = np.argmin(np.abs(x - cen))

        ax1.scatter(
            x[idx],
            peak_sum_s[idx],
            s=36,
            facecolors="white",
            edgecolors="black",
            linewidths=0.8,
            zorder=5
        )

        ax1.axvline(
            cen,
            linestyle="--",
            linewidth=0.7,
            alpha=0.5
        )

    # Texto R²
    ax1.text(
        0.02, 0.93,
        f"$R^2$ = {r2:.4f}",
        transform=ax1.transAxes,
        fontsize=10
    )

    ax1.set_ylabel("Intensity (a.u.)", fontsize=11)

    ax1.tick_params(direction="in", length=4, width=0.9)

    for spine in ax1.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.9)

    ax1.legend(frameon=False, fontsize=8)

    # =====================================================
    # PAINEL INFERIOR — RESIDUAL
    # =====================================================

    ax2.plot(
        x,
        residual,
        color="black",
        linewidth=0.9
    )

    ax2.axhline(0, linestyle="--", linewidth=0.7)

    ax2.set_xlabel("Raman Shift (cm$^{-1}$)", fontsize=11)
    ax2.set_ylabel("Residual", fontsize=10)

    ax2.tick_params(direction="in", length=4, width=0.9)

    for spine in ax2.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.9)

    ax1.margins(x=0)
    ax2.margins(x=0)

    plt.tight_layout(pad=0.4)

    return fig, r2


# =========================================================
# ABA RAMAN
# =========================================================

def render_raman_tab(supabase=None):

    st.header("🧬 Análises Moleculares — Espectroscopia Raman")

    st.markdown("""
    **Subaba 1**  
    Processamento completo do espectro Raman com ajuste Lorentziano multipeak,
    validação estatística (R²) e análise do espectro residual.

    **Subaba 2**  
    PCA multivariada baseada exclusivamente nos fingerprints Raman.
    """)

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
                    result = process_raman_spectrum_with_groups(file_like=file)
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
                    st.markdown("**Espectro bruto**")
                    if "raw" in figures:
                        st.pyplot(figures["raw"], use_container_width=True)

                with col2:
                    st.markdown("**Baseline corrigido**")
                    if "baseline" in figures:
                        st.pyplot(figures["baseline"], use_container_width=True)

                # =================================================
                # FILTRO — PICOS QUÍMICOS
                # =================================================

                if peaks_df is None or peaks_df.empty:
                    st.warning("Nenhum pico molecular identificado.")
                    continue

                peaks_valid = peaks_df[
                    peaks_df["chemical_group"] != "Unassigned"
                ].copy()

                if peaks_valid.empty:
                    st.warning("Nenhum pico com atribuição química válida.")
                    continue

                # =================================================
                # PLOT PAPER + R² + RESIDUAL
                # =================================================

                st.subheader("Decomposição Lorentziana — Validação Estatística")

                fig_paper, r2_value = plot_raman_paper_style(
                    spectrum_df["shift"].values,
                    spectrum_df["intensity_norm"].values,
                    peaks_valid
                )

                st.pyplot(fig_paper, use_container_width=True)

                st.success(f"Coeficiente de determinação do ajuste: R² = {r2_value:.4f}")

                # =================================================
                # TABELA CIENTÍFICA
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
                # FINGERPRINT PARA PCA
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

            if st.button("🗑 Limpar dados Raman", key="clear_raman"):
                st.session_state.raman_peaks = {}
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

        fig, ax = plt.subplots(figsize=(6, 6), dpi=300)

        ax.scatter(scores[:, 0], scores[:, 1], s=60, edgecolors="black")

        for i, label in enumerate(labels):
            ax.text(scores[i, 0], scores[i, 1], label, fontsize=9)

        scale = np.max(np.abs(scores)) * 0.85

        for i in range(loadings.shape[0]):
            ax.arrow(
                0, 0,
                loadings[i, 0] * scale,
                loadings[i, 1] * scale,
                linewidth=0.9,
                length_includes_head=True
            )

        ax.axhline(0, lw=0.8)
        ax.axvline(0, lw=0.8)

        ax.set_xlabel(f"PC1 ({explained[0]:.1f}%)")
        ax.set_ylabel(f"PC2 ({explained[1]:.1f}%)")

        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(0.9)

        ax.set_aspect("equal", adjustable="box")

        plt.tight_layout(pad=0.4)

        st.pyplot(fig)

        st.subheader("Variância explicada")

        st.dataframe(pd.DataFrame({
            "Componente": ["PC1", "PC2"],
            "Variância explicada (%)": explained.round(2)
        }))
