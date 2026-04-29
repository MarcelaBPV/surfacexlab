# =========================================================
# Raman Tab — SurfaceXLab (VERSÃO FINAL CORRIGIDA)
# =========================================================

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
# IMPORT DO MAPEAMENTO (SEM TRY/EXCEPT)
# =========================================================
from raman_mapping_tab import render_mapeamento_molecular_tab


# =========================================================
# PLOT PAPER STYLE (LORENTZ + RESÍDUO + R²)
# =========================================================
def plot_raman_paper_style(x, y_exp, peaks_df):

    def lorentz_plot(x, amp, cen, fwhm):
        gamma = 0.5 * fwhm
        return amp * ((gamma)**2 / ((x - cen)**2 + gamma**2))

    peaks_df = peaks_df.sort_values("center_fit")

    peak_curves = []
    peak_sum = np.zeros_like(x)
    peak_centers = []

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

    if peak_sum.max() > 0:
        peak_sum = peak_sum / peak_sum.max() * y_exp.max()

    y_exp_s = gaussian_filter1d(y_exp, sigma=1.1)
    peak_sum_s = gaussian_filter1d(peak_sum, sigma=1.1)

    peak_curves_s = [
        gaussian_filter1d(curve, sigma=1.1)
        for curve in peak_curves
    ]

    residual = y_exp_s - peak_sum_s
    r2 = r2_score(y_exp_s, peak_sum_s)

    fig, (ax1, ax2) = plt.subplots(
        2, 1,
        figsize=(6.8, 5.4),
        dpi=300,
        gridspec_kw={"height_ratios": [3, 1]},
        sharex=True
    )

    # Espectro
    ax1.plot(x, y_exp_s, color="black", linewidth=1.0, label="Experimental")

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

    ax1.plot(x, peak_sum_s, color="crimson", linewidth=1.6, label="PeakSum")

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
        ax1.axvline(cen, linestyle="--", linewidth=0.7, alpha=0.5)

    ax1.text(
        0.02, 0.93,
        f"$R^2$ = {r2:.4f}",
        transform=ax1.transAxes,
        fontsize=10
    )

    ax1.set_ylabel("Intensity (a.u.)")
    ax1.legend(frameon=False, fontsize=8)

    # Resíduo
    ax2.plot(x, residual, color="black", linewidth=0.9)
    ax2.axhline(0, linestyle="--", linewidth=0.7)

    ax2.set_xlabel("Raman Shift (cm⁻¹)")
    ax2.set_ylabel("Residual")

    plt.tight_layout(pad=0.4)

    return fig, r2


# =========================================================
# TAB PRINCIPAL
# =========================================================
def render_raman_tab(supabase=None):

    st.header("🧬 Análises Moleculares")

    st.markdown("""
    - Ajuste Lorentziano multipeak  
    - Validação estatística (R²)  
    - PCA multivariada  
    - Mapeamento espacial Raman  
    """)

    # =====================================================
    # SESSION STATE
    # =====================================================
    if "raman_peaks" not in st.session_state:
        st.session_state.raman_peaks = {}

    # =====================================================
    # SUBABAS
    # =====================================================
    subtabs = st.tabs([
        "📐 Upload & Processamento",
        "📊 PCA — Raman",
        "🗺️ Mapeamento Raman"
    ])

    # =====================================================
    # SUBABA 1 — PROCESSAMENTO
    # =====================================================
    with subtabs[0]:

        uploaded_files = st.file_uploader(
            "Upload dos espectros Raman",
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

                spectrum_df = result.get("spectrum_df")
                peaks_df = result.get("peaks_df")
                figures = result.get("figures", {})

                col1, col2 = st.columns(2)

                with col1:
                    if "raw" in figures:
                        st.pyplot(figures["raw"])

                with col2:
                    if "baseline" in figures:
                        st.pyplot(figures["baseline"])

                if peaks_df is None or peaks_df.empty:
                    st.warning("Nenhum pico identificado.")
                    continue

                peaks_valid = peaks_df[
                    peaks_df["chemical_group"] != "Unassigned"
                ].copy()

                if peaks_valid.empty:
                    st.warning("Sem picos válidos.")
                    continue

                st.subheader("Decomposição Lorentziana")

                fig_paper, r2_value = plot_raman_paper_style(
                    spectrum_df["shift"].values,
                    spectrum_df["intensity_norm"].values,
                    peaks_valid
                )

                st.pyplot(fig_paper)

                st.success(f"R² = {r2_value:.4f}")

                table_df = peaks_valid[[
                    "center_fit",
                    "amplitude",
                    "fwhm",
                    "chemical_group"
                ]].copy()

                table_df.columns = [
                    "Pico (cm⁻¹)",
                    "Intensidade",
                    "FWHM",
                    "Grupo"
                ]

                st.dataframe(table_df)

                fingerprint = (
                    table_df
                    .groupby("Grupo")["Intensidade"]
                    .mean()
                    .astype(float)
                )

                st.session_state.raman_peaks[file.name] = fingerprint

                st.success("✔ Processado")

        if st.session_state.raman_peaks:

            st.subheader("Fingerprints")

            preview = pd.DataFrame(st.session_state.raman_peaks).fillna(0)
            st.dataframe(preview)

    # =====================================================
    # SUBABA 2 — PCA
    # =====================================================
    with subtabs[1]:

        if len(st.session_state.raman_peaks) < 2:
            st.info("Carregue pelo menos duas amostras.")
            return

        df_fp = pd.DataFrame(st.session_state.raman_peaks).T.fillna(0)

        X = StandardScaler().fit_transform(df_fp.values)

        pca = PCA(n_components=2)
        scores = pca.fit_transform(X)

        fig, ax = plt.subplots()

        ax.scatter(scores[:, 0], scores[:, 1])

        for i, label in enumerate(df_fp.index):
            ax.text(scores[i, 0], scores[i, 1], label)

        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")

        st.pyplot(fig)

    # =====================================================
    # SUBABA 3 — MAPEAMENTO
    # =====================================================
    with subtabs[2]:

        render_mapeamento_molecular_tab(supabase)
