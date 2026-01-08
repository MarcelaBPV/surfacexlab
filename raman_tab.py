# -*- coding: utf-8 -*-
"""
SurfaceXLab — Raman Processing Pipeline
Scientific-grade, PCA-ready
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve
from sklearn.preprocessing import StandardScaler


# =========================================================
# BASELINE — ASLS
# =========================================================
def baseline_asls(y, lam=1e5, p=0.01, niter=10):
    L = len(y)
    D = diags([1, -2, 1], [0, -1, -2], shape=(L, L-2))
    w = np.ones(L)

    for _ in range(niter):
        W = diags(w, 0)
        Z = W + lam * D.dot(D.T)
        z = spsolve(Z, w * y)
        w = p * (y > z) + (1 - p) * (y < z)

    return z


# =========================================================
# FIGURES — PADRÃO ARTIGO
# =========================================================
def article_figure(figsize=(8, 4)):
    fig, ax = plt.subplots(figsize=figsize, dpi=300)
    ax.tick_params(direction="in", top=True, right=True)
    return fig, ax


# =========================================================
# PIPELINE PRINCIPAL
# =========================================================
def process_raman_spectrum_with_groups(
    file_like,
    peak_prominence=0.02
):
    # -----------------------------------------------------
    # 1️⃣ Leitura
    # -----------------------------------------------------
    if file_like.name.endswith(".csv"):
        df = pd.read_csv(file_like)
    elif file_like.name.endswith((".xls", ".xlsx")):
        df = pd.read_excel(file_like)
    else:
        df = pd.read_csv(file_like, sep=None, engine="python")

    df.columns = ["shift", "intensity"]
    df = df.sort_values("shift")

    raw_df = df.copy()

    # -----------------------------------------------------
    # 2️⃣ Baseline
    # -----------------------------------------------------
    baseline = baseline_asls(df["intensity"].values)
    baseline_df = pd.DataFrame({
        "shift": df["shift"],
        "baseline": baseline
    })

    corrected_intensity = df["intensity"] - baseline

    # -----------------------------------------------------
    # 3️⃣ Normalização
    # -----------------------------------------------------
    norm_intensity = corrected_intensity / np.max(corrected_intensity)

    spectrum_df = pd.DataFrame({
        "shift": df["shift"],
        "intensity_norm": norm_intensity
    })

    # -----------------------------------------------------
    # 4️⃣ Detecção de picos
    # -----------------------------------------------------
    peaks, props = find_peaks(
        norm_intensity,
        prominence=peak_prominence
    )

    peaks_df = pd.DataFrame({
        "peak_cm1": df["shift"].iloc[peaks],
        "intensity_norm": norm_intensity[peaks],
        "center_fit": df["shift"].iloc[peaks],
        "amplitude": props["prominences"],
        "width": props.get("widths", np.nan),
        "fwhm": props.get("widths", np.nan),
        "offset": 0.0
    })

    # -----------------------------------------------------
    # 5️⃣ FIGURAS
    # -----------------------------------------------------

    # 🔹 Bruto
    fig_raw, ax = article_figure()
    ax.plot(raw_df["shift"], raw_df["intensity"], color="black")
    ax.set_title("Espectro Raman Bruto")
    ax.set_xlabel("Raman shift (cm⁻¹)")
    ax.set_ylabel("Intensidade (a.u.)")

    # 🔹 Baseline
    fig_baseline, ax = article_figure()
    ax.plot(raw_df["shift"], raw_df["intensity"], color="black", label="Bruto")
    ax.plot(baseline_df["shift"], baseline_df["baseline"],
            "--", color="gray", label="Baseline")
    ax.legend()
    ax.set_title("Correção de Baseline (ASLS)")

    # 🔹 Processado
    fig_processed, ax = article_figure()
    ax.plot(spectrum_df["shift"], spectrum_df["intensity_norm"], color="black")

    for peak in peaks_df["center_fit"]:
        ax.axvline(peak, color="dodgerblue", linestyle="--", alpha=0.5)

    ax.set_title("Espectro Raman Processado")
    ax.set_xlabel("Raman shift (cm⁻¹)")
    ax.set_ylabel("Intensidade normalizada")

    # -----------------------------------------------------
    # 6️⃣ RETORNO COMPLETO
    # -----------------------------------------------------
    return {
        "raw_df": raw_df,
        "baseline_df": baseline_df,
        "corrected_df": pd.DataFrame({
            "shift": df["shift"],
            "intensity": corrected_intensity
        }),
        "spectrum_df": spectrum_df,
        "peaks_df": peaks_df,
        "figures": {
            "raw": fig_raw,
            "baseline": fig_baseline,
            "processed": fig_processed
        }
    }
