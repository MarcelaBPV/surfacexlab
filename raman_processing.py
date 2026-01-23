# -*- coding: utf-8 -*-

from typing import Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.signal import savgol_filter, find_peaks
from scipy.optimize import curve_fit
from scipy import sparse
from scipy.sparse.linalg import spsolve


# =========================================================
# DATABASE — ATRIBUIÇÃO QUÍMICA NR + CaP
# =========================================================

RAMAN_NR_DATABASE = {

    (2820, 3030): "C–H stretch (NR cis-polyisoprene)",
    (1655, 1685): "C=C stretching (NR backbone)",
    (1440, 1470): "CH deformation (NR)",
    (1280, 1320): "Amide III / proteins",

    (935, 955): "PO4 ν1 (Amorphous CaP)",
    (975, 995): "P=O stretching (DCPD / CaP)",
    (995, 1010): "Phenylalanine breathing / phosphate overlap",
}


def classify_raman_group(center):

    for (low, high), label in RAMAN_NR_DATABASE.items():
        if low <= center <= high:
            return label

    return "Unassigned"


# =========================================================
# LEITURA ROBUSTA
# =========================================================

def read_spectrum(file_like) -> Tuple[np.ndarray, np.ndarray]:

    name = getattr(file_like, "name", "").lower()

    if name.endswith((".xls", ".xlsx")):
        df = pd.read_excel(file_like, header=None)
    else:
        try:
            df = pd.read_csv(file_like, sep=None, engine="python", header=None)
        except Exception:
            file_like.seek(0)
            df = pd.read_csv(file_like, delim_whitespace=True, header=None)

    df = df.select_dtypes(include=[np.number])

    if df.shape[1] < 2:
        raise ValueError("Arquivo inválido: mínimo 2 colunas numéricas")

    x = df.iloc[:, 0].values.astype(float)
    y = df.iloc[:, 1].values.astype(float)

    idx = np.argsort(x)

    return x[idx], y[idx]


# =========================================================
# BASELINE — ASLS
# =========================================================

def asls_baseline(y, lam=1e6, p=0.01, niter=10):

    y = np.asarray(y, dtype=float)
    N = len(y)

    D = sparse.diags([1, -2, 1], [0, 1, 2], shape=(N - 2, N))
    w = np.ones(N)

    for _ in range(niter):

        W = sparse.diags(w, 0)
        Z = W + lam * D.T @ D

        z = spsolve(Z, w * y)
        w = p * (y > z) + (1 - p) * (y < z)

    return z


# =========================================================
# MODELO LORENTZ
# =========================================================

def lorentz(x, amp, cen, wid, off):

    return amp * ((0.5 * wid) ** 2 /
                  ((x - cen) ** 2 + (0.5 * wid) ** 2)) + off


def fit_lorentz(x, y, center, window=20):

    mask = (x > center - window / 2) & (x < center + window / 2)

    if mask.sum() < 8:
        return None

    xs, ys = x[mask], y[mask]

    p0 = [
        np.max(ys) - np.min(ys),
        center,
        max((xs.max() - xs.min()) / 6, 2.0),
        np.min(ys)
    ]

    try:

        popt, _ = curve_fit(lorentz, xs, ys, p0=p0, maxfev=8000)

        amp, cen, wid, off = popt

        return {
            "center_fit": float(cen),
            "amplitude": float(amp),
            "width": float(wid),
            "fwhm": float(2 * wid),
            "offset": float(off),
        }

    except Exception:
        return None


# =========================================================
# PIPELINE PRINCIPAL
# =========================================================

def process_raman_pipeline(

    sample_input,
    substrate_input=None,

    resample_points=3000,
    sg_window=11,
    sg_poly=3,

    asls_lambda=1e6,
    asls_p=0.01,

    peak_prominence=0.02,   # <<< AJUSTADO PARA DADOS REAIS
):

    # =============================
    # 1 — Leitura
    # =============================

    x_raw, y_raw = read_spectrum(sample_input)

    x_s = x_raw.copy()
    y_s = y_raw.copy()

    if substrate_input:
        x_b, y_b = read_spectrum(substrate_input)
    else:
        x_b, y_b = x_s, np.zeros_like(y_s)

    # =============================
    # 2 — Harmonização espectral
    # =============================

    x = np.linspace(
        max(x_s.min(), x_b.min()),
        min(x_s.max(), x_b.max()),
        resample_points
    )

    y_s = np.interp(x, x_s, y_s)
    y_b = np.interp(x, x_b, y_b)

    # =============================
    # 3 — Subtração substrato
    # =============================

    A = np.vstack([y_b, np.ones_like(y_b)]).T
    alpha, beta = np.linalg.lstsq(A, y_s, rcond=None)[0]

    alpha = max(alpha, 0.0)

    y_sub = y_s - alpha * y_b - beta

    # =============================
    # 4 — Baseline ASLS
    # =============================

    baseline = asls_baseline(y_sub, lam=asls_lambda, p=asls_p)
    y_corr = y_sub - baseline

    # =============================
    # 5 — Suavização
    # =============================

    if sg_window % 2 == 0:
        sg_window += 1

    y_smooth = savgol_filter(y_corr, sg_window, sg_poly)

    # =============================
    # 6 — Normalização
    # =============================

    norm = np.max(np.abs(y_smooth))
    y_norm = y_smooth / norm if norm > 0 else y_smooth

    # =============================
    # 7 — Detecção física de picos
    # =============================

    peak_idx, _ = find_peaks(
        y_norm,
        prominence=peak_prominence,
        width=6,
        distance=resample_points // 150
    )

    peaks = []

    for idx in peak_idx:

        cen = x[idx]

        fit = fit_lorentz(x, y_norm, cen)

        if not fit:
            continue

        # Filtros físicos
        if fit["amplitude"] < 0.04:
            continue

        if not (3 < fit["width"] < 80):
            continue

        group = classify_raman_group(fit["center_fit"])

        peaks.append({
            "peak_cm1": float(cen),
            "intensity_norm": float(y_norm[idx]),
            "chemical_group": group,
            **fit
        })

    peaks_df = pd.DataFrame(peaks)

    # =============================
    # Fingerprint químico
    # =============================

    if not peaks_df.empty:

        fingerprint_df = peaks_df.pivot_table(
            values="amplitude",
            columns="chemical_group",
            aggfunc="mean"
        ).fillna(0)

    else:
        fingerprint_df = pd.DataFrame()

    # =============================
    # DataFrame espectral
    # =============================

    spectrum_df = pd.DataFrame({
        "shift": x,
        "intensity_norm": y_norm,
        "baseline_norm": baseline / norm if norm > 0 else baseline
    })

    # =============================
    # Figuras base
    # =============================

    figs = {}

    fig_raw, ax = plt.subplots(figsize=(10, 4), dpi=300)
    ax.plot(x_raw, y_raw, lw=1.2)
    ax.set_title("Raw Raman Spectrum")
    figs["raw"] = fig_raw

    fig_base, ax = plt.subplots(figsize=(10, 4), dpi=300)
    ax.plot(x, y_sub, lw=1.2, label="Subtracted")
    ax.plot(x, baseline, "--", lw=1.1, label="ASLS")
    ax.legend(frameon=False)
    figs["baseline"] = fig_base

    fig_proc, ax = plt.subplots(figsize=(10, 4), dpi=300)
    ax.plot(x, y_norm, lw=1.4)
    for _, r in peaks_df.iterrows():
        ax.axvline(r["center_fit"], ls="--", lw=0.9)
    figs["processed"] = fig_proc

    return spectrum_df, peaks_df, fingerprint_df, figs


# =========================================================
# WRAPPER STREAMLIT
# =========================================================

def process_raman_spectrum_with_groups(file_like):

    spectrum_df, peaks_df, fingerprint_df, figures = process_raman_pipeline(
        sample_input=file_like
    )

    return {
        "spectrum_df": spectrum_df,
        "peaks_df": peaks_df,
        "fingerprint_df": fingerprint_df,
        "figures": figures
    }
