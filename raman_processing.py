# raman_processing.py
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.signal import savgol_filter, find_peaks
from scipy.optimize import curve_fit
from scipy import sparse
from scipy.sparse.linalg import spsolve


# =========================================================
# BASE DE DADOS RAMAN — NR + CaP
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


def classify_raman_group(center_cm1):

    for (low, high), label in RAMAN_NR_DATABASE.items():
        if low <= center_cm1 <= high:
            return label

    return "Unassigned"


# =========================================================
# LEITURA
# =========================================================

def read_spectrum(file_like):

    filename = getattr(file_like, "name", "").lower()

    if filename.endswith((".xls", ".xlsx")):
        df = pd.read_excel(file_like, header=None)

    else:
        df = pd.read_csv(file_like, sep=None, engine="python",
                         comment="#", header=None)

    df = df.select_dtypes(include=[np.number])

    if df.shape[1] < 2:
        raise ValueError("Arquivo inválido")

    x = df.iloc[:, 0].values.astype(float)
    y = df.iloc[:, 1].values.astype(float)

    idx = np.argsort(x)

    return x[idx], y[idx]


# =========================================================
# BASELINE — ASLS
# =========================================================

def asls_baseline(y, lam=1e6, p=0.01, niter=10):

    y = np.asarray(y)
    N = len(y)

    D = sparse.diags([1, -2, 1], [0, 1, 2],
                     shape=(N - 2, N))

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

def lorentz(x, amp, cen, wid, offset):

    return amp * ((0.5 * wid)**2 /
                  ((x - cen)**2 + (0.5 * wid)**2)) + offset


def fit_lorentz(x, y, center, window=20):

    mask = (x > center - window / 2) & (x < center + window / 2)

    if mask.sum() < 10:
        return None

    xs, ys = x[mask], y[mask]

    p0 = [
        np.max(ys) - np.min(ys),
        center,
        max((xs.max() - xs.min()) / 6, 2),
        np.min(ys),
    ]

    try:

        popt, _ = curve_fit(lorentz, xs, ys, p0=p0, maxfev=8000)

        amp, cen, wid, off = popt

        return {
            "center_fit": float(cen),
            "amplitude": float(amp),
            "fwhm": float(2 * wid),
            "offset": float(off),
        }

    except Exception:
        return None


# =========================================================
# PIPELINE PRINCIPAL
# =========================================================

def process_raman_pipeline(sample_input):

    # Leitura
    x_raw, y_raw = read_spectrum(sample_input)

    # Baseline
    baseline = asls_baseline(y_raw)

    y_corr = y_raw - baseline

    # Suavização
    y_smooth = savgol_filter(y_corr, 11, 3)

    # Normalização
    y_norm = y_smooth / np.max(np.abs(y_smooth))

    # Detecção picos
    idx_peaks, _ = find_peaks(
        y_norm,
        prominence=0.04,
        width=6
    )

    peaks = []

    for idx in idx_peaks:

        cen = x_raw[idx]

        fit = fit_lorentz(x_raw, y_norm, cen)

        if not fit:
            continue

        if fit["amplitude"] < 0.05:
            continue

        group = classify_raman_group(fit["center_fit"])

        peaks.append({
            "peak_cm1": cen,
            "intensity_norm": y_norm[idx],
            "chemical_group": group,
            **fit
        })

    peaks_df = pd.DataFrame(peaks)

    spectrum_df = pd.DataFrame({
        "shift": x_raw,
        "intensity_norm": y_norm
    })

    figs = {}

    fig_raw, ax = plt.subplots(figsize=(8, 4), dpi=300)
    ax.plot(x_raw, y_raw, lw=1.2)
    ax.set_title("Raw Raman Spectrum")
    figs["raw"] = fig_raw

    fig_base, ax = plt.subplots(figsize=(8, 4), dpi=300)
    ax.plot(x_raw, y_corr, lw=1.2)
    ax.plot(x_raw, baseline, "--", lw=1)
    ax.set_title("Baseline Correction")
    figs["baseline"] = fig_base

    return spectrum_df, peaks_df, None, figs


# =========================================================
# WRAPPER APP
# =========================================================

def process_raman_spectrum_with_groups(file_like):

    spectrum_df, peaks_df, fingerprint_df, figures = process_raman_pipeline(
        sample_input=file_like
    )

    return {
        "spectrum_df": spectrum_df,
        "peaks_df": peaks_df,
        "fingerprint_df": fingerprint_df,
        "figures": figures,
    }
