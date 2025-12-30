# raman_processing.py
# -*- coding: utf-8 -*-
"""
SurfaceXLab — Raman Processing Pipeline (Scientific / CRM-ready)

Pipeline:
1. Leitura e harmonização espectral
2. Subtração de substrato (opcional)
3. Correção de baseline (ASLS)
4. Suavização (Savitzky–Golay)
5. Normalização
6. Detecção automática de picos
7. Ajuste de picos (Lorentziano)

Saídas:
- spectrum_df  → espectro processado
- peaks_df     → tabela de picos (ML + DB ready)
- fig          → figura científica

© 2025 Marcela Veiga
"""

from typing import Tuple, Optional, Dict
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.signal import savgol_filter, find_peaks
from scipy.optimize import curve_fit
from scipy import sparse
from scipy.sparse.linalg import spsolve

# =========================================================
# IO
# =========================================================
def read_spectrum(file_like) -> Tuple[np.ndarray, np.ndarray]:
    """Lê espectro Raman (txt/csv/xlsx)."""
    try:
        df = pd.read_csv(file_like, sep=None, engine="python", comment="#", header=None)
    except Exception:
        file_like.seek(0)
        df = pd.read_csv(file_like, delim_whitespace=True, header=None)

    df = df.select_dtypes(include=[np.number])
    if df.shape[1] < 2:
        raise ValueError("Arquivo deve conter ao menos duas colunas numéricas.")

    x = df.iloc[:, 0].values.astype(float)
    y = df.iloc[:, 1].values.astype(float)

    order = np.argsort(x)
    return x[order], y[order]


# =========================================================
# BASELINE — ASLS
# =========================================================
def asls_baseline(y, lam=1e5, p=0.01, niter=10):
    y = np.asarray(y, dtype=float)
    N = len(y)

    D = sparse.diags(
        [1, -2, 1],
        [0, 1, 2],
        shape=(N - 2, N),
        format="csc",
    )

    w = np.ones(N)
    for _ in range(niter):
        W = sparse.diags(w, 0)
        Z = W + lam * D.T @ D
        z = spsolve(Z, w * y)
        w = p * (y > z) + (1 - p) * (y < z)

    return z


# =========================================================
# MODELO DE PICO
# =========================================================
def lorentz(x, amp, cen, wid, offset):
    return amp * ((0.5 * wid) ** 2 / ((x - cen) ** 2 + (0.5 * wid) ** 2)) + offset


def fit_lorentz(x, y, center, window=20.0):
    mask = (x > center - window / 2) & (x < center + window / 2)
    if mask.sum() < 6:
        return None

    xs, ys = x[mask], y[mask]

    p0 = [
        np.nanmax(ys) - np.nanmin(ys),
        center,
        max((xs.max() - xs.min()) / 6, 1.0),
        np.nanmin(ys),
    ]

    try:
        popt, _ = curve_fit(lorentz, xs, ys, p0=p0, maxfev=5000)
        amp, cen, wid, off = popt
        return {
            "center_fit": cen,
            "amplitude": amp,
            "width": wid,
            "fwhm": 2 * wid,
            "offset": off,
        }
    except Exception:
        return None


# =========================================================
# PIPELINE PRINCIPAL
# =========================================================
def process_raman_pipeline(
    sample_input,
    substrate_input: Optional = None,
    resample_points: int = 3000,
    sg_window: int = 11,
    sg_poly: int = 3,
    asls_lambda: float = 1e5,
    asls_p: float = 0.01,
    peak_prominence: float = 0.02,
) -> Tuple[pd.DataFrame, pd.DataFrame, plt.Figure]:

    # -----------------------------------------------------
    # 1. Leitura
    # -----------------------------------------------------
    x_s, y_s = read_spectrum(sample_input)

    if substrate_input is not None:
        x_b, y_b = read_spectrum(substrate_input)
    else:
        x_b, y_b = x_s, np.zeros_like(y_s)

    # -----------------------------------------------------
    # 2. Harmonização (eixo comum)
    # -----------------------------------------------------
    x_min = max(x_s.min(), x_b.min())
    x_max = min(x_s.max(), x_b.max())
    x = np.linspace(x_min, x_max, resample_points)

    y_s = np.interp(x, x_s, y_s)
    y_b = np.interp(x, x_b, y_b)

    # -----------------------------------------------------
    # 3. Subtração de substrato (regressão linear)
    # -----------------------------------------------------
    A = np.vstack([y_b, np.ones_like(y_b)]).T
    alpha, beta = np.linalg.lstsq(A, y_s, rcond=None)[0]
    alpha = max(alpha, 0)

    y_sub = y_s - alpha * y_b - beta

    # -----------------------------------------------------
    # 4. Baseline ASLS
    # -----------------------------------------------------
    baseline = asls_baseline(y_sub, lam=asls_lambda, p=asls_p)
    y_corr = y_sub - baseline

    # -----------------------------------------------------
    # 5. Suavização
    # -----------------------------------------------------
    if sg_window % 2 == 0:
        sg_window += 1
    y_smooth = savgol_filter(y_corr, sg_window, sg_poly)

    # -----------------------------------------------------
    # 6. Normalização
    # -----------------------------------------------------
    y_norm = y_smooth / np.nanmax(np.abs(y_smooth))

    # -----------------------------------------------------
    # 7. Detecção de picos
    # -----------------------------------------------------
    peak_idx, props = find_peaks(
        y_norm,
        prominence=peak_prominence,
        distance=resample_points // 200,
    )

    peaks = []
    for i, idx in enumerate(peak_idx):
        cen = x[idx]
        inten = y_norm[idx]

        fit = fit_lorentz(x, y_norm, cen)
        if fit:
            peaks.append({
                "peak_cm1": cen,
                "intensity_norm": inten,
                **fit,
            })

    peaks_df = pd.DataFrame(peaks)

    # -----------------------------------------------------
    # 8. Tabela espectral
    # -----------------------------------------------------
    spectrum_df = pd.DataFrame({
        "raman_shift_cm1": x,
        "intensity_norm": y_norm,
        "baseline_norm": baseline / np.nanmax(np.abs(y_smooth)),
    })

    # -----------------------------------------------------
    # 9. Plot científico
    # -----------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(x, y_norm, lw=1.4, label="Espectro processado")
    ax.plot(x, spectrum_df["baseline_norm"], "--", label="Baseline")

    for _, r in peaks_df.iterrows():
        ax.axvline(r["center_fit"], ls="--", lw=1)

    ax.set_xlabel("Raman shift (cm⁻¹)")
    ax.set_ylabel("Intensidade normalizada")
    ax.legend()
    ax.grid(alpha=0.3)

    return spectrum_df, peaks_df, fig
