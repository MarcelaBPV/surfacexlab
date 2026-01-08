# -*- coding: utf-8 -*-
"""
SurfaceXLab — Raman Processing Pipeline (Scientific / ML / DB Ready)

Pipeline:
1. Leitura espectral
2. Harmonização
3. Subtração de substrato
4. Correção de baseline (ASLS)
5. Suavização (Savitzky–Golay)
6. Normalização
7. Detecção automática de picos
8. Ajuste Lorentziano
9. Plot científico padronizado

© 2025 Marcela Veiga — SurfaceXLab
"""

from typing import Tuple, Optional, Dict, Any
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.signal import savgol_filter, find_peaks
from scipy.optimize import curve_fit
from scipy import sparse
from scipy.sparse.linalg import spsolve


# =========================================================
# IO — LEITURA ROBUSTA
# =========================================================
def read_spectrum(file_like) -> Tuple[np.ndarray, np.ndarray]:
    filename = getattr(file_like, "name", "").lower()

    if filename.endswith((".xls", ".xlsx")):
        df = pd.read_excel(file_like, header=None)
    else:
        try:
            df = pd.read_csv(
                file_like,
                sep=None,
                engine="python",
                comment="#",
                header=None
            )
        except Exception:
            file_like.seek(0)
            df = pd.read_csv(
                file_like,
                delim_whitespace=True,
                header=None
            )

    df = df.select_dtypes(include=[np.number])
    if df.shape[1] < 2:
        raise ValueError("Arquivo inválido: mínimo 2 colunas numéricas.")

    x = df.iloc[:, 0].values.astype(float)
    y = df.iloc[:, 1].values.astype(float)

    idx = np.argsort(x)
    return x[idx], y[idx]


# =========================================================
# BASELINE — ASLS
# =========================================================
def asls_baseline(y, lam=1e5, p=0.01, niter=10):
    y = np.asarray(y, dtype=float)
    N = len(y)

    D = sparse.diags([1, -2, 1], [0, 1, 2], shape=(N - 2, N), format="csc")
    w = np.ones(N)

    for _ in range(niter):
        W = sparse.diags(w, 0)
        Z = W + lam * D.T @ D
        z = spsolve(Z, w * y)
        w = p * (y > z) + (1 - p) * (y < z)

    return z


# =========================================================
# MODELO DE PICO — LORENTZ
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
    substrate_input: Optional = None,
    resample_points: int = 3000,
    sg_window: int = 11,
    sg_poly: int = 3,
    asls_lambda: float = 1e5,
    asls_p: float = 0.01,
    peak_prominence: float = 0.02,
):

    # 1️⃣ Leitura (DADOS BRUTOS PRESERVADOS)
    x_raw, y_raw = read_spectrum(sample_input)

    # cópias para processamento
    x_s = x_raw.copy()
    y_s = y_raw.copy()

    if substrate_input is not None:
        x_b, y_b = read_spectrum(substrate_input)
    else:
        x_b, y_b = x_s, np.zeros_like(y_s)

    # 2️⃣ Harmonização
    x = np.linspace(
        max(x_s.min(), x_b.min()),
        min(x_s.max(), x_b.max()),
        resample_points
    )

    y_s = np.interp(x, x_s, y_s)
    y_b = np.interp(x, x_b, y_b)

    # 3️⃣ Subtração de substrato
    A = np.vstack([y_b, np.ones_like(y_b)]).T
    alpha, beta = np.linalg.lstsq(A, y_s, rcond=None)[0]
    alpha = max(alpha, 0.0)
    y_sub = y_s - alpha * y_b - beta

    # 4️⃣ Baseline
    baseline = asls_baseline(y_sub, lam=asls_lambda, p=asls_p)
    y_corr = y_sub - baseline

    # 5️⃣ Suavização
    if sg_window % 2 == 0:
        sg_window += 1
    y_smooth = savgol_filter(y_corr, sg_window, sg_poly)

    # 6️⃣ Normalização
    norm = np.nanmax(np.abs(y_smooth))
    y_norm = y_smooth / norm if norm > 0 else y_smooth

    # 7️⃣ Detecção de picos
    peak_idx, _ = find_peaks(
        y_norm,
        prominence=peak_prominence,
        distance=resample_points // 200
    )

    peaks = []
    for idx in peak_idx:
        cen = x[idx]
        fit = fit_lorentz(x, y_norm, cen)
        if fit:
            peaks.append({
                "peak_cm1": float(cen),
                "intensity_norm": float(y_norm[idx]),
                **fit
            })

    peaks_df = pd.DataFrame(peaks)

    spectrum_df = pd.DataFrame({
        "shift": x,
        "intensity_norm": y_norm,
        "baseline_norm": baseline / norm if norm > 0 else baseline,
    })

    # =====================================================
    # FIGURAS (3 ESTADOS — CONSISTENTES)
    # =====================================================
    figs = {}

    # 🔹 Bruto (dados originais)
    fig_raw, ax = plt.subplots(figsize=(10, 4), dpi=300)
    ax.plot(x_raw, y_raw, color="black")
    ax.set_title("Espectro Raman Bruto")
    ax.set_xlabel("Raman shift (cm⁻¹)")
    ax.set_ylabel("Intensidade (a.u.)")
    figs["raw"] = fig_raw

    # 🔹 Baseline
    fig_base, ax = plt.subplots(figsize=(10, 4), dpi=300)
    ax.plot(x, y_sub, color="black", label="Subtraído")
    ax.plot(x, baseline, "--", color="gray", label="Baseline (ASLS)")
    ax.legend(frameon=False)
    ax.set_title("Correção de Baseline")
    figs["baseline"] = fig_base

    # 🔹 Processado
    fig_proc, ax = plt.subplots(figsize=(10, 4), dpi=300)
    ax.plot(x, y_norm, color="black", label="Processado")

    for _, r in peaks_df.iterrows():
        ax.axvline(r["center_fit"], ls="--", lw=0.9, alpha=0.6)

    ax.set_xlabel("Raman shift (cm⁻¹)")
    ax.set_ylabel("Intensidade normalizada")
    ax.legend(frameon=False)
    figs["processed"] = fig_proc

    return spectrum_df, peaks_df, figs


# =========================================================
# WRAPPER — CONTRATO DO APP
# =========================================================
def process_raman_spectrum_with_groups(
    file_like,
    preprocess_kwargs: Optional[Dict[str, Any]] = None,
    peak_prominence: float = 0.02,
):
    spectrum_df, peaks_df, figures = process_raman_pipeline(
        sample_input=file_like,
        peak_prominence=peak_prominence,
        **(preprocess_kwargs or {})
    )

    return {
        "spectrum_df": spectrum_df,
        "peaks_df": peaks_df,
        "figures": figures,
    }
