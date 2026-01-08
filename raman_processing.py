# -*- coding: utf-8 -*-
"""
SurfaceXLab — Raman Processing Pipeline (Scientific / ML / DB Ready)

Pipeline:
1. Leitura espectral (txt, csv, xls, xlsx)
2. Harmonização espectral
3. Subtração de substrato (opcional)
4. Correção de baseline (ASLS)
5. Suavização (Savitzky–Golay)
6. Normalização
7. Detecção automática de picos
8. Ajuste Lorentziano
9. Plot científico padronizado

Saídas:
- spectrum_df  → espectro processado (ML / PCA ready)
- peaks_df     → tabela de picos
- fig          → figura científica (Streamlit / artigo)

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
# IO — LEITURA ROBUSTA (TXT / CSV / XLS / XLSX)
# =========================================================
def read_spectrum(file_like) -> Tuple[np.ndarray, np.ndarray]:
    """
    Lê espectros Raman a partir de:
    - .txt
    - .csv
    - .xls
    - .xlsx

    Espera ao menos duas colunas numéricas:
    [Raman shift (cm⁻¹), intensidade]
    """

    filename = getattr(file_like, "name", "").lower()

    # -------------------------------
    # Excel (binário) → leitura direta
    # -------------------------------
    if filename.endswith((".xls", ".xlsx")):
        df = pd.read_excel(file_like, header=None)

    # -------------------------------
    # Texto / CSV
    # -------------------------------
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

    # -------------------------------
    # Sanitização
    # -------------------------------
    df = df.select_dtypes(include=[np.number])

    if df.shape[1] < 2:
        raise ValueError(
            "Arquivo inválido: é necessário ao menos duas colunas numéricas "
            "(Raman shift e intensidade)."
        )

    x = df.iloc[:, 0].values.astype(float)
    y = df.iloc[:, 1].values.astype(float)

    order = np.argsort(x)
    return x[order], y[order]


# =========================================================
# BASELINE — ASLS
# =========================================================
def asls_baseline(y, lam=1e5, p=0.01, niter=10):
    """Asymmetric Least Squares baseline correction"""
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
# MODELO DE PICO — LORENTZ
# =========================================================
def lorentz(x, amp, cen, wid, offset):
    return amp * ((0.5 * wid) ** 2 / ((x - cen) ** 2 + (0.5 * wid) ** 2)) + offset


def fit_lorentz(x, y, center, window=20.0):
    """Ajuste Lorentziano local ao redor do pico detectado"""
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
) -> Tuple[pd.DataFrame, pd.DataFrame, plt.Figure]:

    # 1️⃣ Leitura
    x_s, y_s = read_spectrum(sample_input)

    if substrate_input is not None:
        x_b, y_b = read_spectrum(substrate_input)
    else:
        x_b, y_b = x_s, np.zeros_like(y_s)

    # 2️⃣ Harmonização espectral
    x_min = max(x_s.min(), x_b.min())
    x_max = min(x_s.max(), x_b.max())
    x = np.linspace(x_min, x_max, resample_points)

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
        distance=resample_points // 200,
    )

    peaks = []
    for idx in peak_idx:
        cen = x[idx]
        inten = y_norm[idx]
        fit = fit_lorentz(x, y_norm, cen)
        if fit:
            peaks.append({
                "peak_cm1": float(cen),
                "intensity_norm": float(inten),
                **fit,
            })

    peaks_df = pd.DataFrame(peaks)

    spectrum_df = pd.DataFrame({
        "raman_shift_cm1": x,
        "intensity_norm": y_norm,
        "baseline_norm": baseline / norm if norm > 0 else baseline,
    })

    # =====================================================
    # PLOT CIENTÍFICO
    # =====================================================
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(x, y_norm, lw=1.4, color="black", label="Espectro processado")
    ax.plot(x, spectrum_df["baseline_norm"], "--", lw=1, color="gray", label="Baseline")

    for _, r in peaks_df.iterrows():
        ax.axvline(r["center_fit"], ls="--", lw=0.9, alpha=0.6)

    ax.set_xlabel("Raman shift (cm⁻¹)")
    ax.set_ylabel("Intensidade normalizada")
    ax.grid(alpha=0.3)
    ax.legend(frameon=False)
    fig.tight_layout()

    return spectrum_df, peaks_df, fig


# =========================================================
# WRAPPER — USADO PELO APP (raman_tab.py)
# =========================================================
def process_raman_spectrum_with_groups(
    file_like,
    preprocess_kwargs: Optional[Dict[str, Any]] = None,
    peak_prominence: float = 0.02,
):
    """Wrapper padrão esperado pelo app SurfaceXLab"""
    spectrum_df, peaks_df, fig = process_raman_pipeline(
        sample_input=file_like,
        peak_prominence=peak_prominence,
        **(preprocess_kwargs or {})
    )

    return {
        "spectrum_df": spectrum_df,
        "peaks_df": peaks_df,
        "figure": fig,
    }
