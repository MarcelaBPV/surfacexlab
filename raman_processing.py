# =========================================================
# Raman Processing — SurfaceXLab (FINAL ESTÁVEL E CIENTÍFICO)
# =========================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.signal import find_peaks


# =========================================================
# LEITURA UNIVERSAL (ROBUSTA)
# =========================================================
def read_raman_file(file_like):

    name = file_like.name.lower()

    if name.endswith(".xlsx") or name.endswith(".xls"):
        df = pd.read_excel(file_like)
    else:
        try:
            df = pd.read_csv(file_like, sep=None, engine="python", encoding="utf-8")
        except:
            try:
                df = pd.read_csv(file_like, sep=None, engine="python", encoding="latin1")
            except:
                df = pd.read_csv(file_like, sep=None, engine="python", encoding="cp1252")

    # limpa e garante numérico
    df = df.apply(pd.to_numeric, errors="coerce").dropna()

    if df.shape[1] < 2:
        raise ValueError("Arquivo inválido para Raman")

    df = df.iloc[:, :2]
    df.columns = ["shift", "intensity"]

    return df


# =========================================================
# BASELINE ALS (CORRIGIDO — SEM ERRO DE DIMENSÃO)
# =========================================================
def baseline_als(y, lam=1e5, p=0.01, niter=10):

    L = len(y)

    # operador segunda derivada correto
    D = np.diff(np.eye(L), n=2, axis=0)
    DTD = D.T @ D

    w = np.ones(L)

    for _ in range(niter):
        W = np.diag(w)

        # solve é mais estável que inv
        Z = np.linalg.solve(W + lam * DTD, w * y)

        w = p * (y > Z) + (1 - p) * (y < Z)

    return Z


# =========================================================
# FUNÇÕES LORENTZIANAS
# =========================================================
def lorentz(x, a, x0, g):
    return a * (g**2 / ((x - x0)**2 + g**2))


def multi_lorentz(x, *params):

    y = np.zeros_like(x)

    for i in range(0, len(params), 3):
        y += lorentz(x, params[i], params[i+1], params[i+2])

    return y


# =========================================================
# CLASSIFICAÇÃO QUÍMICA
# =========================================================
def classify_peak(pos):

    if 1500 < pos < 1650:
        return "G band (C=C)"
    elif 1300 < pos < 1400:
        return "D band"
    elif 2600 < pos < 2800:
        return "2D band"
    else:
        return "Unassigned"


# =========================================================
# PIPELINE PRINCIPAL
# =========================================================
def process_raman_spectrum_with_groups(file_like):

    df = read_raman_file(file_like)

    x = df["shift"].values
    y = df["intensity"].values

    # =====================================================
    # BASELINE
    # =====================================================
    # ajuste automático para poucos pontos
    lam = 1e5 if len(y) > 100 else 1e4

    baseline = baseline_als(y, lam=lam)

    y_corr = y - baseline

    # =====================================================
    # NORMALIZAÇÃO
    # =====================================================
    y_norm = y_corr / np.max(y_corr)

    df["intensity_norm"] = y_norm

    # =====================================================
    # DETECÇÃO DE PICOS
    # =====================================================
    peaks, _ = find_peaks(y_norm, height=0.2, distance=20)

    if len(peaks) < 2:
        peaks = np.argsort(y_norm)[-3:]

    # =====================================================
    # PARÂMETROS INICIAIS
    # =====================================================
    p0 = []

    for p in peaks:
        p0.extend([y_norm[p], x[p], 10])

    # =====================================================
    # FIT LORENTZIANO
    # =====================================================
    try:
        popt, _ = curve_fit(
            multi_lorentz,
            x,
            y_norm,
            p0=p0,
            maxfev=10000
        )
        y_fit = multi_lorentz(x, *popt)

    except:
        popt = p0
        y_fit = multi_lorentz(x, *popt)

    # =====================================================
    # R²
    # =====================================================
    ss_res = np.sum((y_norm - y_fit) ** 2)
    ss_tot = np.sum((y_norm - np.mean(y_norm)) ** 2)

    r2 = 1 - ss_res / ss_tot

    # =====================================================
    # PEAKS
    # =====================================================
    peaks_data = []

    for i in range(0, len(popt), 3):
        peaks_data.append({
            "center_fit": popt[i+1],
            "amplitude": popt[i],
            "fwhm": popt[i+2],
            "chemical_group": classify_peak(popt[i+1])
        })

    peaks_df = pd.DataFrame(peaks_data)

    # =====================================================
    # FIGURA 1 — RAW + BASELINE
    # =====================================================
    fig1, ax1 = plt.subplots(figsize=(6,4), dpi=300)

    ax1.plot(x, y, label="Raw")
    ax1.plot(x, baseline, '--', label="Baseline")

    ax1.set_title("Raw Raman Spectrum")
    ax1.set_xlabel("Raman shift (cm⁻¹)")
    ax1.set_ylabel("Intensity")

    ax1.legend()
    ax1.grid(alpha=0.3)

    # =====================================================
    # FIGURA 2 — FIT + RESÍDUO
    # =====================================================
    fig2, (ax2, ax3) = plt.subplots(
        2, 1,
        figsize=(6,5),
        dpi=300,
        sharex=True,
        gridspec_kw={"height_ratios": [3,1]}
    )

    ax2.plot(x, y_norm, label="Experimental", color="black")
    ax2.plot(x, y_fit, label=f"Fit (R²={r2:.4f})", color="red")

    ax2.set_ylabel("Intensity (a.u.)")
    ax2.legend()
    ax2.grid(alpha=0.3)

    residual = y_norm - y_fit

    ax3.plot(x, residual, color="black")
    ax3.axhline(0, linestyle="--")

    ax3.set_xlabel("Raman shift (cm⁻¹)")
    ax3.set_ylabel("Residual")

    return {
        "spectrum_df": df,
        "peaks_df": peaks_df,
        "figures": {
            "raw": fig1,
            "baseline": fig2
        }
    }
