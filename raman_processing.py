import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.signal import find_peaks


# =========================================================
# FUNÇÃO LORENTZIANA
# =========================================================
def lorentz(x, a, x0, g):
    return a * (g**2 / ((x - x0)**2 + g**2))


# =========================================================
# BASELINE ALS (REAL)
# =========================================================
def baseline_als(y, lam=1e5, p=0.01, niter=10):

    L = len(y)
    D = np.diff(np.eye(L), 2)
    D = D.T @ D

    w = np.ones(L)

    for _ in range(niter):
        W = np.diag(w)
        Z = np.linalg.inv(W + lam * D) @ (w * y)
        w = p * (y > Z) + (1 - p) * (y < Z)

    return Z


# =========================================================
# LEITURA
# =========================================================
def read_raman_file(file_like):

    df = pd.read_csv(file_like, sep=None, engine="python")

    x = pd.to_numeric(df.iloc[:, 0], errors="coerce")
    y = pd.to_numeric(df.iloc[:, 1], errors="coerce")

    df = pd.DataFrame({"shift": x, "intensity": y}).dropna()

    return df


# =========================================================
# AJUSTE MULTIPEAK
# =========================================================
def multi_lorentz(x, *params):

    y = np.zeros_like(x)

    for i in range(0, len(params), 3):
        a = params[i]
        x0 = params[i+1]
        g = params[i+2]

        y += lorentz(x, a, x0, g)

    return y


# =========================================================
# PIPELINE
# =========================================================
def process_raman_spectrum_with_groups(file_like):

    df = read_raman_file(file_like)

    x = df["shift"].values
    y = df["intensity"].values

    # =========================
    # BASELINE REAL
    # =========================
    baseline = baseline_als(y)
    y_corr = y - baseline

    y_norm = y_corr / np.max(y_corr)

    df["intensity_norm"] = y_norm

    # =========================
    # DETECÇÃO DE PICOS
    # =========================
    peaks, _ = find_peaks(y_norm, height=0.2, distance=20)

    if len(peaks) < 2:
        peaks = np.argsort(y_norm)[-3:]  # fallback

    # =========================
    # PARAMETROS INICIAIS
    # =========================
    p0 = []

    for p in peaks:
        p0.extend([y_norm[p], x[p], 10])

    # =========================
    # FIT
    # =========================
    try:
        popt, _ = curve_fit(multi_lorentz, x, y_norm, p0=p0)
        y_fit = multi_lorentz(x, *popt)
    except:
        popt = p0
        y_fit = multi_lorentz(x, *popt)

    # =========================
    # R²
    # =========================
    ss_res = np.sum((y_norm - y_fit) ** 2)
    ss_tot = np.sum((y_norm - np.mean(y_norm)) ** 2)

    r2 = 1 - ss_res / ss_tot

    # =========================
    # PEAKS DF
    # =========================
    peaks_data = []

    for i in range(0, len(popt), 3):
        peaks_data.append({
            "center_fit": popt[i+1],
            "amplitude": popt[i],
            "fwhm": popt[i+2],
            "chemical_group": classify_peak(popt[i+1])
        })

    peaks_df = pd.DataFrame(peaks_data)

    # =========================
    # PLOTS
    # =========================
    fig1, ax1 = plt.subplots()

    ax1.plot(x, y, label="Raw")
    ax1.plot(x, baseline, label="Baseline")

    ax1.set_title("Raw Raman Spectrum")

    fig2, ax2 = plt.subplots()

    ax2.plot(x, y_norm, label="Experimental")
    ax2.plot(x, y_fit, label=f"Fit (R²={r2:.4f})")

    ax2.legend()

    return {
        "spectrum_df": df,
        "peaks_df": peaks_df,
        "figures": {
            "raw": fig1,
            "baseline": fig2
        }
    }


# =========================================================
# CLASSIFICAÇÃO
# =========================================================
def classify_peak(pos):

    if 1500 < pos < 1650:
        return "C=C (G band)"
    elif 1300 < pos < 1400:
        return "D band"
    elif 2600 < pos < 2800:
        return "2D band"
    else:
        return "Unassigned"
