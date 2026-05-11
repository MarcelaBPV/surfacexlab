# =========================================================
# Raman Processing — SurfaceXLab
# Deconvolução Científica estilo Paper
# =========================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.optimize import curve_fit

from scipy.signal import (
    savgol_filter,
    find_peaks
)

from scipy.sparse import diags
from scipy.sparse.linalg import spsolve

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from raman_database import classify_peak


# =========================================================
# LEITURA UNIVERSAL
# =========================================================
def read_raman_file(file_like):

    name = file_like.name.lower()

    # =====================================================
    # EXCEL
    # =====================================================
    if name.endswith((".xlsx", ".xls")):

        df = pd.read_excel(file_like)

    # =====================================================
    # CSV / TXT / LOG
    # =====================================================
    else:

        try:

            df = pd.read_csv(
                file_like,
                sep=None,
                engine="python",
                encoding="utf-8"
            )

        except:

            df = pd.read_csv(
                file_like,
                sep=None,
                engine="python",
                encoding="latin1"
            )

    # =====================================================
    # LIMPEZA
    # =====================================================
    df = df.apply(
        pd.to_numeric,
        errors="coerce"
    ).dropna()

    df = df.iloc[:, :2]

    df.columns = [
        "shift",
        "intensity"
    ]

    return df


# =========================================================
# BASELINE ALS
# =========================================================
def baseline_als(
    y,
    lam=1e5,
    p=0.001,
    niter=10
):

    L = len(y)

    D = diags(
        [1, -2, 1],
        [0, -1, -2],
        shape=(L, L-2)
    )

    w = np.ones(L)

    for _ in range(niter):

        W = diags(w, 0)

        Z = W + lam * D.dot(D.transpose())

        z = spsolve(Z, w * y)

        w = p * (y > z) + (1-p) * (y < z)

    return z


# =========================================================
# LORENTZIANA
# =========================================================
def lorentzian(
    x,
    amp,
    cen,
    wid
):

    return amp * (
        wid**2 /
        ((x-cen)**2 + wid**2)
    )


# =========================================================
# MULTI LORENTZ
# =========================================================
def multi_lorentzian(x, *params):

    y = np.zeros_like(x)

    for i in range(0, len(params), 3):

        amp = params[i]
        cen = params[i+1]
        wid = params[i+2]

        y += lorentzian(
            x,
            amp,
            cen,
            wid
        )

    return y


# =========================================================
# PROCESSAMENTO PRINCIPAL
# =========================================================
def process_raman_spectrum_with_groups(file_like):

    # =====================================================
    # LEITURA
    # =====================================================
    df = read_raman_file(file_like)

    x = df["shift"].values
    y = df["intensity"].values


    # =====================================================
    # SUAVIZAÇÃO
    # =====================================================
    y_smooth = savgol_filter(

        y,

        window_length=15,

        polyorder=3
    )


    # =====================================================
    # BASELINE
    # =====================================================
    baseline = baseline_als(y_smooth)

    y_corr = y_smooth - baseline


    # =====================================================
    # NORMALIZAÇÃO
    # =====================================================
    y_norm = y_corr / np.max(y_corr)


    # =====================================================
    # DETECÇÃO DE PICOS
    # =====================================================
    peaks, properties = find_peaks(

        y_norm,

        prominence=0.05,

        distance=20
    )


    # =====================================================
    # PARÂMETROS INICIAIS
    # =====================================================
    p0 = []

    lower = []
    upper = []

    for p in peaks:

        p0.extend([

            y_norm[p],
            x[p],
            12
        ])

        lower.extend([

            0,
            x[p]-15,
            3
        ])

        upper.extend([

            2,
            x[p]+15,
            60
        ])


    # =====================================================
    # FITTING
    # =====================================================
    try:

        popt, _ = curve_fit(

            multi_lorentzian,

            x,

            y_norm,

            p0=p0,

            bounds=(lower, upper),

            maxfev=50000
        )

    except:

        popt = p0


    # =====================================================
    # CURVA FIT
    # =====================================================
    y_fit = multi_lorentzian(

        x,

        *popt
    )


    # =====================================================
    # RESÍDUO
    # =====================================================
    residual = y_norm - y_fit


    # =====================================================
    # R²
    # =====================================================
    ss_res = np.sum(residual**2)

    ss_tot = np.sum(

        (y_norm - np.mean(y_norm))**2
    )

    r2 = 1 - (ss_res / ss_tot)


    # =====================================================
    # PEAK TABLE
    # =====================================================
    peaks_data = []

    for i in range(0, len(popt), 3):

        amp = popt[i]
        center = popt[i+1]
        width = popt[i+2]

        fwhm = 2 * width

        matches = classify_peak(center)

        if matches:

            for match in matches:

                peaks_data.append({

                    "Peak (cm⁻¹)": round(center, 2),

                    "Intensity": round(amp, 4),

                    "FWHM": round(fwhm, 2),

                    "Assignment": match["group"],

                    "Category": match["category"],

                    "Description": match["description"]
                })

        else:

            peaks_data.append({

                "Peak (cm⁻¹)": round(center, 2),

                "Intensity": round(amp, 4),

                "FWHM": round(fwhm, 2),

                "Assignment": "Unknown",

                "Category": "Unknown",

                "Description": "Unknown peak"
            })

    peaks_df = pd.DataFrame(peaks_data)


    # =====================================================
    # FIGURA PAPER STYLE
    # =====================================================
    fig, (ax1, ax2) = plt.subplots(

        2, 1,

        figsize=(7,5),

        dpi=300,

        sharex=True,

        gridspec_kw={
            "height_ratios": [4,1]
        }
    )

    # =====================================================
    # EXPERIMENTAL
    # =====================================================
    ax1.plot(

        x,

        y_norm,

        color="black",

        linewidth=1.2,

        label="Experimental"
    )

    # =====================================================
    # COMPONENTES
    # =====================================================
    colors = plt.cm.Set2.colors

    for idx, i in enumerate(range(0, len(popt), 3)):

        peak_curve = lorentzian(

            x,

            popt[i],

            popt[i+1],

            popt[i+2]
        )

        ax1.plot(

            x,

            peak_curve,

            color=colors[idx % len(colors)],

            linewidth=1.0,

            alpha=0.9
        )

        # posição do pico
        peak_pos = popt[i+1]

        peak_amp = popt[i]

        ax1.text(

            peak_pos,

            peak_amp + 0.03,

            f"{int(peak_pos)}",

            fontsize=7,

            ha="center"
        )

    # =====================================================
    # FIT TOTAL
    # =====================================================
    ax1.plot(

        x,

        y_fit,

        color="red",

        linewidth=1.5,

        label=f"Fit (R²={r2:.4f})"
    )

    # =====================================================
    # PEAKS EXPERIMENTAIS
    # =====================================================
    ax1.scatter(

        x[peaks],

        y_norm[peaks],

        color="red",

        s=15,

        zorder=5
    )

    ax1.set_ylabel(
        "Normalized Intensity"
    )

    ax1.legend()

    ax1.grid(alpha=0.2)


    # =====================================================
    # RESÍDUO
    # =====================================================
    ax2.plot(

        x,

        residual,

        color="black",

        linewidth=0.8
    )

    ax2.axhline(

        0,

        linestyle="--",

        color="gray"
    )

    ax2.set_xlabel(
        "Raman Shift (cm⁻¹)"
    )

    ax2.set_ylabel(
        "Residual"
    )

    ax2.grid(alpha=0.2)


    # =====================================================
    # FIGURA RAW
    # =====================================================
    fig_raw, ax_raw = plt.subplots(

        figsize=(7,4),

        dpi=300
    )

    ax_raw.plot(

        x,

        y_corr,

        color="black"
    )

    ax_raw.scatter(

        x[peaks],

        y_corr[peaks],

        color="red",

        s=15
    )

    # labels
    for p in peaks:

        ax_raw.text(

            x[p],

            y_corr[p] + 0.03*np.max(y_corr),

            f"{int(x[p])}",

            fontsize=7,

            ha="center"
        )

    ax_raw.set_xlabel(
        "Raman shift (cm⁻¹)"
    )

    ax_raw.set_ylabel(
        "Corrected Intensity"
    )

    ax_raw.grid(alpha=0.2)


    # =====================================================
    # SUMMARY
    # =====================================================
    summary = {

        "Sample": file_like.name,

        "R²": round(r2, 4),

        "N Peaks": len(peaks),

        "Main Peak": round(
            np.max(x[peaks]), 2
        )
    }


    # =====================================================
    # RETORNO
    # =====================================================
    return {

        "summary": summary,

        "peaks_df": peaks_df,

        "figures": {

            "spectrum": fig_raw,

            "deconvolution": fig
        }
    }


# =========================================================
# PCA RAMAN
# =========================================================
def run_raman_pca(df):

    numeric_df = df.select_dtypes(
        include=[np.number]
    ).fillna(0)

    X = StandardScaler().fit_transform(
        numeric_df
    )

    pca = PCA(n_components=2)

    scores = pca.fit_transform(X)

    explained = (
        pca.explained_variance_ratio_ * 100
    )

    scores_df = pd.DataFrame({

        "PC1": scores[:,0],

        "PC2": scores[:,1]
    })

    loadings_df = pd.DataFrame(

        pca.components_.T,

        columns=["PC1", "PC2"],

        index=numeric_df.columns
    )

    # =====================================================
    # FIGURA PCA
    # =====================================================
    fig, ax = plt.subplots(

        figsize=(6,5),

        dpi=300
    )

    ax.scatter(

        scores[:,0],

        scores[:,1]
    )

    for i, txt in enumerate(df.index):

        ax.text(

            scores[i,0],

            scores[i,1],

            str(txt),

            fontsize=8
        )

    ax.set_xlabel(
        f"PC1 ({explained[0]:.1f}%)"
    )

    ax.set_ylabel(
        f"PC2 ({explained[1]:.1f}%)"
    )

    ax.grid(alpha=0.3)

    return {

        "scores": scores_df,

        "loadings": loadings_df,

        "explained": explained,

        "figure": fig
    }
