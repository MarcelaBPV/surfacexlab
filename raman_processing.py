# =========================================================
# raman_processing.py
# SurfaceXLab — Raman Scientific Processing
# PSEUDO-VOIGT VERSION
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

from sklearn.preprocessing import (
    StandardScaler
)

from sklearn.decomposition import PCA

from raman_database import (
    classify_peak
)


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
    )

    df = df.dropna()

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

        z = spsolve(Z, w*y)

        w = p * (y > z) + (1-p) * (y < z)

    return z


# =========================================================
# PSEUDO-VOIGT
# =========================================================
def pseudo_voigt(

    x,

    amp,

    cen,

    sigma,

    eta
):

    # =====================================================
    # LORENTZIANA
    # =====================================================
    lorentz = (

        sigma**2 /

        ((x-cen)**2 + sigma**2)
    )

    # =====================================================
    # GAUSSIANA
    # =====================================================
    gauss = np.exp(

        -((x-cen)**2) /

        (2*sigma**2)
    )

    # =====================================================
    # COMBINAÇÃO
    # =====================================================
    return amp * (

        eta * lorentz +

        (1-eta) * gauss
    )


# =========================================================
# MULTI PSEUDO-VOIGT
# =========================================================
def multi_pseudo_voigt(x, *params):

    y = np.zeros_like(x)

    for i in range(0, len(params), 4):

        amp = params[i]

        cen = params[i+1]

        sigma = params[i+2]

        eta = params[i+3]

        y += pseudo_voigt(

            x,

            amp,

            cen,

            sigma,

            eta
        )

    return y


# =========================================================
# PROCESSAMENTO PRINCIPAL
# =========================================================
def process_raman_spectrum_with_groups(
    file_like
):

    # =====================================================
    # LEITURA
    # =====================================================
    df = read_raman_file(file_like)

    x = df["shift"].values.astype(float)

    y = df["intensity"].values.astype(float)

    # =====================================================
    # VALIDAÇÃO
    # =====================================================
    if len(x) < 20:

        raise ValueError(
            "Espectro muito pequeno."
        )

    # =====================================================
    # ORDENAÇÃO
    # =====================================================
    order = np.argsort(x)

    x = x[order]

    y = y[order]

    # =====================================================
    # SAVITZKY-GOLAY
    # =====================================================
    window = min(

        15,

        len(y)-1 if len(y) % 2 == 0 else len(y)
    )

    if window % 2 == 0:

        window -= 1

    if window < 5:

        window = 5

    y_smooth = savgol_filter(

        y,

        window_length=window,

        polyorder=3
    )

    # =====================================================
    # BASELINE
    # =====================================================
    baseline = baseline_als(
        y_smooth
    )

    y_corr = y_smooth - baseline

    y_corr = y_corr - np.min(y_corr)

    # =====================================================
    # NORMALIZAÇÃO
    # =====================================================
    max_val = np.max(y_corr)

    if max_val == 0:

        max_val = 1e-9

    y_norm = y_corr / max_val

    # =====================================================
    # DETECÇÃO DE PICOS
    # =====================================================
    peaks, properties = find_peaks(

        y_norm,

        prominence=0.03,

        distance=20
    )

    # =====================================================
    # FALLBACK
    # =====================================================
    if len(peaks) == 0:

        peaks = np.array([

            np.argmax(y_norm)
        ])

    # =====================================================
    # PARÂMETROS INICIAIS
    # =====================================================
    p0 = []

    lower = []

    upper = []

    for p in peaks:

        amp0 = max(
            y_norm[p],
            0.05
        )

        cen0 = x[p]

        sigma0 = 10

        eta0 = 0.5

        p0.extend([

            amp0,
            cen0,
            sigma0,
            eta0
        ])

        lower.extend([

            0,
            cen0 - 20,
            2,
            0
        ])

        upper.extend([

            2,
            cen0 + 20,
            80,
            1
        ])

    # =====================================================
    # FITTING
    # =====================================================
    try:

        popt, _ = curve_fit(

            multi_pseudo_voigt,

            x,

            y_norm,

            p0=p0,

            bounds=(lower, upper),

            maxfev=100000
        )

    except Exception:

        popt = np.array(p0)

    # =====================================================
    # FIT FINAL
    # =====================================================
    y_fit = multi_pseudo_voigt(

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

    if ss_tot == 0:

        r2 = 0

    else:

        r2 = 1 - (ss_res / ss_tot)

    # =====================================================
    # TABELA DE PICOS
    # =====================================================
    peaks_data = []

    for i in range(0, len(popt), 4):

        amp = popt[i]

        center = popt[i+1]

        sigma = popt[i+2]

        eta = popt[i+3]

        # aproximação FWHM pseudo-Voigt
        fwhm = 2.355 * sigma

        matches = classify_peak(center)

        if matches:

            for match in matches:

                peaks_data.append({

                    "Peak (cm⁻¹)":
                        round(center, 2),

                    "Intensity":
                        round(amp, 4),

                    "FWHM":
                        round(fwhm, 2),

                    "Eta":
                        round(eta, 3),

                    "Assignment":
                        match["group"],

                    "Category":
                        match["category"],

                    "Description":
                        match["description"]
                })

        else:

            peaks_data.append({

                "Peak (cm⁻¹)":
                    round(center, 2),

                "Intensity":
                    round(amp, 4),

                "FWHM":
                    round(fwhm, 2),

                "Eta":
                    round(eta, 3),

                "Assignment":
                    "Unknown",

                "Category":
                    "Unknown",

                "Description":
                    "Unknown peak"
            })

    peaks_df = pd.DataFrame(
        peaks_data
    )

    # =====================================================
    # FIGURA PAPER STYLE
    # =====================================================
    fig_deconv, (ax1, ax2) = plt.subplots(

        2, 1,

        figsize=(7,5),

        dpi=300,

        sharex=True,

        gridspec_kw={
            "height_ratios": [4,1]
        }
    )

    # =====================================================
    # ESPECTRO EXPERIMENTAL
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

    for idx, i in enumerate(
        range(0, len(popt), 4)
    ):

        peak_curve = pseudo_voigt(

            x,

            popt[i],

            popt[i+1],

            popt[i+2],

            popt[i+3]
        )

        ax1.plot(

            x,

            peak_curve,

            color=colors[
                idx % len(colors)
            ],

            linewidth=1.0,

            alpha=0.9
        )

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
    # PEAKS
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
    fig_spec, ax_spec = plt.subplots(

        figsize=(7,4),

        dpi=300
    )

    ax_spec.plot(

        x,

        y_corr,

        color="black"
    )

    ax_spec.scatter(

        x[peaks],

        y_corr[peaks],

        color="red",

        s=15
    )

    for p in peaks:

        ax_spec.text(

            x[p],

            y_corr[p] +
            0.03*np.max(y_corr),

            f"{int(x[p])}",

            fontsize=7,

            ha="center"
        )

    ax_spec.set_xlabel(
        "Raman Shift (cm⁻¹)"
    )

    ax_spec.set_ylabel(
        "Corrected Intensity"
    )

    ax_spec.grid(alpha=0.2)

    # =====================================================
    # SUMMARY
    # =====================================================
    summary = {

        "Sample":
            file_like.name,

        "R²":
            round(r2, 4),

        "N Peaks":
            len(peaks),

        "Main Peak":
            round(
                x[np.argmax(y_norm)],
                2
            )
    }

    # =====================================================
    # RETORNO
    # =====================================================
    return {

        "summary":
            summary,

        "peaks_df":
            peaks_df,

        "figures": {

            "spectrum":
                fig_spec,

            "deconvolution":
                fig_deconv
        }
    }


# =========================================================
# PCA RAMAN
# =========================================================
def run_raman_pca(df):

    numeric_df = df.select_dtypes(
        include=[np.number]
    ).fillna(0)

    if numeric_df.empty:

        raise ValueError(
            "Sem dados numéricos para PCA."
        )

    X = StandardScaler().fit_transform(
        numeric_df
    )

    pca = PCA(
        n_components=2
    )

    scores = pca.fit_transform(X)

    explained = (
        pca.explained_variance_ratio_ * 100
    )

    scores_df = pd.DataFrame({

        "PC1":
            scores[:,0],

        "PC2":
            scores[:,1]
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

    ax.set_title(
        "Raman PCA"
    )

    ax.grid(alpha=0.3)

    return {

        "scores":
            scores_df,

        "loadings":
            loadings_df,

        "explained":
            explained,

        "figure":
            fig
    }
