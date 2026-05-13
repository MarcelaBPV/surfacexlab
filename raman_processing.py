# =========================================================
# raman_processing.py
# SurfaceXLab — Raman Scientific Pipeline
# Pseudo-Voigt + SERS + Biomédico + Materiais
# =========================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.optimize import curve_fit

from scipy.signal import (
    savgol_filter,
    find_peaks,
    medfilt
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
    # CSV / TXT
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
    # CONVERSÃO NUMÉRICA
    # =====================================================
    df = df.apply(
        pd.to_numeric,
        errors="coerce"
    )

    df = df.dropna()

    # =====================================================
    # PRIMEIRAS DUAS COLUNAS
    # =====================================================
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
    lam=1e7,
    p=0.001,
    niter=20
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
# PSEUDO VOIGT
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

    return amp * (
        eta * lorentz +
        (1-eta) * gauss
    )


# =========================================================
# MULTI PSEUDO VOIGT
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
# BANDAS GUIADAS
# =========================================================
def get_known_bands(
    shift_min,
    shift_max
):

    # =====================================================
    # BIOMÉDICO
    # =====================================================
    if shift_min >= 1500 and shift_max <= 1700:

        return [

            1538,
            1560,
            1580,
            1601,
            1621,
            1631,
            1663
        ]

    # =====================================================
    # CARBONO / GRAFENO
    # =====================================================
    elif shift_max > 2500:

        return [

            1350,
            1580,
            2700
        ]

    # =====================================================
    # GERAL
    # =====================================================
    else:

        return None


# =========================================================
# PROCESSAMENTO PRINCIPAL
# =========================================================
def process_raman_spectrum_with_groups(

    file_like,

    shift_min=1500,
    shift_max=1700,

    sg_window=11,
    sg_poly=3,

    baseline_lambda=1e7,
    baseline_p=0.001,

    prominence=0.03,
    distance=20
):

    # =====================================================
    # LEITURA
    # =====================================================
    df = read_raman_file(file_like)

    x = df["shift"].values.astype(float)
    y = df["intensity"].values.astype(float)

    # =====================================================
    # ORDENAÇÃO
    # =====================================================
    order = np.argsort(x)

    x = x[order]
    y = y[order]

    # =====================================================
    # CORTE ESPECTRAL
    # =====================================================
    mask = (
        (x >= shift_min) &
        (x <= shift_max)
    )

    x = x[mask]
    y = y[mask]

    # =====================================================
    # REMOÇÃO DE COSMIC RAY
    # =====================================================
    y_med = medfilt(
        y,
        kernel_size=5
    )

    diff = np.abs(y - y_med)

    threshold = 5 * np.std(diff)

    y = np.where(
        diff > threshold,
        y_med,
        y
    )

    # =====================================================
    # VALIDAÇÃO WINDOW
    # =====================================================
    if sg_window >= len(y):

        sg_window = len(y)-1

    if sg_window % 2 == 0:

        sg_window -= 1

    if sg_window < 5:

        sg_window = 5

    # =====================================================
    # SUAVIZAÇÃO
    # =====================================================
    y_smooth = savgol_filter(
        y,
        window_length=sg_window,
        polyorder=sg_poly
    )

    # =====================================================
    # BASELINE
    # =====================================================
    baseline = baseline_als(
        y_smooth,
        lam=baseline_lambda,
        p=baseline_p
    )

    # =====================================================
    # CORREÇÃO
    # =====================================================
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
    # PICOS GUIADOS
    # =====================================================
    known_bands = get_known_bands(
        shift_min,
        shift_max
    )

    p0 = []
    lower = []
    upper = []

    # =====================================================
    # BANDAS CONHECIDAS
    # =====================================================
    if known_bands is not None:

        for band in known_bands:

            idx = np.argmin(
                np.abs(x - band)
            )

            amp0 = max(
                y_norm[idx],
                0.05
            )

            cen0 = band

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
                band - 5,
                2,
                0
            ])

            upper.extend([
                2,
                band + 5,
                50,
                1
            ])

    # =====================================================
    # DETECÇÃO AUTOMÁTICA
    # =====================================================
    else:

        peaks, properties = find_peaks(
            y_norm,
            prominence=prominence,
            distance=distance
        )

        for p in peaks:

            amp0 = y_norm[p]

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
                cen0 - 10,
                2,
                0
            ])

            upper.extend([
                2,
                cen0 + 10,
                50,
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

    except:

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

    r2 = 1 - (ss_res / ss_tot)

    # =====================================================
    # PEAK TABLE
    # =====================================================
    peaks_data = []

    features = {

        "Sample":
            file_like.name
    }

    # =====================================================
    # FIGURA PAPER STYLE
    # =====================================================
    fig_deconv, (ax1, ax2) = plt.subplots(

        2, 1,

        figsize=(7,6),

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
    # CORES
    # =====================================================
    colors = [

        "#ff4d4d",
        "#4d79ff",
        "#ffcc00",
        "#33cc66",
        "#9999ff",
        "#bfbfbf",
        "#ff9999"
    ]

    # =====================================================
    # COMPONENTES
    # =====================================================
    for idx, i in enumerate(
        range(0, len(popt), 4)
    ):

        amp = popt[i]
        center = popt[i+1]
        sigma = popt[i+2]
        eta = popt[i+3]

        peak_curve = pseudo_voigt(

            x,

            amp,

            center,

            sigma,

            eta
        )

        # =================================================
        # ÁREA
        # =================================================
        area = np.trapz(
            peak_curve,
            x
        )

        # =================================================
        # FWHM
        # =================================================
        fwhm = 2.355 * sigma

        # =================================================
        # FEATURES
        # =================================================
        features[
            f"Area_{int(center)}"
        ] = area

        features[
            f"Eta_{int(center)}"
        ] = eta

        features[
            f"FWHM_{int(center)}"
        ] = fwhm

        # =================================================
        # CLASSIFICAÇÃO
        # =================================================
        matches = classify_peak(center)

        if matches:

            label = matches[0]["group"]
            desc = matches[0]["description"]
            category = matches[0]["category"]

        else:

            label = "Unknown"
            desc = "Unknown"
            category = "Unknown"

        peaks_data.append({

            "Peak (cm⁻¹)":
                round(center, 2),

            "Intensity":
                round(amp, 4),

            "Area":
                round(area, 4),

            "FWHM":
                round(fwhm, 2),

            "Eta":
                round(eta, 3),

            "Lorentzian %":
                round(eta * 100, 2),

            "Gaussian %":
                round((1-eta) * 100, 2),

            "Assignment":
                label,

            "Category":
                category,

            "Description":
                desc
        })

        # =================================================
        # COMPONENTES COLORIDAS
        # =================================================
        ax1.fill_between(

            x,

            peak_curve,

            alpha=0.55,

            color=colors[
                idx % len(colors)
            ]
        )

        ax1.plot(

            x,

            peak_curve,

            linewidth=1.0,

            color=colors[
                idx % len(colors)
            ]
        )

        # =================================================
        # LINHA CENTRAL
        # =================================================
        ax1.axvline(

            center,

            linestyle="--",

            linewidth=0.7,

            color="black",

            alpha=0.6
        )

        # =================================================
        # TEXTO
        # =================================================
        ax1.text(

            center,

            amp + 0.03,

            f"{int(center)}",

            fontsize=7,

            ha="center"
        )

    # =====================================================
    # FIT TOTAL
    # =====================================================
    ax1.plot(

        x,

        y_fit,

        linestyle="--",

        color="red",

        linewidth=1.5,

        label=f"Fit (R²={r2:.4f})"
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

    plt.tight_layout()

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

        color="black",

        linewidth=1.0
    )

    ax_spec.set_xlabel(
        "Raman Shift (cm⁻¹)"
    )

    ax_spec.set_ylabel(
        "Corrected Intensity"
    )

    ax_spec.grid(alpha=0.2)

    plt.tight_layout()

    # =====================================================
    # PEAKS DF
    # =====================================================
    peaks_df = pd.DataFrame(
        peaks_data
    )

    # =====================================================
    # SUMMARY
    # =====================================================
    summary = {

        "Sample":
            file_like.name,

        "R²":
            round(r2, 4),

        "N Peaks":
            int(len(popt)/4),

        "Main Peak":
            round(
                x[np.argmax(y_norm)],
                2
            )
    }

    return {

        "summary":
            summary,

        "peaks_df":
            peaks_df,

        "features":
            features,

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

    # =====================================================
    # NORMALIZAÇÃO
    # =====================================================
    X = StandardScaler().fit_transform(
        numeric_df
    )

    # =====================================================
    # PCA
    # =====================================================
    pca = PCA(
        n_components=2
    )

    scores = pca.fit_transform(X)

    explained = (
        pca.explained_variance_ratio_ * 100
    )

    # =====================================================
    # DATAFRAMES
    # =====================================================
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
    # FIGURA
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

    plt.tight_layout()

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
