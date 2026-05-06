# =========================================================
# Raman Processing — SurfaceXLab
# Pipeline Científico Raman
# =========================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.optimize import curve_fit

from scipy.signal import (
    find_peaks,
    savgol_filter
)

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
    if name.endswith(".xlsx") or name.endswith(".xls"):

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

            try:

                df = pd.read_csv(
                    file_like,
                    sep=None,
                    engine="python",
                    encoding="latin1"
                )

            except:

                df = pd.read_csv(
                    file_like,
                    sep=None,
                    engine="python",
                    encoding="cp1252"
                )


    # =====================================================
    # LIMPEZA
    # =====================================================
    df = df.apply(
        pd.to_numeric,
        errors="coerce"
    ).dropna()

    if df.shape[1] < 2:

        raise ValueError(
            "Arquivo inválido para Raman"
        )

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
    p=0.01,
    niter=10
):

    L = len(y)

    D = np.diff(
        np.eye(L),
        n=2,
        axis=0
    )

    DTD = D.T @ D

    w = np.ones(L)

    for _ in range(niter):

        W = np.diag(w)

        Z = np.linalg.solve(
            W + lam * DTD,
            w * y
        )

        w = (
            p * (y > Z)
            +
            (1-p) * (y < Z)
        )

    return Z


# =========================================================
# LORENTZIANA
# =========================================================
def lorentz(
    x,
    amplitude,
    center,
    gamma
):

    return amplitude * (
        gamma**2 /
        ((x-center)**2 + gamma**2)
    )


# =========================================================
# MULTI-LORENTZ
# =========================================================
def multi_lorentz(x, *params):

    y = np.zeros_like(x)

    for i in range(0, len(params), 3):

        amplitude = params[i]

        center = params[i+1]

        gamma = params[i+2]

        y += lorentz(
            x,
            amplitude,
            center,
            gamma
        )

    return y


# =========================================================
# VALIDAÇÃO ESPECTRAL
# =========================================================
def validate_spectrum(x, y):

    if len(x) < 50:

        return False, "Poucos pontos espectrais"

    if np.max(np.abs(y)) < 1e-6:

        return False, "Intensidade inválida"

    if np.isnan(y).sum() > 0:

        return False, "Valores NaN encontrados"

    return True, "OK"


# =========================================================
# FEATURE ENGINEERING
# =========================================================
def extract_raman_features(peaks_df):

    features = {}

    for _, row in peaks_df.iterrows():

        group = row["assignment"]

        features[
            f"{group}_position"
        ] = row["position"]

        features[
            f"{group}_intensity"
        ] = row["intensity"]

        features[
            f"{group}_fwhm"
        ] = row["fwhm"]

    return features


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
    # VALIDAÇÃO
    # =====================================================
    valid, message = validate_spectrum(x, y)

    if not valid:

        raise ValueError(message)


    # =====================================================
    # SUAVIZAÇÃO
    # =====================================================
    y_smooth = savgol_filter(
        y,
        window_length=11,
        polyorder=3
    )


    # =====================================================
    # BASELINE ALS
    # =====================================================
    lam = 1e5 if len(y) > 100 else 1e4

    baseline = baseline_als(
        y_smooth,
        lam=lam
    )

    y_corr = y_smooth - baseline


    # =====================================================
    # NORMALIZAÇÃO
    # =====================================================
    max_int = np.max(np.abs(y_corr))

    if max_int == 0:

        raise ValueError(
            "Falha na normalização"
        )

    y_norm = y_corr / max_int

    df["intensity_norm"] = y_norm


    # =====================================================
    # DETECÇÃO DE PICOS
    # =====================================================
    peaks, _ = find_peaks(
        y_norm,
        prominence=0.05,
        distance=20
    )

    if len(peaks) < 2:

        peaks = np.argsort(y_norm)[-3:]


    # =====================================================
    # PARÂMETROS INICIAIS
    # =====================================================
    p0 = []

    lower_bounds = []

    upper_bounds = []

    for p in peaks:

        p0.extend([
            y_norm[p],
            x[p],
            10
        ])

        lower_bounds.extend([
            0,
            x[p]-20,
            1
        ])

        upper_bounds.extend([
            2,
            x[p]+20,
            200
        ])


    # =====================================================
    # FITTING LORENTZIANO
    # =====================================================
    try:

        popt, _ = curve_fit(

            multi_lorentz,

            x,

            y_norm,

            p0=p0,

            bounds=(
                lower_bounds,
                upper_bounds
            ),

            maxfev=20000
        )

        y_fit = multi_lorentz(
            x,
            *popt
        )

    except:

        popt = p0

        y_fit = multi_lorentz(
            x,
            *popt
        )


    # =====================================================
    # R²
    # =====================================================
    residual = y_norm - y_fit

    ss_res = np.sum(
        residual**2
    )

    ss_tot = np.sum(
        (y_norm - np.mean(y_norm))**2
    )

    r2 = 1 - (ss_res / ss_tot)


    # =====================================================
    # QUALITY FLAG
    # =====================================================
    quality_flag = "GOOD"

    if r2 < 0.85:

        quality_flag = "LOW_QUALITY"


    # =====================================================
    # PEAKS DATAFRAME
    # =====================================================
    peaks_data = []

    for i in range(0, len(popt), 3):

        position = popt[i+1]

        intensity = popt[i]

        gamma = popt[i+2]

        fwhm = 2 * gamma

        assignments = classify_peak(position)

        if assignments:

            for match in assignments:

                peaks_data.append({

                    "position": position,

                    "intensity": intensity,

                    "fwhm": fwhm,

                    "assignment": match["group"],

                    "category": match["category"],

                    "description": match["description"]
                })

        else:

            peaks_data.append({

                "position": position,

                "intensity": intensity,

                "fwhm": fwhm,

                "assignment": "Unknown",

                "category": "Unknown",

                "description": "Unassigned peak"
            })


    peaks_df = pd.DataFrame(peaks_data)


    # =====================================================
    # FEATURE ENGINEERING
    # =====================================================
    fingerprint = extract_raman_features(
        peaks_df
    )


    # =====================================================
    # FIGURA 1 — RAW
    # =====================================================
    fig_raw, ax_raw = plt.subplots(
        figsize=(6,4),
        dpi=300
    )

    ax_raw.plot(
        x,
        y,
        linewidth=1.2
    )

    ax_raw.set_title(
        "Raw Raman Spectrum"
    )

    ax_raw.set_xlabel(
        "Raman Shift (cm⁻¹)"
    )

    ax_raw.set_ylabel(
        "Intensity"
    )

    ax_raw.grid(alpha=0.3)


    # =====================================================
    # FIGURA 2 — BASELINE
    # =====================================================
    fig_base, ax_base = plt.subplots(
        figsize=(6,4),
        dpi=300
    )

    ax_base.plot(
        x,
        y_norm,
        label="Corrected"
    )

    ax_base.plot(
        x,
        baseline/max_int,
        "--",
        label="ASLS Baseline"
    )

    ax_base.set_title(
        "Baseline Correction"
    )

    ax_base.set_xlabel(
        "Raman Shift (cm⁻¹)"
    )

    ax_base.set_ylabel(
        "Normalized Intensity"
    )

    ax_base.legend()

    ax_base.grid(alpha=0.3)


    # =====================================================
    # FIGURA 3 — FITTING
    # =====================================================
    fig_fit, (ax1, ax2) = plt.subplots(

        2, 1,

        figsize=(7,5),

        dpi=300,

        sharex=True,

        gridspec_kw={
            "height_ratios": [3,1]
        }
    )

    # ================================================
    # FITTING
    # ================================================
    ax1.plot(
        x,
        y_norm,
        "k.",
        markersize=3,
        label="Experimental"
    )

    colors = plt.cm.tab10.colors

    for idx, i in enumerate(range(0, len(popt), 3)):

        peak_curve = lorentz(

            x,

            popt[i],

            popt[i+1],

            popt[i+2]
        )

        ax1.plot(
            x,
            peak_curve,
            color=colors[idx % len(colors)],
            alpha=0.8
        )

    ax1.plot(
        x,
        y_fit,
        "r-",
        linewidth=2,
        label=f"Fit (R²={r2:.4f})"
    )

    ax1.legend()

    ax1.set_ylabel(
        "Normalized Intensity"
    )

    ax1.grid(alpha=0.3)


    # ================================================
    # RESÍDUO
    # ================================================
    ax2.plot(
        x,
        residual,
        color="black"
    )

    ax2.axhline(
        0,
        linestyle="--"
    )

    ax2.set_xlabel(
        "Raman Shift (cm⁻¹)"
    )

    ax2.set_ylabel(
        "Residual"
    )

    ax2.grid(alpha=0.3)


    # =====================================================
    # RETORNO
    # =====================================================
    return {

        "spectrum_df": df,

        "peaks_df": peaks_df,

        "fingerprint": fingerprint,

        "r2": r2,

        "quality_flag": quality_flag,

        "figures": {

            "raw": fig_raw,

            "baseline": fig_base,

            "fit": fig_fit
        }
    }


# =========================================================
# PCA RAMAN
# =========================================================
def run_raman_pca(samples):

    fingerprints = []

    labels = []

    for sample_id, sample_data in samples.items():

        if "raman" not in sample_data:

            continue

        fp = sample_data["raman"].get(
            "fingerprint"
        )

        if fp is None:

            continue

        fingerprints.append(fp)

        labels.append(sample_id)

    if len(fingerprints) < 2:

        raise ValueError(
            "Poucas amostras para PCA"
        )

    df = pd.DataFrame(
        fingerprints
    ).fillna(0)

    X = StandardScaler().fit_transform(df)

    pca = PCA(n_components=2)

    scores = pca.fit_transform(X)

    explained = (
        pca.explained_variance_ratio_
    )

    return {

        "scores": scores,

        "labels": labels,

        "explained_variance": explained,

        "dataframe": df
    }
