# =========================================================
# raman_mapping.py
# Advanced Raman Molecular Mapping
# =========================================================

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.signal import savgol_filter, find_peaks
from scipy import sparse
from scipy.sparse.linalg import spsolve
from scipy.optimize import curve_fit

# =========================================================
# RAMAN DATABASE
# =========================================================
RAMAN_DATABASE = {

    # =====================================================
    # HEMOGLOBIN / HEME
    # =====================================================
    "Hemoglobina": [
        1547,
        1562,
        1582,
        1604,
        1620
    ],

    "Grupo Heme": [
        754,
        1212,
        1375,
        1547,
        1562,
        1582
    ],

    "Porfirinas": [
        750,
        1127,
        1310,
        1580,
        1622
    ],

    # =====================================================
    # PROTEINS
    # =====================================================
    "Proteínas": [
        1003,
        1208,
        1448,
        1605,
        1655
    ],

    "Amida I": [
        1635,
        1650,
        1660,
        1670
    ],

    "Amida II": [
        1540,
        1555,
        1570
    ],

    "Amida III": [
        1230,
        1245,
        1265,
        1280
    ],

    "Fenilalanina": [
        1003,
        1032,
        1208,
        1585
    ],

    "Triptofano": [
        758,
        880,
        1010,
        1340,
        1555
    ],

    "Tirosina": [
        830,
        850,
        1175,
        1615
    ],

    # =====================================================
    # LIPIDS
    # =====================================================
    "Lipídios": [
        1060,
        1265,
        1300,
        1440,
        1655,
        1735
    ],

    "Fosfolipídios": [
        718,
        1065,
        1090,
        1302,
        1445
    ],

    "Colesterol": [
        701,
        1440,
        1670
    ],

    # =====================================================
    # NUCLEIC ACIDS
    # =====================================================
    "DNA": [
        725,
        785,
        1092,
        1335,
        1578
    ],

    "RNA": [
        813,
        1100,
        1338,
        1485
    ],

    "Nucleotídeos": [
        725,
        780,
        1095,
        1340
    ],

    # =====================================================
    # GLUCOSE / METABOLITES
    # =====================================================
    "Glicose": [
        911,
        1060,
        1125,
        1375
    ],

    "Lactato": [
        853,
        1045,
        1455
    ],

    "Ureia": [
        1005,
        1160,
        1460
    ],

    "Creatinina": [
        680,
        846,
        908,
        1420
    ],

    # =====================================================
    # CELLULAR COMPONENTS
    # =====================================================
    "Hemácias": [
        754,
        1212,
        1375,
        1547,
        1582
    ],

    "Leucócitos": [
        785,
        1090,
        1445,
        1655
    ],

    "Plaquetas": [
        1003,
        1245,
        1450,
        1660
    ],

    # =====================================================
    # OXIDATIVE STRESS
    # =====================================================
    "Espécies Oxidativas": [
        875,
        1450,
        1660
    ],

    "Peroxidação Lipídica": [
        1265,
        1302,
        1440,
        1655,
        1745
    ],

    # =====================================================
    # CARBON
    # =====================================================
    "Carbono D": [
        1320,
        1345,
        1360
    ],

    "Carbono G": [
        1560,
        1580,
        1600
    ],

    # =====================================================
    # SERS
    # =====================================================
    "Nanotubos de Carbono": [
        1340,
        1580,
        1610
    ],

    "SERS Hotspots": [
        1550,
        1580,
        1605
    ]
}

# =========================================================
# BASELINE ALS
# =========================================================
def baseline_als(y, lam=1e5, p=0.01, niter=10):

    L = len(y)

    D = sparse.diags(
        [1, -2, 1],
        [0, -1, -2],
        shape=(L, L)
    )

    D = lam * D.dot(D.transpose())

    w = np.ones(L)

    for i in range(niter):

        W = sparse.spdiags(w, 0, L, L)

        Z = W + D

        z = spsolve(Z, w * y)

        w = p * (y > z) + (1 - p) * (y < z)

    return z

# =========================================================
# PSEUDO VOIGT
# =========================================================
def pseudo_voigt(x, A, x0, sigma, gamma, eta):

    gaussian = np.exp(
        -((x - x0)**2) / (2 * sigma**2)
    )

    lorentz = 1 / (
        1 + ((x - x0)/gamma)**2
    )

    return A * (
        eta * lorentz +
        (1 - eta) * gaussian
    )

# =========================================================
# MULTI PEAK
# =========================================================
def multi_pseudo_voigt(x, *params):

    y = np.zeros_like(x)

    n_peaks = len(params) // 5

    for i in range(n_peaks):

        A = params[i*5]
        x0 = params[i*5 + 1]
        sigma = params[i*5 + 2]
        gamma = params[i*5 + 3]
        eta = params[i*5 + 4]

        y += pseudo_voigt(
            x,
            A,
            x0,
            sigma,
            gamma,
            eta
        )

    return y

# =========================================================
# IDENTIFY PEAK
# =========================================================
def identify_peak(shift):

    for molecule, peaks in RAMAN_DATABASE.items():

        for p in peaks:

            if abs(shift - p) <= 10:

                return molecule

    return "Não identificado"

# =========================================================
# MAIN FUNCTION
# =========================================================
def render_raman_mapping_tab():

    st.title("🧬 Advanced Raman Molecular Mapping")

    # =====================================================
    # SIDEBAR
    # =====================================================
    st.sidebar.header("⚙️ Raman Mapping")

    shift_min = st.sidebar.number_input(
        "Shift mínimo",
        value=1500
    )

    shift_max = st.sidebar.number_input(
        "Shift máximo",
        value=1800
    )

    prominence = st.sidebar.slider(
        "Peak prominence",
        0.01,
        0.5,
        0.05
    )

    # =====================================================
    # FILE
    # =====================================================
    uploaded_file = st.file_uploader(

        "Upload Raman Mapping",

        type=["txt", "csv", "xlsx"]
    )

    if uploaded_file is None:

        st.info("Faça upload do arquivo Raman.")
        return

    # =====================================================
    # READ
    # =====================================================
    try:

        if uploaded_file.name.endswith(
            (".txt", ".csv")
        ):

            df = pd.read_csv(

                uploaded_file,

                sep=r"\s+",

                engine="python"
            )

        else:

            df = pd.read_excel(
                uploaded_file
            )

    except Exception as e:

        st.error("Erro ao ler arquivo.")
        st.exception(e)
        return

    # =====================================================
    # ORGANIZE
    # =====================================================
    df.columns = [

        str(c).strip().lower()

        for c in df.columns
    ]

    required = [

        "x",
        "y",
        "wave",
        "intensity"
    ]

    for c in required:

        if c not in df.columns:

            st.error(f"Faltando coluna: {c}")
            st.write(df.columns)
            return

    for c in required:

        df[c] = pd.to_numeric(
            df[c],
            errors="coerce"
        )

    df = df.dropna()

    # =====================================================
    # FILTER REGION
    # =====================================================
    df = df[

        (df["wave"] >= shift_min) &
        (df["wave"] <= shift_max)

    ]

    # =====================================================
    # GROUP
    # =====================================================
    grouped = list(

        df.groupby(["x", "y"])
    )

    st.success(
        f"{len(grouped)} espectros detectados."
    )

    # =====================================================
    # HEATMAP
    # =====================================================
    heatmap = []

    for idx, ((x, y), group) in enumerate(grouped):

        intensity = group[
            "intensity"
        ].values

        heatmap.append([

            x,
            y,
            np.max(intensity)
        ])

    # =====================================================
    # HEATMAP FIGURE
    # =====================================================
    st.subheader("🔥 Raman Intensity Map")

    heat_df = pd.DataFrame(

        heatmap,

        columns=[
            "X",
            "Y",
            "Intensity"
        ]
    )

    pivot = heat_df.pivot(

        index="Y",

        columns="X",

        values="Intensity"
    )

    fig_map, ax_map = plt.subplots(

        figsize=(12,6),

        dpi=300
    )

    im = ax_map.imshow(

        pivot,

        cmap="magma",

        interpolation="bicubic",

        aspect="auto",

        origin="lower"
    )

    cbar = plt.colorbar(
        im,
        ax=ax_map
    )

    cbar.set_label(
        "Relative Raman Intensity"
    )

    ax_map.set_title(
        "Spatial Raman Molecular Distribution"
    )

    ax_map.set_xlabel(
        "X Position"
    )

    ax_map.set_ylabel(
        "Y Position"
    )

    st.pyplot(fig_map)

    # =====================================================
    # SELECT SPECTRUM
    # =====================================================
    st.subheader("🔬 Detailed Spectrum Analysis")

    selected_spectrum = st.selectbox(

        "Selecionar espectro",

        options=list(range(len(grouped))),

        format_func=lambda x: f"Espectro {x}"
    )

    idx, ((x, y), group) = list(
        enumerate(grouped)
    )[selected_spectrum]

    group = group.sort_values(
        "wave"
    )

    wave = group["wave"].values

    intensity = group[
        "intensity"
    ].values

    # =====================================================
    # BASELINE
    # =====================================================
    baseline = baseline_als(
        intensity
    )

    corrected = intensity - baseline

    # =====================================================
    # SMOOTH
    # =====================================================
    smooth = savgol_filter(

        corrected,

        21,

        3
    )

    # =====================================================
    # NORMALIZE
    # =====================================================
    norm = (

        smooth - np.min(smooth)

    ) / (

        np.max(smooth)

        - np.min(smooth)

        + 1e-9
    )

    # =====================================================
    # PEAKS
    # =====================================================
    peaks, props = find_peaks(

        norm,

        prominence=prominence
    )

    # =====================================================
    # INITIAL PARAMS
    # =====================================================
    initial_params = []

    for peak in peaks:

        initial_params.extend([

            norm[peak],
            wave[peak],
            5,
            5,
            0.5
        ])

    # =====================================================
    # FIT
    # =====================================================
    try:

        popt, _ = curve_fit(

            multi_pseudo_voigt,

            wave,

            norm,

            p0=initial_params,

            maxfev=20000
        )

        fit = multi_pseudo_voigt(
            wave,
            *popt
        )

    except:

        fit = norm
        popt = initial_params

    # =====================================================
    # RESIDUAL
    # =====================================================
    residual = norm - fit

    ss_res = np.sum(residual**2)

    ss_tot = np.sum(

        (norm - np.mean(norm))**2
    )

    r2 = 1 - (ss_res / ss_tot)

    # =====================================================
    # FIGURE
    # =====================================================
    fig = plt.figure(

        figsize=(14,8),

        dpi=300
    )

    gs = fig.add_gridspec(

        2,

        1,

        height_ratios=[4,1],

        hspace=0.15
    )

    ax1 = fig.add_subplot(gs[0])

    ax2 = fig.add_subplot(gs[1])

    # =====================================================
    # MAIN CURVES
    # =====================================================
    ax1.plot(

        wave,

        norm,

        color="black",

        linewidth=1.5,

        label="Experimental"
    )

    ax1.plot(

        wave,

        fit,

        "--",

        color="red",

        linewidth=1.5,

        label=f"Fit (R²={r2:.4f})"
    )

    # =====================================================
    # TABLE
    # =====================================================
    peak_table = []

    # =====================================================
    # COMPONENTS
    # =====================================================
    n_peaks = len(popt)//5

    for i in range(n_peaks):

        A = popt[i*5]
        x0 = popt[i*5 + 1]
        sigma = popt[i*5 + 2]
        gamma = popt[i*5 + 3]
        eta = popt[i*5 + 4]

        component = pseudo_voigt(

            wave,

            A,

            x0,

            sigma,

            gamma,

            eta
        )

        molecule = identify_peak(x0)

        fwhm = 0.5346 * (2*gamma) + np.sqrt(

            0.2166*(2*gamma)**2 +

            (2.355*sigma)**2
        )

        ax1.fill_between(

            wave,

            component,

            alpha=0.4
        )

        ax1.text(

            x0,

            np.max(component),

            f"{int(x0)}",

            fontsize=8
        )

        ax1.annotate(

            molecule,

            (

                x0,

                np.max(component)

            ),

            fontsize=8,

            rotation=90
        )

        peak_table.append({

            "Peak (cm⁻¹)": round(x0,2),

            "Molecular Group": molecule,

            "Intensity": round(A,4),

            "FWHM": round(fwhm,2),

            "Gaussian σ": round(sigma,2),

            "Lorentzian γ": round(gamma,2),

            "η Mixing": round(eta,2)
        })

    # =====================================================
    # RESIDUAL
    # =====================================================
    ax2.plot(

        wave,

        residual,

        color="gray",

        linewidth=0.8
    )

    ax2.axhline(

        0,

        linestyle="--",

        linewidth=0.5
    )

    # =====================================================
    # LABELS
    # =====================================================
    ax1.set_title(

        f"Raman Spectrum {selected_spectrum} | X={x} | Y={y}"
    )

    ax1.set_ylabel(
        "Normalized Intensity"
    )

    ax1.legend()

    ax1.grid(alpha=0.2)

    ax2.set_xlabel(
        "Raman Shift (cm⁻¹)"
    )

    ax2.set_ylabel(
        "Residual"
    )

    ax2.grid(alpha=0.2)

    st.pyplot(fig)

    # =====================================================
    # TABLE
    # =====================================================
    st.subheader("📊 Identified Raman Peaks")

    peak_df = pd.DataFrame(
        peak_table
    )

    st.dataframe(

        peak_df,

        use_container_width=True
    )

    # =====================================================
    # DOWNLOAD
    # =====================================================
    csv = peak_df.to_csv(
        index=False
    )

    st.download_button(

        "⬇ Download Peak Table",

        data=csv,

        file_name=f"raman_peaks_spectrum_{selected_spectrum}.csv",

        mime="text/csv"
    )
