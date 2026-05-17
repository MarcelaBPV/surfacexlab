# =========================================================
# raman_mapping.py
# Advanced Raman Molecular Mapping + Pseudo-Voigt
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
# PAGE
# =========================================================
st.set_page_config(layout="wide")

# =========================================================
# MOLECULAR DATABASE
# =========================================================
# =========================================================
# ADVANCED RAMAN BLOOD DATABASE
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
    # CARBON-BASED SIGNALS
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
    # NANOSTRUCTURES / SERS
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
# IDENTIFICATION
# =========================================================
def identify_peak(shift):

    for molecule, peaks in RAMAN_DATABASE.items():

        for p in peaks:

            if abs(shift - p) <= 10:

                return molecule

    return "Unknown"

# =========================================================
# MAIN FUNCTION
# =========================================================
def render_raman_mapping_tab():

    st.title("🧬 Advanced Raman Molecular Mapping")

    # =====================================================
    # SIDEBAR
    # =====================================================
    st.sidebar.header("⚙ Configuration")

    shift_min = st.sidebar.number_input(
        "Shift Min",
        value=1500
    )

    shift_max = st.sidebar.number_input(
        "Shift Max",
        value=1800
    )

    prominence = st.sidebar.slider(
        "Peak Prominence",
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

        st.info("Upload Raman mapping file.")
        return

    # =====================================================
    # READ FILE
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

        st.error("Error reading file.")
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

            st.error(f"Missing column: {c}")
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
        f"{len(grouped)} spectra detected."
    )

    # =====================================================
    # SELECT SPECTRA
    # =====================================================
    selected = st.multiselect(

        "Select spectra",

        options=list(range(len(grouped))),

        default=list(range(min(18, len(grouped))))
    )

    # =====================================================
    # HEATMAP
    # =====================================================
    heatmap = []

    # =====================================================
    # SUBPLOTS
    # =====================================================
    n_spectra = len(selected)

    cols = 3

    rows = int(np.ceil(n_spectra / cols))

    fig = plt.figure(

        figsize=(18, rows * 5),

        dpi=300
    )

    plot_idx = 1

    # =====================================================
    # LOOP
    # =====================================================
    for idx, ((x, y), group) in enumerate(grouped):

        if idx not in selected:
            continue

        group = group.sort_values(
            "wave"
        )

        wave = group["wave"].values

        intensity = group[
            "intensity"
        ].values

        # =================================================
        # BASELINE
        # =================================================
        baseline = baseline_als(
            intensity
        )

        corrected = intensity - baseline

        # =================================================
        # SMOOTHING
        # =================================================
        smooth = savgol_filter(

            corrected,

            21,

            3
        )

        # =================================================
        # NORMALIZATION
        # =================================================
        norm = (

            smooth - np.min(smooth)

        ) / (

            np.max(smooth)

            - np.min(smooth)

            + 1e-9
        )

        # =================================================
        # PEAK DETECTION
        # =================================================
        peaks, props = find_peaks(

            norm,

            prominence=prominence
        )

        peak_positions = wave[peaks]

        # =================================================
        # INITIAL PARAMS
        # =================================================
        initial_params = []

        for peak in peaks:

            initial_params.extend([

                norm[peak],
                wave[peak],
                5,
                5,
                0.5
            ])

        # =================================================
        # FIT
        # =================================================
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

        # =================================================
        # R²
        # =================================================
        residual = norm - fit

        ss_res = np.sum(residual**2)

        ss_tot = np.sum(

            (norm - np.mean(norm))**2
        )

        r2 = 1 - (ss_res / ss_tot)

        # =================================================
        # HEATMAP
        # =================================================
        heatmap.append([

            x,
            y,
            np.max(norm)
        ])

        # =================================================
        # SPECTRUM PLOT
        # =================================================
        gs = fig.add_gridspec(

            rows * 2,

            cols,

            hspace=0.5
        )

        row_base = ((plot_idx - 1)//cols) * 2
        col = (plot_idx - 1)%cols

        ax1 = fig.add_subplot(
            gs[row_base, col]
        )

        ax2 = fig.add_subplot(
            gs[row_base+1, col]
        )

        # =================================================
        # MAIN CURVE
        # =================================================
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

            linewidth=1.2,

            label=f"Fit (R²={r2:.4f})"
        )

        # =================================================
        # COMPONENTS
        # =================================================
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

            ax1.fill_between(

                wave,

                component,

                alpha=0.4
            )

            ax1.text(

                x0,

                np.max(component),

                f"{int(x0)}",

                fontsize=7
            )

            ax1.annotate(

                molecule,

                (

                    x0,

                    np.max(component)

                ),

                fontsize=6,

                rotation=90
            )

        # =================================================
        # RESIDUAL
        # =================================================
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

        # =================================================
        # TITLES
        # =================================================
        ax1.set_title(

            f"Spectrum {idx} | Y={y}"
        )

        ax1.set_ylabel(
            "Normalized Intensity"
        )

        ax1.legend(fontsize=6)

        ax1.grid(alpha=0.2)

        ax2.set_xlabel(
            "Raman Shift (cm⁻¹)"
        )

        ax2.set_ylabel(
            "Residual"
        )

        ax2.grid(alpha=0.2)

        plot_idx += 1

    plt.tight_layout()

    st.pyplot(fig)

    # =====================================================
    # HEATMAP
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

    fig2, ax2 = plt.subplots(

        figsize=(12,6),

        dpi=300
    )

    im = ax2.imshow(

        pivot,

        cmap="inferno",

        interpolation="bicubic",

        aspect="auto",

        origin="lower"
    )

    cbar = plt.colorbar(

        im,

        ax=ax2
    )

    cbar.set_label(
        "Relative Raman Intensity"
    )

    ax2.set_title(
        "Spatial Raman Molecular Distribution"
    )

    ax2.set_xlabel(
        "X Position"
    )

    ax2.set_ylabel(
        "Y Position"
    )

    st.pyplot(fig2)
