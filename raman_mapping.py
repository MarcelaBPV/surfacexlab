# =========================================================
# raman_mapping.py
# SurfaceXLab — Raman Molecular Mapping
# COMPLETE CORRECTED VERSION
# =========================================================

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.optimize import curve_fit
from scipy.signal import savgol_filter, medfilt
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve

# =========================================================
# RAMAN DATABASE
# =========================================================
RAMAN_DATABASE = {

    "Hemoglobina": [1547,1562,1582,1604,1620],
    "Grupo Heme": [754,1212,1375,1547,1562,1582],
    "Porfirinas": [750,1127,1310,1580,1622],

    "Proteínas": [1003,1208,1448,1605,1655],
    "Amida I": [1635,1650,1660,1670],
    "Amida II": [1540,1555,1570],
    "Amida III": [1230,1245,1265,1280],

    "Fenilalanina": [1003,1032,1208,1585],
    "Triptofano": [758,880,1010,1340,1555],
    "Tirosina": [830,850,1175,1615],

    "Lipídios": [1060,1265,1300,1440,1655,1735],

    "DNA": [725,785,1092,1335,1578],
    "RNA": [813,1100,1338,1485],

    "Hemácias": [754,1212,1375,1547,1582],

    "SERS Hotspots": [1550,1580,1605]
}

# =========================================================
# IDENTIFY PEAK
# =========================================================
def identify_peak(shift):

    matches = []

    for group, peaks in RAMAN_DATABASE.items():

        for p in peaks:

            if abs(shift - p) <= 10:

                matches.append(group)

    if matches:

        return ", ".join(matches)

    return "Unknown"

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

        [1,-2,1],

        [0,-1,-2],

        shape=(L,L-2),

        dtype=float
    )

    w = np.ones(L)

    for _ in range(niter):

        W = diags(w,0)

        Z = W + lam * D.dot(D.transpose())

        z = spsolve(

            Z.tocsc(),

            w*y
        )

        w = p*(y>z) + (1-p)*(y<z)

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

    lorentz = (

        sigma**2 /

        ((x-cen)**2 + sigma**2)
    )

    gauss = np.exp(

        -((x-cen)**2) /

        (2*sigma**2)
    )

    return amp * (

        eta*lorentz +

        (1-eta)*gauss
    )

# =========================================================
# MULTI VOIGT
# =========================================================
def multi_pseudo_voigt(x,*params):

    y = np.zeros_like(x)

    for i in range(0,len(params),4):

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
# BLOOD BANDS
# =========================================================
def get_blood_bands():

    return [

        1533,
        1555,
        1580,
        1604,
        1620,
        1628,
        1658
    ]

# =========================================================
# MAIN
# =========================================================
def render_raman_mapping_tab():

    st.title("🧬 Raman Molecular Mapping")

    # =====================================================
    # SIDEBAR
    # =====================================================
    st.sidebar.header("⚙️ Mapping Settings")

    shift_min = st.sidebar.number_input(

        "Shift Min",

        value=1500
    )

    shift_max = st.sidebar.number_input(

        "Shift Max",

        value=1700
    )

    # =====================================================
    # FILE
    # =====================================================
    uploaded_file = st.file_uploader(

        "Upload Raman Mapping",

        type=["txt","csv","xlsx"]
    )

    if uploaded_file is None:

        st.info("Upload Raman mapping file.")
        return

    # =====================================================
    # READ FILE
    # =====================================================
    try:

        if uploaded_file.name.endswith(

            (".xlsx",".xls")
        ):

            df = pd.read_excel(
                uploaded_file
            )

        else:

            try:

                df = pd.read_csv(

                    uploaded_file,

                    sep=r"\s+",

                    engine="python"
                )

            except:

                df = pd.read_csv(

                    uploaded_file,

                    sep="\t",

                    engine="python"
                )

    except Exception as e:

        st.error("Error reading file.")
        st.exception(e)
        return

    # =====================================================
    # FIX COLUMNS
    # =====================================================
    df.columns = [

        str(c)
        .replace("\t","")
        .replace(" ","")
        .lower()
        .strip()

        for c in df.columns
    ]

    # =====================================================
    # REQUIRED
    # =====================================================
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

    # =====================================================
    # NUMERIC
    # =====================================================
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

        df.groupby(["x","y"])
    )

    st.success(

        f"{len(grouped)} spectra detected."
    )

    # =====================================================
    # HEATMAP
    # =====================================================
    heatmap = []

    for idx, ((x,y), group) in enumerate(grouped):

        intensity = group[
            "intensity"
        ].values

        heatmap.append([

            x,
            y,
            np.max(intensity)
        ])

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

    st.subheader("🔥 Raman Intensity Map")

    fig_map, ax_map = plt.subplots(

        figsize=(10,5),

        dpi=400
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

    ax_map.set_xlabel(
        "X Position"
    )

    ax_map.set_ylabel(
        "Y Position"
    )

    plt.tight_layout()

    st.pyplot(fig_map)

    # =====================================================
    # SPECTRUM SELECTION
    # =====================================================
    options = ["All Spectra"] + [

        f"Spectrum {i+1}"

        for i in range(len(grouped))
    ]

    selected = st.selectbox(

        "Choose Spectrum",

        options
    )

    if selected == "All Spectra":

        spectra_to_process = range(
            len(grouped)
        )

    else:

        idx = int(
            selected.split()[-1]
        ) - 1

        spectra_to_process = [idx]

    # =====================================================
    # PROCESS LOOP
    # =====================================================
    for selected_idx in spectra_to_process:

        idx, ((x_pos,y_pos), group) = list(

            enumerate(grouped)

        )[selected_idx]

        group = group.sort_values(
            "wave"
        )

        x = group["wave"].values

        y = group["intensity"].values

        # =================================================
        # COSMIC RAY REMOVAL
        # =================================================
        y_med = medfilt(

            y,

            kernel_size=5
        )

        diff = np.abs(y - y_med)

        threshold = 5*np.std(diff)

        y = np.where(

            diff > threshold,

            y_med,

            y
        )

        # =================================================
        # SMOOTH
        # =================================================
        y_smooth = savgol_filter(

            y,

            11,

            3
        )

        # =================================================
        # BASELINE
        # =================================================
        baseline = baseline_als(
            y_smooth
        )

        y_corr = y_smooth - baseline

        y_corr = y_corr - np.min(y_corr)

        # =================================================
        # NORMALIZE
        # =================================================
        y_norm = y_corr / np.max(y_corr)

        # =================================================
        # FIT
        # =================================================
        bands = get_blood_bands()

        p0 = []
        lower = []
        upper = []

        for band in bands:

            idx_band = np.argmin(
                np.abs(x-band)
            )

            amp0 = max(

                y_norm[idx_band],

                0.05
            )

            p0.extend([

                amp0,
                band,
                8,
                0.5
            ])

            lower.extend([

                0,
                band-5,
                2,
                0.2
            ])

            upper.extend([

                2,
                band+5,
                25,
                0.8
            ])

        try:

            popt,_ = curve_fit(

                multi_pseudo_voigt,

                x,

                y_norm,

                p0=p0,

                bounds=(lower,upper),

                maxfev=100000
            )

        except:

            popt = np.array(p0)

        y_fit = multi_pseudo_voigt(

            x,

            *popt
        )

        residual = y_norm - y_fit

        # =================================================
        # R²
        # =================================================
        ss_res = np.sum(residual**2)

        ss_tot = np.sum(

            (y_norm - np.mean(y_norm))**2
        )

        r2 = 1 - (ss_res/ss_tot)

        # =================================================
        # FIGURE
        # =================================================
        fig = plt.figure(

            figsize=(10,7),

            dpi=500
        )

        gs = fig.add_gridspec(

            2,
            1,

            height_ratios=[4,1],

            hspace=0.08
        )

        ax1 = fig.add_subplot(gs[0])

        ax2 = fig.add_subplot(gs[1])

        # =================================================
        # EXPERIMENTAL
        # =================================================
        ax1.plot(

            x,

            y_norm,

            color="black",

            linewidth=1.8,

            label="Experimental"
        )

        # =================================================
        # FIT
        # =================================================
        ax1.plot(

            x,

            y_fit,

            "--",

            color="red",

            linewidth=1.5,

            label=f"Fit (R²={r2:.4f})"
        )

        colors = [

            "#f94144",
            "#577590",
            "#43aa8b",
            "#f9c74f",
            "#9b5de5",
            "#f3722c",
            "#90be6d"
        ]

        peak_table = []

        # =================================================
        # COMPONENTS
        # =================================================
        for peak_idx,i in enumerate(

            range(0,len(popt),4)
        ):

            amp = popt[i]
            cen = popt[i+1]
            sigma = popt[i+2]
            eta = popt[i+3]

            peak_curve = pseudo_voigt(

                x,

                amp,

                cen,

                sigma,

                eta
            )

            ax1.fill_between(

                x,

                0,

                peak_curve,

                alpha=0.30,

                color=colors[
                    peak_idx % len(colors)
                ]
            )

            ax1.plot(

                x,

                peak_curve,

                linewidth=1.0,

                color=colors[
                    peak_idx % len(colors)
                ]
            )

            ax1.text(

                cen,

                np.max(peak_curve)+0.02,

                f"{int(cen)}",

                fontsize=8,

                ha="center"
            )

            fwhm = 2.355 * sigma

            peak_table.append({

                "Peak":
                    round(cen,2),

                "Assignment":
                    identify_peak(cen),

                "Intensity":
                    round(amp,4),

                "FWHM":
                    round(fwhm,2)
            })

        # =================================================
        # STYLE
        # =================================================
        ax1.set_title(

            f"Spectrum {selected_idx+1} | "
            f"X={x_pos} Y={y_pos}"
        )

        ax1.set_ylabel(
            "Normalized Intensity"
        )

        ax2.set_ylabel(
            "Residual"
        )

        ax2.set_xlabel(
            "Raman Shift (cm⁻¹)"
        )

        ax1.grid(alpha=0.15)
        ax2.grid(alpha=0.15)

        ax1.legend()

        ax1.spines["top"].set_visible(False)
        ax1.spines["right"].set_visible(False)

        ax2.spines["top"].set_visible(False)
        ax2.spines["right"].set_visible(False)

        # =================================================
        # RESIDUAL
        # =================================================
        ax2.plot(

            x,

            residual,

            color="black",

            linewidth=1
        )

        ax2.axhline(

            0,

            linestyle="--",

            color="gray"
        )

        plt.tight_layout()

        st.pyplot(fig)

        # =================================================
        # TABLE
        # =================================================
        peak_df = pd.DataFrame(
            peak_table
        )

        st.dataframe(

            peak_df,

            width="stretch",

            height=300
        )
