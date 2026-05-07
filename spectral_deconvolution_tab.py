# =========================================================
# SurfaceXLab — Intelligent Raman Analysis Platform
# Automatic Peak Detection + Molecular Identification
# =========================================================

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.signal import savgol_filter, find_peaks
from scipy import sparse
from scipy.sparse.linalg import spsolve

from lmfit.models import (
    LorentzianModel,
    GaussianModel,
    PseudoVoigtModel
)

# =========================================================
# BLOOD RAMAN DATABASE
# =========================================================

BLOOD_RAMAN_DATABASE = {

    720: "Adenine",

    752: "Tryptophan / Hemoglobin",

    785: "DNA/RNA",

    830: "Tyrosine",

    852: "Proteins",

    935: "α-helix proteins",

    1003: "Phenylalanine",

    1030: "Phenylalanine",

    1080: "Lipids / DNA",

    1125: "Lipids",

    1155: "Carotenoids",

    1170: "Tyrosine",

    1207: "Tryptophan/Fenylalanine",

    1245: "Amide III",

    1300: "Lipids",

    1335: "DNA / Proteins",

    1365: "Hemoglobin",

    1445: "CH₂ lipids/proteins",

    1547: "Hemoglobin",

    1562: "Tryptophan",

    1575: "Hemoglobin",

    1602: "Phenylalanine/Tyrosine",

    1620: "Hemoglobin",

    1655: "Amide I",

    1660: "Unsaturated lipids"
}

# =========================================================
# BASELINE CORRECTION
# =========================================================

def baseline_asls(
    y,
    lam=1e8,
    p=0.0001,
    niter=10
):

    y = np.asarray(y, dtype=float)

    L = len(y)

    D = sparse.diags(
        [1, -2, 1],
        [0, -1, -2],
        shape=(L, L - 2),
        dtype=float
    ).tocsc()

    w = np.ones(L)

    for _ in range(niter):

        W = sparse.spdiags(
            w,
            0,
            L,
            L
        ).tocsc()

        Z = W + lam * D.dot(D.transpose())

        z = spsolve(
            Z,
            w * y
        )

        w = p * (y > z) + (1 - p) * (y < z)

    return z

# =========================================================
# MAIN FUNCTION
# =========================================================

def render_spectral_deconvolution_tab():

    st.title(
        "🧬 Intelligent Raman Molecular Analysis"
    )

    st.markdown("""

    Plataforma automatizada para:

    - Raman Mapping
    - Peak Detection
    - Molecular Identification
    - Automatic Spectral Fitting
    - Residual Analysis
    - Origin Validation
    - Raman Intelligence

    """)

    st.divider()

    # =====================================================
    # CONFIG
    # =====================================================

    st.subheader(
        "⚙ Processing Configuration"
    )

    col1, col2, col3 = st.columns(3)

    with col1:

        model_type = st.selectbox(

            "Fit Model",

            [
                "Pseudo-Voigt",
                "Lorentzian",
                "Gaussian"
            ]
        )

    with col2:

        smooth_window = st.slider(

            "Savitzky-Golay Window",

            5,

            51,

            11,

            step=2
        )

    with col3:

        polyorder = st.slider(

            "Polynomial Order",

            2,

            5,

            3
        )

    # =====================================================
    # FILE UPLOAD
    # =====================================================

    uploaded_file = st.file_uploader(

        "Upload Raman Mapping",

        type=[
            "txt",
            "csv",
            "xlsx"
        ]
    )

    if uploaded_file is None:

        st.info(
            "Upload Raman mapping file."
        )

        return

    # =====================================================
    # LOAD FILE
    # =====================================================

    try:

        if uploaded_file.name.endswith(".xlsx"):

            df = pd.read_excel(
                uploaded_file
            )

        else:

            df = pd.read_csv(

                uploaded_file,

                sep=r"\s+|,|;",

                engine="python"
            )

    except Exception as e:

        st.error(
            f"File loading error: {e}"
        )

        return

    st.success(
        "File loaded successfully."
    )

    df.columns = [
        str(c).strip()
        for c in df.columns
    ]

    st.write(df.head())

    # =====================================================
    # REQUIRED COLUMNS
    # =====================================================

    if not all(col in df.columns for col in ['x', 'y']):

        st.error(
            "File must contain x and y columns."
        )

        return

    # =====================================================
    # COLUMN SELECTION
    # =====================================================

    cols = df.columns.tolist()

    col1, col2 = st.columns(2)

    with col1:

        x_col = st.selectbox(

            "Raman Shift Column",

            cols,

            index=min(2, len(cols)-1)
        )

    with col2:

        y_col = st.selectbox(

            "Intensity Column",

            cols,

            index=min(3, len(cols)-1)
        )

    # =====================================================
    # GROUPED DATA
    # =====================================================

    grouped = df.groupby(['x', 'y'])

    coords = list(grouped.groups.keys())

    # =====================================================
    # TABS
    # =====================================================

    tab1, tab2, tab3 = st.tabs([

        "🔬 Raman Mapping",

        "⚙ Spectral Processing",

        "📉 Validation"
    ])

    # =====================================================
    # TAB 1 — RAMAN MAPPING
    # =====================================================

    with tab1:

        st.header(
            "🔬 Raman Mapping Viewer"
        )

        st.success(
            f"{len(coords)} spectra detected."
        )

        n_spectra = st.slider(

            "Number of spectra",

            min_value=1,

            max_value=min(36, len(coords)),

            value=9
        )

        coords_plot = coords[:n_spectra]

        ncols = 3
        nrows = int(np.ceil(n_spectra / ncols))

        fig_map, axes = plt.subplots(

            nrows=nrows,

            ncols=ncols,

            figsize=(15, 4*nrows)
        )

        axes = np.array(axes).flatten()

        for idx, (x_pos, y_pos) in enumerate(coords_plot):

            sub = grouped.get_group(
                (x_pos, y_pos)
            )

            wave = sub[x_col].values
            inten = sub[y_col].values

            order = np.argsort(wave)

            wave = wave[order]
            inten = inten[order]

            axes[idx].plot(

                wave,

                inten,

                color='black',

                lw=1
            )

            axes[idx].set_title(
                f"Y = {y_pos:.0f} µm"
            )

            axes[idx].grid(alpha=0.2)

        for j in range(idx+1, len(axes)):

            fig_map.delaxes(
                axes[j]
            )

        plt.tight_layout()

        st.pyplot(fig_map)

    # =====================================================
    # TAB 2 — PROCESSING
    # =====================================================

    with tab2:

        st.header(
            "⚙ Spectral Processing"
        )

        # =================================================
        # SELECT SPECTRUM
        # =================================================

        st.subheader(
            "🎯 Select Spectrum"
        )

        coord_options = [

            f"X={x} | Y={y}"

            for x,y in coords
        ]

        selected_coord = st.selectbox(

            "Choose spectrum",

            coord_options
        )

        selected_idx = coord_options.index(
            selected_coord
        )

        x_sel, y_sel = coords[selected_idx]

        selected_spectrum = grouped.get_group(
            (x_sel, y_sel)
        )

        st.success(
            f"Selected spectrum: X={x_sel} | Y={y_sel}"
        )

        # =================================================
        # EXTRACT
        # =================================================

        x = selected_spectrum[x_col].values
        y = selected_spectrum[y_col].values

        order = np.argsort(x)

        x = x[order]
        y = y[order]

        # =================================================
        # REGION
        # =================================================

        st.subheader(
            "🔍 Spectral Region"
        )

        col1, col2 = st.columns(2)

        with col1:

            x_min = st.number_input(

                "Min Raman",

                value=600.0
            )

        with col2:

            x_max = st.number_input(

                "Max Raman",

                value=1800.0
            )

        mask = (

            (x >= x_min) &

            (x <= x_max)
        )

        x = x[mask]
        y = y[mask]

        # =================================================
        # PREPROCESSING
        # =================================================

        st.subheader(
            "📊 Spectral Preprocessing"
        )

        y_smooth = savgol_filter(

            y,

            smooth_window,

            polyorder
        )

        baseline = baseline_asls(

            y_smooth,

            lam=1e8,

            p=0.0001
        )

        y_corr = y_smooth - baseline

        y_corr = y_corr - np.min(y_corr)

        y_corr = y_corr / np.max(y_corr)

        # =================================================
        # COMPACT PREPROCESSING FIGURE
        # =================================================

        fig_pre, (ax1, ax2) = plt.subplots(

            2,

            1,

            figsize=(8,5),

            sharex=True,

            gridspec_kw={
                'height_ratios': [1, 1]
            }
        )

        ax1.plot(

            x,

            y,

            color='gray',

            lw=1.2,

            label='Raw'
        )

        ax1.plot(

            x,

            y_smooth,

            color='orange',

            lw=1.8,

            label='Smoothed'
        )

        ax1.plot(

            x,

            baseline,

            color='green',

            lw=2,

            label='Baseline'
        )

        ax1.legend(
            fontsize=8
        )

        ax1.grid(alpha=0.2)

        ax1.set_ylabel(
            "Intensity",
            fontsize=9
        )

        ax2.plot(

            x,

            y_corr,

            color='red',

            lw=2
        )

        ax2.set_xlabel(
            "Raman shift (cm⁻¹)",
            fontsize=10
        )

        ax2.set_ylabel(
            "Normalized",
            fontsize=9
        )

        ax2.grid(alpha=0.2)

        plt.tight_layout()

        st.pyplot(fig_pre)

        # =================================================
        # AUTOMATIC PEAK DETECTION
        # =================================================

        st.subheader(
            "🧠 Automatic Raman Analysis"
        )

        peak_idx, props = find_peaks(

            y_corr,

            prominence=0.03,

            width=3,

            distance=8
        )

        peak_positions = x[peak_idx]

        peak_heights = y_corr[peak_idx]

        # =================================================
        # PEAK IDENTIFICATION
        # =================================================

        identified_peaks = []

        for peak, height in zip(

            peak_positions,

            peak_heights
        ):

            matched = False

            for known_peak, assignment in BLOOD_RAMAN_DATABASE.items():

                if abs(peak - known_peak) <= 8:

                    identified_peaks.append({

                        "Detected Peak":

                            round(peak,1),

                        "Reference":

                            known_peak,

                        "Assignment":

                            assignment,

                        "Intensity":

                            round(height,4)
                    })

                    matched = True

            if not matched:

                identified_peaks.append({

                    "Detected Peak":

                        round(peak,1),

                    "Reference":

                        "-",

                    "Assignment":

                        "Unknown",

                    "Intensity":

                        round(height,4)
                })

        identified_df = pd.DataFrame(
            identified_peaks
        )

        # =================================================
        # BUILD MODEL
        # =================================================

        model = None
        params = None

        for i, peak in enumerate(peak_positions):

            prefix = f"p{i}_"

            if model_type == "Lorentzian":

                peak_model = LorentzianModel(
                    prefix=prefix
                )

            elif model_type == "Gaussian":

                peak_model = GaussianModel(
                    prefix=prefix
                )

            else:

                peak_model = PseudoVoigtModel(
                    prefix=prefix
                )

            if model is None:

                model = peak_model

            else:

                model += peak_model

            pars = peak_model.make_params()

            pars[prefix+'center'].set(

                value=peak,

                min=peak-8,

                max=peak+8
            )

            pars[prefix+'sigma'].set(

                value=6,

                min=1,

                max=30
            )

            pars[prefix+'amplitude'].set(

                value=peak_heights[i],

                min=0
            )

            if model_type == "Pseudo-Voigt":

                pars[prefix+'fraction'].set(

                    value=0.5,

                    min=0,

                    max=1
                )

            if params is None:

                params = pars

            else:

                params.update(pars)

        # =================================================
        # FIT
        # =================================================

        result = model.fit(

            y_corr,

            params,

            x=x
        )

        components = result.eval_components(
            x=x
        )

        # =================================================
        # RESIDUAL
        # =================================================

        residual = y_corr - result.best_fit

        rmse = np.sqrt(
            np.mean(residual**2)
        )

        # =================================================
        # PUBLICATION-GRADE FIT FIGURE
        # =================================================

        fig_fit, ax_fit = plt.subplots(
            figsize=(10,6)
        )

        ax_fit.plot(

            x,

            y_corr,

            color='gray',

            lw=2.5,

            label='Experimental'
        )

        ax_fit.plot(

            x,

            result.best_fit,

            '--',

            color='red',

            lw=2.5,

            label='Global Fit'
        )

        colors = plt.cm.Set2.colors

        for i, peak in enumerate(peak_positions):

            prefix = f"p{i}_"

            comp = components[prefix]

            ax_fit.fill_between(

                x,

                0,

                comp,

                alpha=0.5,

                color=colors[
                    i % len(colors)
                ]
            )

            ax_fit.plot(

                x,

                comp,

                lw=1.3,

                color=colors[
                    i % len(colors)
                ]
            )

        # =================================================
        # LABELS
        # =================================================

        for _, row in identified_df.iterrows():

            peak = row["Detected Peak"]

            label = row["Assignment"]

            idx = np.argmin(
                np.abs(x - peak)
            )

            ypos = y_corr[idx]

            ax_fit.annotate(

                f"{peak:.0f}\n{label}",

                xy=(peak, ypos),

                xytext=(

                    peak + 5,

                    ypos + 0.05
                ),

                fontsize=9,

                fontweight='bold',

                arrowprops=dict(
                    arrowstyle='->',
                    lw=1
                )
            )

        ax_fit.set_xlabel(
            "Raman shift (cm⁻¹)",
            fontsize=14
        )

        ax_fit.set_ylabel(
            "Normalized Intensity",
            fontsize=14
        )

        ax_fit.legend(
            fontsize=10
        )

        ax_fit.grid(alpha=0.2)

        plt.tight_layout()

        st.pyplot(fig_fit)

        # =================================================
        # RESIDUAL
        # =================================================

        st.subheader(
            "📉 Residual"
        )

        fig_res, ax_res = plt.subplots(
            figsize=(10,2.5)
        )

        ax_res.plot(

            x,

            residual,

            color='black'
        )

        ax_res.axhline(

            0,

            linestyle='--',

            color='red'
        )

        ax_res.grid(alpha=0.2)

        ax_res.set_xlabel(
            "Raman shift (cm⁻¹)"
        )

        ax_res.set_ylabel(
            "Residual"
        )

        plt.tight_layout()

        st.pyplot(fig_res)

        # =================================================
        # RMSE
        # =================================================

        st.metric(
            "RMSE",
            f"{rmse:.6f}"
        )

        # =================================================
        # IDENTIFIED PEAKS
        # =================================================

        st.subheader(
            "🧬 Automatic Molecular Identification"
        )

        st.dataframe(

            identified_df,

            width='stretch'
        )
