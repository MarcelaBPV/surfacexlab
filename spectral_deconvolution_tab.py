# =========================================================
# SurfaceXLab — Spectral Deconvolution Module
# Raman Mapping + Origin Validation
# =========================================================

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.signal import savgol_filter
from scipy import sparse
from scipy.sparse.linalg import spsolve

from lmfit.models import (
    LorentzianModel,
    GaussianModel,
    PseudoVoigtModel
)

# =========================================================
# KNOWN RAMAN BANDS
# =========================================================

RAMAN_BANDS = {

    1536: "ν11",

    1556: "ν19",

    1577: "ν37 CaCm",

    1600: "ν C=C",

    1617: "ν CaCβ",

    1626: "ν10 CaCm",

    1663: "Amide I"
}

# =========================================================
# BASELINE ASLS
# =========================================================

def baseline_asls(
    y,
    lam=1e6,
    p=0.001,
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

        z = spsolve(Z, w * y)

        w = p * (y > z) + (1 - p) * (y < z)

    return z

# =========================================================
# MAIN TAB
# =========================================================

def render_spectral_deconvolution_tab():

    st.title("🧪 Spectral Deconvolution")

    st.markdown("""
    Módulo avançado de deconvolução espectral Raman
    com validação quantitativa contra Origin.
    """)

    st.divider()

    # =====================================================
    # CONFIG
    # =====================================================

    st.subheader("⚙ Configurações")

    col1, col2, col3 = st.columns(3)

    with col1:

        model_type = st.selectbox(
            "Modelo",
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
            "Polyorder",
            2,
            5,
            3
        )

    st.divider()

    # =====================================================
    # UPLOAD RAMAN
    # =====================================================

    uploaded_file = st.file_uploader(

        "Upload Raman File",

        type=[
            "txt",
            "csv",
            "xlsx"
        ]
    )

    if uploaded_file is None:

        st.info("Faça upload do espectro Raman.")

        return

    # =====================================================
    # LOAD FILE
    # =====================================================

    try:

        if uploaded_file.name.endswith(".xlsx"):

            df = pd.read_excel(uploaded_file)

        else:

            df = pd.read_csv(
                uploaded_file,
                sep=r"\s+|,|;",
                engine="python"
            )

    except Exception as e:

        st.error(f"Erro leitura arquivo: {e}")

        return

    st.success("Arquivo carregado.")

    df.columns = [
        str(c).strip()
        for c in df.columns
    ]

    st.write(df.head())

    # =====================================================
    # SELECT COLUMNS
    # =====================================================

    cols = df.columns.tolist()

    col1, col2 = st.columns(2)

    with col1:

        x_col = st.selectbox(
            "Raman Shift",
            cols,
            index=min(2, len(cols)-1)
        )

    with col2:

        y_col = st.selectbox(
            "Intensity",
            cols,
            index=min(3, len(cols)-1)
        )

    x = df[x_col].values
    y = df[y_col].values

    # =====================================================
    # REGION
    # =====================================================

    st.subheader("🔍 Região Espectral")

    col1, col2 = st.columns(2)

    with col1:

        x_min = st.number_input(
            "Min",
            value=1500.0
        )

    with col2:

        x_max = st.number_input(
            "Max",
            value=1700.0
        )

    mask = (x >= x_min) & (x <= x_max)

    x = x[mask]
    y = y[mask]

    # =====================================================
    # SORT
    # =====================================================

    order = np.argsort(x)

    x = x[order]
    y = y[order]

    # =====================================================
    # PREPROCESS
    # =====================================================

    y_smooth = savgol_filter(
        y,
        smooth_window,
        polyorder
    )

    baseline = baseline_asls(
        y_smooth
    )

    y_corr = y_smooth - baseline

    y_corr = (
        y_corr /
        np.max(y_corr)
    )

    # =====================================================
    # PREPROCESS PLOT
    # =====================================================

    st.subheader("📊 Pré-processamento")

    fig_pre, ax_pre = plt.subplots(
        figsize=(10,5)
    )

    ax_pre.plot(
        x,
        y,
        label='Raw',
        alpha=0.5
    )

    ax_pre.plot(
        x,
        y_smooth,
        label='Smoothed'
    )

    ax_pre.plot(
        x,
        baseline,
        label='Baseline'
    )

    ax_pre.plot(
        x,
        y_corr,
        label='Corrected'
    )

    ax_pre.legend()

    ax_pre.invert_xaxis()

    ax_pre.grid(alpha=0.2)

    st.pyplot(fig_pre)

    # =====================================================
    # GUIDED FITTING
    # =====================================================

    st.subheader("🧠 Guided Raman Deconvolution")

    expected_peaks = list(
        RAMAN_BANDS.keys()
    )

    model = None
    params = None

    for i, center in enumerate(expected_peaks):

        prefix = f"p{i}_"

        # ================================================
        # MODEL TYPE
        # ================================================

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

        # ================================================
        # BUILD MODEL
        # ================================================

        if model is None:

            model = peak_model

        else:

            model += peak_model

        # ================================================
        # PARAMETERS
        # ================================================

        pars = peak_model.make_params()

        pars[prefix + 'center'].set(

            value=center,

            min=center - 8,

            max=center + 8
        )

        pars[prefix + 'sigma'].set(

            value=10,

            min=1,

            max=40
        )

        pars[prefix + 'amplitude'].set(

            value=0.5,

            min=0
        )

        if model_type == "Pseudo-Voigt":

            pars[prefix + 'fraction'].set(

                value=0.5,

                min=0,

                max=1
            )

        if params is None:

            params = pars

        else:

            params.update(pars)

    # =====================================================
    # FIT
    # =====================================================

    result = model.fit(

        y_corr,

        params,

        x=x
    )

    components = result.eval_components(
        x=x
    )

    # =====================================================
    # RESIDUAL
    # =====================================================

    residual = y_corr - result.best_fit

    rmse = np.sqrt(
        np.mean(residual**2)
    )

    # =====================================================
    # FIT PLOT
    # =====================================================

    st.subheader("📉 Publication-Grade Fitting")

    fig_fit, ax_fit = plt.subplots(
        figsize=(12,7)
    )

    ax_fit.plot(

        x,

        y_corr,

        color='black',

        lw=1.5,

        label='Experimental'
    )

    ax_fit.plot(

        x,

        result.best_fit,

        '--',

        color='red',

        lw=2,

        label='Global Fit'
    )

    cmap = plt.cm.tab10.colors

    for i, center_ref in enumerate(expected_peaks):

        prefix = f"p{i}_"

        comp = components[prefix]

        ax_fit.plot(

            x,

            comp,

            color=cmap[i % len(cmap)],

            lw=1.2
        )

        fitted_center = result.params[
            prefix + 'center'
        ].value

        idx_peak = np.argmin(
            np.abs(x - fitted_center)
        )

        ypos = comp[idx_peak]

        ax_fit.annotate(

            f"{fitted_center:.0f}\n({RAMAN_BANDS[center_ref]})",

            xy=(fitted_center, ypos),

            xytext=(
                fitted_center + 5,
                ypos + 0.05
            ),

            fontsize=8,

            arrowprops=dict(
                arrowstyle='->'
            )
        )

    ax_fit.set_xlabel(
        "Raman shift (cm⁻¹)"
    )

    ax_fit.set_ylabel(
        "Normalized Intensity"
    )

    ax_fit.legend()

    ax_fit.grid(alpha=0.2)

    ax_fit.invert_xaxis()

    st.pyplot(fig_fit)

    # =====================================================
    # RESIDUAL PLOT
    # =====================================================

    st.subheader("📉 Residual Analysis")

    fig_res, ax_res = plt.subplots(
        figsize=(12,3)
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

    ax_res.set_xlabel(
        "Raman shift (cm⁻¹)"
    )

    ax_res.set_ylabel(
        "Residual"
    )

    ax_res.invert_xaxis()

    ax_res.grid(alpha=0.2)

    st.pyplot(fig_res)

    # =====================================================
    # RMSE
    # =====================================================

    st.metric(
        "RMSE",
        f"{rmse:.6f}"
    )

    # =====================================================
    # PEAK TABLE
    # =====================================================

    st.subheader("📋 Peak Parameters")

    peak_results = []

    for i, center_ref in enumerate(expected_peaks):

        prefix = f"p{i}_"

        peak_results.append({

            "Peak":

                RAMAN_BANDS[
                    center_ref
                ],

            "Center (cm⁻¹)":

                result.params[
                    prefix + 'center'
                ].value,

            "FWHM":

                result.params[
                    prefix + 'fwhm'
                ].value,

            "Amplitude":

                result.params[
                    prefix + 'amplitude'
                ].value
        })

    peak_df = pd.DataFrame(
        peak_results
    )

    st.dataframe(
        peak_df,
        width='stretch'
    )

    # =====================================================
    # ORIGIN VALIDATION
    # =====================================================

    st.divider()

    st.header(
        "🔬 Origin vs SurfaceXLab"
    )

    origin_file = st.file_uploader(

        "Upload Origin Export",

        type=["csv", "txt"]
    )

    if origin_file is not None:

        try:

            origin_df = pd.read_csv(
                origin_file
            )

            st.write(origin_df.head())

            origin_x = origin_df.iloc[:,0].values
            origin_y = origin_df.iloc[:,1].values

            # ============================================
            # INTERPOLATION
            # ============================================

            origin_interp = np.interp(

                x,

                origin_x,

                origin_y
            )

            # normalize
            origin_interp = (
                origin_interp /
                np.max(origin_interp)
            )

            # ============================================
            # RMSE
            # ============================================

            rmse_origin = np.sqrt(

                np.mean(
                    (
                        origin_interp -
                        result.best_fit
                    )**2
                )
            )

            # ============================================
            # R²
            # ============================================

            ss_res = np.sum(

                (
                    origin_interp -
                    result.best_fit
                )**2
            )

            ss_tot = np.sum(

                (
                    origin_interp -
                    np.mean(origin_interp)
                )**2
            )

            r2 = 1 - (
                ss_res / ss_tot
            )

            # ============================================
            # PEARSON
            # ============================================

            pearson = np.corrcoef(

                origin_interp,

                result.best_fit

            )[0,1]

            # ============================================
            # METRICS
            # ============================================

            col1, col2, col3 = st.columns(3)

            col1.metric(
                "RMSE",
                f"{rmse_origin:.6f}"
            )

            col2.metric(
                "R²",
                f"{r2:.5f}"
            )

            col3.metric(
                "Pearson",
                f"{pearson:.5f}"
            )

            # ============================================
            # OVERLAY
            # ============================================

            fig_cmp, ax_cmp = plt.subplots(
                figsize=(12,6)
            )

            ax_cmp.plot(

                x,

                origin_interp,

                color='black',

                lw=2,

                label='Origin Manual'
            )

            ax_cmp.plot(

                x,

                result.best_fit,

                '--',

                color='red',

                lw=2,

                label='SurfaceXLab'
            )

            ax_cmp.set_xlabel(
                "Raman shift (cm⁻¹)"
            )

            ax_cmp.set_ylabel(
                "Normalized Intensity"
            )

            ax_cmp.legend()

            ax_cmp.invert_xaxis()

            ax_cmp.grid(alpha=0.2)

            st.pyplot(fig_cmp)

        except Exception as e:

            st.error(
                f"Erro validação Origin: {e}"
            )

    # =====================================================
    # DOWNLOAD
    # =====================================================

    csv = peak_df.to_csv(
        index=False
    ).encode()

    st.download_button(

        "⬇ Download Peak Table",

        data=csv,

        file_name="peak_results.csv",

        mime="text/csv"
    )

    st.success(
        "Deconvolução concluída."
    )
