# =========================================================
# SurfaceXLab — Spectral Deconvolution Module
# Raman Mapping + Guided Deconvolution + Origin Validation
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
# RAMAN PEAK DATABASE
# center, amplitude, sigma
# =========================================================

RAMAN_PEAKS = [

    (1536, 2000, 8),

    (1556, 4500, 12),

    (1577, 5500, 15),

    (1600, 3800, 9),

    (1617, 2400, 7),

    (1626, 3000, 10),

    (1663, 1800, 18)
]

# =========================================================
# BASELINE ASLS
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

        z = spsolve(Z, w * y)

        w = p * (y > z) + (1 - p) * (y < z)

    return z

# =========================================================
# MAIN FUNCTION
# =========================================================

def render_spectral_deconvolution_tab():

    st.title("🧪 Spectral Deconvolution")

    st.markdown("""
    Workflow espectroscópico completo para:

    - Raman Mapping
    - Seleção individual de espectros
    - Baseline correction
    - Noise removal
    - Guided Raman fitting
    - Residual analysis
    - Validation against Origin
    """)

    st.divider()

    # =====================================================
    # CONFIG
    # =====================================================

    st.subheader("⚙ Configurações Gerais")

    col1, col2, col3 = st.columns(3)

    with col1:

        model_type = st.selectbox(
            "Modelo de fitting",
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

        st.info("Faça upload do arquivo Raman.")

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

        st.error(f"Erro leitura arquivo: {e}")

        return

    st.success("Arquivo carregado.")

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
            "Arquivo precisa possuir colunas x e y."
        )

        return

    # =====================================================
    # COLUMN SELECTION
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

        st.header("🔬 Raman Mapping Viewer")

        st.success(
            f"{len(coords)} espectros detectados."
        )

        n_spectra = st.slider(

            "Quantidade de espectros",

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

            fig_map.delaxes(axes[j])

        plt.tight_layout()

        st.pyplot(fig_map)

    # =====================================================
    # TAB 2 — PROCESSING
    # =====================================================

    with tab2:

        st.header("⚙ Spectral Processing")

        # =================================================
        # SELECT SPECTRUM
        # =================================================

        st.subheader("🎯 Select Spectrum")

        coord_options = [

            f"X={x} | Y={y}"

            for x,y in coords
        ]

        selected_coord = st.selectbox(

            "Escolha o espectro para análise",

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
            f"Espectro selecionado: X={x_sel} | Y={y_sel}"
        )

        # =================================================
        # EXTRACT SPECTRUM
        # =================================================

        x = selected_spectrum[x_col].values
        y = selected_spectrum[y_col].values

        order = np.argsort(x)

        x = x[order]
        y = y[order]

        # =================================================
        # REGION
        # =================================================

        st.subheader("🔍 Região Espectral")

        col1, col2 = st.columns(2)

        with col1:

            x_min = st.number_input(
                "Min Raman",
                value=1500.0
            )

        with col2:

            x_max = st.number_input(
                "Max Raman",
                value=1700.0
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

        st.subheader("📊 Preprocessing")

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

        # corrected spectrum
        y_corr = y_smooth - baseline

        # publication-grade normalization
        y_corr = y_corr - np.min(y_corr)
        y_corr = y_corr / np.max(y_corr)

        # =================================================
        # PREPROCESSING PLOT
        # =================================================

        fig_pre, (ax1, ax2) = plt.subplots(

            2,

            1,

            figsize=(10,8),

            sharex=True
        )

        # ==========================================
        # RAW + BASELINE
        # ==========================================

        ax1.plot(

            x,

            y,

            color='gray',

            lw=1.5,

            label='Raw'
        )

        ax1.plot(

            x,

            y_smooth,

            color='orange',

            lw=2,

            label='Smoothed'
        )

        ax1.plot(

            x,

            baseline,

            color='green',

            lw=2,

            label='Baseline'
        )

        ax1.set_ylabel("Intensity")

        ax1.legend()

        ax1.grid(alpha=0.2)

        # ==========================================
        # CORRECTED SPECTRUM
        # ==========================================

        ax2.plot(

            x,

            y_corr,

            color='red',

            lw=2
        )

        ax2.set_xlabel(
            "Raman shift (cm⁻¹)"
        )

        ax2.set_ylabel(
            "Corrected"
        )

        ax2.grid(alpha=0.2)

        plt.tight_layout()

        st.pyplot(fig_pre)

        # =================================================
        # RAMAN DECONVOLUTION
        # =================================================

        st.subheader("🧠 Raman Deconvolution")

        model = None
        params = None

        for i, (center, amp_guess, sigma_guess) in enumerate(RAMAN_PEAKS):

            prefix = f"p{i}_"

            # =============================================
            # MODEL TYPE
            # =============================================

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

            # =============================================
            # BUILD MODEL
            # =============================================

            if model is None:

                model = peak_model

            else:

                model += peak_model

            # =============================================
            # PARAMS
            # =============================================

            pars = peak_model.make_params()

            pars[prefix+'center'].set(

                value=center,

                min=center-10,

                max=center+10
            )

            pars[prefix+'sigma'].set(

                value=sigma_guess,

                min=2,

                max=40
            )

            pars[prefix+'amplitude'].set(

                value=amp_guess,

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
        # FIT PLOT
        # =================================================

        fig_fit, ax_fit = plt.subplots(
            figsize=(12,7)
        )

        # experimental
        ax_fit.plot(

            x,

            y_corr,

            color='gray',

            lw=2.5,

            label='Experimental'
        )

        # global fit
        ax_fit.plot(

            x,

            result.best_fit,

            '--',

            color='red',

            lw=2,

            label='Global Fit'
        )

        # colors
        component_colors = [

            '#ef2929',

            '#204a87',

            '#edd400',

            '#4e9a06',

            '#75507b',

            '#888a85',

            '#f4a3a3'
        ]

        # components
        for i, (center_ref, _, _) in enumerate(RAMAN_PEAKS):

            prefix = f"p{i}_"

            comp = components[prefix]

            ax_fit.fill_between(

                x,

                0,

                comp,

                alpha=0.45,

                color=component_colors[
                    i % len(component_colors)
                ]
            )

            ax_fit.plot(

                x,

                comp,

                lw=1,

                color=component_colors[
                    i % len(component_colors)
                ]
            )

            fitted_center = result.params[
                prefix+'center'
            ].value

            idx_peak = np.argmin(
                np.abs(x - fitted_center)
            )

            ypos = comp[idx_peak]

            ax_fit.annotate(

                f"{fitted_center:.0f}\n({RAMAN_BANDS[int(center_ref)]})",

                xy=(fitted_center, ypos),

                xytext=(

                    fitted_center + 8,

                    ypos + np.max(y_corr)*0.05
                ),

                fontsize=10,

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
            "Intensity",
            fontsize=14
        )

        ax_fit.legend()

        ax_fit.grid(alpha=0.15)

        plt.tight_layout()

        st.pyplot(fig_fit)

        # =================================================
        # RESIDUAL PLOT
        # =================================================

        st.subheader("📉 Residual")

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
        # PEAK TABLE
        # =================================================

        st.subheader("📋 Peak Parameters")

        peak_results = []

        for i, (center_ref, _, _) in enumerate(RAMAN_PEAKS):

            prefix = f"p{i}_"

            peak_results.append({

                "Peak":

                    RAMAN_BANDS[
                        int(center_ref)
                    ],

                "Center (cm⁻¹)":

                    result.params[
                        prefix+'center'
                    ].value,

                "FWHM":

                    result.params[
                        prefix+'fwhm'
                    ].value,

                "Amplitude":

                    result.params[
                        prefix+'amplitude'
                    ].value
            })

        peak_df = pd.DataFrame(
            peak_results
        )

        st.dataframe(
            peak_df,
            width='stretch'
        )

        # save session
        st.session_state[
            "result_best_fit"
        ] = result.best_fit

        st.session_state[
            "x_processed"
        ] = x

    # =====================================================
    # TAB 3 — VALIDATION
    # =====================================================

    with tab3:

        st.header("📉 Origin Validation")

        if "result_best_fit" not in st.session_state:

            st.warning(
                "Execute o fitting primeiro."
            )

        else:

            origin_file = st.file_uploader(

                "Upload Origin Export",

                type=["csv","txt"]
            )

            if origin_file is not None:

                try:

                    origin_df = pd.read_csv(
                        origin_file
                    )

                    x_proc = st.session_state[
                        "x_processed"
                    ]

                    fit_proc = st.session_state[
                        "result_best_fit"
                    ]

                    origin_x = origin_df.iloc[:,0].values
                    origin_y = origin_df.iloc[:,1].values

                    origin_interp = np.interp(

                        x_proc,

                        origin_x,

                        origin_y
                    )

                    origin_interp = (
                        origin_interp /
                        np.max(origin_interp)
                    )

                    # metrics
                    rmse_origin = np.sqrt(

                        np.mean(
                            (
                                origin_interp -
                                fit_proc
                            )**2
                        )
                    )

                    ss_res = np.sum(

                        (
                            origin_interp -
                            fit_proc
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

                    pearson = np.corrcoef(

                        origin_interp,

                        fit_proc

                    )[0,1]

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

                    # overlay
                    fig_cmp, ax_cmp = plt.subplots(
                        figsize=(12,6)
                    )

                    ax_cmp.plot(

                        x_proc,

                        origin_interp,

                        color='black',

                        lw=2,

                        label='Origin Manual'
                    )

                    ax_cmp.plot(

                        x_proc,

                        fit_proc,

                        '--',

                        color='red',

                        lw=2,

                        label='SurfaceXLab'
                    )

                    ax_cmp.legend()

                    ax_cmp.grid(alpha=0.2)

                    st.pyplot(fig_cmp)

                except Exception as e:

                    st.error(
                        f"Erro validação Origin: {e}"
                    )
