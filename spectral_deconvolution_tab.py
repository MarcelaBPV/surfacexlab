# =========================================================
# SurfaceXLab — Spectral Deconvolution Module
# Raman Mapping + Multi-Peak Fitting
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
    Ajuste espectral multi-pico automatizado utilizando:
    
    - Correção de linha de base
    - Suavização espectral
    - Detecção automática de picos
    - Fitting Lorentziano
    - Fitting Pseudo-Voigt
    - Extração de FWHM
    - Raman Mapping
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
    # UPLOAD
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

        st.info("Faça upload do arquivo Raman.")

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

    st.write(df.head())

    # =====================================================
    # STANDARDIZE COLUMN NAMES
    # =====================================================

    df.columns = [
        str(c).strip()
        for c in df.columns
    ]

    # =====================================================
    # RAMAN MAPPING
    # =====================================================

    if all(col in df.columns for col in ['x', 'y']):

        st.divider()

        st.header("🧬 Raman Mapping — 18 Spectra")

        st.success("Mapeamento Raman detectado.")

        # ================================================
        # GROUP SPECTRA
        # ================================================

        grouped = df.groupby(['x', 'y'])

        coords = list(grouped.groups.keys())

        st.write(
            f"Total de espectros encontrados: {len(coords)}"
        )

        n_spectra = st.slider(

            "Quantidade de espectros",

            min_value=1,

            max_value=min(36, len(coords)),

            value=min(18, len(coords))
        )

        coords = coords[:n_spectra]

        # ================================================
        # GRID
        # ================================================

        ncols = 3
        nrows = int(np.ceil(n_spectra / ncols))

        fig, axes = plt.subplots(

            nrows=nrows,

            ncols=ncols,

            figsize=(15, 4 * nrows)
        )

        axes = np.array(axes).flatten()

        # ================================================
        # LOOP
        # ================================================

        for idx, (x_pos, y_pos) in enumerate(coords):

            sub = grouped.get_group(
                (x_pos, y_pos)
            )

            # ============================================
            # COLUMN DETECTION
            # ============================================

            wave_col = None
            inten_col = None

            for c in df.columns:

                if 'wave' in c.lower():

                    wave_col = c

                if 'intensity' in c.lower():

                    inten_col = c

            if wave_col is None:
                wave_col = df.columns[2]

            if inten_col is None:
                inten_col = df.columns[3]

            wave = sub[wave_col].values
            inten = sub[inten_col].values

            # ordenar
            order = np.argsort(wave)

            wave = wave[order]
            inten = inten[order]

            # smoothing
            inten_smooth = savgol_filter(
                inten,
                smooth_window,
                polyorder
            )

            # baseline
            baseline = baseline_asls(
                inten_smooth
            )

            inten_corr = (
                inten_smooth - baseline
            )

            # normalize
            max_val = np.max(inten_corr)

            if max_val != 0:

                inten_corr = (
                    inten_corr / max_val
                )

            ax = axes[idx]

            ax.plot(

                wave,

                inten_corr,

                color='black',

                lw=1.2
            )

            ax.set_title(

                f"Y = {y_pos:.0f} µm",

                fontsize=11
            )

            ax.set_xlabel(
                "Raman shift (cm⁻¹)"
            )

            ax.set_ylabel(
                "Intensity"
            )

            ax.grid(alpha=0.2)

            ax.invert_xaxis()

        # ================================================
        # REMOVE EMPTY AXES
        # ================================================

        for j in range(idx + 1, len(axes)):

            fig.delaxes(axes[j])

        plt.tight_layout()

        st.pyplot(fig)

        # ================================================
        # SAVE
        # ================================================

        fig.savefig(

            "raman_mapping_18_spectra.png",

            dpi=600,

            bbox_inches='tight'
        )

        st.success(
            "18 espectros Raman plotados."
        )

    # =====================================================
    # SINGLE SPECTRUM
    # =====================================================

    st.divider()

    st.header("📈 Single Spectrum Analysis")

    cols = df.columns.tolist()

    col1, col2 = st.columns(2)

    with col1:

        x_col = st.selectbox(
            "Eixo Raman",
            cols,
            index=min(2, len(cols)-1)
        )

    with col2:

        y_col = st.selectbox(
            "Intensidade",
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
            "Raman Min",
            value=float(np.min(x))
        )

    with col2:

        x_max = st.number_input(
            "Raman Max",
            value=float(np.max(x))
        )

    mask = (x >= x_min) & (x <= x_max)

    x = x[mask]
    y = y[mask]

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

    max_val = np.max(y_corr)

    if max_val != 0:

        y_corr = y_corr / max_val

    # =====================================================
    # PREPROCESS PLOT
    # =====================================================

    st.subheader("📊 Pré-processamento")

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.plot(
        x,
        y,
        label="Raw",
        alpha=0.5
    )

    ax.plot(
        x,
        y_smooth,
        label="Smoothed"
    )

    ax.plot(
        x,
        baseline,
        label="Baseline"
    )

    ax.plot(
        x,
        y_corr,
        label="Corrected"
    )

    ax.legend()

    ax.set_xlabel("Raman shift")

    ax.set_ylabel("Intensity")

    ax.invert_xaxis()

    st.pyplot(fig)

    # =====================================================
    # PEAK DETECTION
    # =====================================================

    st.subheader("📍 Peak Detection")

    peaks, props = find_peaks(

        y_corr,

        prominence=0.03,

        distance=20,

        width=5
    )

    peak_x = x[peaks]
    peak_y = y_corr[peaks]

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.plot(x, y_corr)

    ax.scatter(
        peak_x,
        peak_y,
        color='red'
    )

    ax.invert_xaxis()

    st.pyplot(fig)

    # =====================================================
    # FITTING
    # =====================================================

    st.subheader("🧠 Multi-Peak Fitting")

    model = None
    params = None

    for i, center in enumerate(peak_x):

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

        pars[prefix + 'center'].set(

            value=center,

            min=center - 10,

            max=center + 10
        )

        pars[prefix + 'sigma'].set(

            value=8,

            min=1,

            max=50
        )

        pars[prefix + 'amplitude'].set(

            value=0.5,

            min=0
        )

        if params is None:

            params = pars

        else:

            params.update(pars)

    result = model.fit(

        y_corr,

        params,

        x=x
    )

    components = result.eval_components(x=x)

    # =====================================================
    # FIT PLOT
    # =====================================================

    st.subheader("📉 Publication-Grade Fitting")

    fig, ax = plt.subplots(figsize=(12, 7))

    ax.plot(

        x,

        y_corr,

        color='black',

        lw=1.5,

        label='Experimental'
    )

    ax.plot(

        x,

        result.best_fit,

        '--',

        color='red',

        lw=2,

        label='Global Fit'
    )

    cmap = plt.cm.tab10.colors

    for i in range(len(peak_x)):

        prefix = f"p{i}_"

        comp = components[prefix]

        ax.plot(

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

        ax.annotate(

            f"{fitted_center:.0f}",

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

    ax.set_xlabel(
        "Raman shift (cm⁻¹)"
    )

    ax.set_ylabel(
        "Normalized Intensity"
    )

    ax.legend(loc='upper right')

    ax.grid(alpha=0.2)

    ax.invert_xaxis()

    st.pyplot(fig)

    # =====================================================
    # PEAK TABLE
    # =====================================================

    st.subheader("📋 Peak Parameters")

    peak_results = []

    for i in range(len(peak_x)):

        prefix = f"p{i}_"

        peak_results.append({

            "Peak": i + 1,

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
