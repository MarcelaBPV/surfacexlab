# =========================================================
# raman_mapping.py
# SurfaceXLab — Raman Molecular Mapping
# =========================================================

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.signal import savgol_filter, find_peaks
from scipy import sparse
from scipy.sparse.linalg import spsolve


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
# MAIN TAB
# =========================================================
def render_raman_mapping_tab():

    st.subheader("🗺️ Raman Molecular Mapping")

    st.markdown("""
    Plataforma para reconstrução espacial de espectros Raman
    adquiridos em múltiplas posições da amostra.

    Funcionalidades:
    - correção automática de linha de base;
    - suavização espectral;
    - normalização;
    - identificação automática de picos;
    - reconstrução espacial molecular.
    """)

    # =====================================================
    # FILE UPLOAD
    # =====================================================
    uploaded_file = st.file_uploader(

        "Upload Raman Mapping",

        type=["txt", "csv", "xlsx"]
    )

    if uploaded_file is None:

        st.info(
            "Faça upload do arquivo Raman Mapping."
        )

        return

    # =====================================================
    # READ FILE
    # =====================================================
    try:

        # ================================================
        # TXT / CSV
        # ================================================
        if uploaded_file.name.endswith(

            (".txt", ".csv")

        ):

            df = pd.read_csv(

                uploaded_file,

                sep=r"\s+",

                engine="python",

                header=0
            )

        # ================================================
        # XLSX
        # ================================================
        else:

            df = pd.read_excel(
                uploaded_file
            )

    except Exception as e:

        st.error(
            "Erro ao carregar arquivo."
        )

        st.exception(e)

        return

    # =====================================================
    # ORGANIZE COLUMNS
    # =====================================================
    df.columns = [

        str(c).strip().lower()

        for c in df.columns
    ]

    # =====================================================
    # COLUMN FIX
    # =====================================================
    rename_map = {

        "wave": "wave",

        "wavenumber": "wave",

        "ramanshift": "wave",

        "intensity": "intensity",

        "x": "x",

        "y": "y"
    }

    df.rename(

        columns=rename_map,

        inplace=True
    )

    # =====================================================
    # CHECK COLUMNS
    # =====================================================
    required = [

        "x",
        "y",
        "wave",
        "intensity"
    ]

    for c in required:

        if c not in df.columns:

            st.error(
                f"Coluna obrigatória ausente: {c}"
            )

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
    # GROUP SPECTRA
    # =====================================================
    grouped = list(

        df.groupby(["x", "y"])
    )

    st.success(

        f"{len(grouped)} espectros detectados."
    )

    # =====================================================
    # SPECTRA FIGURE
    # =====================================================
    fig, ax = plt.subplots(

        figsize=(12,7),

        dpi=300
    )

    heatmap = []

    features = []

    # =====================================================
    # PROCESS EACH SPECTRUM
    # =====================================================
    for idx, ((x, y), group) in enumerate(grouped):

        group = group.sort_values(
            "wave"
        )

        wave = group["wave"].values

        intensity = group[
            "intensity"
        ].values

        # =================================================
        # BASELINE CORRECTION
        # =================================================
        baseline = baseline_als(
            intensity
        )

        corrected = intensity - baseline

        # =================================================
        # SMOOTHING
        # =================================================
        if len(corrected) > 21:

            smooth = savgol_filter(

                corrected,

                21,

                3
            )

        else:

            smooth = corrected

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

            prominence=0.05
        )

        # =================================================
        # MAIN PEAK
        # =================================================
        if len(peaks) > 0:

            peak_idx = peaks[
                np.argmax(norm[peaks])
            ]

            peak_wave = wave[
                peak_idx
            ]

            peak_intensity = norm[
                peak_idx
            ]

        else:

            peak_wave = 0
            peak_intensity = 0

        # =================================================
        # FEATURES
        # =================================================
        features.append({

            "Spectrum": idx + 1,

            "X": x,

            "Y": y,

            "Main Peak": peak_wave,

            "Max Intensity": peak_intensity
        })

        heatmap.append([

            x,
            y,
            peak_intensity
        ])

        # =================================================
        # PLOT
        # =================================================
        ax.plot(

            wave,

            norm,

            linewidth=1
        )

        ax.scatter(

            wave[peaks],

            norm[peaks],

            s=10
        )

    # =====================================================
    # FINAL SPECTRA PLOT
    # =====================================================
    ax.set_title(
        "Raman Spectral Mapping"
    )

    ax.set_xlabel(
        "Raman Shift (cm⁻¹)"
    )

    ax.set_ylabel(
        "Normalized Intensity"
    )

    ax.grid(alpha=0.2)

    st.pyplot(fig)

    # =====================================================
    # FEATURES TABLE
    # =====================================================
    st.subheader(
        "📊 Spectral Features"
    )

    features_df = pd.DataFrame(
        features
    )

    st.dataframe(
        features_df
    )

    # =====================================================
    # HEATMAP
    # =====================================================
    st.subheader(
        "🔥 Raman Molecular Map"
    )

    heat_df = pd.DataFrame(

        heatmap,

        columns=[
            "X",
            "Y",
            "Intensity"
        ]
    )

    # =====================================================
    # PIVOT
    # =====================================================
    pivot = heat_df.pivot(

        index="Y",

        columns="X",

        values="Intensity"
    )

    # =====================================================
    # HEATMAP FIGURE
    # =====================================================
    fig2, ax2 = plt.subplots(

        figsize=(8,6),

        dpi=300
    )

    im = ax2.imshow(

        pivot,

        cmap="inferno",

        origin="lower",

        aspect="auto"
    )

    cbar = plt.colorbar(

        im,

        ax=ax2
    )

    cbar.set_label(
        "Relative Raman Intensity"
    )

    ax2.set_title(
        "Spatial Molecular Distribution"
    )

    ax2.set_xlabel(
        "X Position"
    )

    ax2.set_ylabel(
        "Y Position"
    )

    st.pyplot(fig2)
