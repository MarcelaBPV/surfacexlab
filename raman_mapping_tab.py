# =========================================================
# raman_mapping.py
# SurfaceXLab — Molecular Mapping Module
# =========================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

from scipy.signal import savgol_filter, find_peaks
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


# =========================================================
# BASELINE ASLS
# =========================================================
def baseline_asls(y, lam=1e5, p=0.001, niter=10):

    L = len(y)

    D = diags([1, -2, 1], [0, -1, -2], shape=(L, L-2))

    w = np.ones(L)

    for _ in range(niter):

        W = diags(w, 0)

        Z = W + lam * D.dot(D.transpose())

        z = spsolve(Z, w * y)

        w = p * (y > z) + (1 - p) * (y < z)

    return z


# =========================================================
# LEITOR MAPEAMENTO
# =========================================================
def load_mapping_file(file):

    df = pd.read_csv(
        file,
        sep=r"\s+",
        engine="python"
    )

    df.columns = [
        c.strip().lower()
        for c in df.columns
    ]

    return df


# =========================================================
# ORGANIZA ESPECTROS
# =========================================================
def organize_spectra(df):

    grouped = {}

    coords = df[["x", "y"]].drop_duplicates()

    for _, row in coords.iterrows():

        x = row["x"]
        y = row["y"]

        sub = df[
            (df["x"] == x) &
            (df["y"] == y)
        ]

        wave = sub["wave"].values
        intensity = sub["intensity"].values

        grouped[(x, y)] = {
            "wave": wave,
            "intensity": intensity
        }

    return grouped


# =========================================================
# PROCESSAMENTO ESPECTRAL
# =========================================================
def process_spectrum(wave, intensity):

    # suavização
    smooth = savgol_filter(
        intensity,
        window_length=21,
        polyorder=3
    )

    # baseline
    baseline = baseline_asls(smooth)

    corrected = smooth - baseline

    # normalização
    corrected = (
        corrected - corrected.min()
    ) / (
        corrected.max() - corrected.min()
    )

    return corrected, baseline


# =========================================================
# DETECÇÃO PICOS
# =========================================================
def detect_peaks(wave, intensity):

    peaks, props = find_peaks(
        intensity,
        prominence=0.08,
        distance=15
    )

    return peaks


# =========================================================
# PLOT ESPECTROS
# =========================================================
def plot_mapping_spectra(grouped):

    fig, ax = plt.subplots(
        figsize=(8, 5)
    )

    selected = list(grouped.items())[:9]

    for idx, ((x, y), data) in enumerate(selected):

        wave = data["wave"]
        intensity = data["processed"]

        peaks = data["peaks"]

        ax.plot(
            wave,
            intensity + idx * 1.2,
            linewidth=1.5
        )

        ax.scatter(
            wave[peaks],
            intensity[peaks] + idx * 1.2,
            color="red",
            s=15
        )

    ax.set_xlabel("Raman Shift (cm⁻¹)")
    ax.set_ylabel("Normalized Intensity")

    ax.set_title(
        "Raman Spectra Along Blood Drop Diameter"
    )

    return fig


# =========================================================
# MAPA RAMAN
# =========================================================
def build_intensity_map(grouped):

    positions = []
    intensities = []

    for (x, y), data in grouped.items():

        positions.append(y)

        intensities.append(
            np.sum(data["processed"])
        )

    positions = np.array(positions)
    intensities = np.array(intensities)

    idx = np.argsort(positions)

    positions = positions[idx]
    intensities = intensities[idx]

    matrix = np.tile(
        intensities,
        (20, 1)
    )

    fig, ax = plt.subplots(
        figsize=(7, 4)
    )

    im = ax.imshow(
        matrix,
        aspect="auto",
        origin="lower"
    )

    ax.set_title("Raman Intensity Map")

    ax.set_xlabel("Mapping Position")
    ax.set_ylabel("Y Position")

    plt.colorbar(im)

    return fig


# =========================================================
# PCA
# =========================================================
def run_mapping_pca(grouped):

    spectra = []
    labels = []

    for (x, y), data in grouped.items():

        spectra.append(data["processed"])

        labels.append(f"Y={int(y)}")

    X = np.array(spectra)

    X = StandardScaler().fit_transform(X)

    pca = PCA(n_components=2)

    scores = pca.fit_transform(X)

    fig, ax = plt.subplots(
        figsize=(6, 5)
    )

    ax.scatter(
        scores[:, 0],
        scores[:, 1]
    )

    for i, label in enumerate(labels):

        ax.text(
            scores[i, 0],
            scores[i, 1],
            label,
            fontsize=8
        )

    ax.set_title(
        "PCA — Molecular Fingerprint"
    )

    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")

    return fig


# =========================================================
# STREAMLIT TAB
# =========================================================
def render_mapping_tab():

    st.subheader(
        "🗺️ Molecular Raman Mapping"
    )

    file = st.file_uploader(
        "Upload mapping file",
        type=["txt", "csv"]
    )

    if file is None:
        return

    df = load_mapping_file(file)

    grouped = organize_spectra(df)

    # processamento
    for key in grouped:

        wave = grouped[key]["wave"]
        intensity = grouped[key]["intensity"]

        processed, baseline = process_spectrum(
            wave,
            intensity
        )

        peaks = detect_peaks(
            wave,
            processed
        )

        grouped[key]["processed"] = processed
        grouped[key]["baseline"] = baseline
        grouped[key]["peaks"] = peaks

    # =====================================================
    # FIGURA 1 — ESPECTROS
    # =====================================================
    st.markdown("### 📐 Raman Spectra")

    fig1 = plot_mapping_spectra(grouped)

    st.pyplot(fig1)

    # =====================================================
    # FIGURA 2 — MAPA
    # =====================================================
    st.markdown("### 🗺️ Raman Intensity Map")

    fig2 = build_intensity_map(grouped)

    st.pyplot(fig2)

    # =====================================================
    # FIGURA 3 — PCA
    # =====================================================
    st.markdown("### 📊 PCA")

    fig3 = run_mapping_pca(grouped)

    st.pyplot(fig3)
