# =========================================================
# RAMAN MAPPING — VERSÃO CIENTÍFICA
# =========================================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# =========================================================
# LEITURA DO ARQUIVO
# =========================================================
def read_mapping(file):

    df = pd.read_csv(file, sep=r"\s+|\t+", engine="python")

    df.columns = ["y", "x", "wave", "intensity"]

    df = df.apply(pd.to_numeric, errors="coerce")

    return df.dropna()


# =========================================================
# ORGANIZA ESPECTROS POR POSIÇÃO Y
# =========================================================
def extract_spectra(df):

    spectra = []

    for y_val, group in df.groupby("y"):

        group = group.sort_values("wave")

        spectra.append({
            "y": y_val,
            "wave": group["wave"].values,
            "intensity": group["intensity"].values
        })

    return spectra


# =========================================================
# PLOT INDIVIDUAL (PADRÃO PAPER)
# =========================================================
def plot_single_spectrum(spec):

    fig, ax = plt.subplots(figsize=(4.5, 3.2), dpi=300)

    ax.plot(spec["wave"], spec["intensity"], color="black", linewidth=1.2)

    # picos simples (máximos locais)
    from scipy.signal import find_peaks
    peaks, _ = find_peaks(spec["intensity"], distance=20)

    ax.scatter(
        spec["wave"][peaks],
        spec["intensity"][peaks],
        color="red",
        s=15
    )

    ax.invert_xaxis()

    ax.set_xlabel("Raman Shift (cm⁻¹)")
    ax.set_ylabel("Intensity (a.u.)")

    ax.set_title(f"Y = {spec['y']} µm")

    return fig


# =========================================================
# PLOT GRID (18 ESPECTROS)
# =========================================================
def plot_grid_spectra(spectra):

    n = len(spectra)
    cols = 3
    rows = int(np.ceil(n / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(10, rows*3), dpi=300)

    axes = axes.flatten()

    for i, spec in enumerate(spectra):

        ax = axes[i]

        ax.plot(spec["wave"], spec["intensity"], color="black")

        ax.invert_xaxis()
        ax.set_title(f"Y={spec['y']}")

    # remove eixos vazios
    for j in range(i+1, len(axes)):
        axes[j].axis("off")

    plt.tight_layout()

    return fig


# =========================================================
# HEATMAP REAL (PADRÃO PAPER)
# =========================================================
def plot_heatmap(df):

    pivot = df.pivot_table(
        index="y",
        columns="wave",
        values="intensity",
        aggfunc="mean"
    )

    pivot = pivot.sort_index()
    pivot = pivot.sort_index(axis=1, ascending=False)

    fig, ax = plt.subplots(figsize=(7,4), dpi=300)

    im = ax.imshow(
        pivot.values,
        aspect="auto",
        cmap="inferno",
        origin="lower",
        extent=[
            pivot.columns.min(),
            pivot.columns.max(),
            pivot.index.min(),
            pivot.index.max()
        ]
    )

    ax.set_title("Raman intensity map")
    ax.set_xlabel("Raman shift (cm⁻¹)")
    ax.set_ylabel("Y position")

    cbar = fig.colorbar(im)
    cbar.set_label("Intensity")

    return fig


# =========================================================
# TAB PRINCIPAL
# =========================================================
def render_mapeamento_molecular_tab(supabase=None):

    st.subheader("🗺️ Raman Mapping — Análise Espacial")

    file = st.file_uploader(
        "Upload arquivo de mapeamento Raman",
        type=["txt", "csv"]
    )

    if not file:
        st.info("Faça upload do arquivo de mapeamento.")
        return

    df = read_mapping(file)

    spectra = extract_spectra(df)

    st.success(f"{len(spectra)} espectros identificados")

    # =====================================================
    # SUBABAS INTERNAS
    # =====================================================
    tabs = st.tabs([
        "📊 Grid de espectros",
        "📈 Espectros individuais",
        "🔥 Mapa Raman"
    ])

    # GRID
    with tabs[0]:
        st.pyplot(plot_grid_spectra(spectra))

    # INDIVIDUAL
    with tabs[1]:
        for spec in spectra:
            st.pyplot(plot_single_spectrum(spec))

    # HEATMAP
    with tabs[2]:
        st.pyplot(plot_heatmap(df))
