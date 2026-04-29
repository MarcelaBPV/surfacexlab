# =========================================================
# Raman Mapping — Subaba
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
# SEPARAR ESPECTROS
# =========================================================
def extract_spectra(df):

    spectra = {}

    grouped = df.groupby(["x", "y"])

    for (x, y), g in grouped:
        spectra[(x, y)] = g.sort_values("wave")

    return spectra


# =========================================================
# PLOT DOS 18 ESPECTROS
# =========================================================
def plot_all_spectra(spectra):

    fig, ax = plt.subplots(figsize=(7,5), dpi=300)

    for (x, y), s in spectra.items():
        ax.plot(s["wave"], s["intensity"], alpha=0.4)

    ax.set_xlabel("Número de onda (cm⁻¹)")
    ax.set_ylabel("Intensidade")
    ax.set_title("Espectros Raman — Mapeamento")

    return fig


# =========================================================
# MAPA DE INTENSIDADE
# =========================================================
def build_intensity_map(spectra):

    coords = []
    values = []

    for (x, y), s in spectra.items():

        # intensidade média ou pode trocar por pico específico
        intensity = s["intensity"].mean()

        coords.append((x, y))
        values.append(intensity)

    coords = np.array(coords)
    values = np.array(values)

    x_unique = np.unique(coords[:,0])
    y_unique = np.unique(coords[:,1])

    grid = np.zeros((len(y_unique), len(x_unique)))

    for (x, y), val in zip(coords, values):
        xi = np.where(x_unique == x)[0][0]
        yi = np.where(y_unique == y)[0][0]
        grid[yi, xi] = val

    return grid, x_unique, y_unique


# =========================================================
# PLOT HEATMAP
# =========================================================
def plot_heatmap(grid):

    fig, ax = plt.subplots(figsize=(5,5), dpi=300)

    im = ax.imshow(grid, origin="lower", aspect="auto")

    ax.set_title("Mapa de Intensidade Raman")

    plt.colorbar(im)

    return fig


# =========================================================
# TAB PRINCIPAL
# =========================================================
def render_mapeamento_molecular_tab():

    st.subheader("🧬 Raman Mapping — Mapeamento Molecular")

    file = st.file_uploader(
        "Upload arquivo de mapeamento (.txt)",
        type=["txt"]
    )

    if not file:
        return

    df = read_mapping(file)

    st.success("Arquivo carregado")

    spectra = extract_spectra(df)

    st.write(f"Total de espectros: {len(spectra)}")

    # =====================================================
    # PLOT ESPECTROS
    # =====================================================
    st.markdown("### 📈 Espectros Raman")

    fig1 = plot_all_spectra(spectra)
    st.pyplot(fig1)

    # =====================================================
    # HEATMAP
    # =====================================================
    st.markdown("### 🌈 Mapa de Intensidade")

    grid, _, _ = build_intensity_map(spectra)

    fig2 = plot_heatmap(grid)
    st.pyplot(fig2)
