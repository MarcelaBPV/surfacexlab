# =========================================================
# Raman Mapping — SurfaceXLab (VERSÃO FINAL)
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

    df = df.apply(lambda col: pd.to_numeric(col, errors="coerce"))

    return df.dropna()


# =========================================================
# EXTRAÇÃO DOS ESPECTROS
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
# GRID DE ESPECTROS
# =========================================================
def plot_grid_spectra(spectra):

    n = len(spectra)
    cols = 3
    rows = int(np.ceil(n / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(10, rows*3), dpi=300)

    axes = axes.flatten()

    for i, spec in enumerate(spectra):

        ax = axes[i]

        ax.plot(spec["wave"], spec["intensity"], color="black", linewidth=1)

        ax.invert_xaxis()
        ax.set_title(f"Y = {spec['y']} µm")

    for j in range(i+1, len(axes)):
        axes[j].axis("off")

    plt.tight_layout()

    return fig


# =========================================================
# ESPECTROS INDIVIDUAIS (ESTILO ARTIGO)
# =========================================================
def plot_single_spectrum(spec):

    fig, ax = plt.subplots(figsize=(5,3), dpi=300)

    ax.plot(spec["wave"], spec["intensity"], color="black", linewidth=1.2)

    # marca picos simples
    from scipy.signal import find_peaks
    peaks, _ = find_peaks(spec["intensity"], distance=20)

    ax.scatter(
        spec["wave"][peaks],
        spec["intensity"][peaks],
        color="red",
        s=18
    )

    ax.invert_xaxis()

    ax.set_xlabel("Raman Shift (cm⁻¹)")
    ax.set_ylabel("Intensity (a.u.)")

    ax.set_title(f"Espectro (Y = {spec['y']} µm)")

    return fig


# =========================================================
# HEATMAP (MAPA RAMAN)
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

    # 🔥 UPLOAD INDEPENDENTE (CORRIGIDO)
    uploaded_file = st.file_uploader(
        "📂 Upload do arquivo de mapeamento Raman",
        type=["txt", "csv"],
        key="raman_mapping_upload"
    )

    if uploaded_file is None:
        st.info("Faça upload do arquivo para visualizar os espectros e o mapa Raman.")
        return

    # =====================================================
    # LEITURA
    # =====================================================
    try:
        df = read_mapping(uploaded_file)
    except Exception as e:
        st.error("Erro ao ler o arquivo")
        st.exception(e)
        return

    st.success("Arquivo carregado com sucesso")

    st.write("Pré-visualização:")
    st.dataframe(df.head())

    # =====================================================
    # EXTRAÇÃO
    # =====================================================
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

    # =====================================================
    # GRID
    # =====================================================
    with tabs[0]:
        st.pyplot(plot_grid_spectra(spectra))

    # =====================================================
    # INDIVIDUAIS
    # =====================================================
    with tabs[1]:
        for spec in spectra:
            st.pyplot(plot_single_spectrum(spec))

    # =====================================================
    # HEATMAP
    # =====================================================
    with tabs[2]:
        st.pyplot(plot_heatmap(df))
