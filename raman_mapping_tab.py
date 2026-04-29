# =========================================================
# Raman Mapping — SurfaceXLab (FINAL COMPLETO)
# =========================================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# =========================================================
# LEITURA ROBUSTA
# =========================================================
def read_mapping(file):

    df = pd.read_csv(file, sep=r"\s+|\t+", engine="python")

    if len(df.columns) >= 4:
        df = df.iloc[:, :4]
        df.columns = ["y", "x", "wave", "intensity"]
    else:
        raise ValueError("Arquivo inválido — precisa de 4 colunas")

    df = df.apply(lambda col: pd.to_numeric(col, errors="coerce"))

    return df.dropna()


# =========================================================
# EXTRAÇÃO DOS ESPECTROS
# =========================================================
def extract_spectra(df):

    spectra = []

    for (x_val, y_val), group in df.groupby(["x", "y"]):

        group = group.sort_values("wave")

        spectra.append({
            "x": x_val,
            "y": y_val,
            "wave": group["wave"].values,
            "intensity": group["intensity"].values
        })

    return spectra


# =========================================================
# GRID DOS ESPECTROS
# =========================================================
def plot_grid_spectra(spectra):

    n = len(spectra)
    cols = 6
    rows = int(np.ceil(n / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(14, rows*2.5), dpi=300)
    axes = axes.flatten()

    for i, spec in enumerate(spectra):

        ax = axes[i]
        ax.plot(spec["wave"], spec["intensity"], linewidth=0.8)

        ax.set_title(f"x={spec['x']}, y={spec['y']}", fontsize=8)
        ax.invert_xaxis()

    for j in range(i+1, len(axes)):
        axes[j].axis("off")

    plt.tight_layout()

    return fig


# =========================================================
# ESPECTRO INDIVIDUAL COM PICOS
# =========================================================
def plot_single_spectrum(spec):

    from scipy.signal import find_peaks

    fig, ax = plt.subplots(figsize=(5,3), dpi=300)

    ax.plot(spec["wave"], spec["intensity"], linewidth=1.2)

    peaks, _ = find_peaks(spec["intensity"], distance=20)

    ax.scatter(
        spec["wave"][peaks],
        spec["intensity"][peaks],
        s=18
    )

    ax.invert_xaxis()
    ax.set_xlabel("Raman Shift (cm⁻¹)")
    ax.set_ylabel("Intensity")
    ax.set_title(f"x={spec['x']} | y={spec['y']}")

    return fig


# =========================================================
# MAPA ESPACIAL
# =========================================================
def plot_spatial_map(df):

    pivot = df.groupby(["y", "x"])["intensity"].mean().unstack()

    fig, ax = plt.subplots(figsize=(6,5), dpi=300)

    im = ax.imshow(
        pivot.values,
        origin="lower",
        aspect="auto",
        cmap="inferno"
    )

    ax.set_title("Mapa Raman (Intensidade média)")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")

    plt.colorbar(im, ax=ax)

    return fig


# =========================================================
# FUNÇÃO PRINCIPAL (COM CONTROLE DE PROCESSAMENTO)
# =========================================================
def render_mapeamento_molecular_tab(supabase=None):

    st.subheader("🗺️ Raman Mapping — Análise Espacial")

    # =====================================================
    # UPLOAD
    # =====================================================
    uploaded_file = st.file_uploader(
        "📂 Upload do arquivo de mapeamento Raman",
        type=["txt", "csv"],
        key="raman_mapping_upload"
    )

    if uploaded_file is None:
        st.info("Faça upload do arquivo para iniciar.")
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

    # =====================================================
    # PREVIEW
    # =====================================================
    st.markdown("### 🔍 Preview dos dados")
    st.dataframe(df.head())

    # =====================================================
    # BOTÃO DE PROCESSAMENTO
    # =====================================================
    processar = st.button("🚀 Processar Mapeamento")

    if not processar:
        st.warning("Clique no botão para gerar os espectros e o mapa.")
        return

    # =====================================================
    # EXTRAÇÃO
    # =====================================================
    spectra = extract_spectra(df)

    st.success(f"{len(spectra)} espectros detectados")

    # =====================================================
    # SUBABAS INTERNAS
    # =====================================================
    tabs = st.tabs([
        "📊 18 Espectros",
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
    # MAPA
    # =====================================================
    with tabs[2]:
        st.pyplot(plot_spatial_map(df))
