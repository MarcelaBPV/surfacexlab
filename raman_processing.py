# =========================================================
# Raman Mapping — SurfaceXLab (VERSÃO FINAL FUNCIONAL)
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

    if df.shape[1] < 4:
        raise ValueError("Arquivo inválido — precisa de 4 colunas")

    df = df.iloc[:, :4]
    df.columns = ["y", "x", "wave", "intensity"]

    # garante numérico
    df = df.apply(pd.to_numeric, errors="coerce")

    return df.dropna()


# =========================================================
# ORGANIZAÇÃO DOS ESPECTROS
# =========================================================
def extract_spectra(df):

    spectra = []

    for (y_val, x_val), g in df.groupby(["y", "x"]):

        g = g.sort_values("wave")

        spectra.append({
            "y": float(y_val),
            "x": float(x_val),
            "wave": g["wave"].values,
            "intensity": g["intensity"].values
        })

    # ordena por Y → padrão do artigo
    spectra = sorted(spectra, key=lambda s: s["y"])

    return spectra


# =========================================================
# GRID 18 ESPECTROS (PADRÃO ARTIGO)
# =========================================================
def plot_grid_spectra(spectra):

    fig, axes = plt.subplots(6, 3, figsize=(12, 14), dpi=300)
    axes = axes.flatten()

    for i, spec in enumerate(spectra[:18]):

        ax = axes[i]

        ax.plot(spec["wave"], spec["intensity"], color="black", linewidth=1)

        # padrão Raman
        ax.invert_xaxis()

        ax.set_title(f"Y = {int(spec['y'])} µm", fontsize=8)

    # limpa espaços vazios
    for j in range(len(spectra), len(axes)):
        axes[j].axis("off")

    plt.tight_layout()

    return fig


# =========================================================
# MAPA DE INTENSIDADE
# =========================================================
def plot_intensity_map(df):

    pivot = df.pivot_table(
        values="intensity",
        index="y",
        columns="x",
        aggfunc="mean"
    )

    fig, ax = plt.subplots(figsize=(6, 5), dpi=300)

    im = ax.imshow(
        pivot.values,
        origin="lower",
        aspect="auto",
        cmap="inferno"
    )

    ax.set_title("Raman Intensity Map")
    ax.set_xlabel("X position")
    ax.set_ylabel("Y position")

    plt.colorbar(im, ax=ax)

    return fig


# =========================================================
# FUNÇÃO PRINCIPAL
# =========================================================
def render_mapeamento_molecular_tab(supabase=None):

    st.subheader("🗺️ Raman Mapping — Análise Espacial")

    # =====================================================
    # UPLOAD
    # =====================================================
    uploaded_file = st.file_uploader(
        "📂 Upload do arquivo Raman Mapping",
        type=["txt", "csv"],
        key="raman_mapping_upload"
    )

    if uploaded_file is None:
        st.info("Faça upload do arquivo para visualizar os espectros.")
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
    # BOTÃO (CONTROLE CIENTÍFICO)
    # =====================================================
    if not st.button("🚀 Gerar Espectros e Mapa"):
        st.warning("Clique no botão para processar os dados")
        return

    # =====================================================
    # EXTRAÇÃO
    # =====================================================
    spectra = extract_spectra(df)

    st.success(f"{len(spectra)} espectros identificados")

    # =====================================================
    # PLOT — 18 ESPECTROS
    # =====================================================
    st.markdown("### 📊 Espectros Raman (18 posições)")

    fig1 = plot_grid_spectra(spectra)
    st.pyplot(fig1)

    # =====================================================
    # MAPA
    # =====================================================
    st.markdown("### 🔥 Mapa de Intensidade Raman")

    fig2 = plot_intensity_map(df)
    st.pyplot(fig2)
