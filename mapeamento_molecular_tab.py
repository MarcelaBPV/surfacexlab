# mapeamento_molecular_tab.py
# -*- coding: utf-8 -*-

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.signal import find_peaks
from scipy import sparse
from scipy.sparse.linalg import spsolve


# =========================================================
# DATABASE RAMAN — BIOMOLÉCULAS DO SANGUE
# =========================================================
RAMAN_DATABASE = {
    (720, 730): "Adenine",
    (750, 760): "Tryptophan",
    (1000, 1006): "Phenylalanine",
    (1240, 1300): "Amide III",
    (1330, 1370): "Hemoglobin",
    (1440, 1470): "Lipids",
    (1540, 1580): "Amide II",
    (1640, 1680): "Amide I",
}


# =========================================================
# CLASSIFICAÇÃO
# =========================================================
def classify_raman_group(center):
    for (low, high), label in RAMAN_DATABASE.items():
        if low <= center <= high:
            return label
    return None


# =========================================================
# BASELINE ASLS
# =========================================================
def asls_baseline(y, lam=1e6, p=0.01, niter=10):

    if len(y) < 10:
        return np.zeros_like(y)

    y = np.asarray(y, dtype=float)
    N = len(y)

    D = sparse.diags([1, -2, 1], [0, 1, 2], shape=(N - 2, N))
    w = np.ones(N)

    for _ in range(niter):
        W = sparse.diags(w, 0)
        Z = W + lam * D.T @ D
        z = spsolve(Z, w * y)
        w = p * (y > z) + (1 - p) * (y < z)

    return z


# =========================================================
# LEITURA ROBUSTA
# =========================================================
def read_mapping_file(uploaded_file):

    name = uploaded_file.name.lower()

    # Excel continua normal
    if name.endswith((".xls", ".xlsx")):
        df = pd.read_excel(uploaded_file)

    else:
        # leitura MUITO robusta Raman txt
        uploaded_file.seek(0)

        df = pd.read_csv(
            uploaded_file,
            sep=r"\s+|\t+|,",   # aceita espaço, tab ou vírgula
            engine="python",
            comment=None,
            header=None,
            skip_blank_lines=True,
            encoding="latin1"
        )

        # se tiver 4 colunas assume formato Raman mapping
        if df.shape[1] >= 4:
            df = df.iloc[:, :4]
            df.columns = ["y", "x", "wave", "intensity"]

        else:
            raise ValueError(
                "Arquivo não possui 4 colunas esperadas: y, x, wave, intensity."
            )

    # converte tudo para número
    df = df.apply(pd.to_numeric, errors="coerce")
    df = df.dropna()

    if df.empty:
        raise ValueError("Arquivo sem dados numéricos válidos.")

    return df


# =========================================================
# PLOT ESTILO ARTIGO — PAINEL A
# =========================================================
def plot_single_spectrum(spec):

    fig, ax = plt.subplots(figsize=(6,4), dpi=300)

    ax.plot(
        spec["wave"],
        spec["intensity"],
        color="black",
        lw=1.4
    )

    for cen in spec["peak_positions"]:

        ax.axvline(cen, color="gray", lw=0.8, ls="--")

        ax.text(
            cen,
            max(spec["intensity"]) * 0.85,
            f"{cen:.0f}",
            rotation=90,
            fontsize=7,
            ha="center"
        )

    ax.set_xlabel("Número de onda / cm⁻¹")
    ax.set_ylabel("Intensidade Raman (a.u.)")
    ax.invert_xaxis()

    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_linewidth(1)

    return fig


# =========================================================
# PLOT ESTILO ARTIGO — PAINEL B
# =========================================================
def plot_multi_spectra(spectra_list):

    fig, ax = plt.subplots(figsize=(6,4), dpi=300)

    cmap = plt.cm.viridis

    for i, spec in enumerate(spectra_list):

        ax.plot(
            spec["wave"],
            spec["intensity"],
            color=cmap(i / len(spectra_list)),
            lw=1,
            alpha=0.9,
            label=f"Y={spec['y']:.0f}"
        )

    ax.set_xlabel("Número de onda / cm⁻¹")
    ax.set_ylabel("Intensidade Raman (a.u.)")
    ax.invert_xaxis()

    ax.legend(frameon=False, fontsize=7)
    ax.grid(False)

    for spine in ax.spines.values():
        spine.set_linewidth(1)

    return fig


# =========================================================
# TAB STREAMLIT
# =========================================================
def render_mapeamento_molecular_tab(supabase):

    st.header("🗺️ Mapeamento Molecular Raman")

    uploaded_file = st.file_uploader(
        "Upload Raman Mapping",
        type=["txt", "csv", "xls", "xlsx"]
    )

    if not uploaded_file:
        return

    try:

        df = read_mapping_file(uploaded_file)
        grouped = df.groupby(["y", "x"])

        spectra_list = []

        for (y_val, x_val), group in grouped:

            group = group.sort_values("wave")

            x = group["wave"].values
            y = group["intensity"].values

            baseline = asls_baseline(y)
            y_corr = y - baseline

            peak_idx, _ = find_peaks(
                y_corr,
                prominence=np.max(y_corr) * 0.08
            )

            peak_positions = []

            for idx in peak_idx:
                cen = x[idx]
                if classify_raman_group(cen):
                    peak_positions.append(cen)

            spectra_list.append({
                "y": y_val,
                "wave": x,
                "intensity": y_corr,
                "peak_positions": peak_positions
            })

        if not spectra_list:
            st.warning("Nenhum espectro encontrado.")
            return

        # =============================
        # SELEÇÃO DO ESPECTRO
        # =============================
        selected_index = st.slider(
            "Selecionar ponto do mapa",
            0,
            len(spectra_list)-1,
            0
        )

        spec = spectra_list[selected_index]

        st.subheader("Painel A — Espectro Individual")
        fig_single = plot_single_spectrum(spec)
        st.pyplot(fig_single)

        st.subheader("Painel B — Comparação Espacial")
        fig_multi = plot_multi_spectra(spectra_list)
        st.pyplot(fig_multi)

    except Exception as e:
        st.error(f"Erro ao processar arquivo: {e}")
