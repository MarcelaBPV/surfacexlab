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
# DATABASE RAMAN — FOCO BIOMOLÉCULAS SANGUE
# =========================================================
RAMAN_DATABASE = {

    (720, 730): "Adenine (DNA/RNA)",
    (750, 760): "Tryptophan (Proteins)",
    (1000, 1006): "Phenylalanine",
    (1240, 1300): "Amide III (Proteins)",
    (1330, 1370): "Hemoglobin",
    (1440, 1470): "Lipids",
    (1540, 1580): "Amide II",
    (1640, 1680): "Amide I",
}


# =========================================================
# CLASSIFICAÇÃO QUÍMICA
# =========================================================
def classify_raman_group(center):

    for (low, high), label in RAMAN_DATABASE.items():
        if low <= center <= high:
            return label

    return "Unassigned"


# =========================================================
# BASELINE ASLS (ROBUSTO)
# =========================================================
def asls_baseline(y, lam=1e6, p=0.01, niter=10):

    if len(y) < 10:
        return np.zeros_like(y)

    y = np.asarray(y, dtype=float)
    N = len(y)

    D = sparse.diags([1, -2, 1], [0, 1, 2], shape=(N-2, N))
    w = np.ones(N)

    for _ in range(niter):
        W = sparse.diags(w, 0)
        Z = W + lam * D.T @ D
        z = spsolve(Z, w*y)
        w = p*(y > z) + (1-p)*(y < z)

    return z


# =========================================================
# LEITURA ROBUSTA ARQUIVO RAMAN MAPPING
# =========================================================
def read_mapping_file(uploaded_file):

    name = uploaded_file.name.lower()

    if name.endswith((".xls", ".xlsx")):
        df = pd.read_excel(uploaded_file)

    else:
        try:
            df = pd.read_csv(
                uploaded_file,
                sep=None,
                engine="python",
                comment="#",
                skip_blank_lines=True,
                low_memory=False
            )
        except Exception:
            uploaded_file.seek(0)
            df = pd.read_csv(
                uploaded_file,
                delim_whitespace=True
            )

    df.columns = [
        c.replace("#", "").strip().lower()
        for c in df.columns
    ]

    df = df[["y","x","wave","intensity"]]
    df = df.apply(pd.to_numeric, errors="coerce")
    df = df.dropna()

    if df.empty:
        raise ValueError("Arquivo sem dados Raman válidos.")

    return df


# =========================================================
# STREAMLIT TAB PRINCIPAL
# =========================================================
def render_mapeamento_molecular_tab(supabase):

    st.header("🗺️ Mapeamento Molecular Raman")

    uploaded_file = st.file_uploader(
        "Upload arquivo Raman Mapping",
        type=["txt","csv","xls","xlsx"]
    )

    if not uploaded_file:
        return

    try:

        df = read_mapping_file(uploaded_file)

        grouped = df.groupby(["y","x"])

        spectra_list = []

        # ==========================================
        # PROCESSAMENTO DOS ESPECTROS
        # ==========================================
        for (y_val, x_val), group in grouped:

            group = group.sort_values("wave")

            x = group["wave"].values
            y = group["intensity"].values

            baseline = asls_baseline(y)
            y_corr = y - baseline

            # DETECÇÃO DE PICOS
            peak_idx, _ = find_peaks(
                y_corr,
                prominence=np.max(y_corr)*0.05
            )

            peaks = []

            for idx in peak_idx:

                cen = x[idx]
                group_name = classify_raman_group(cen)

                peaks.append((cen, group_name))

            spectra_list.append({
                "y": y_val,
                "x": x_val,
                "wave": x,
                "intensity": y_corr,
                "peaks": peaks
            })

        if not spectra_list:
            st.warning("Nenhum espectro válido encontrado.")
            return

        # ==========================================
        # 1️⃣ CARROSSEL ESPECTROS INDIVIDUAIS
        # ==========================================
        st.subheader("Espectros Individuais")

        index = st.slider(
            "Navegar pelo mapa Raman",
            0,
            len(spectra_list)-1,
            0
        )

        spec = spectra_list[index]

        fig, ax = plt.subplots(figsize=(7,4), dpi=300)

        ax.plot(
            spec["wave"],
            spec["intensity"],
            'k-',
            lw=1.2
        )

        for cen, group_name in spec["peaks"]:

            ax.axvline(
                cen,
                linestyle="--",
                alpha=0.6
            )

            ax.text(
                cen,
                max(spec["intensity"])*0.9,
                f"{cen:.0f}\n{group_name}",
                rotation=90,
                fontsize=8,
                ha="center"
            )

        ax.set_title(f"Ponto Y={spec['y']:.0f} µm")
        ax.set_xlabel("Raman Shift (cm⁻¹)")
        ax.set_ylabel("Intensity (a.u.)")

        ax.invert_xaxis()
        ax.grid(alpha=0.3)

        st.pyplot(fig)

        # ==========================================
        # 2️⃣ TODOS ESPECTROS SOBREPOSTOS
        # ==========================================
        st.subheader("Comparação Global dos Espectros")

        fig_all, ax_all = plt.subplots(figsize=(7,4), dpi=300)

        for spec in spectra_list:
            ax_all.plot(
                spec["wave"],
                spec["intensity"],
                alpha=0.5
            )

        ax_all.set_xlabel("Raman Shift (cm⁻¹)")
        ax_all.set_ylabel("Intensity (a.u.)")
        ax_all.set_title("Todos os Espectros Raman")

        ax_all.invert_xaxis()
        ax_all.grid(alpha=0.3)

        st.pyplot(fig_all)

    except Exception as e:
        st.error(f"Erro ao processar arquivo: {e}")
