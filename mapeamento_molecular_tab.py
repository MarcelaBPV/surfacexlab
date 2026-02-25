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
            df = pd.read_csv(uploaded_file, delim_whitespace=True)

    df.columns = [
        c.replace("#", "").strip().lower()
        for c in df.columns
    ]

    df = df[["y", "x", "wave", "intensity"]]
    df = df.apply(pd.to_numeric, errors="coerce")
    df = df.dropna()

    if df.empty:
        raise ValueError("Arquivo sem dados Raman válidos.")

    return df


# =========================================================
# TAB PRINCIPAL
# =========================================================
def render_mapeamento_molecular_tab(supabase):

    st.header("🗺️ Mapeamento Molecular Raman")

    uploaded_file = st.file_uploader(
        "Upload arquivo Raman Mapping",
        type=["txt", "csv", "xls", "xlsx"]
    )

    if not uploaded_file:
        return

    try:

        df = read_mapping_file(uploaded_file)
        grouped = df.groupby(["y", "x"])
        spectra_list = []

        # =========================
        # PROCESSAMENTO
        # =========================
        for (y_val, x_val), group in grouped:

            group = group.sort_values("wave")

            x = group["wave"].values
            y = group["intensity"].values

            baseline = asls_baseline(y)
            y_corr = y - baseline

            peak_idx, _ = find_peaks(
                y_corr,
                prominence=np.max(y_corr)*0.05
            )

            peaks = []

            for idx in peak_idx:
                cen = x[idx]
                peaks.append((cen, classify_raman_group(cen)))

            spectra_list.append({
                "y": y_val,
                "wave": x,
                "intensity": y_corr,
                "peaks": peaks
            })

        if not spectra_list:
            st.warning("Nenhum espectro válido encontrado.")
            return


        # =====================================================
        # MINIATURAS DOS ESPECTROS
        # =====================================================
        st.subheader("Galeria de Espectros Raman")

        if "selected_spec" not in st.session_state:
            st.session_state.selected_spec = 0

        cols = st.columns(4)

        for i, spec in enumerate(spectra_list):

            with cols[i % 4]:

                fig_small, ax_small = plt.subplots(figsize=(3,2))

                ax_small.plot(
                    spec["wave"],
                    spec["intensity"],
                    'k-', lw=1
                )

                ax_small.set_xticks([])
                ax_small.set_yticks([])
                ax_small.set_title(f"Y={spec['y']:.0f}")

                st.pyplot(fig_small)

                if st.button("Abrir", key=f"btn_{i}"):
                    st.session_state.selected_spec = i


        # =====================================================
        # ESPECTRO GRANDE
        # =====================================================
        st.divider()
        st.subheader("Espectro Selecionado")

        spec = spectra_list[st.session_state.selected_spec]

        fig_big, ax_big = plt.subplots(figsize=(10,5), dpi=300)

        ax_big.plot(
            spec["wave"],
            spec["intensity"],
            'k-', lw=1.4
        )

        for cen, group_name in spec["peaks"]:

            ax_big.axvline(cen, linestyle="--", alpha=0.6)

            ax_big.text(
                cen,
                max(spec["intensity"])*0.9,
                f"{cen:.0f}\n{group_name}",
                rotation=90,
                fontsize=8,
                ha="center"
            )

        ax_big.set_xlabel("Raman Shift (cm⁻¹)")
        ax_big.set_ylabel("Intensity (a.u.)")
        ax_big.set_title(f"Ponto Y={spec['y']:.0f} µm")

        ax_big.invert_xaxis()
        ax_big.grid(alpha=0.3)

        st.pyplot(fig_big)


        # =====================================================
        # TODOS ESPECTROS SOBREPOSTOS
        # =====================================================
        st.subheader("Comparação Global dos Espectros")

        fig_all, ax_all = plt.subplots(figsize=(8,4), dpi=300)

        for spec in spectra_list:
            ax_all.plot(spec["wave"], spec["intensity"], alpha=0.4)

        ax_all.set_xlabel("Raman Shift (cm⁻¹)")
        ax_all.set_ylabel("Intensity (a.u.)")
        ax_all.set_title("Todos os Espectros Raman")
        ax_all.invert_xaxis()
        ax_all.grid(alpha=0.3)

        st.pyplot(fig_all)

    except Exception as e:
        st.error(f"Erro ao processar arquivo: {e}")
