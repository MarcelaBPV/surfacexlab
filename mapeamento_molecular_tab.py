# mapeamento_molecular_tab.py
# -*- coding: utf-8 -*-

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.signal import savgol_filter, find_peaks
from scipy.optimize import curve_fit
from scipy import sparse
from scipy.sparse.linalg import spsolve


# =========================================================
# DATABASE RAMAN QUÍMICA
# =========================================================
RAMAN_DATABASE = {

    (650, 680): "C–S Proteins",
    (720, 730): "Adenine",
    (750, 760): "Tryptophan",
    (820, 850): "Tyrosine",
    (930, 950): "Protein Backbone",
    (1000, 1006): "Phenylalanine",
    (1240, 1300): "Amide III",
    (1440, 1470): "Lipids CH2",
    (1540, 1580): "Amide II",
    (1640, 1680): "Amide I",

    (1320, 1360): "Carbon D Band",
    (1570, 1605): "Carbon G Band",
    (2650, 2720): "Graphene 2D",
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
# MODELO LORENTZIANO
# =========================================================
def lorentz(x, amp, cen, wid, off):
    return amp * ((0.5 * wid)**2 /
                 ((x - cen)**2 + (0.5 * wid)**2)) + off


def fit_lorentz(x, y, center, window=20):

    mask = (x > center-window/2) & (x < center+window/2)

    if mask.sum() < 8:
        return None

    xs = x[mask]
    ys = y[mask]

    p0 = [
        np.max(ys) - np.min(ys),
        center,
        10,
        np.min(ys)
    ]

    try:
        popt, _ = curve_fit(lorentz, xs, ys, p0=p0, maxfev=8000)

        amp, cen, wid, off = popt

        return {
            "center": float(cen),
            "amplitude": float(amp),
            "fwhm": float(2 * wid)
        }

    except Exception:
        return None


# =========================================================
# LEITURA ROBUSTA DE ARQUIVO RAMAN MAPPING
# =========================================================
def read_mapping_file(uploaded_file):

    name = uploaded_file.name.lower()

    # Excel
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
                delim_whitespace=True,
                low_memory=False
            )

    # padroniza colunas
    df.columns = [
        c.replace("#", "").strip().lower()
        for c in df.columns
    ]

    # garante colunas corretas
    df = df[["y", "x", "wave", "intensity"]]

    # numérico
    df = df.apply(pd.to_numeric, errors="coerce")

    df = df.dropna()

    if df.empty:
        raise ValueError("Arquivo sem dados Raman válidos.")

    return df


# =========================================================
# PROCESSAMENTO DO MAPA RAMAN
# =========================================================
def process_mapping(df):

    grouped = df.groupby(["y", "x"])

    all_peaks = []
    chemical_maps = {}

    for (y_val, x_val), group in grouped:

        group = group.sort_values("wave")

        x = group["wave"].values
        y = group["intensity"].values

        if len(y) < 10:
            continue

        # BASELINE
        baseline = asls_baseline(y)
        y_corr = y - baseline

        # SUAVIZAÇÃO
        y_smooth = savgol_filter(y_corr, 11, 3)

        # NORMALIZAÇÃO
        norm = np.max(np.abs(y_smooth))
        y_norm = y_smooth / norm if norm > 0 else y_smooth

        # DETECÇÃO DE PICOS
        peak_idx, _ = find_peaks(
            y_norm,
            prominence=0.02,
            width=5
        )

        for idx in peak_idx:

            cen = x[idx]
            fit = fit_lorentz(x, y_norm, cen)

            if not fit:
                continue

            group_name = classify_raman_group(fit["center"])

            all_peaks.append({
                "x": x_val,
                "y": y_val,
                "peak_cm1": fit["center"],
                "amplitude": fit["amplitude"],
                "fwhm": fit["fwhm"],
                "chemical_group": group_name
            })

            chemical_maps.setdefault(group_name, []).append(
                (x_val, y_val, fit["amplitude"])
            )

    peaks_df = pd.DataFrame(all_peaks)

    return peaks_df, chemical_maps


# =========================================================
# HEATMAP QUÍMICO
# =========================================================
def plot_maps(chemical_maps):

    figs = {}

    for group, values in chemical_maps.items():

        df = pd.DataFrame(values, columns=["x", "y", "amp"])

        pivot = df.pivot_table(
            index="y",
            columns="x",
            values="amp",
            aggfunc="mean"
        )

        fig, ax = plt.subplots(figsize=(6,5), dpi=300)

        im = ax.imshow(
            pivot.values,
            origin="lower",
            aspect="auto"
        )

        ax.set_title(group)
        ax.set_xlabel("X (µm)")
        ax.set_ylabel("Y (µm)")

        plt.colorbar(im, ax=ax, label="Intensity")

        figs[group] = fig

    return figs


# =========================================================
# STREAMLIT TAB
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

        peaks_df, chemical_maps = process_mapping(df)

        st.subheader("Tabela de Picos Raman")
        st.dataframe(peaks_df)

        st.subheader("Mapas Químicos")

        figs = plot_maps(chemical_maps)

        for fig in figs.values():
            st.pyplot(fig)

    except Exception as e:
        st.error(f"Erro ao processar arquivo: {e}")
