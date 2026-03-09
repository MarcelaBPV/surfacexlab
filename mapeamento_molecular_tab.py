# mapeamento_molecular_tab.py
# -*- coding: utf-8 -*-

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.signal import find_peaks, savgol_filter
from scipy import sparse
from scipy.sparse.linalg import spsolve


# =========================================================
# BASELINE ASLS
# =========================================================
def asls_baseline(y, lam=1e7, p=0.01, niter=15):

    y = np.asarray(y)

    N = len(y)

    D = sparse.diags([1,-2,1],[0,1,2], shape=(N-2,N))
    w = np.ones(N)

    for _ in range(niter):

        W = sparse.diags(w,0)
        Z = W + lam * D.T @ D

        z = spsolve(Z, w*y)

        w = p*(y>z)+(1-p)*(y<z)

    return z


# =========================================================
# SUAVIZAÇÃO
# =========================================================
def smooth_signal(y, window=11, poly=3):

    if window % 2 == 0:
        window += 1

    if len(y) < window:
        return y

    return savgol_filter(y, window, poly)


# =========================================================
# NORMALIZAÇÃO
# =========================================================
def normalize(y):

    max_val = np.max(np.abs(y))

    if max_val == 0:
        return y

    return y / max_val


# =========================================================
# LEITURA DO ARQUIVO
# =========================================================
def read_mapping_file(uploaded_file):

    uploaded_file.seek(0)

    df = pd.read_csv(
        uploaded_file,
        sep=r"\s+|\t+|;|,",
        engine="python",
        header=None,
        decimal=","
    )

    df = df.dropna()

    # formato mapping (y x wave intensity)

    if df.shape[1] >= 4:

        df = df.iloc[:,:4]
        df.columns = ["y","x","wave","intensity"]

    else:

        df.columns = ["wave","intensity"]
        df["y"] = 0
        df["x"] = 0

    return df


# =========================================================
# PROCESSAMENTO DO ESPECTRO
# =========================================================
def process_spectrum(x, y):

    idx = np.argsort(x)

    x = x[idx]
    y = y[idx]

    # suavização
    y_smooth = smooth_signal(y)

    # baseline
    baseline = asls_baseline(y_smooth)

    y_corr = y_smooth - baseline

    # normalização
    y_norm = normalize(y_corr)

    return x, y_norm, baseline


# =========================================================
# DETECÇÃO DE PICOS
# =========================================================
def detect_peaks(x, y):

    peak_idx,_ = find_peaks(
        y,
        prominence=0.05,
        distance=20
    )

    peaks = []

    for idx in peak_idx:

        peaks.append({
            "Peak (cm⁻¹)": round(x[idx],2),
            "Intensity": round(y[idx],3)
        })

    return peak_idx, pd.DataFrame(peaks)


# =========================================================
# PLOT
# =========================================================
def plot_spectrum(x, y, peak_idx):

    fig, ax = plt.subplots(figsize=(6,4), dpi=200)

    ax.plot(x, y, "k-", lw=1.2)

    ax.scatter(x[peak_idx], y[peak_idx], color="red", zorder=3)

    for i in peak_idx:

        ax.text(
            x[i],
            y[i]+0.02,
            f"{x[i]:.0f}",
            fontsize=8,
            ha="center"
        )

    ax.set_xlabel("Raman Shift (cm⁻¹)")
    ax.set_ylabel("Normalized Intensity")

    ax.invert_xaxis()

    return fig


# =========================================================
# STREAMLIT TAB
# =========================================================
def render_mapeamento_molecular_tab(supabase):

    st.header("🔬 Raman Mapping – 17 Spectra")

    uploaded_file = st.file_uploader(
        "Upload Raman Mapping",
        type=["txt","csv"]
    )

    if not uploaded_file:
        return

    df = read_mapping_file(uploaded_file)

    grouped = df.groupby(["y","x"])

    spectra_list = []

    for (y_val, x_val), group in grouped:

        group = group.sort_values("wave")

        spectra_list.append({
            "y": y_val,
            "x": x_val,
            "wave": group["wave"].values,
            "intensity": group["intensity"].values
        })

    st.write(f"Total de espectros detectados: **{len(spectra_list)}**")

    cols = st.columns(2)

    for i, spec in enumerate(spectra_list):

        x, y, baseline = process_spectrum(
            spec["wave"],
            spec["intensity"]
        )

        peak_idx, peak_df = detect_peaks(x, y)

        with cols[i % 2]:

            st.subheader(f"Espectro {i+1} — Y={spec['y']}")

            fig = plot_spectrum(x, y, peak_idx)

            st.pyplot(fig)

            st.dataframe(
                peak_df,
                use_container_width=True
            )

        if i % 2 == 1:
            cols = st.columns(2)
