# mapeamento_molecular_tab.py
# -*- coding: utf-8 -*-

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.signal import savgol_filter, find_peaks
from scipy import sparse
from scipy.sparse.linalg import spsolve


# =========================================================
# BASELINE ASLS
# =========================================================
def asls_baseline(y, lam=1e7, p=0.01, niter=15):

    y = np.asarray(y, dtype=float)
    N = len(y)

    D = sparse.diags([1,-2,1],[0,1,2], shape=(N-2,N))
    w = np.ones(N)

    for _ in range(niter):

        W = sparse.diags(w,0)
        Z = W + lam*D.T@D

        z = spsolve(Z, w*y)

        w = p*(y>z)+(1-p)*(y<z)

    return z


# =========================================================
# SUAVIZAÇÃO
# =========================================================
def smooth_signal(y):

    window = 11

    if window % 2 == 0:
        window += 1

    if len(y) < window:
        return y

    return savgol_filter(y, window, 3)


# =========================================================
# NORMALIZAÇÃO
# =========================================================
def normalize(y):

    max_val = np.max(np.abs(y))

    if max_val == 0:
        return y

    return y/max_val


# =========================================================
# LEITURA DO ARQUIVO
# =========================================================
def read_mapping(uploaded_file):

    uploaded_file.seek(0)

    df = pd.read_csv(
        uploaded_file,
        sep=r"\s+|\t+|;|,",
        engine="python",
        header=None,
        decimal=","
    )

    df = df.dropna()

    df = df.iloc[:,:4]

    df.columns = ["y","x","wave","intensity"]

    df["y"] = pd.to_numeric(df["y"], errors="coerce")
    df["wave"] = pd.to_numeric(df["wave"], errors="coerce")
    df["intensity"] = pd.to_numeric(df["intensity"], errors="coerce")

    df = df.dropna()

    return df


# =========================================================
# PROCESSAR ESPECTRO
# =========================================================
def process_spectrum(wave, intensity):

    idx = np.argsort(wave)

    x = wave[idx]
    y = intensity[idx]

    y = smooth_signal(y)

    baseline = asls_baseline(y)

    y_corr = y - baseline

    y_norm = normalize(y_corr)

    return x, y_norm


# =========================================================
# DETECTAR PICOS
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
            "Peak (cm⁻¹)": float(round(float(x[idx]),2)),
            "Intensity": float(round(float(y[idx]),3))
        })

    return peak_idx, pd.DataFrame(peaks)


# =========================================================
# PLOT PEQUENO
# =========================================================
def plot_small(x,y):

    fig, ax = plt.subplots(figsize=(3,2), dpi=120)

    ax.plot(x,y,"k",lw=1)

    ax.set_xticks([])
    ax.set_yticks([])

    ax.invert_xaxis()

    return fig


# =========================================================
# PLOT GRANDE
# =========================================================
def plot_large(x,y,peak_idx):

    fig, ax = plt.subplots(figsize=(6,4), dpi=200)

    ax.plot(x,y,"k",lw=1.2)

    ax.scatter(x[peak_idx],y[peak_idx],color="red")

    ax.set_xlabel("Raman Shift (cm⁻¹)")
    ax.set_ylabel("Normalized Intensity")

    ax.invert_xaxis()

    return fig


# =========================================================
# STREAMLIT
# =========================================================
def render_mapeamento_molecular_tab(supabase):

    st.header("🔬 Raman Mapping")

    uploaded_file = st.file_uploader(
        "Upload arquivo Raman Mapping",
        type=["txt","csv"]
    )

    if not uploaded_file:
        return

    df = read_mapping(uploaded_file)

    # agrupar apenas pelo Y
    spectra = []

    for y_val, group in df.groupby("y"):

        group = group.sort_values("wave")

        spectra.append({
            "y": y_val,
            "wave": group["wave"].values,
            "intensity": group["intensity"].values
        })

    st.write(f"Total de espectros detectados: **{len(spectra)}**")

    cols = st.columns(4)

    for i, spec in enumerate(spectra):

        x, y = process_spectrum(
            spec["wave"],
            spec["intensity"]
        )

        peak_idx, peak_df = detect_peaks(x,y)

        with cols[i%4]:

            st.caption(f"Y = {spec['y']}")

            fig_small = plot_small(x,y)

            if st.button(f"Abrir {i+1}"):

                st.pyplot(plot_large(x,y,peak_idx))

                st.dataframe(
                    peak_df,
                    use_container_width=True
                )

            else:

                st.pyplot(fig_small)

        if i%4==3:
            cols = st.columns(4)
