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
# DATABASE RAMAN – SANGUE (LITERATURA)
# =========================================================

RAMAN_BLOOD_DB = [

    (720,"DNA/RNA Adenine"),
    (754,"Hemoglobin breathing"),
    (760,"Tryptophan"),
    (785,"DNA/RNA Cytosine"),
    (830,"Tyrosine"),
    (855,"Glucose"),
    (960,"Phosphate / ATP"),
    (1003,"Phenylalanine"),
    (1095,"DNA backbone"),
    (1127,"Hemoglobin pyrrole"),
    (1245,"Amide III proteins"),
    (1300,"Lipids CH2"),
    (1335,"Hemoglobin oxidation"),
    (1445,"Lipids CH2 bending"),
    (1543,"Amide II proteins"),
    (1562,"Hemoglobin ν19"),
    (1581,"Aromatic C=C"),
    (1602,"C=C stretching"),
    (1620,"Tyrosine / proteins"),
    (1632,"Protein side chains"),
    (1655,"Amide I proteins"),
]


def identify_molecule(peak):

    best = "Unknown"
    min_diff = 1e9

    for ref,label in RAMAN_BLOOD_DB:

        diff = abs(peak-ref)

        if diff < min_diff and diff < 15:
            min_diff = diff
            best = label

    return best


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
def smooth(y):

    window = 11

    if window % 2 == 0:
        window += 1

    if len(y) < window:
        return y

    return savgol_filter(y,window,3)


# =========================================================
# NORMALIZAÇÃO
# =========================================================
def normalize(y):

    m = np.max(np.abs(y))

    if m == 0:
        return y

    return y/m


# =========================================================
# LEITURA ARQUIVO
# =========================================================
def read_mapping(file):

    file.seek(0)

    df = pd.read_csv(
        file,
        sep=r"\s+|\t+|;|,",
        engine="python",
        header=None,
        decimal=","
    )

    df = df.iloc[:,:4]

    df.columns = ["y","x","wave","intensity"]

    df["y"] = pd.to_numeric(df["y"],errors="coerce")
    df["wave"] = pd.to_numeric(df["wave"],errors="coerce")
    df["intensity"] = pd.to_numeric(df["intensity"],errors="coerce")

    df = df.dropna()

    return df


# =========================================================
# PROCESSAMENTO
# =========================================================
def process_spectrum(wave,intensity):

    idx = np.argsort(wave)

    x = wave[idx]
    y = intensity[idx]

    y = smooth(y)

    baseline = asls_baseline(y)

    y_corr = y-baseline

    y_norm = normalize(y_corr)

    return x,y_norm


# =========================================================
# DETECÇÃO DE PICOS
# =========================================================
def detect_peaks(x,y):

    peaks,_ = find_peaks(
        y,
        prominence=0.05,
        distance=20
    )

    table=[]

    for p in peaks:

        pos = float(x[p])

        table.append({
            "Peak (cm⁻¹)":round(pos,1),
            "Intensity":round(float(y[p]),3),
            "Molécula provável":identify_molecule(pos)
        })

    return peaks,pd.DataFrame(table)


# =========================================================
# PLOT PEQUENO
# =========================================================
def plot_small(x,y):

    fig,ax = plt.subplots(figsize=(3,2),dpi=120)

    ax.plot(x,y,"k",lw=1)

    ax.invert_xaxis()

    ax.set_xticks([])
    ax.set_yticks([])

    return fig


# =========================================================
# PLOT GRANDE
# =========================================================
def plot_large(x,y,peaks):

    fig,ax = plt.subplots(figsize=(6,4),dpi=200)

    ax.plot(x,y,"k",lw=1.2)

    ax.scatter(x[peaks],y[peaks],color="red")

    ax.set_xlabel("Raman Shift (cm⁻¹)")
    ax.set_ylabel("Normalized Intensity")

    ax.invert_xaxis()

    return fig


# =========================================================
# STREAMLIT TAB
# =========================================================
def render_mapeamento_molecular_tab(supabase):

    st.header("🔬 Raman Molecular Mapping")

    file = st.file_uploader(
        "Upload Raman mapping file",
        type=["txt","csv"]
    )

    if not file:
        return

    df = read_mapping(file)

    spectra=[]

    for y_val,group in df.groupby("y"):

        group = group.sort_values("wave")

        spectra.append({
            "y":y_val,
            "wave":group["wave"].values,
            "intensity":group["intensity"].values
        })

    st.write(f"Total de espectros detectados: **{len(spectra)}**")

    cols = st.columns(4)

    for i,spec in enumerate(spectra):

        x,y = process_spectrum(
            spec["wave"],
            spec["intensity"]
        )

        peaks,table = detect_peaks(x,y)

        with cols[i%4]:

            st.caption(f"Y = {spec['y']}")

            fig_small = plot_small(x,y)

            if st.button(f"Abrir {i+1}"):

                st.pyplot(plot_large(x,y,peaks))

                st.dataframe(
                    table,
                    use_container_width=True
                )

            else:

                st.pyplot(fig_small)

        if i%4==3:
            cols = st.columns(4)
