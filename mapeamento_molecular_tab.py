# -*- coding: utf-8 -*-

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.signal import savgol_filter, find_peaks
from scipy import sparse
from scipy.sparse.linalg import spsolve
from scipy.special import wofz
from sklearn.decomposition import PCA


# =========================================================
# DATABASE MOLECULAR RAMAN SANGUE (LITERATURA)
# =========================================================

RAMAN_DB = [

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
    (1620,"Tyrosine proteins"),
    (1632,"Protein side chains"),
    (1655,"Amide I proteins"),
]


# =========================================================
# IDENTIFICAÇÃO MOLECULAR
# =========================================================

def identify_molecule(peak):

    best="Unknown"
    diff_min=1e9

    for ref,label in RAMAN_DB:

        diff=abs(peak-ref)

        if diff<diff_min and diff<15:
            diff_min=diff
            best=label

    return best


# =========================================================
# BASELINE ASLS
# =========================================================

def baseline_asls(y, lam=1e7, p=0.01, niter=15):

    y=np.asarray(y)

    N=len(y)

    D=sparse.diags([1,-2,1],[0,1,2], shape=(N-2,N))
    w=np.ones(N)

    for _ in range(niter):

        W=sparse.diags(w,0)

        Z=W+lam*D.T@D

        z=spsolve(Z,w*y)

        w=p*(y>z)+(1-p)*(y<z)

    return z


# =========================================================
# SUAVIZAÇÃO
# =========================================================

def smooth(y):

    if len(y)<11:
        return y

    return savgol_filter(y,11,3)


# =========================================================
# NORMALIZAÇÃO
# =========================================================

def normalize(y):

    m=np.max(np.abs(y))

    if m==0:
        return y

    return y/m


# =========================================================
# PERFIL VOIGT
# =========================================================

def voigt(x,a,c,s,g):

    z=((x-c)+1j*g)/(s*np.sqrt(2))

    return a*np.real(wofz(z))/(s*np.sqrt(2*np.pi))


# =========================================================
# LEITURA DO ARQUIVO
# =========================================================

def read_mapping(file):

    file.seek(0)

    df=pd.read_csv(
        file,
        sep=r"\s+|\t+|;|,",
        engine="python",
        header=None,
        decimal=","
    )

    df=df.iloc[:,:4]

    df.columns=["y","x","wave","intensity"]

    df["y"]=pd.to_numeric(df["y"],errors="coerce")
    df["wave"]=pd.to_numeric(df["wave"],errors="coerce")
    df["intensity"]=pd.to_numeric(df["intensity"],errors="coerce")

    df=df.dropna()

    return df


# =========================================================
# PROCESSAMENTO DO ESPECTRO 
# =========================================================

def process_spectrum(wave,intensity):

    idx=np.argsort(wave)

    x=wave[idx]
    y=intensity[idx]

    y=smooth(y)

    baseline=baseline_asls(y)

    y_corr=y-baseline

    y_norm=normalize(y_corr)

    return x,y_norm


# =========================================================
# DETECÇÃO DE PICOS
# =========================================================

def detect_peaks(x,y):

    peaks,_=find_peaks(
        y,
        prominence=0.05,
        distance=20
    )

    table=[]

    for p in peaks:

        pos=float(x[p])

        table.append({
            "Peak (cm⁻¹)":round(pos,1),
            "Intensity":round(float(y[p]),3),
            "Molécula provável":identify_molecule(pos)
        })

    return peaks,pd.DataFrame(table)


# =========================================================
# HEATMAP - - alterar eixos de intensidade
# =========================================================

def plot_heatmap(df):

    pivot = df.pivot_table(
        index="y",
        columns="wave",
        values="intensity"
    )

    # ordenar eixos
    pivot = pivot.sort_index(axis=0)
    pivot = pivot.sort_index(axis=1)

    y_vals = pivot.index.values
    x_vals = pivot.columns.values

    fig, ax = plt.subplots(figsize=(6,4))

    im = ax.imshow(
        pivot.values,
        aspect="auto",
        cmap="inferno",
        origin="lower",
        extent=[
            x_vals.min(), x_vals.max(),   # eixo X = Raman shift
            y_vals.min(), y_vals.max()    # eixo Y = posição
        ]
    )

    # =====================================================
    # EIXOS CORRETOS
    # =====================================================
    ax.set_xlabel("Raman shift (cm⁻¹)")
    ax.set_ylabel("Posição Y (plano cartesiano)")

    ax.set_title("Raman intensity map")

    cbar = fig.colorbar(im)
    cbar.set_label("Intensidade")

    return fig

# =========================================================
# PCA
# =========================================================

def run_pca(spectra):

    X=[]

    for s in spectra:

        X.append(s["intensity"])

    X=np.array(X)

    pca=PCA(n_components=2)

    scores=pca.fit_transform(X)

    fig,ax=plt.subplots()

    ax.scatter(scores[:,0],scores[:,1])

    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")

    ax.set_title("PCA Raman spectra")

    return fig


# =========================================================
# STREAMLIT
# =========================================================

def render_mapeamento_molecular_tab(supabase):

    st.header("🧬 Raman Molecular Mapping")

    # =========================================================
    # UPLOAD
    # =========================================================
    file = st.file_uploader(
        "Upload Raman mapping",
        type=["txt","csv"]
    )

    if not file:
        return

    df = read_mapping(file)

    # =========================================================
    # PROCESSAMENTO
    # =========================================================
    spectra = []

    for y_val, group in df.groupby("y"):

        group = group.sort_values("wave")

        x, y = process_spectrum(
            group["wave"].values,
            group["intensity"].values
        )

        spectra.append({
            "y": y_val,
            "wave": x,
            "intensity": y
        })

    st.write(f"Total espectros: **{len(spectra)}**")

    # =========================================================
    # SUBABAS
    # =========================================================
    subtabs = st.tabs([
        "Espectros",
        "Mapa Raman",
        "PCA",
        "Grupos (1500–1750 cm⁻¹)"
    ])

# =========================================================
# ABA 1 — ESPECTROS
# =========================================================
    with subtabs[0]:

        cols = st.columns(4)

        for i, spec in enumerate(spectra):

            peaks, table = detect_peaks(
                spec["wave"],
                spec["intensity"]
            )

            with cols[i % 4]:

                st.caption(f"Y = {spec['y']}")

                fig, ax = plt.subplots(figsize=(3,2))

                ax.plot(
                    spec["wave"],
                    spec["intensity"],
                    "k",
                    lw=1
                )

                ax.invert_xaxis()

                st.pyplot(fig)

                if st.button(f"Expandir {i}"):

                    fig2, ax2 = plt.subplots(figsize=(6,4))

                    ax2.plot(
                        spec["wave"],
                        spec["intensity"],
                        "k"
                    )

                    ax2.scatter(
                        spec["wave"][peaks],
                        spec["intensity"][peaks],
                        color="red"
                    )

                    ax2.invert_xaxis()

                    st.pyplot(fig2)

                    st.dataframe(table)

            if i % 4 == 3:
                cols = st.columns(4)

# =========================================================
# ABA 2 — HEATMAP
# =========================================================
    with subtabs[1]:

        st.subheader("Mapa Raman")
        st.pyplot(plot_heatmap(df))

# =========================================================
# ABA 3 — PCA
# =========================================================
    with subtabs[2]:

        st.subheader("PCA espectral")
        st.pyplot(run_pca(spectra))

# =========================================================
# ABA 4 — GRUPOS região 1500–1750 cm⁻¹
# =========================================================
    with subtabs[3]:

        st.subheader("Espectros por região (1500–1750 cm⁻¹)")

        st.pyplot(plot_raman_groups(spectra))
