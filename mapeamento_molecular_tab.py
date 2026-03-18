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
# DATABASE MOLECULAR RAMAN
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
# BASELINE
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
# PROCESSAMENTO
# =========================================================
def smooth(y):
    return savgol_filter(y,11,3) if len(y)>=11 else y


def normalize(y):
    m=np.max(np.abs(y))
    return y if m==0 else y/m


def process_spectrum(wave,intensity):

    idx=np.argsort(wave)

    x=wave[idx]
    y=intensity[idx]

    y=smooth(y)
    base=baseline_asls(y)

    y_corr=y-base
    y_norm=normalize(y_corr)

    return x,y_norm


# =========================================================
# DETECÇÃO DE PICOS
# =========================================================
def detect_peaks(x,y):

    peaks,_=find_peaks(y,prominence=0.05,distance=20)

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
# LEITURA
# =========================================================
def read_mapping(file):

    df=pd.read_csv(file,sep=r"\s+|,|;",engine="python",header=None)

    df=df.iloc[:,:4]
    df.columns=["y","x","wave","intensity"]

    df=df.apply(pd.to_numeric,errors="coerce")
    df=df.dropna()

    return df


# =========================================================
# HEATMAP
# =========================================================
def plot_heatmap(df):

    pivot=df.pivot_table(index="y",columns="wave",values="intensity")
    pivot=pivot.sort_index().sort_index(axis=1)

    fig,ax=plt.subplots(figsize=(6,4))

    im=ax.imshow(
        pivot.values,
        aspect="auto",
        cmap="inferno",
        origin="lower",
        extent=[
            pivot.columns.min(),
            pivot.columns.max(),
            pivot.index.min(),
            pivot.index.max()
        ]
    )

    ax.set_xlabel("Raman shift (cm⁻¹)")
    ax.set_ylabel("Posição Y")
    ax.set_title("Raman intensity map")

    cbar=fig.colorbar(im)
    cbar.set_label("Intensidade")

    return fig


# =========================================================
# PCA
# =========================================================
def run_pca(spectra):

    X=[s["intensity"] for s in spectra]

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
# GRUPOS L1–L4 (ESTILO ARTIGO)
# =========================================================
def plot_raman_groups(spectra):

    groups={
        "L1 (0–200)":(0,200),
        "L2 (0–400)":(0,400),
        "L3 (0–600)":(0,600),
        "L4 (0–800)":(0,800),
    }

    fig,axes=plt.subplots(4,1,figsize=(6,10),sharex=True)

    for ax,(label,(ymin,ymax)) in zip(axes,groups.items()):

        for spec in spectra:

            if ymin<=spec["y"]<=ymax:

                mask=(spec["wave"]>=1500)&(spec["wave"]<=1750)

                x=spec["wave"][mask]
                y=spec["intensity"][mask]

                if len(x)==0:
                    continue

                ax.plot(x,y,lw=1)

        ax.set_title(label)
        ax.grid(alpha=0.3)

    axes[-1].set_xlabel("Raman shift (cm⁻¹)")
    fig.text(0.02,0.5,"Intensity",rotation=90)

    plt.tight_layout()

    return fig


# =========================================================
# STREAMLIT
# =========================================================
def render_mapeamento_molecular_tab(supabase):

    st.header("🧬 Raman Molecular Mapping")

    file=st.file_uploader("Upload Raman mapping",type=["txt","csv"])

    if not file:
        return

    df=read_mapping(file)

    spectra=[]

    for y_val,group in df.groupby("y"):

        group=group.sort_values("wave")

        x,y=process_spectrum(group["wave"].values,group["intensity"].values)

        spectra.append({
            "y":y_val,
            "wave":x,
            "intensity":y
        })

    st.write(f"Total espectros: **{len(spectra)}**")

    subtabs=st.tabs([
        "Espectros",
        "Mapa Raman",
        "PCA",
        "Grupos (1500–1750 cm⁻¹)"
    ])

# ESPECTROS
    with subtabs[0]:

        cols=st.columns(4)

        for i,spec in enumerate(spectra):

            peaks,table=detect_peaks(spec["wave"],spec["intensity"])

            with cols[i%4]:

                st.caption(f"Y = {spec['y']}")

                fig,ax=plt.subplots(figsize=(3,2))

                ax.plot(spec["wave"],spec["intensity"],"k")
                ax.invert_xaxis()

                st.pyplot(fig)

                if st.button(f"Expandir {i}"):

                    fig2,ax2=plt.subplots()

                    ax2.plot(spec["wave"],spec["intensity"])
                    ax2.scatter(spec["wave"][peaks],spec["intensity"][peaks],color="red")

                    ax2.invert_xaxis()

                    st.pyplot(fig2)
                    st.dataframe(table)

            if i%4==3:
                cols=st.columns(4)

# HEATMAP
    with subtabs[1]:
        st.pyplot(plot_heatmap(df))

# PCA
    with subtabs[2]:
        st.pyplot(run_pca(spectra))

# GRUPOS
    with subtabs[3]:
        st.pyplot(plot_raman_groups(spectra))
