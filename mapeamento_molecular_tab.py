# -*- coding: utf-8 -*-

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.signal import find_peaks
from scipy.optimize import curve_fit
from scipy.special import wofz
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# =========================================================
# DATABASE MOLECULAR AVANÇADO
# =========================================================
RAMAN_DATABASE = [

    (1658,10,"Amide I (Proteins)"),
    (1602,8,"C=C stretching"),
    (1581,8,"Hemoglobin mode"),
    (1562,8,"Heme vibration"),
    (1543,8,"Hemoglobin ν11"),
    (1445,10,"Lipids CH2"),
    (1335,12,"Hemoglobin oxidation"),
    (1003,6,"Phenylalanine"),
    (750,6,"Tryptophan"),
]


def classify_raman_peak(center):

    best_label="Unassigned"
    best_diff=1e9

    for ref,tol,label in RAMAN_DATABASE:

        diff=abs(center-ref)

        if diff<tol and diff<best_diff:
            best_diff=diff
            best_label=label

    return best_label


# =========================================================
# PERFIL VOIGT
# =========================================================
def voigt(x,amp,cen,sigma,gamma):
    z=((x-cen)+1j*gamma)/(sigma*np.sqrt(2))
    return amp*np.real(wofz(z))/(sigma*np.sqrt(2*np.pi))


# =========================================================
# DECONVOLUÇÃO MULTIPICO
# =========================================================
def fit_multi_voigt(x,y):

    peak_idx,_=find_peaks(y,prominence=np.max(y)*0.05,distance=15)

    curves=[]
    table=[]

    for idx in peak_idx:

        cen_guess=x[idx]
        amp_guess=y[idx]

        try:

            popt,_=curve_fit(
                voigt,x,y,
                p0=[amp_guess,cen_guess,8,8],
                maxfev=12000
            )

            curve=voigt(x,*popt)
            cen=popt[1]

            curves.append(curve)

            table.append({
                "Peak":round(cen,1),
                "Grupo":classify_raman_peak(cen),
                "Amplitude":round(float(popt[0]),2)
            })

        except:
            continue

    return curves,pd.DataFrame(table)


# =========================================================
# PCA AUTOMÁTICO
# =========================================================
def run_pca(spectra):

    X=np.vstack(spectra)

    scaler=StandardScaler()
    X_scaled=scaler.fit_transform(X)

    pca=PCA(n_components=2)
    scores=pca.fit_transform(X_scaled)

    return scores,pca.explained_variance_ratio_


# =========================================================
# PLOT FITTING
# =========================================================
def plot_spectrum(x,y):

    curves,table=fit_multi_voigt(x,y)

    fig,ax=plt.subplots(figsize=(6,4),dpi=300)

    ax.plot(x,y,"k",lw=1.3,label="Experimental")

    for c in curves:
        ax.plot(x,c,alpha=0.7)

    ax.set_xlabel("Raman Shift (cm⁻¹)")
    ax.set_ylabel("Intensity")
    ax.invert_xaxis()

    return fig,table


# =========================================================
# STREAMLIT TAB PRINCIPAL
# =========================================================
def render_mapeamento_molecular_tab(supabase):

    st.header("🧬 Raman Molecular Mapping — Advanced")

    uploaded=st.file_uploader(
        "Upload Raman mapping",
        type=["txt","csv","xls","xlsx"]
    )

    if not uploaded:
        return

    df=pd.read_csv(
        uploaded,
        sep=r"\s+|\t+|;|,",
        engine="python",
        header=None,
        decimal=","
    )

    df=df.iloc[:,:4]
    df.columns=["y","x","wave","intensity"]

    df=df.replace(",",".",regex=True)
    df=df.apply(pd.to_numeric,errors="coerce")
    df=df.dropna()

    grouped=df.groupby(["y","x"])

    spectra=[]
    waves=None

    cols=st.columns(2)

    for i,((y,x),g) in enumerate(grouped):

        g=g.sort_values("wave")

        wave=g.wave.values
        intensity=g.intensity.values

        spectra.append(intensity)
        waves=wave

        with cols[i%2]:

            st.markdown(f"### Ponto Y={y}")

            fig,table=plot_spectrum(wave,intensity)

            st.pyplot(fig)
            st.dataframe(table,use_container_width=True)

        if i%2==1:
            cols=st.columns(2)

    # =====================================================
    # PCA AUTOMÁTICO
    # =====================================================
    st.subheader("📊 PCA Molecular")

    scores,var=run_pca(spectra)

    fig,ax=plt.subplots(figsize=(5,4),dpi=300)

    ax.scatter(scores[:,0],scores[:,1])

    ax.set_xlabel(f"PC1 ({var[0]*100:.1f}%)")
    ax.set_ylabel(f"PC2 ({var[1]*100:.1f}%)")

    st.pyplot(fig)
