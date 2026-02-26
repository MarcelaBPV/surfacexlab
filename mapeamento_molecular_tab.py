# mapeamento_molecular_tab.py
# -*- coding: utf-8 -*-

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.signal import find_peaks, savgol_filter
from scipy.optimize import curve_fit
from scipy import sparse
from scipy.sparse.linalg import spsolve
from scipy.special import wofz


# =========================================================
# COMPATIBILIDADE NUMPY
# =========================================================
def integrate_area(y, x):
    try:
        return np.trapezoid(y, x)
    except AttributeError:
        return np.trapz(y, x)


# =========================================================
# DATABASE RAMAN BIOMOLECULAR
# =========================================================
RAMAN_DATABASE = {

    (720,735): "Adenine DNA/RNA",
    (780,800): "Uracil/Cytosine",
    (1070,1100): "DNA backbone PO2",
    (1570,1605): "Guanine/Adenine ring",

    (1000,1006): "Phenylalanine",
    (1240,1300): "Amide III proteins",
    (1540,1580): "Amide II proteins",
    (1640,1680): "Amide I proteins",
    (830,860): "Tyrosine",

    (1300,1315): "CH2 twisting lipids",
    (1440,1475): "CH2 bending lipids",
    (1650,1670): "C=C lipids",

    (850,920): "Glucose/carbohydrates",
    (1040,1060): "C-O carbohydrates",
    (1120,1150): "C-C carbohydrates",

    (1330,1370): "Hemoglobin",
    (1545,1565): "Hemoglobin porphyrin",
    (620,650): "Lactate",
    (750,770): "Cytochrome",

    (1080,1095): "PO2 phospholipids",
    (700,710): "Cholesterol",
    (1450,1465): "Proteins/lipids CH2",
}


def classify_raman_peak(center):
    for (low, high), label in RAMAN_DATABASE.items():
        if low <= center <= high:
            return label
    return "Unassigned"


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
# NORMALIZAÇÃO
# =========================================================
def normalize_snv(y):
    std = np.std(y)
    if std == 0:
        return y
    return (y - np.mean(y)) / std


def normalize_area(y, x):
    area = integrate_area(y, x)
    if area == 0:
        return y
    return y / area


# =========================================================
# SUAVIZAÇÃO
# =========================================================
def smooth_savgol(y, window=11, poly=3):

    if window % 2 == 0:
        window += 1

    if len(y) < window:
        return y

    return savgol_filter(y, window, poly)


# =========================================================
# PERFIL VOIGT
# =========================================================
def voigt_profile(x, amp, cen, sigma, gamma):

    z = ((x-cen)+1j*gamma)/(sigma*np.sqrt(2))
    return amp*np.real(wofz(z))/(sigma*np.sqrt(2*np.pi))


def multi_voigt(x, *params):

    y = np.zeros_like(x)

    for i in range(0, len(params), 4):
        amp, cen, sigma, gamma = params[i:i+4]
        y += voigt_profile(x, amp, cen, sigma, gamma)

    return y


# =========================================================
# FWHM
# =========================================================
def calc_fwhm(x, curve):

    half = np.max(curve)/2
    idx = np.where(curve >= half)[0]

    if len(idx) < 2:
        return np.nan

    return abs(x[idx[-1]] - x[idx[0]])


# =========================================================
# LEITURA ARQUIVO
# =========================================================
def read_mapping_file(uploaded_file):

    name = uploaded_file.name.lower()

    if name.endswith((".xls",".xlsx")):
        df = pd.read_excel(uploaded_file)

    else:
        uploaded_file.seek(0)
        df = pd.read_csv(
            uploaded_file,
            sep=r"\s+|\t+|;|,",
            engine="python",
            header=None,
            decimal=","
        )

        df = df.iloc[:,:4]
        df.columns=["y","x","wave","intensity"]

    df = df.replace(",",".",regex=True)
    df = df.apply(pd.to_numeric,errors="coerce")
    df = df.dropna()

    return df


# =========================================================
# FIT GLOBAL MULTIPICO
# =========================================================
def plot_fit(spec, smooth=False, window=11, poly=3,
             normalization="Nenhuma",
             region_min=None, region_max=None):

    x = np.array(spec["wave"])
    y = np.array(spec["intensity"])

    if region_min is not None and region_max is not None:
        mask = (x>=region_min)&(x<=region_max)
        x = x[mask]
        y = y[mask]

    if len(x) < 10:
        return None, pd.DataFrame()

    if smooth:
        y = smooth_savgol(y,window,poly)

    baseline = asls_baseline(y)
    y_corr = y - baseline

    if normalization=="SNV":
        y_corr=normalize_snv(y_corr)

    elif normalization=="Área":
        y_corr=normalize_area(y_corr,x)

    prominence=np.std(y_corr)*3

    peak_idx,_=find_peaks(
        y_corr,
        prominence=prominence,
        distance=20
    )

    if len(peak_idx)==0:
        return None,pd.DataFrame()

    init_params=[]
    for idx in peak_idx:
        init_params += [y_corr[idx],x[idx],8,8]

    try:
        popt,pcov=curve_fit(
            multi_voigt,
            x,y_corr,
            p0=init_params,
            maxfev=40000
        )

        y_fit=multi_voigt(x,*popt)
        perr=np.sqrt(np.diag(pcov))

    except Exception:
        return None,pd.DataFrame()

    residual=y_corr-y_fit

    peak_table=[]

    for i in range(0,len(popt),4):

        amp,cen,sigma,gamma=popt[i:i+4]
        err_cen=perr[i+1]

        curve=voigt_profile(x,amp,cen,sigma,gamma)

        peak_table.append({
            "Peak (cm⁻¹)":round(cen,1),
            "Erro ±":round(err_cen,2),
            "FWHM":round(calc_fwhm(x,curve),2),
            "Área":round(integrate_area(curve,x),2),
            "Grupo molecular":classify_raman_peak(cen)
        })

    fig,axes=plt.subplots(
        2,1,figsize=(6,5),dpi=300,
        sharex=True,
        gridspec_kw={"height_ratios":[3,1]}
    )

    axes[0].plot(x,y_corr,"k-",lw=1,label="Experimental")
    axes[0].plot(x,y_fit,"r--",lw=1,label="Fit global")
    axes[0].invert_xaxis()
    axes[0].legend()

    axes[1].plot(x,residual,"k-")
    axes[1].axhline(0,ls="--")

    axes[1].set_xlabel("Raman Shift (cm⁻¹)")
    axes[1].set_ylabel("Residual")

    return fig,pd.DataFrame(peak_table)


# =========================================================
# STREAMLIT TAB
# =========================================================
def render_mapeamento_molecular_tab(supabase):

    st.header("🧬 Mapeamento Molecular Raman")

    st.subheader("Pré-processamento")

    smooth=st.checkbox("Suavização Savitzky-Golay")

    if smooth:
        window=st.slider("Janela",5,31,11,step=2)
        poly=st.slider("Polinômio",2,5,3)
    else:
        window,poly=11,3

    normalization=st.radio(
        "Normalização:",
        ["Nenhuma","SNV","Área"]
    )

    region_choice=st.radio(
        "Região espectral:",
        ["Completo","Fingerprint 800–1800","Personalizado"]
    )

    if region_choice=="Fingerprint 800–1800":
        region_min,region_max=800,1800
    elif region_choice=="Personalizado":
        region_min=st.number_input("Min",800)
        region_max=st.number_input("Max",1800)
    else:
        region_min,region_max=None,None

    uploaded_file=st.file_uploader(
        "Upload Raman Mapping",
        type=["txt","csv","xls","xlsx"]
    )

    if not uploaded_file:
        return

    df=read_mapping_file(uploaded_file)
    grouped=df.groupby(["y","x"])

    cols = st.columns(2)

    for i,((y_pos,x_pos),group) in enumerate(grouped):

        spec={
            "wave":group["wave"].values,
            "intensity":group["intensity"].values
        }

        titulo = f"Espectro {i+1} ({y_pos:.0f},{x_pos:.0f})"

        fig,peak_df=plot_fit(
            spec,
            smooth=smooth,
            window=window,
            poly=poly,
            normalization=normalization,
            region_min=region_min,
            region_max=region_max
        )

        if fig:
            with cols[i%2]:

                st.markdown(f"#### {titulo}")

                with st.expander("🔍 Expandir espectro"):
                    st.pyplot(fig,use_container_width=True)

                with st.expander("📊 Dados do espectro"):
                    st.dataframe(peak_df,use_container_width=True)

        if i%2==1:
            cols = st.columns(2)
