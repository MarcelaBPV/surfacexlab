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
# DATABASE RAMAN
# =========================================================
RAMAN_DATABASE = {
    (720,735): "Adenine DNA/RNA",
    (780,800): "Uracil/Cytosine",
    (1000,1006): "Phenylalanine",
    (1240,1300): "Amide III",
    (1330,1370): "Hemoglobin",
    (1440,1475): "Lipids",
    (1540,1580): "Amide II",
    (1640,1680): "Amide I",
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

        # REFINAMENTO FÍSICO
        sigma = max(sigma, 4)
        gamma = max(gamma, 4)

        if amp < 0:
            amp = 0

        y += voigt_profile(x, amp, cen, sigma, gamma)

    return y


# =========================================================
# MÉTRICAS ESTATÍSTICAS
# =========================================================
def compute_statistics(y_exp, y_fit, k):

    residual = y_exp - y_fit
    n = len(y_exp)

    ss_res = np.sum(residual**2)
    ss_tot = np.sum((y_exp - np.mean(y_exp))**2)

    r2 = 1 - ss_res/ss_tot if ss_tot != 0 else 0
    rmse = np.sqrt(np.mean(residual**2))

    aic = n*np.log(ss_res/n) + 2*k
    bic = n*np.log(ss_res/n) + k*np.log(n)

    snr = np.max(y_exp) / np.std(residual)

    return r2, rmse, aic, bic, snr


# =========================================================
# FIT GLOBAL REFINADO
# =========================================================
def plot_fit(spec, smooth=False, window=11, poly=3,
             region_min=None, region_max=None):

    x = np.array(spec["wave"])
    y = np.array(spec["intensity"])

    if region_min is not None and region_max is not None:
        mask = (x>=region_min)&(x<=region_max)
        x = x[mask]
        y = y[mask]

    if smooth:
        y = smooth_savgol(y,window,poly)

    baseline = asls_baseline(y)
    y_corr = y - baseline

    prominence = np.std(y_corr)*3

    peak_idx,_ = find_peaks(
        y_corr,
        prominence=prominence,
        distance=25
    )

    # REMOVER PICOS MUITO FRACOS (SNR)
    peak_idx = [
        idx for idx in peak_idx
        if y_corr[idx] > np.std(y_corr)*2
    ]

    if len(peak_idx)==0:
        return None, pd.DataFrame()

    init_params=[]
    lower=[]
    upper=[]

    for idx in peak_idx:
        init_params += [y_corr[idx], x[idx], 10, 10]
        lower += [0, x[idx]-20, 4, 4]
        upper += [np.max(y_corr)*3, x[idx]+20, 80, 80]

    try:
        popt,pcov = curve_fit(
            multi_voigt,
            x,y_corr,
            p0=init_params,
            bounds=(lower,upper),
            maxfev=50000
        )

        y_fit = multi_voigt(x,*popt)
        perr = np.sqrt(np.diag(pcov))

    except:
        return None, pd.DataFrame()

    r2,rmse,aic,bic,snr = compute_statistics(
        y_corr,
        y_fit,
        len(popt)
    )

    peak_table=[]

    for i in range(0,len(popt),4):

        amp,cen,sigma,gamma = popt[i:i+4]
        err_cen = perr[i+1]

        curve = voigt_profile(x,amp,cen,sigma,gamma)

        peak_table.append({
            "Peak (cm⁻¹)":round(cen,1),
            "Erro ±":round(err_cen,2),
            "FWHM":round(abs(2.355*sigma),2),
            "Área":round(integrate_area(curve,x),2),
            "Grupo molecular":classify_raman_peak(cen)
        })

    fig,axes = plt.subplots(
        2,1,figsize=(6,5),dpi=300,
        sharex=True,
        gridspec_kw={"height_ratios":[3,1]}
    )

    axes[0].plot(x,y_corr,"k-",lw=1,label="Experimental")
    axes[0].plot(x,y_fit,"r--",lw=1,label="Fit refinado")
    axes[0].legend()
    axes[0].invert_xaxis()

    axes[1].plot(x,y_corr-y_fit,"k-")
    axes[1].axhline(0,ls="--")

    axes[1].set_xlabel("Raman Shift (cm⁻¹)")
    axes[1].set_ylabel("Residual")

    st.markdown(
        f"""
        **R²:** {r2:.4f}  
        **RMSE:** {rmse:.4f}  
        **AIC:** {aic:.2f}  
        **BIC:** {bic:.2f}  
        **SNR:** {snr:.2f}
        """
    )

    return fig, pd.DataFrame(peak_table)
