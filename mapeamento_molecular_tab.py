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
# DATABASE RAMAN PROBABILÍSTICO (baseado na sua figura)
# =========================================================
RAMAN_DATABASE = [
    {"center": 1003, "label": "Phenylalanine"},
    {"center": 1245, "label": "Amide III proteins"},
    {"center": 1335, "label": "Hemoglobin"},
    {"center": 1445, "label": "Lipids CH2"},
    {"center": 1543, "label": "Amide II (v11)"},
    {"center": 1562, "label": "Nucleic acids (v19)"},
    {"center": 1581, "label": "Aromatic C=C (v37)"},
    {"center": 1602, "label": "C=C stretching"},
    {"center": 1621, "label": "Tyrosine / CαCβ"},
    {"center": 1632, "label": "Protein side chains"},
    {"center": 1658, "label": "Amide I"},
]


# =========================================================
# BASELINE ALS
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

        sigma = max(sigma, 4)
        gamma = max(gamma, 4)
        amp = max(amp, 0)

        y += voigt_profile(x, amp, cen, sigma, gamma)

    return y


# =========================================================
# ESTATÍSTICA
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

    snr = np.max(y_exp) / (np.std(residual)+1e-9)

    return r2, rmse, aic, bic, snr


# =========================================================
# IDENTIFICAÇÃO MOLECULAR PROBABILÍSTICA
# =========================================================
def molecular_probability(center, error, fwhm, area):

    if error <= 0:
        error = 5

    probabilities=[]

    for ref in RAMAN_DATABASE:

        ref_center = ref["center"]

        dist_term = np.exp(-((center-ref_center)**2)/(2*error**2))
        width_term = np.exp(-(fwhm-40)**2/800)
        area_term = np.log1p(abs(area))

        score = dist_term * width_term * area_term

        probabilities.append({
            "Molécula":ref["label"],
            "Centro ref (cm⁻¹)":ref_center,
            "Score":score
        })

    df=pd.DataFrame(probabilities)

    total=df["Score"].sum()

    if total>0:
        df["Probabilidade (%)"]=100*df["Score"]/total
    else:
        df["Probabilidade (%)"]=0

    return df.sort_values("Probabilidade (%)",ascending=False).head(5)


# =========================================================
# FIT GLOBAL
# =========================================================
def plot_fit(spec, smooth=False):

    x = np.array(spec["wave"])
    y = np.array(spec["intensity"])

    if smooth:
        y = smooth_savgol(y)

    baseline = asls_baseline(y)
    y_corr = y - baseline

    prominence = np.std(y_corr)*3

    peak_idx,_ = find_peaks(
        y_corr,
        prominence=prominence,
        distance=25
    )

    if len(peak_idx)==0:
        return None, pd.DataFrame()

    init_params=[]
    lower=[]
    upper=[]

    for idx in peak_idx:
        init_params += [y_corr[idx], x[idx], 10, 10]
        lower += [0, x[idx]-20, 4, 4]
        upper += [np.max(y_corr)*3, x[idx]+20, 80, 80]

    popt,pcov = curve_fit(
        multi_voigt,
        x,y_corr,
        p0=init_params,
        bounds=(lower,upper),
        maxfev=50000
    )

    y_fit = multi_voigt(x,*popt)
    perr = np.sqrt(np.diag(pcov))

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

        fwhm_val = abs(2.355*sigma)
        area_val = integrate_area(curve,x)

        prob_df = molecular_probability(
            cen, err_cen, fwhm_val, area_val
        )

        top_molecule = prob_df.iloc[0]["Molécula"]
        top_prob = prob_df.iloc[0]["Probabilidade (%)"]

        peak_table.append({
            "Peak (cm⁻¹)":round(cen,1),
            "Erro ±":round(err_cen,2),
            "FWHM":round(fwhm_val,2),
            "Área":round(area_val,2),
            "Molécula provável":top_molecule,
            "Confiança (%)":round(top_prob,1)
        })

    fig,axes = plt.subplots(
        2,1,figsize=(6,5),dpi=300,
        sharex=True,
        gridspec_kw={"height_ratios":[3,1]}
    )

    axes[0].plot(x,y_corr,"k-",lw=1,label="Experimental")
    axes[0].plot(x,y_fit,"r--",lw=1,label="Fit global")
    axes[0].invert_xaxis()
    axes[0].legend()

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


# =========================================================
# STREAMLIT
# =========================================================
def render_mapeamento_molecular_tab(supabase):

    st.header("🧬 Análise Raman com Identificação Probabilística")

    smooth = st.checkbox("Suavização Savitzky-Golay")

    uploaded_file = st.file_uploader(
        "Upload espectro Raman",
        type=["txt","csv"]
    )

    if not uploaded_file:
        return

    df = pd.read_csv(uploaded_file)

    spec={
        "wave":df.iloc[:,0].values,
        "intensity":df.iloc[:,1].values
    }

    fig,peak_df = plot_fit(spec,smooth=smooth)

    if fig:
        st.pyplot(fig,use_container_width=True)
        st.dataframe(peak_df,use_container_width=True)

        st.subheader("Identificação molecular detalhada")

        for _,row in peak_df.iterrows():

            prob_df = molecular_probability(
                row["Peak (cm⁻¹)"],
                row["Erro ±"],
                row["FWHM"],
                row["Área"]
            )

            st.markdown(f"### Pico {row['Peak (cm⁻¹)']} cm⁻¹")
            st.dataframe(prob_df,use_container_width=True)
