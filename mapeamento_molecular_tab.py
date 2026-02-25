# mapeamento_molecular_tab.py
# -*- coding: utf-8 -*-

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.signal import find_peaks
from scipy.optimize import curve_fit
from scipy import sparse
from scipy.sparse.linalg import spsolve
from scipy.special import wofz


# =========================================================
# DATABASE RAMAN — BIOMOLÉCULAS SANGUE
# =========================================================
RAMAN_DATABASE = {

    # ===============================
    # ÁCIDOS NUCLEICOS / BASES
    # ===============================
    (720, 735): "Adenine (DNA/RNA)",
    (780, 800): "Uracil / Cytosine (RNA/DNA)",
    (1335, 1345): "Nucleic acids (DNA/RNA backbone)",

    # ===============================
    # AMINOÁCIDOS / PROTEÍNAS
    # ===============================
    (750, 760): "Tryptophan (Proteins)",
    (830, 850): "Tyrosine (Proteins)",
    (870, 880): "C–C stretch (Proteins)",
    (930, 950): "α-Helix C–C (Proteins)",
    (1000, 1006): "Phenylalanine (Proteins)",
    (1030, 1045): "C–H in-plane bending (Proteins)",
    (1120, 1140): "C–N stretching (Proteins)",
    (1200, 1230): "Amide III (Proteins)",
    (1240, 1300): "Amide III (Proteins)",
    (1540, 1580): "Amide II (Proteins)",
    (1640, 1680): "Amide I (Proteins)",

    # ===============================
    # LIPÍDIOS / FOSFOLIPÍDIOS
    # ===============================
    (1060, 1085): "C–C stretch (Lipids)",
    (1295, 1310): "CH2 twisting (Lipids)",
    (1440, 1475): "CH2/CH3 bending (Lipids)",
    (1650, 1665): "C=C stretch (Unsaturated Lipids)",
    (2850, 2870): "CH2 symmetric stretch (Lipids)",
    (2880, 2900): "CH2 asymmetric stretch (Lipids)",

    # ===============================
    # HEMOGLOBINA / HEME
    # ===============================
    (750, 755): "Hemoglobin (heme breathing)",
    (1125, 1145): "Hemoglobin (pyrrole deformation)",
    (1340, 1380): "Hemoglobin (oxidation state)",
    (1545, 1565): "Hemoglobin ν(C=C)",
    (1600, 1630): "Hemoglobin (spin state)",

    # ===============================
    # CARBOIDRATOS / GLICOSE
    # ===============================
    (480, 520): "Glucose (C–C/C–O)",
    (850, 920): "Glucose (C–O–C)",
    (1040, 1060): "Glucose (C–O stretch)",

    # ===============================
    # METABÓLITOS / OUTROS
    # ===============================
    (620, 650): "Lactate",
    (950, 980): "Phosphate (ATP/ADP)",
    (1080, 1100): "Phospholipids / Phosphate",
}


def classify_raman_peak(center):
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

    D = sparse.diags([1, -2, 1], [0, 1, 2], shape=(N-2, N))
    w = np.ones(N)

    for _ in range(niter):
        W = sparse.diags(w, 0)
        Z = W + lam * D.T @ D
        z = spsolve(Z, w*y)
        w = p*(y > z) + (1-p)*(y < z)

    return z


# =========================================================
# MODELOS ESPECTRAIS
# =========================================================
def lorentz(x, amp, cen, wid):
    return amp*((0.5*wid)**2 / ((x-cen)**2 + (0.5*wid)**2))


def voigt_profile(x, amp, cen, sigma, gamma):
    z = ((x - cen) + 1j*gamma) / (sigma*np.sqrt(2))
    return amp*np.real(wofz(z))/(sigma*np.sqrt(2*np.pi))


# =========================================================
# LEITURA ROBUSTA COM VÍRGULA DECIMAL
# =========================================================
def read_mapping_file(uploaded_file):

    name = uploaded_file.name.lower()

    if name.endswith((".xls", ".xlsx")):
        df = pd.read_excel(uploaded_file)

    else:
        uploaded_file.seek(0)

        df = pd.read_csv(
            uploaded_file,
            sep=r"\s+|\t+|;|,",
            engine="python",
            header=None,
            encoding="latin1",
            decimal=","
        )

        df = df.iloc[:, :4]
        df.columns = ["y","x","wave","intensity"]

    df = df.replace(",", ".", regex=True)
    df = df.apply(pd.to_numeric, errors="coerce")
    df = df.dropna()

    if df.empty:
        raise ValueError("Arquivo sem dados válidos.")

    return df


# =========================================================
# FITTING + TABELA
# =========================================================
def plot_fit(spec, model="voigt"):

    x = spec["wave"]
    y = spec["intensity"]

    baseline = asls_baseline(y)
    y_corr = y - baseline

    peak_idx, _ = find_peaks(
        y_corr,
        prominence=np.max(y_corr)*0.08,
        distance=15
    )

    fits = []
    y_sum = np.zeros_like(x)
    peak_table = []

    for idx in peak_idx:

        cen_guess = x[idx]
        amp_guess = y_corr[idx]

        try:
            if model == "lorentz":

                popt, _ = curve_fit(
                    lorentz,
                    x,
                    y_corr,
                    p0=[amp_guess, cen_guess, 15],
                    maxfev=8000
                )

                curve = lorentz(x, *popt)
                cen = popt[1]

            else:

                popt, _ = curve_fit(
                    voigt_profile,
                    x,
                    y_corr,
                    p0=[amp_guess, cen_guess, 8, 8],
                    maxfev=8000
                )

                curve = voigt_profile(x, *popt)
                cen = popt[1]

            fits.append((cen, curve))
            y_sum += curve

            group = classify_raman_peak(cen)

            peak_table.append({
                "Peak (cm⁻¹)": round(cen,1),
                "Grupo molecular": group,
                "Intensidade": round(float(np.max(curve)),2),
                "Modelo": model
            })

        except:
            continue

    residual = y_corr - y_sum

    fig, axes = plt.subplots(
        2,1,
        figsize=(6,5),
        dpi=300,
        sharex=True,
        gridspec_kw={"height_ratios":[3,1]}
    )

    ax = axes[0]

    ax.plot(x, y_corr, "k-", lw=1.2, label="Experimental")

    colors = plt.cm.tab10.colors

    for i, (cen, curve) in enumerate(fits):

        ax.plot(x, curve, color=colors[i % 10], lw=1)
        ax.axvline(cen, ls="--", lw=0.7, color="gray")

        # LABEL HORIZONTAL
        ax.text(
            cen,
            max(y_corr)*0.95,
            classify_raman_peak(cen),
            fontsize=7,
            ha="center"
        )

    #ax.plot(x, y_sum, "r-", lw=1.3, label="PeakSum")

    ax.legend(frameon=False, fontsize=8)
    ax.set_ylabel("Intensity (a.u.)")
    ax.invert_xaxis()
    ax.grid(False)

    axes[1].plot(x, residual, "k-", lw=1)
    axes[1].axhline(0, ls="--")
    axes[1].set_xlabel("Raman Shift (cm⁻¹)")
    axes[1].set_ylabel("Residual")

    return fig, pd.DataFrame(peak_table)


# =========================================================
# STREAMLIT TAB
# =========================================================
def render_mapeamento_molecular_tab(supabase):

    st.header("🗺️ Mapeamento Molecular Raman")

    model_choice = st.radio(
        "Tipo de ajuste espectral:",
        ["Voigt (recomendado)", "Lorentziano"]
    )

    model = "voigt" if "Voigt" in model_choice else "lorentz"

    uploaded_file = st.file_uploader(
        "Upload Raman Mapping",
        type=["txt","csv","xls","xlsx"]
    )

    if not uploaded_file:
        return

    try:

        df = read_mapping_file(uploaded_file)
        grouped = df.groupby(["y","x"])

        spectra_list = []

        for (y_val, x_val), group in grouped:
            group = group.sort_values("wave")

            spectra_list.append({
                "y": y_val,
                "wave": group["wave"].values,
                "intensity": group["intensity"].values
            })

        st.subheader("Ajuste Raman")

        cols = st.columns(2)

        for i, spec in enumerate(spectra_list):

            with cols[i % 2]:

                st.markdown(
                    f"**Espectro {i+1} — Y={spec['y']:.0f} µm**"
                )

                fig, peak_df = plot_fit(spec, model=model)
                st.pyplot(fig)

                st.dataframe(
                    peak_df,
                    use_container_width=True
                )

            if i % 2 == 1:
                cols = st.columns(2)

    except Exception as e:
        st.error(f"Erro ao processar arquivo: {e}")
