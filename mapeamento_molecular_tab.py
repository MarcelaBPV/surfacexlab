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


# =========================================================
# BASELINE ASLS ROBUSTO
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
# MODELO LORENTZIANO
# =========================================================
def lorentz(x, amp, cen, wid):
    return amp*((0.5*wid)**2 /
               ((x-cen)**2 + (0.5*wid)**2))


# =========================================================
# LEITURA ROBUSTA ARQUIVO RAMAN
# =========================================================
def read_mapping_file(uploaded_file):

    name = uploaded_file.name.lower()

    if name.endswith((".xls", ".xlsx")):
        df = pd.read_excel(uploaded_file)

    else:
        uploaded_file.seek(0)

        df = pd.read_csv(
            uploaded_file,
            sep=r"\s+|\t+|,",
            engine="python",
            header=None,
            skip_blank_lines=True,
            encoding="latin1"
        )

        if df.shape[1] < 4:
            raise ValueError(
                "Arquivo deve ter 4 colunas: y, x, wave, intensity"
            )

        df = df.iloc[:, :4]
        df.columns = ["y", "x", "wave", "intensity"]

    df = df.apply(pd.to_numeric, errors="coerce")
    df = df.dropna()

    if df.empty:
        raise ValueError("Arquivo sem dados numéricos válidos.")

    return df


# =========================================================
# PLOT CIENTÍFICO COM AJUSTE + RESIDUAL
# =========================================================
def plot_raman_fit_publication(spec):

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

    for idx in peak_idx:

        cen_guess = x[idx]
        amp_guess = y_corr[idx]
        wid_guess = 15

        try:
            popt, _ = curve_fit(
                lorentz,
                x,
                y_corr,
                p0=[amp_guess, cen_guess, wid_guess],
                maxfev=6000
            )

            amp, cen, wid = popt

            peak_curve = lorentz(x, amp, cen, wid)

            fits.append((amp, cen, wid, peak_curve))
            y_sum += peak_curve

        except Exception:
            continue

    residual = y_corr - y_sum

    # ===== FIGURA CIENTÍFICA =====
    fig, axes = plt.subplots(
        2, 1,
        figsize=(6,5),
        dpi=300,
        sharex=True,
        gridspec_kw={"height_ratios":[3,1]}
    )

    ax = axes[0]

    ax.plot(x, y_corr, "k-", lw=1.2, label="Experimental")

    colors = plt.cm.tab10.colors

    for i, (_, cen, _, curve) in enumerate(fits):

        ax.plot(
            x,
            curve,
            color=colors[i % 10],
            lw=1
        )

        ax.axvline(cen, ls="--", lw=0.7, color="gray")

    ax.plot(x, y_sum, "r-", lw=1.3, label="PeakSum")

    ax.legend(frameon=False, fontsize=8)
    ax.set_ylabel("Intensity (a.u.)")
    ax.invert_xaxis()
    ax.grid(False)

    # residual
    axes[1].plot(x, residual, "k-", lw=1)
    axes[1].axhline(0, ls="--")
    axes[1].set_xlabel("Raman Shift (cm⁻¹)")
    axes[1].set_ylabel("Residual")

    return fig


# =========================================================
# STREAMLIT TAB
# =========================================================
def render_mapeamento_molecular_tab(supabase):

    st.header("🗺️ Mapeamento Molecular Raman")

    uploaded_file = st.file_uploader(
        "Upload Raman Mapping",
        type=["txt", "csv", "xls", "xlsx"]
    )

    if not uploaded_file:
        return

    try:

        df = read_mapping_file(uploaded_file)
        grouped = df.groupby(["y", "x"])

        spectra_list = []

        for (y_val, x_val), group in grouped:

            group = group.sort_values("wave")

            spectra_list.append({
                "y": y_val,
                "wave": group["wave"].values,
                "intensity": group["intensity"].values
            })

        st.subheader("Ajuste Raman — Todos os Espectros")

        for i, spec in enumerate(spectra_list):

            st.markdown(
                f"### Espectro {i+1} (Y={spec['y']:.0f} µm)"
            )

            fig = plot_raman_fit_publication(spec)
            st.pyplot(fig)

    except Exception as e:
        st.error(f"Erro ao processar arquivo: {e}")
