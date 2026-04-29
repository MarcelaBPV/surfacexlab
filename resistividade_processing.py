# =========================================================
# PROCESSAMENTO ELÉTRICO — VERSÃO CIENTÍFICA (4 PONTAS)
# =========================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# =========================================================
# LEITURA ROBUSTA
# =========================================================
def read_iv_file(file_like):

    df = pd.read_csv(file_like, sep=None, engine="python")

    df.columns = [c.lower() for c in df.columns]

    col_I = [c for c in df.columns if "i" in c][0]
    col_V = [c for c in df.columns if "v" in c][0]

    df = df[[col_I, col_V]]
    df.columns = ["I", "V"]

    df = df.apply(pd.to_numeric, errors="coerce").dropna()

    return df


# =========================================================
# AJUSTE LINEAR (REGIÃO CENTRAL — FÍSICO)
# =========================================================
def linear_region_fit(V, I):

    # usa região central → evita contatos / não linearidade
    mask = (V > np.percentile(V, 20)) & (V < np.percentile(V, 80))

    V_lin = V[mask]
    I_lin = I[mask]

    A = np.vstack([V_lin, np.ones_like(V_lin)]).T
    slope, intercept = np.linalg.lstsq(A, I_lin, rcond=None)[0]

    I_pred = slope * V + intercept

    ss_res = np.sum((I - I_pred) ** 2)
    ss_tot = np.sum((I - np.mean(I)) ** 2)

    R2 = 1 - ss_res / ss_tot

    return slope, intercept, R2, I_pred


# =========================================================
# RESISTIVIDADE (4 PONTAS)
# =========================================================
def resistivity_4p(R, thickness):

    k = np.pi / np.log(2)
    return k * R * thickness


# =========================================================
# CLASSIFICAÇÃO
# =========================================================
def classify(sigma):

    if sigma > 1e5:
        return "Metal"
    elif sigma > 1e-6:
        return "Semicondutor"
    else:
        return "Isolante"


# =========================================================
# PIPELINE PRINCIPAL
# =========================================================
def process_resistivity(file_like, thickness_m, geometry="four_point_film"):

    df = read_iv_file(file_like)

    I = df["I"].values
    V = df["V"].values

    # =====================================================
    # AJUSTE CORRETO
    # =====================================================
    slope, offset, R2, I_pred = linear_region_fit(V, I)

    # Física correta → I = V/R
    R = 1 / slope if slope != 0 else np.nan

    rho = resistivity_4p(R, thickness_m)
    sigma = 1 / rho if rho > 0 else np.nan
    Rs = rho / thickness_m

    # =====================================================
    # RESISTÊNCIA DIFERENCIAL
    # =====================================================
    dV = np.gradient(V)
    dI = np.gradient(I)
    R_diff = dV / dI

    # =====================================================
    # DIAGNÓSTICO FÍSICO
    # =====================================================
    regime = "Ôhmico"

    if R2 < 0.98:
        regime = "Não ôhmico"

    if abs(offset) > 0.01:
        regime += " + Contato não ideal"

    classe = classify(sigma)

    # =====================================================
    # GRÁFICO PADRÃO PAPER
    # =====================================================
    fig, ax = plt.subplots(figsize=(6,4), dpi=300)

    ax.scatter(V, I, s=12, label="Experimental")
    ax.plot(V, I_pred, linewidth=1.5, label=f"Ajuste (R²={R2:.4f})")

    ax.set_xlabel("Tensão (V)")
    ax.set_ylabel("Corrente (A)")
    ax.set_title("Curva I–V")

    ax.grid(alpha=0.3)
    ax.legend()

    # =====================================================
    # SUMMARY
    # =====================================================
    summary = {
        "Resistência (Ω)": R,
        "Resistividade (Ω·m)": rho,
        "Condutividade (S/m)": sigma,
        "Sheet Resistance (Ω/sq)": Rs,
        "R²": R2,
        "Offset (A)": offset,
        "Regime": regime,
        "Classe": classe,
    }

    return {
        "df": df,
        "summary": summary,
        "figure": fig,
        "R_diff": R_diff
    }
