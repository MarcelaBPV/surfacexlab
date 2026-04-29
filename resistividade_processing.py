# =========================================================
# PROCESSAMENTO ELÉTRICO — VERSÃO CIENTÍFICA (4 PONTAS)
# =========================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# =========================================================
# LEITURA
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
# AJUSTE LINEAR
# =========================================================
def linear_fit(I, V):

    A = np.vstack([I, np.ones_like(I)]).T
    slope, intercept = np.linalg.lstsq(A, V, rcond=None)[0]

    V_pred = slope * I + intercept

    ss_res = np.sum((V - V_pred) ** 2)
    ss_tot = np.sum((V - np.mean(V)) ** 2)

    R2 = 1 - ss_res / ss_tot

    return slope, intercept, R2, V_pred


# =========================================================
# RESISTIVIDADE (4 PONTAS)
# =========================================================
def resistivity_4p(R, thickness):

    k = np.pi / np.log(2)
    return k * R * thickness


# =========================================================
# CLASSIFICAÇÃO FÍSICA
# =========================================================
def classify(sigma):

    if sigma > 1e5:
        return "Metal"
    elif sigma > 1e-6:
        return "Semicondutor"
    else:
        return "Isolante"


# =========================================================
# PIPELINE
# =========================================================
def process_resistivity(file_like, thickness_m, geometry="four_point_film"):

    df = read_iv_file(file_like)

    I = df["I"].values
    V = df["V"].values

    # -------------------------
    # Ajuste linear
    # -------------------------
    R, offset, R2, V_pred = linear_fit(I, V)

    # -------------------------
    # Resistividade
    # -------------------------
    rho = resistivity_4p(R, thickness_m)
    sigma = 1 / rho if rho > 0 else np.nan
    Rs = rho / thickness_m

    # -------------------------
    # Resistência diferencial
    # -------------------------
    dV = np.gradient(V)
    dI = np.gradient(I)
    R_diff = dV / dI

    # -------------------------
    # DIAGNÓSTICO FÍSICO
    # -------------------------
    regime = "Ôhmico"

    if R2 < 0.98:
        regime = "Não ôhmico"

    if abs(offset) > 0.01:
        regime += " + Contato não ideal"

    classe = classify(sigma)

    # -------------------------
    # GRÁFICO
    # -------------------------
    fig, ax = plt.subplots(figsize=(7,4), dpi=300)

    ax.scatter(I, V, color="black", label="Experimental")
    ax.plot(I, V_pred, color="red", label=f"Ajuste (R²={R2:.4f})")

    ax.set_xlabel("Corrente (A)")
    ax.set_ylabel("Tensão (V)")
    ax.legend()
    ax.grid(alpha=0.3)

    # -------------------------
    # SUMMARY
    # -------------------------
    summary = {
        "Resistência (Ω)": R,
        "Resistividade (Ω·m)": rho,
        "Condutividade (S/m)": sigma,
        "Sheet Resistance (Ω/sq)": Rs,
        "R²": R2,
        "Offset (V)": offset,
        "Regime": regime,
        "Classe": classe,
    }

    return {
        "df": df,
        "summary": summary,
        "figure": fig,
        "R_diff": R_diff
    }
