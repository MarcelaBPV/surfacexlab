# =========================================================
# tensiometria_processing.py
# SurfaceXLab — Tensiometria Científica
# =========================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from scipy.signal import savgol_filter


# =========================================================
# LEITURA UNIVERSAL
# =========================================================
def read_tensiometry_file(file_like):

    name = file_like.name.lower()

    # =====================================================
    # LOG KRUSS / OCA
    # =====================================================
    if name.endswith(".log"):

        df = pd.read_csv(
            file_like,
            delim_whitespace=True,
            skiprows=1
        )

        df.columns = [str(c).strip().lower() for c in df.columns]

        required = ["time", "theta(l)", "theta(r)", "mean"]

        for col in required:
            if col not in df.columns:
                raise ValueError(f"Coluna obrigatória ausente: {col}")

        df = df.rename(columns={
            "theta(l)": "theta_left",
            "theta(r)": "theta_right",
            "mean": "theta_mean"
        })

    # =====================================================
    # XLSX / CSV
    # =====================================================
    else:

        if name.endswith((".xlsx", ".xls")):
            df = pd.read_excel(file_like)

        else:
            df = pd.read_csv(file_like)

        df.columns = [str(c).strip().lower() for c in df.columns]

        theta_col = None

        for c in df.columns:
            if "angle" in c or "theta" in c:
                theta_col = c

        if theta_col is None:
            raise ValueError("Coluna de ângulo não encontrada")

        df["theta_mean"] = pd.to_numeric(
            df[theta_col],
            errors="coerce"
        )

        df["theta_left"] = df["theta_mean"]
        df["theta_right"] = df["theta_mean"]

        df["time"] = np.arange(len(df))

    # =====================================================
    # LIMPEZA
    # =====================================================
    cols = ["time", "theta_left", "theta_right", "theta_mean"]

    df = df[cols]

    for c in cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna()

    # remove absurdos
    df = df[
        (df["theta_mean"] > 5) &
        (df["theta_mean"] < 170)
    ]

    return df.reset_index(drop=True)


# =========================================================
# REMOÇÃO DE OUTLIERS
# =========================================================
def remove_outliers(theta):

    q1 = np.percentile(theta, 25)
    q3 = np.percentile(theta, 75)

    iqr = q3 - q1

    mask = (
        (theta > q1 - 1.5 * iqr) &
        (theta < q3 + 1.5 * iqr)
    )

    return theta[mask]


# =========================================================
# ENERGIA SUPERFICIAL SIMPLIFICADA
# =========================================================
def estimate_surface_energy(theta_final):

    # aproximação física simples
    # suficiente para dissertação

    if theta_final < 30:
        gamma = 72

    elif theta_final < 60:
        gamma = 55

    elif theta_final < 90:
        gamma = 40

    else:
        gamma = 25

    gamma_d = gamma * 0.65
    gamma_p = gamma * 0.35

    return gamma, gamma_d, gamma_p


# =========================================================
# CLASSIFICAÇÃO
# =========================================================
def classify_surface(theta):

    if theta < 30:
        return "Superhidrofílica"

    elif theta < 90:
        return "Hidrofílica"

    elif theta < 150:
        return "Hidrofóbica"

    else:
        return "Superhidrofóbica"


# =========================================================
# PROCESSAMENTO PRINCIPAL
# =========================================================
def process_tensiometry(file_like):

    # =====================================================
    # LEITURA
    # =====================================================
    df = read_tensiometry_file(file_like)

    time = df["time"].values

    theta_left = df["theta_left"].values
    theta_right = df["theta_right"].values
    theta_mean = df["theta_mean"].values

    # =====================================================
    # OUTLIERS
    # =====================================================
    theta_mean = remove_outliers(theta_mean)

    n = len(theta_mean)

    time = time[:n]
    theta_left = theta_left[:n]
    theta_right = theta_right[:n]

    # =====================================================
    # SUAVIZAÇÃO
    # =====================================================
    if n > 11:

        theta_smooth = savgol_filter(
            theta_mean,
            11,
            3
        )

    else:
        theta_smooth = theta_mean

    # =====================================================
    # ESTADO ESTÁVEL
    # =====================================================
    stable_region = theta_smooth[int(0.7 * n):]

    theta_final = np.mean(stable_region)

    theta_std = np.std(stable_region)

    hysteresis = np.mean(
        np.abs(theta_left - theta_right)
    )

    # =====================================================
    # ENERGIA SUPERFICIAL
    # =====================================================
    gamma, gamma_d, gamma_p = estimate_surface_energy(
        theta_final
    )

    # =====================================================
    # CLASSIFICAÇÃO
    # =====================================================
    wetting_class = classify_surface(theta_final)

    # =====================================================
    # DIAGNÓSTICO
    # =====================================================
    diagnostic = "Superfície estável"

    if theta_std > 5:
        diagnostic = "Instabilidade da gota"

    # =====================================================
    # FIGURA 1 — θ(t)
    # =====================================================
    fig_theta, ax = plt.subplots(
        figsize=(6, 4),
        dpi=300
    )

    ax.plot(
        time,
        theta_mean,
        linewidth=1,
        alpha=0.4,
        label="Experimental"
    )

    ax.plot(
        time,
        theta_smooth,
        linewidth=2,
        label="SavGol"
    )

    ax.axhline(
        theta_final,
        linestyle="--",
        linewidth=1.2,
        label=f"θ* = {theta_final:.1f}°"
    )

    ax.set_xlabel("Tempo (s)")
    ax.set_ylabel("Ângulo de contato (°)")
    ax.set_title("Dinâmica do ângulo de contato")

    ax.grid(alpha=0.3)

    ax.legend()

    # =====================================================
    # FIGURA 2 — ENERGIA SUPERFICIAL
    # =====================================================
    fig_energy, ax2 = plt.subplots(
        figsize=(5, 4),
        dpi=300
    )

    labels = [
        "γ Total",
        "γ Dispersiva",
        "γ Polar"
    ]

    values = [
        gamma,
        gamma_d,
        gamma_p
    ]

    ax2.bar(labels, values)

    ax2.set_ylabel("Energia superficial (mJ/m²)")

    ax2.set_title("Componentes da energia superficial")

    ax2.grid(alpha=0.3)

    # =====================================================
    # SUMMARY
    # =====================================================
    summary = {

        "Amostra":
            file_like.name,

        "Theta final (°)":
            round(theta_final, 2),

        "Desvio padrão":
            round(theta_std, 2),

        "Histerese":
            round(hysteresis, 2),

        "Energia superficial":
            round(gamma, 2),

        "Componente dispersiva":
            round(gamma_d, 2),

        "Componente polar":
            round(gamma_p, 2),

        "Classe":
            wetting_class,

        "Diagnóstico":
            diagnostic
    }

    return {

        "df": df,

        "summary": summary,

        "fig_theta": fig_theta,

        "fig_energy": fig_energy
    }
