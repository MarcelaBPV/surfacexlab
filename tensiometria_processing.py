# =========================================================
# tensiometria_processing.py
# SurfaceXLab — Processamento Tensiométrico
# =========================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.stats import linregress

# =========================================================
# FUNÇÃO PRINCIPAL
# =========================================================
def process_tensiometry(file):

    # =====================================================
    # LEITURA DOS DADOS
    # =====================================================
    try:

        if file.name.endswith(".csv"):

            df = pd.read_csv(file)

        elif file.name.endswith(".xlsx"):

            df = pd.read_excel(file)

        elif file.name.endswith(".log"):

            df = pd.read_csv(
                file,
                sep=None,
                engine="python"
            )

        else:

            raise ValueError(
                "Formato não suportado."
            )

    except Exception as e:

        raise Exception(
            f"Erro ao ler arquivo: {e}"
        )

    # =====================================================
    # PADRONIZAÇÃO
    # =====================================================
    df.columns = [

        str(c).strip().lower()

        for c in df.columns
    ]

    # =====================================================
    # IDENTIFICAÇÃO DAS COLUNAS
    # =====================================================
    possible_theta = [

        "theta",
        "ângulo",
        "angulo",
        "contact angle",
        "angle"
    ]

    theta_col = None

    for c in df.columns:

        if c.lower() in possible_theta:

            theta_col = c
            break

    if theta_col is None:

        theta_col = df.columns[0]

    theta = pd.to_numeric(

        df[theta_col],

        errors="coerce"
    ).dropna()

    if len(theta) == 0:

        raise Exception(
            "Nenhum valor válido encontrado."
        )

    # =====================================================
    # ESTATÍSTICA
    # =====================================================
    theta_mean = np.mean(theta)

    theta_std = np.std(theta)

    hysteresis = np.max(theta) - np.min(theta)

    # =====================================================
    # CLASSIFICAÇÃO
    # =====================================================
    if theta_mean < 30:

        classe = "Superhidrofílica"

    elif theta_mean < 90:

        classe = "Hidrofílica"

    elif theta_mean < 150:

        classe = "Hidrofóbica"

    else:

        classe = "Superhidrofóbica"

    # =====================================================
    # DIAGNÓSTICO
    # =====================================================
    if hysteresis < 10:

        diagnostico = (
            "Baixa histerese e boa homogeneidade superficial."
        )

    elif hysteresis < 30:

        diagnostico = (
            "Histerese moderada."
        )

    else:

        diagnostico = (
            "Alta heterogeneidade superficial."
        )

    # =====================================================
    # ENERGIA SUPERFICIAL (MODELO SIMPLIFICADO)
    # =====================================================
    gamma_total = 72.8 * np.cos(
        np.radians(theta_mean)
    )

    gamma_total = abs(gamma_total)

    gamma_disp = gamma_total * 0.7

    gamma_polar = gamma_total * 0.3

    # =====================================================
    # REGRESSÃO
    # =====================================================
    x_reg = np.arange(len(theta))

    y_reg = theta.values

    slope, intercept, r_value, _, _ = linregress(

        x_reg,

        y_reg
    )

    r2 = r_value**2

    # =====================================================
    # FIGURA — ÂNGULO
    # =====================================================
    fig_theta, ax = plt.subplots(

        figsize=(6,4),

        dpi=300
    )

    ax.plot(

        theta.values,

        marker="o",

        linewidth=1.5
    )

    ax.axhline(

        theta_mean,

        linestyle="--"
    )

    ax.set_xlabel("Medição")

    ax.set_ylabel("Ângulo de contato (°)")

    ax.set_title("Molhabilidade")

    ax.grid(alpha=0.2)

    ax.spines["top"].set_visible(False)

    ax.spines["right"].set_visible(False)

    plt.tight_layout()

    # =====================================================
    # FIGURA — ENERGIA
    # =====================================================
    fig_energy, ax2 = plt.subplots(

        figsize=(5,4),

        dpi=300
    )

    labels = [

        "Dispersiva",
        "Polar"
    ]

    values = [

        gamma_disp,
        gamma_polar
    ]

    ax2.bar(

        labels,

        values
    )

    ax2.set_ylabel("mJ/m²")

    ax2.set_title("Energia Superficial")

    ax2.spines["top"].set_visible(False)

    ax2.spines["right"].set_visible(False)

    plt.tight_layout()

    # =====================================================
    # SUMMARY
    # =====================================================
    summary = {

        "Amostra":
            file.name,

        "Theta final (°)":
            float(theta_mean),

        "Desvio padrão":
            float(theta_std),

        "Histerese":
            float(hysteresis),

        "Classe":
            classe,

        "Diagnóstico":
            diagnostico,

        "Energia superficial":
            float(gamma_total),

        "Componente dispersiva":
            float(gamma_disp),

        "Componente polar":
            float(gamma_polar),

        "R²":
            float(r2)
    }

    # =====================================================
    # RETURN
    # =====================================================
    return {

        "summary":
            summary,

        "fig_theta":
            fig_theta,

        "fig_energy":
            fig_energy
    }
