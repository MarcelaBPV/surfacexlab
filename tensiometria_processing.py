# tensiometria_processing.py
# -*- coding: utf-8 -*-

"""
Processamento de Análises Físico-Mecânicas por Tensiometria Óptica
Método: Ângulo de contato + OWRK

Entrada:
- Arquivo .LOG / .TXT / .CSV de goniômetro óptico
  (Time, Theta(L), Theta(R), Mean, Dev., ...)

Saídas:
- Estatísticas robustas do ângulo de contato
- Energia livre de superfície (OWRK)
- Componentes polar e dispersiva
- Classificação de molhabilidade
- DataFrames prontos para Supabase e ML
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, Tuple

# ---------------------------------------------------------
# CONSTANTES — líquidos padrão (25 °C)
# ---------------------------------------------------------
LIQUIDS = {
    "water": {
        "gamma_L": 72.8,
        "gamma_d": 21.8,
        "gamma_p": 51.0,
    },
    "diiodomethane": {
        "gamma_L": 50.8,
        "gamma_d": 50.8,
        "gamma_p": 0.0,
    },
    "formamide": {
        "gamma_L": 58.0,
        "gamma_d": 39.0,
        "gamma_p": 19.0,
    },
}

# ---------------------------------------------------------
# LEITURA ROBUSTA DO LOG
# ---------------------------------------------------------
def read_contact_angle_log(file_like) -> pd.DataFrame:
    df = pd.read_csv(
        file_like,
        sep=None,
        engine="python",
        comment="#",
        skip_blank_lines=True,
    )

    df.columns = [c.strip() for c in df.columns]

    # normalização de nomes
    rename_map = {
        "Theta(L)": "theta_L",
        "Theta(R)": "theta_R",
        "Mean": "theta_mean",
        "Dev.": "theta_std",
        "Time": "time_s",
        "Messages": "messages",
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

    return df


# ---------------------------------------------------------
# LIMPEZA DOS DADOS
# ---------------------------------------------------------
def clean_contact_angle(df: pd.DataFrame) -> pd.DataFrame:
    dfc = df.copy()

    # remove leituras inválidas
    for col in ["theta_L", "theta_R", "theta_mean"]:
        if col in dfc.columns:
            dfc[col] = pd.to_numeric(dfc[col], errors="coerce")

    # descarta ângulos fisicamente inválidos
    dfc = dfc[
        (dfc["theta_mean"] > 0) &
        (dfc["theta_mean"] < 180)
    ]

    # remove pontos com erro explícito do equipamento
    if "messages" in dfc.columns:
        dfc = dfc[~dfc["messages"].astype(str).str.contains("Error", na=False)]

    return dfc.reset_index(drop=True)


# ---------------------------------------------------------
# ESTATÍSTICAS DO ÂNGULO DE CONTATO
# ---------------------------------------------------------
def contact_angle_statistics(df: pd.DataFrame) -> Dict:
    theta = df["theta_mean"].values

    return {
        "theta_mean_deg": float(np.mean(theta)),
        "theta_std_deg": float(np.std(theta, ddof=1)),
        "theta_median_deg": float(np.median(theta)),
        "n_points": int(len(theta)),
    }


# ---------------------------------------------------------
# OWRK — Owens–Wendt–Rabel–Kaelble
# ---------------------------------------------------------
def owkr_surface_energy(theta_by_liquid: Dict[str, float]) -> Dict:
    """
    theta_by_liquid: dict
        ex: {"water": 75.2, "diiodomethane": 42.1}
    """

    X = []
    Y = []

    for liq, theta in theta_by_liquid.items():
        props = LIQUIDS[liq]
        gamma_L = props["gamma_L"]
        gamma_d = props["gamma_d"]
        gamma_p = props["gamma_p"]

        cos_theta = np.cos(np.deg2rad(theta))
        y = gamma_L * (1 + cos_theta) / (2 * np.sqrt(gamma_d))
        x = np.sqrt(gamma_p / gamma_d) if gamma_d > 0 else 0

        X.append(x)
        Y.append(y)

    X = np.array(X)
    Y = np.array(Y)

    # regressão linear
    A = np.vstack([X, np.ones_like(X)]).T
    slope, intercept = np.linalg.lstsq(A, Y, rcond=None)[0]

    gamma_s_d = intercept ** 2
    gamma_s_p = slope ** 2
    gamma_s_total = gamma_s_d + gamma_s_p

    # R²
    Y_pred = slope * X + intercept
    ss_res = np.sum((Y - Y_pred) ** 2)
    ss_tot = np.sum((Y - np.mean(Y)) ** 2)
    R2 = 1 - ss_res / ss_tot if ss_tot != 0 else np.nan

    return {
        "gamma_s_total": float(gamma_s_total),
        "gamma_s_d": float(gamma_s_d),
        "gamma_s_p": float(gamma_s_p),
        "polar_fraction": float(gamma_s_p / gamma_s_total) if gamma_s_total > 0 else np.nan,
        "R2": float(R2),
    }


# ---------------------------------------------------------
# CLASSIFICAÇÃO DE MOLHABILIDADE
# ---------------------------------------------------------
def classify_wettability(theta_mean: float, polar_fraction: float) -> str:
    if theta_mean > 90 and polar_fraction < 0.3:
        return "Hidrofóbica"
    if theta_mean < 60 and polar_fraction > 0.5:
        return "Hidrofílica"
    return "Intermediária / Anfifílica"


# ---------------------------------------------------------
# FUNÇÃO PRINCIPAL
# ---------------------------------------------------------
def process_tensiometry(
    file_like,
    liquid_name: str,
) -> Dict:
    """
    Processamento completo da tensiometria
    """

    df_raw = read_contact_angle_log(file_like)
    df_clean = clean_contact_angle(df_raw)

    stats = contact_angle_statistics(df_clean)

    theta_by_liquid = {
        liquid_name: stats["theta_mean_deg"]
    }

    owkr = owkr_surface_energy(theta_by_liquid)

    wettability = classify_wettability(
        stats["theta_mean_deg"],
        owkr["polar_fraction"],
    )

    # -----------------------------------------------------
    # PLOT
    # -----------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(df_clean["time_s"], df_clean["theta_mean"], lw=1.5)
    ax.set_xlabel("Tempo (s)")
    ax.set_ylabel("Ângulo de contato (°)")
    ax.set_title("Evolução do ângulo de contato")
    ax.grid(True, linestyle="--", alpha=0.4)

    return {
        "df_raw": df_raw,
        "df_clean": df_clean,
        "statistics": stats,
        "owrk": owkr,
        "wettability": wettability,
        "figure": fig,
    }
