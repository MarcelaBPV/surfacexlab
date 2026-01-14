# tensiometria_processing.py
# -*- coding: utf-8 -*-

"""
Processamento de Análises Físico-Mecânicas por Tensiometria Óptica
Método: Ângulo de contato + OWRK (Owens–Wendt–Rabel–Kaelble)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict
from io import StringIO


# =========================================================
# CONSTANTES — líquidos padrão (25 °C)
# =========================================================
LIQUIDS = {
    "water": {"gamma_L": 72.8, "gamma_d": 21.8, "gamma_p": 51.0},
    "diiodomethane": {"gamma_L": 50.8, "gamma_d": 50.8, "gamma_p": 0.0},
    "formamide": {"gamma_L": 58.0, "gamma_d": 39.0, "gamma_p": 19.0},
}


# =========================================================
# LEITURA ROBUSTA DE LOG
# =========================================================
def read_contact_angle_log(file_like) -> pd.DataFrame:
    file_like.seek(0)
    raw = file_like.read().decode("latin1", errors="ignore")
    lines = raw.splitlines()

    header_idx = None
    for i, line in enumerate(lines):
        if "Mean" in line and ("Theta" in line or "Time" in line):
            header_idx = i
            break

    if header_idx is None:
        return pd.DataFrame()

    buffer = StringIO("\n".join(lines[header_idx:]))

    try:
        df = pd.read_csv(buffer, sep=";", engine="python", on_bad_lines="skip")
        if df.shape[1] < 2:
            buffer.seek(0)
            df = pd.read_csv(buffer, sep="\t", engine="python", on_bad_lines="skip")
    except Exception:
        return pd.DataFrame()

    df.columns = [c.strip() for c in df.columns]

    rename_map = {
        "Time": "time_s",
        "Mean": "theta_mean",
        "Dev.": "theta_std",
        "Theta(L)": "theta_L",
        "Theta(R)": "theta_R",
        "Messages": "messages",
    }

    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

    for col in df.columns:
        df[col] = (
            df[col].astype(str)
            .str.replace(",", ".", regex=False)
        )
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


# =========================================================
# LIMPEZA
# =========================================================
def clean_contact_angle(df: pd.DataFrame) -> pd.DataFrame:
    dfc = df.copy()

    if "theta_mean" not in dfc.columns:
        return pd.DataFrame()

    dfc = dfc[
        (dfc["theta_mean"] > 0) &
        (dfc["theta_mean"] < 180)
    ]

    if "messages" in dfc.columns:
        dfc = dfc[~dfc["messages"].astype(str).str.contains("Error", na=False)]

    return dfc.reset_index(drop=True)


# =========================================================
# ESTATÍSTICAS
# =========================================================
def contact_angle_statistics(df: pd.DataFrame) -> Dict:
    theta = df["theta_mean"].values

    return {
        "Theta médio (°)": float(np.mean(theta)),
        "Theta std (°)": float(np.std(theta, ddof=1)),
        "Theta mediano (°)": float(np.median(theta)),
        "N pontos": int(len(theta)),
    }


# =========================================================
# OWRK
# =========================================================
def owkr_surface_energy(theta_by_liquid: Dict[str, float]) -> Dict:

    if len(theta_by_liquid) < 2:
        raise ValueError("OWRK requer pelo menos dois líquidos.")

    X, Y = [], []

    for liq, theta in theta_by_liquid.items():
        props = LIQUIDS[liq]

        cos_t = np.cos(np.deg2rad(theta))
        y = props["gamma_L"] * (1 + cos_t) / (2 * np.sqrt(props["gamma_d"]))
        x = np.sqrt(props["gamma_p"] / props["gamma_d"]) if props["gamma_d"] > 0 else 0

        X.append(x)
        Y.append(y)

    X = np.array(X)
    Y = np.array(Y)

    A = np.vstack([X, np.ones_like(X)]).T
    slope, intercept = np.linalg.lstsq(A, Y, rcond=None)[0]

    gamma_d = intercept ** 2
    gamma_p = slope ** 2
    gamma_total = gamma_d + gamma_p

    Y_pred = slope * X + intercept
    ss_res = np.sum((Y - Y_pred) ** 2)
    ss_tot = np.sum((Y - np.mean(Y)) ** 2)
    R2 = 1 - ss_res / ss_tot if ss_tot != 0 else np.nan

    return {
        "Energia superficial (mJ/m²)": float(gamma_total),
        "Componente dispersiva (mJ/m²)": float(gamma_d),
        "Componente polar (mJ/m²)": float(gamma_p),
        "Fração polar": float(gamma_p / gamma_total) if gamma_total > 0 else np.nan,
        "OWRK R²": float(R2),
    }


# =========================================================
# CLASSIFICAÇÃO
# =========================================================
def classify_wettability(theta: float, polar_fraction: float) -> str:
    if theta > 90 and polar_fraction < 0.3:
        return "Hidrofóbica"
    if theta < 60 and polar_fraction > 0.5:
        return "Hidrofílica"
    return "Intermediária"


# =========================================================
# PIPELINE PRINCIPAL
# =========================================================
def process_tensiometry(
    file_like,
    theta_by_liquid: Dict[str, float],
) -> Dict:

    df_raw = read_contact_angle_log(file_like)
    df_clean = clean_contact_angle(df_raw)

    stats = contact_angle_statistics(df_clean)
    owkr = owkr_surface_energy(theta_by_liquid)

    wettability = classify_wettability(
        stats["Theta médio (°)"],
        owkr["Fração polar"],
    )

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(df_clean["time_s"], df_clean["theta_mean"], lw=1.5)
    ax.set_xlabel("Tempo (s)")
    ax.set_ylabel("Ângulo de contato (°)")
    ax.set_title("Evolução do ângulo de contato")
    ax.grid(True, linestyle="--", alpha=0.4)

    return {
        "df_raw": df_raw,
        "df_clean": df_clean,
        **stats,
        **owkr,
        "Molhabilidade": wettability,
        "figure": fig,
    }
