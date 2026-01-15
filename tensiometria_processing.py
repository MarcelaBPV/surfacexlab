# tensiometria_processing.py
# -*- coding: utf-8 -*-

"""
SurfaceXLab — Tensiometria Óptica
Processamento físico + OWRK + integração Raman / Topografia
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
# LEITURA ROBUSTA DO LOG
# =========================================================
def read_contact_angle_log(file_like) -> pd.DataFrame:
    file_like.seek(0)
    raw = file_like.read().decode("latin1", errors="ignore")
    lines = raw.splitlines()

    header_idx = None
    for i, line in enumerate(lines):
        l = line.lower()
        if "time" in l and "mean" in l and "theta" in l:
            header_idx = i
            break

    if header_idx is None:
        raise ValueError("Cabeçalho do LOG não identificado.")

    buffer = StringIO("\n".join(lines[header_idx:]))

    df = pd.read_csv(buffer, sep=r"[;\t\s]+", engine="python")
    df.columns = [c.strip() for c in df.columns]

    rename_map = {
        "Time": "time_s",
        "Mean": "theta_mean",
        "Dev.": "theta_std",
        "Theta(L)": "theta_L",
        "Theta(R)": "theta_R",
        "Height": "height",
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
    if "theta_mean" not in df.columns:
        return pd.DataFrame()

    dfc = df[
        (df["theta_mean"] > 0) &
        (df["theta_mean"] < 180)
    ]

    if "messages" in dfc.columns:
        dfc = dfc[~dfc["messages"].astype(str).str.contains("Error", na=False)]

    return dfc.reset_index(drop=True)


# =========================================================
# Rrms — RUGOSIDADE EFETIVA DA LINHA DE CONTATO
# =========================================================
def compute_rrms(df: pd.DataFrame) -> float:
    """
    Prioridade:
    1️⃣ Altura da gota (Height) → Rrms geométrico
    2️⃣ Fallback: flutuação angular
    Retorna em milímetros (mm)
    """

    if "height" in df.columns and df["height"].notna().sum() > 5:
        x = df["height"].dropna().values
    else:
        x = df["theta_mean"].dropna().values

    mean_x = np.mean(x)
    rrms = np.sqrt(np.mean((x - mean_x) ** 2))

    return float(rrms)


# =========================================================
# q* — ÂNGULO CARACTERÍSTICO
# =========================================================
def compute_q_star(df: pd.DataFrame) -> float:
    n = len(df)
    if n < 10:
        return float(df["theta_mean"].mean())

    tail = df.iloc[int(0.7 * n):]
    return float(tail["theta_mean"].mean())


# =========================================================
# OWRK — ENERGIA DE SUPERFÍCIE
# =========================================================
def owkr_surface_energy(theta_by_liquid: Dict[str, float]) -> Dict:
    if len(theta_by_liquid) < 2:
        raise ValueError("OWRK requer pelo menos dois líquidos.")

    X, Y = [], []

    for liq, theta in theta_by_liquid.items():
        p = LIQUIDS[liq]

        cos_t = np.cos(np.deg2rad(theta))
        y = p["gamma_L"] * (1 + cos_t) / (2 * np.sqrt(p["gamma_d"]))
        x = np.sqrt(p["gamma_p"] / p["gamma_d"]) if p["gamma_d"] > 0 else 0

        X.append(x)
        Y.append(y)

    X, Y = np.array(X), np.array(Y)

    A = np.vstack([X, np.ones_like(X)]).T
    slope, intercept = np.linalg.lstsq(A, Y, rcond=None)[0]

    gamma_d = intercept ** 2
    gamma_p = slope ** 2

    return {
        "Energia superficial (mJ/m²)": gamma_d + gamma_p,
        "Componente dispersiva (mJ/m²)": gamma_d,
        "Componente polar (mJ/m²)": gamma_p,
        "Fração polar": gamma_p / (gamma_d + gamma_p),
    }


# =========================================================
# PIPELINE PRINCIPAL
# =========================================================
def process_tensiometry(
    file_like,
    theta_by_liquid: Dict[str, float],
    ID_IG: float,
    I2D_IG: float,
) -> Dict:

    df_raw = read_contact_angle_log(file_like)
    df_clean = clean_contact_angle(df_raw)

    rrms = compute_rrms(df_clean)
    q_star = compute_q_star(df_clean)
    owkr = owkr_surface_energy(theta_by_liquid)

    # -------------------------------
    # Plot
    # -------------------------------
    fig, ax = plt.subplots(figsize=(8, 4), dpi=300)
    ax.plot(df_clean["time_s"], df_clean["theta_mean"], lw=1.5)
    ax.axhline(q_star, color="red", linestyle="--", label="q*")
    ax.set_xlabel("Tempo (s)")
    ax.set_ylabel("Ângulo de contato (°)")
    ax.set_title("Evolução do ângulo de contato")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # -------------------------------
    # SAÍDA FINAL (PCA / ML)
    # -------------------------------
    summary = {
        "Rrms (mm)": rrms,
        "ID/IG": ID_IG,
        "I2D/IG": I2D_IG,
        "q* (°)": q_star,
        **owkr,
    }

    return {
        "df_raw": df_raw,
        "df_clean": df_clean,
        "summary": summary,
        "figure": fig,
    }
