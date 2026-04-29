# tensiometria_processing.py
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict
from io import StringIO


# =========================================================
# CONSTANTES — líquidos padrão
# =========================================================
LIQUIDS = {
    "water": {"gamma_L": 72.8, "gamma_d": 21.8, "gamma_p": 51.0},
    "diiodomethane": {"gamma_L": 50.8, "gamma_d": 50.8, "gamma_p": 0.0},
    "formamide": {"gamma_L": 58.0, "gamma_d": 39.0, "gamma_p": 19.0},
}


# =========================================================
# LEITURA
# =========================================================
def read_contact_angle_log(file_like):

    file_like.seek(0)
    raw = file_like.read().decode("latin1", errors="ignore")
    lines = raw.splitlines()

    header_idx = None
    for i, line in enumerate(lines):
        if "time" in line.lower() and "mean" in line.lower():
            header_idx = i
            break

    if header_idx is None:
        raise ValueError("Cabeçalho não encontrado")

    buffer = StringIO("\n".join(lines[header_idx:]))

    df = pd.read_csv(buffer, sep=r"[;\t\s]+", engine="python")
    df.columns = [c.strip() for c in df.columns]

    rename_map = {
        "Time": "time_s",
        "Mean": "theta_mean",
        "Theta(L)": "theta_L",
        "Theta(R)": "theta_R",
        "Height": "height"
    }

    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

    for col in df.columns:
        df[col] = pd.to_numeric(df[col].astype(str).str.replace(",", "."), errors="coerce")

    return df.dropna()


# =========================================================
# LIMPEZA
# =========================================================
def clean(df):
    return df[(df["theta_mean"] > 0) & (df["theta_mean"] < 180)]


# =========================================================
# OWRK
# =========================================================
def owkr(theta_by_liquid):

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

    gamma_d = intercept**2
    gamma_p = slope**2

    return gamma_d + gamma_p, gamma_d, gamma_p


# =========================================================
# PROCESSAMENTO PRINCIPAL
# =========================================================
def process_tensiometry(file_like, theta_by_liquid, ID_IG, I2D_IG):

    df = clean(read_contact_angle_log(file_like))

    # ==========================
    # FÍSICA
    # ==========================
    q_star = df["theta_mean"].iloc[int(0.7 * len(df)):].mean()
    rrms = np.std(df["height"]) if "height" in df else np.std(df["theta_mean"])

    theta_initial = df["theta_mean"].iloc[0]
    theta_final = df["theta_mean"].iloc[-1]

    spread = theta_initial - theta_final
    stability = np.std(df["theta_mean"])

    hysteresis = np.mean(df["theta_L"] - df["theta_R"]) if "theta_L" in df else 0

    # ==========================
    # OWRK
    # ==========================
    gamma_total, gamma_d, gamma_p = owkr(theta_by_liquid)

    # ==========================
    # CLASSIFICAÇÃO
    # ==========================
    wetting = "Hidrofílico" if q_star < 90 else "Hidrofóbico"

    diagnostic = "Superfície estável"

    if stability > 2:
        diagnostic = "Instável"

    if spread > 10:
        diagnostic += " + alta molhabilidade dinâmica"

    if abs(hysteresis) > 2:
        diagnostic += " + heterogênea"

    # ==========================
    # PLOT
    # ==========================
    fig, ax = plt.subplots(figsize=(7,4), dpi=300)

    ax.plot(df["time_s"], df["theta_mean"], label="θ médio")
    ax.axhline(q_star, color="red", linestyle="--", label="q*")

    ax.set_xlabel("Tempo (s)")
    ax.set_ylabel("Ângulo (°)")
    ax.legend()
    ax.grid(alpha=0.3)

    # ==========================
    # SUMMARY
    # ==========================
    summary = {
        "q* (°)": q_star,
        "Rrms (mm)": rrms,
        "Molhabilidade": wetting,
        "Histerese (°)": hysteresis,
        "Espalhamento (°)": spread,
        "Estabilidade (°)": stability,
        "Energia superficial (mJ/m²)": gamma_total,
        "Componente dispersiva (mJ/m²)": gamma_d,
        "Componente polar (mJ/m²)": gamma_p,
        "Diagnóstico": diagnostic,
        "ID/IG": ID_IG,
        "I2D/IG": I2D_IG,
    }

    return {
        "df": df,
        "summary": summary,
        "figure": fig
    }
