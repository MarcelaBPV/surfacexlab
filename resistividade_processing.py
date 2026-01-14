# resistividade_processing.py
# -*- coding: utf-8 -*-

"""
Processamento de Análises Elétricas — SurfaceXLab
Método: Corrente × Tensão (I–V)

Referência:
Smits, F. M. (1958). Measurement of sheet resistivities with the four-point probe.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict


# =========================================================
# LEITURA ROBUSTA DO ARQUIVO I–V
# =========================================================
def read_iv_file(file_like) -> pd.DataFrame:
    """
    Lê CSV / TXT / XLS / XLSX e identifica automaticamente
    colunas de corrente e tensão.
    """

    # -----------------------------
    # Leitura genérica
    # -----------------------------
    name = str(file_like.name).lower()

    if name.endswith((".xls", ".xlsx")):
        df = pd.read_excel(file_like)
    else:
        df = pd.read_csv(
            file_like,
            sep=None,
            engine="python",
            comment="#",
            skip_blank_lines=True,
        )

    if df.empty or df.shape[1] < 2:
        raise ValueError("Arquivo vazio ou inválido.")

    # -----------------------------
    # Normalização de nomes
    # -----------------------------
    df.columns = [c.strip().lower() for c in df.columns]

    col_I = None
    col_V = None

    for c in df.columns:
        if any(k in c for k in ["current", "corrente", " i", "i(", "i["]):
            col_I = c
        if any(k in c for k in ["voltage", "tensao", "tensão", " v", "v(", "v["]):
            col_V = c

    # fallback simples
    if col_I is None and "i" in df.columns:
        col_I = "i"
    if col_V is None and "v" in df.columns:
        col_V = "v"

    if col_I is None or col_V is None:
        raise ValueError(
            f"Arquivo inválido: colunas de corrente e tensão não encontradas.\n"
            f"Colunas detectadas: {list(df.columns)}"
        )

    df = df[[col_I, col_V]].copy()
    df.columns = ["current_a", "voltage_v"]

    # -----------------------------
    # Conversão numérica robusta
    # -----------------------------
    for col in df.columns:
        df[col] = (
            df[col]
            .astype(str)
            .str.replace(",", ".", regex=False)
        )

    df = df.apply(pd.to_numeric, errors="coerce")
    df = df.dropna().reset_index(drop=True)

    if len(df) < 3:
        raise ValueError("Número insuficiente de pontos válidos para ajuste linear.")

    return df


# =========================================================
# AJUSTE LINEAR I–V
# =========================================================
def linear_iv_fit(I: np.ndarray, V: np.ndarray) -> Dict:
    """
    Ajuste linear V = R·I + b
    """

    A = np.vstack([I, np.ones_like(I)]).T
    slope, intercept = np.linalg.lstsq(A, V, rcond=None)[0]

    V_pred = slope * I + intercept

    ss_res = np.sum((V - V_pred) ** 2)
    ss_tot = np.sum((V - np.mean(V)) ** 2)

    R2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan

    return {
        "R_ohm": float(slope),
        "offset_v": float(intercept),
        "R2": float(R2),
        "V_pred": V_pred,
    }


# =========================================================
# RESISTIVIDADE
# =========================================================
def compute_resistivity(
    R_ohm: float,
    thickness_m: float,
    geometry: str = "four_point_film",
) -> float:

    if thickness_m <= 0:
        raise ValueError("Espessura do filme deve ser maior que zero.")

    if geometry == "four_point_film":
        k = np.pi / np.log(2)  # Smits
        return k * R_ohm * thickness_m

    if geometry == "bulk":
        return R_ohm

    raise ValueError("Geometria inválida.")


# =========================================================
# CLASSIFICAÇÃO DO MATERIAL
# =========================================================
def classify_material(sigma_S_m: float) -> str:

    if not np.isfinite(sigma_S_m):
        return "Indefinido"

    if sigma_S_m >= 1e4:
        return "Condutor"
    if 1e-2 <= sigma_S_m < 1e4:
        return "Semicondutor"
    return "Isolante"


# =========================================================
# PIPELINE PRINCIPAL
# =========================================================
def process_resistivity(
    file_like,
    thickness_m: float,
    geometry: str = "four_point_film",
) -> Dict:

    # -------------------------------
    # Leitura
    # -------------------------------
    df = read_iv_file(file_like)

    I = df["current_a"].values
    V = df["voltage_v"].values

    # -------------------------------
    # Ajuste linear
    # -------------------------------
    fit = linear_iv_fit(I, V)

    # ⚠️ ALERTA, NÃO BLOQUEIO
    ohmic_warning = fit["R2"] < 0.90

    R = fit["R_ohm"]
    rho = compute_resistivity(R, thickness_m, geometry)
    sigma = 1.0 / rho if rho > 0 else np.nan
    classe = classify_material(sigma)

    # -------------------------------
    # Plot I × V
    # -------------------------------
    fig, ax = plt.subplots(figsize=(7, 4), dpi=300)

    ax.scatter(I, V, color="black", s=30, label="Dados experimentais")
    ax.plot(
        I,
        fit["V_pred"],
        color="red",
        linewidth=2,
        label=f"Ajuste linear (R² = {fit['R2']:.4f})",
    )

    ax.set_xlabel("Corrente (A)")
    ax.set_ylabel("Tensão (V)")
    ax.set_title("Curva Corrente × Tensão")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend()

    # -------------------------------
    # Summary pronto para PCA / ML
    # -------------------------------
    summary = {
        "Resistência (Ω)": R,
        "Resistividade (Ω·m)": rho,
        "Condutividade (S/m)": sigma,
        "R²": fit["R2"],
        "Ajuste_ohmico_alerta": ohmic_warning,
    }

    return {
        "df": df,
        "fit": fit,
        "summary": summary,
        "classe": classe,
        "figure": fig,
    }
