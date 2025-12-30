# resistividade_processing.py
# -*- coding: utf-8 -*-

"""
Processamento de Análises Elétricas — SurfaceXLab
Método: Corrente × Tensão (I–V), com foco em quatro pontas (filme fino)

Entrada:
- Arquivo CSV/TXT com colunas de corrente e tensão

Saídas:
- Ajuste linear V = R·I + b
- Resistência elétrica (R)
- Resistividade elétrica (ρ)
- Condutividade elétrica (σ)
- Coeficiente de determinação R² (validação ôhmica)
- Classificação física do material

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
    Lê arquivo CSV/TXT com colunas de corrente e tensão.
    Aceita variações comuns de cabeçalho.
    """
    df = pd.read_csv(
        file_like,
        sep=None,
        engine="python",
        comment="#",
        skip_blank_lines=True,
    )

    # normaliza nomes
    df.columns = [c.strip().lower() for c in df.columns]

    rename_map = {
        "i": "current_a",
        "current": "current_a",
        "current_a": "current_a",
        "corrente": "current_a",
        "v": "voltage_v",
        "voltage": "voltage_v",
        "voltage_v": "voltage_v",
        "tensao": "voltage_v",
    }

    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

    if "current_a" not in df.columns or "voltage_v" not in df.columns:
        raise ValueError(
            "Arquivo inválido: colunas de corrente e tensão não encontradas."
        )

    df = df[["current_a", "voltage_v"]]
    df = df.apply(pd.to_numeric, errors="coerce")
    df = df.dropna().reset_index(drop=True)

    if len(df) < 3:
        raise ValueError("Número insuficiente de pontos para ajuste linear.")

    return df


# =========================================================
# AJUSTE LINEAR I–V E MÉTRICAS
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
    R2 = 1 - ss_res / ss_tot if ss_tot != 0 else np.nan

    return {
        "R_ohm": float(slope),
        "offset_v": float(intercept),
        "R2": float(R2),
        "V_pred": V_pred,
    }


# =========================================================
# CÁLCULO DA RESISTIVIDADE
# =========================================================
def compute_resistivity(
    R_ohm: float,
    thickness_m: float,
    geometry: str = "four_point_film",
) -> float:
    """
    geometry:
    - 'four_point_film' → filme fino (Smits, 1958)
    - 'bulk'            → material volumétrico
    """
    if geometry == "four_point_film":
        k = np.pi / np.log(2)
        return k * R_ohm * thickness_m

    elif geometry == "bulk":
        return R_ohm

    else:
        raise ValueError("Geometria inválida.")


# =========================================================
# CLASSIFICAÇÃO DO MATERIAL
# =========================================================
def classify_material(sigma_S_m: float) -> str:
    """
    Classificação física baseada na condutividade elétrica.
    """
    if not np.isfinite(sigma_S_m):
        return "Indefinido"

    if sigma_S_m >= 1e4:
        return "Condutor"
    if 1e-2 <= sigma_S_m < 1e4:
        return "Semicondutor"
    return "Isolante"


# =========================================================
# FUNÇÃO PRINCIPAL
# =========================================================
def process_resistivity(
    file_like,
    thickness_m: float,
    geometry: str = "four_point_film",
) -> Dict:
    """
    Pipeline completo de análise elétrica.
    """

    # -------------------------------
    # Leitura dos dados
    # -------------------------------
    df = read_iv_file(file_like)

    I = df["current_a"].values
    V = df["voltage_v"].values

    # -------------------------------
    # Ajuste linear
    # -------------------------------
    fit = linear_iv_fit(I, V)

    R = fit["R_ohm"]
    rho = compute_resistivity(R, thickness_m, geometry)
    sigma = 1.0 / rho if rho > 0 else np.nan
    classe = classify_material(sigma)

    # -------------------------------
    # Plot I × V
    # -------------------------------
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(I, V, color="black", s=25, label="Dados experimentais")
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
    # Retorno estruturado
    # -------------------------------
    return {
        "df": df,
        "fit": fit,
        "R_ohm": float(R),
        "rho_ohm_m": float(rho),
        "sigma_S_m": float(sigma),
        "classe": classe,
        "figure": fig,
    }
