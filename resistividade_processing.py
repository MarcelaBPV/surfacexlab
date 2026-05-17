# =========================================================
# resistividade_processing.py
# SurfaceXLab — Pipeline Elétrico Avançado
# Método 4 Pontas — Física Correta
# =========================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.stats import linregress
from scipy.signal import savgol_filter

from resistividade_database import (

    classify_resistivity,

    infer_surface_physics
)


# =========================================================
# LEITURA UNIVERSAL
# =========================================================
def read_iv_file(file_like):

    try:

        df = pd.read_csv(
            file_like,
            sep=None,
            engine="python"
        )

    except:

        df = pd.read_table(file_like)

    # =====================================================
    # NORMALIZA COLUNAS
    # =====================================================
    df.columns = [
        c.lower().strip()
        for c in df.columns
    ]

    # =====================================================
    # DETECÇÃO AUTOMÁTICA
    # =====================================================
    current_cols = [

        c for c in df.columns

        if (
            "i" in c or
            "current" in c
        )
    ]

    voltage_cols = [

        c for c in df.columns

        if (
            "v" in c or
            "voltage" in c
        )
    ]

    if not current_cols:

        raise ValueError(
            """
            Coluna de corrente não encontrada.
            """
        )

    if not voltage_cols:

        raise ValueError(
            """
            Coluna de tensão não encontrada.
            """
        )

    # =====================================================
    # SELEÇÃO
    # =====================================================
    df = df[[
        current_cols[0],
        voltage_cols[0]
    ]]

    df.columns = ["I", "V"]

    # =====================================================
    # CONVERSÃO
    # =====================================================
    df["I"] = pd.to_numeric(
        df["I"],
        errors="coerce"
    )

    df["V"] = pd.to_numeric(
        df["V"],
        errors="coerce"
    )

    # =====================================================
    # LIMPEZA
    # =====================================================
    df = df.dropna()

    df = df[np.isfinite(df["I"])]
    df = df[np.isfinite(df["V"])]

    if len(df) < 3:

        raise ValueError(
            """
            Poucos pontos experimentais.
            """
        )

    return df


# =========================================================
# PRÉ PROCESSAMENTO
# =========================================================
def preprocess_iv_data(df):

    df = df.copy()

    # =====================================================
    # REMOVE DUPLICADOS
    # =====================================================
    df = df.drop_duplicates()

    # =====================================================
    # ORDENA CORRENTE
    # =====================================================
    df = df.sort_values("I")

    # =====================================================
    # SUAVIZAÇÃO
    # =====================================================
    if len(df) > 11:

        df["V_smooth"] = savgol_filter(

            df["V"],

            window_length=11,

            polyorder=3
        )

    else:

        df["V_smooth"] = df["V"]

    return df


# =========================================================
# DETECÇÃO REGIÃO ÔHMICA
# =========================================================
def detect_linear_region(df):

    I = df["I"].values

    V = df["V_smooth"].values

    # =====================================================
    # LIMPEZA
    # =====================================================
    mask = (
        np.isfinite(I) &
        np.isfinite(V)
    )

    I = I[mask]
    V = V[mask]

    # =====================================================
    # DADOS INSUFICIENTES
    # =====================================================
    if len(I) < 5:

        return I, V

    # =====================================================
    # REGIÃO CENTRAL
    # =====================================================
    lower = np.percentile(I, 20)

    upper = np.percentile(I, 80)

    mask_lin = (
        (I >= lower) &
        (I <= upper)
    )

    I_lin = I[mask_lin]

    V_lin = V[mask_lin]

    # =====================================================
    # FALLBACK
    # =====================================================
    if len(np.unique(I_lin)) < 2:

        return I, V

    return I_lin, V_lin


# =========================================================
# AJUSTE 4 PONTAS
# V = f(I)
# =========================================================
def robust_linear_fit(I, V):

    # =====================================================
    # LIMPEZA
    # =====================================================
    mask = (
        np.isfinite(I) &
        np.isfinite(V)
    )

    I = I[mask]

    V = V[mask]

    # =====================================================
    # VALIDAÇÃO
    # =====================================================
    if len(I) < 2:

        raise ValueError(
            """
            Poucos pontos para regressão.
            """
        )

    if np.std(I) == 0:

        raise ValueError(
            """
            Corrente constante detectada.

            O método 4 pontas requer
            sweep de corrente.
            """
        )

    # =====================================================
    # REGRESSÃO
    # =====================================================
    result = linregress(I, V)

    slope = result.slope

    intercept = result.intercept

    r_value = result.rvalue

    R2 = r_value ** 2

    # =====================================================
    # RESISTÊNCIA
    # =====================================================
    resistance = slope

    # =====================================================
    # CURVA AJUSTADA
    # =====================================================
    V_fit = (
        slope * I +
        intercept
    )

    return (

        slope,

        intercept,

        R2,

        V_fit,

        resistance
    )


# =========================================================
# RESISTIVIDADE 4 PONTAS
# =========================================================
def calculate_resistivity_4p(
    resistance,
    thickness
):

    correction_factor = (
        np.pi / np.log(2)
    )

    rho = (
        correction_factor *
        resistance *
        thickness
    )

    return rho


# =========================================================
# FEATURES SUPERFICIAIS
# =========================================================
def extract_surface_features(df):

    I = df["I"].values

    V = df["V_smooth"].values

    # =====================================================
    # DERIVADAS
    # =====================================================
    dV_dI = np.gradient(V, I)

    d2V_dI2 = np.gradient(dV_dI, I)

    # =====================================================
    # NÃO LINEARIDADE
    # =====================================================
    denominator = np.mean(
        np.abs(dV_dI)
    )

    if denominator == 0:

        nonlinearity = 0

    else:

        nonlinearity = (
            np.std(dV_dI) /
            denominator
        )

    # =====================================================
    # FEATURES
    # =====================================================
    features = {

        "V_max":
            np.max(V),

        "V_min":
            np.min(V),

        "V_mean":
            np.mean(V),

        "V_std":
            np.std(V),

        "dV_dI_mean":
            np.mean(dV_dI),

        "dV_dI_std":
            np.std(dV_dI),

        "d2V_dI2_mean":
            np.mean(d2V_dI2),

        "d2V_dI2_std":
            np.std(d2V_dI2),

        "nonlinearity_index":
            nonlinearity
    }

    return features


# =========================================================
# PLOT CIENTÍFICO
# =========================================================
def generate_publication_plot(

    df,

    I_fit,

    V_fit,

    R2,

    sample_name
):

    fig, ax = plt.subplots(

        figsize=(6,4),

        dpi=300
    )

    # =====================================================
    # EXPERIMENTAL
    # =====================================================
    ax.scatter(

        df["I"],

        df["V"],

        s=18,

        alpha=0.8,

        label="Experimental"
    )

    # =====================================================
    # AJUSTE
    # =====================================================
    ax.plot(

        I_fit,

        V_fit,

        linewidth=2,

        label=f"Linear Fit (R²={R2:.4f})"
    )

    # =====================================================
    # LABELS
    # =====================================================
    ax.set_xlabel("Current (A)")

    ax.set_ylabel("Voltage (V)")

    ax.set_title(
        f"4-Point Probe — {sample_name}"
    )

    ax.grid(alpha=0.25)

    ax.legend(frameon=False)

    plt.tight_layout()

    return fig


# =========================================================
# PIPELINE PRINCIPAL
# =========================================================
def process_resistivity(

    file_like,

    thickness_m,

    sample_name="F200"
):

    # =====================================================
    # LEITURA
    # =====================================================
    df = read_iv_file(file_like)

    # =====================================================
    # PRÉ PROCESSAMENTO
    # =====================================================
    df = preprocess_iv_data(df)

    # =====================================================
    # REGIÃO ÔHMICA
    # =====================================================
    I_lin, V_lin = detect_linear_region(df)

    # =====================================================
    # AJUSTE
    # =====================================================
    (
        slope,
        intercept,
        R2,
        V_fit,
        resistance
    ) = robust_linear_fit(

        I_lin,

        V_lin
    )

    # =====================================================
    # RESISTIVIDADE
    # =====================================================
    rho = calculate_resistivity_4p(

        resistance,

        thickness_m
    )

    # =====================================================
    # CONDUTIVIDADE
    # =====================================================
    sigma = (

        1 / rho

        if rho > 0

        else np.nan
    )

    # =====================================================
    # RESISTÊNCIA DE FOLHA
    # =====================================================
    sheet_resistance = (
        rho / thickness_m
    )

    # =====================================================
    # FEATURES
    # =====================================================
    features = extract_surface_features(df)

    # =====================================================
    # ASSIMETRIA
    # =====================================================
    positive_voltage = np.mean(
        df[df["I"] > 0]["V_smooth"]
    )

    negative_voltage = np.mean(
        np.abs(
            df[df["I"] < 0]["V_smooth"]
        )
    )

    if negative_voltage == 0:

        asymmetry_index = 1

    else:

        asymmetry_index = (
            positive_voltage /
            negative_voltage
        )

    # =====================================================
    # CLASSIFICAÇÃO
    # =====================================================
    classification = classify_resistivity(

        resistivity=rho,

        r_squared=R2,

        slope=slope
    )

    # =====================================================
    # INTERPRETAÇÃO FÍSICO-QUÍMICA
    # =====================================================
    surface_physics = infer_surface_physics(

        resistivity=rho,

        r_squared=R2,

        slope=slope,

        nonlinearity_index=
            features[
                "nonlinearity_index"
            ],

        dI_dV_std=
            features[
                "dV_dI_std"
            ],

        asymmetry_index=
            asymmetry_index
    )

    # =====================================================
    # FIGURA
    # =====================================================
    fig = generate_publication_plot(

        df=df,

        I_fit=I_lin,

        V_fit=V_fit,

        R2=R2,

        sample_name=sample_name
    )

    # =====================================================
    # SUMMARY
    # =====================================================
    summary = {

        "Sample":
            sample_name,

        "Resistance_Ohm":
            resistance,

        "Resistivity_Ohm_m":
            rho,

        "Conductivity_S_m":
            sigma,

        "Sheet_Resistance_Ohm_sq":
            sheet_resistance,

        "Slope_V_A":
            slope,

        "Intercept_V":
            intercept,

        "R_squared":
            R2,

        **classification,

        **features,

        **surface_physics
    }

    # =====================================================
    # RETURN
    # =====================================================
    return {

        "dataframe":
            df,

        "summary":
            summary,

        "figure":
            fig,

        "features":
            features
    }
