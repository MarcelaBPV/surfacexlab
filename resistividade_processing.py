# =========================================================
# resistividade_processing.py
# SurfaceXLab — Pipeline Elétrico Avançado
# Método 4 Pontas + Caracterização Interfacial
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
# LEITURA UNIVERSAL AVANÇADA
# =========================================================
def read_iv_file(file_like):

    # =====================================================
    # LEITURA CSV
    # =====================================================
    df = pd.read_csv(

        file_like,

        sep=";",

        engine="python"
    )

    # =====================================================
    # NORMALIZA NOMES
    # =====================================================
    df.columns = [

        c.strip().lower()

        for c in df.columns
    ]

    # =====================================================
    # DETECÇÃO COLUNAS
    # =====================================================
    voltage_col = None

    current_col = None

    resistance_col = None

    for col in df.columns:

        if "voltage" in col:

            voltage_col = col

        if "current" in col:

            current_col = col

        if "resistance" in col:

            resistance_col = col

    # =====================================================
    # VALIDAÇÃO
    # =====================================================
    if voltage_col is None:

        raise ValueError(
            "Coluna de tensão não encontrada."
        )

    if current_col is None:

        raise ValueError(
            "Coluna de corrente não encontrada."
        )

    # =====================================================
    # SELEÇÃO
    # =====================================================
    selected_cols = [

        current_col,

        voltage_col
    ]

    if resistance_col is not None:

        selected_cols.append(
            resistance_col
        )

    df = df[selected_cols]

    # =====================================================
    # RENOMEIA
    # =====================================================
    rename_dict = {

        current_col: "I",

        voltage_col: "V"
    }

    if resistance_col is not None:

        rename_dict[
            resistance_col
        ] = "R_inst"

    df = df.rename(
        columns=rename_dict
    )

    # =====================================================
    # CORREÇÃO FORMATO CIENTÍFICO
    # =====================================================
    numeric_cols = ["I", "V"]

    if "R_inst" in df.columns:

        numeric_cols.append(
            "R_inst"
        )

    for col in numeric_cols:

        df[col] = (

            df[col]

            .astype(str)

            .str.strip()

            # converte espaço decimal
            .str.replace(
                r'(?<=\d)\s+(?=\d)',
                '.',
                regex=True
            )

            .str.replace(",", ".")
        )

        df[col] = pd.to_numeric(

            df[col],

            errors="coerce"
        )

    # =====================================================
    # REMOVE NAN
    # =====================================================
    df = df.dropna()

    # =====================================================
    # REMOVE INF
    # =====================================================
    df = df[np.isfinite(df["I"])]

    df = df[np.isfinite(df["V"])]

    # =====================================================
    # CONVERSÃO AUTOMÁTICA UNIDADES
    # =====================================================
    max_current = np.abs(df["I"]).max()

    # µA -> A
    if max_current > 1:

        df["I"] = (
            df["I"] * 1e-6
        )

    # =====================================================
    # VALIDAÇÃO
    # =====================================================
    if len(df) < 5:

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
            Sweep de corrente inválido.
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
    # AJUSTE
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
# FIGURA CIENTÍFICA
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
    # FORMAT
    # =====================================================
    ax.ticklabel_format(

        style='sci',

        axis='both',

        scilimits=(0,0)
    )

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
    # RESISTÊNCIA FOLHA
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
    # INTERPRETAÇÃO SUPERFICIAL
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
