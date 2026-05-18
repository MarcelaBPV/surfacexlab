# =========================================================
# resistividade_processing.py
# SurfaceXLab — Electrical Physics Pipeline
# Compatível:
# - Agilent / Keysight SMU
# - Sweep de tensão
# - Sweep de corrente
# - Método 4 pontas
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

    # =====================================================
    # CSV
    # =====================================================
    df = pd.read_csv(

        file_like,

        sep=";",

        engine="python"
    )

    # =====================================================
    # NORMALIZA COLUNAS
    # =====================================================
    df.columns = [

        c.strip().lower()

        for c in df.columns
    ]

    # =====================================================
    # DETECÇÃO AUTOMÁTICA
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
    # CONVERSÃO CIENTÍFICA
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
    # REMOVE NAN/INF
    # =====================================================
    df = df.dropna()

    df = df[np.isfinite(df["I"])]

    df = df[np.isfinite(df["V"])]

    # =====================================================
    # VALIDAÇÃO
    # =====================================================
    if len(df) < 5:

        raise ValueError(
            "Poucos pontos experimentais."
        )

    return df


# =========================================================
# PRÉ PROCESSAMENTO
# =========================================================
def preprocess_iv_data(

    df,

    mode="voltage_sweep"
):

    df = df.copy()

    df = df.drop_duplicates()

    # =====================================================
    # ORDENA
    # =====================================================
    if mode == "voltage_sweep":

        df = df.sort_values("V")

    else:

        df = df.sort_values("I")

    # =====================================================
    # SUAVIZAÇÃO
    # =====================================================
    if len(df) > 11:

        if mode == "voltage_sweep":

            df["I_smooth"] = savgol_filter(

                df["I"],

                window_length=11,

                polyorder=3
            )

        else:

            df["V_smooth"] = savgol_filter(

                df["V"],

                window_length=11,

                polyorder=3
            )

    else:

        df["I_smooth"] = df["I"]

        df["V_smooth"] = df["V"]

    return df


# =========================================================
# REGIÃO ÔHMICA
# =========================================================
def detect_linear_region(

    df,

    mode="voltage_sweep"
):

    if mode == "voltage_sweep":

        X = df["V"].values

        Y = df["I_smooth"].values

    else:

        X = df["I"].values

        Y = df["V_smooth"].values

    # =====================================================
    # LIMPEZA
    # =====================================================
    mask = (
        np.isfinite(X) &
        np.isfinite(Y)
    )

    X = X[mask]

    Y = Y[mask]

    # =====================================================
    # REGIÃO CENTRAL
    # =====================================================
    lower = np.percentile(X, 20)

    upper = np.percentile(X, 80)

    mask_lin = (
        (X >= lower) &
        (X <= upper)
    )

    X_lin = X[mask_lin]

    Y_lin = Y[mask_lin]

    # =====================================================
    # FALLBACK
    # =====================================================
    if len(np.unique(X_lin)) < 2:

        return X, Y

    return X_lin, Y_lin


# =========================================================
# AJUSTE LINEAR
# =========================================================
def robust_linear_fit(

    X,

    Y
):

    mask = (
        np.isfinite(X) &
        np.isfinite(Y)
    )

    X = X[mask]

    Y = Y[mask]

    if len(X) < 2:

        raise ValueError(
            "Poucos pontos experimentais."
        )

    if np.std(X) == 0:

        raise ValueError(
            "Sweep inválido."
        )

    # =====================================================
    # REGRESSÃO
    # =====================================================
    result = linregress(X, Y)

    slope = result.slope

    intercept = result.intercept

    r_value = result.rvalue

    R2 = r_value ** 2

    Y_fit = (
        slope * X +
        intercept
    )

    return (

        slope,

        intercept,

        R2,

        Y_fit
    )


# =========================================================
# RESISTIVIDADE
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
# FEATURES
# =========================================================
def extract_surface_features(

    df,

    mode="voltage_sweep"
):

    # =====================================================
    # VOLTAGE SWEEP
    # =====================================================
    if mode == "voltage_sweep":

        X = df["V"].values

        Y = df["I_smooth"].values

        dY_dX = np.gradient(Y, X)

        d2Y_dX2 = np.gradient(
            dY_dX,
            X
        )

    # =====================================================
    # CURRENT SWEEP
    # =====================================================
    else:

        X = df["I"].values

        Y = df["V_smooth"].values

        dY_dX = np.gradient(Y, X)

        d2Y_dX2 = np.gradient(
            dY_dX,
            X
        )

    # =====================================================
    # NÃO LINEARIDADE
    # =====================================================
    denominator = np.mean(
        np.abs(dY_dX)
    )

    if denominator == 0:

        nonlinearity = 0

    else:

        nonlinearity = (
            np.std(dY_dX) /
            denominator
        )

    # =====================================================
    # FEATURES
    # =====================================================
    features = {

        "signal_max":
            np.max(Y),

        "signal_min":
            np.min(Y),

        "signal_mean":
            np.mean(Y),

        "signal_std":
            np.std(Y),

        "derivative_mean":
            np.mean(dY_dX),

        "derivative_std":
            np.std(dY_dX),

        "second_derivative_mean":
            np.mean(d2Y_dX2),

        "second_derivative_std":
            np.std(d2Y_dX2),

        "nonlinearity_index":
            nonlinearity
    }

    return features


# =========================================================
# PLOT
# =========================================================
def generate_publication_plot(

    df,

    X_fit,

    Y_fit,

    R2,

    sample_name,

    mode="voltage_sweep"
):

    fig, ax = plt.subplots(

        figsize=(6,4),

        dpi=300
    )

    # =====================================================
    # VOLTAGE SWEEP
    # =====================================================
    if mode == "voltage_sweep":

        ax.scatter(

            df["V"],

            df["I"],

            s=18,

            alpha=0.8,

            label="Experimental"
        )

        ax.plot(

            X_fit,

            Y_fit,

            linewidth=2,

            label=f"Linear Fit (R²={R2:.4f})"
        )

        ax.set_xlabel("Voltage (V)")

        ax.set_ylabel("Current (A)")

    # =====================================================
    # CURRENT SWEEP
    # =====================================================
    else:

        ax.scatter(

            df["I"],

            df["V"],

            s=18,

            alpha=0.8,

            label="Experimental"
        )

        ax.plot(

            X_fit,

            Y_fit,

            linewidth=2,

            label=f"Linear Fit (R²={R2:.4f})"
        )

        ax.set_xlabel("Current (A)")

        ax.set_ylabel("Voltage (V)")

    # =====================================================
    # FORMAT
    # =====================================================
    ax.ticklabel_format(

        style='sci',

        axis='both',

        scilimits=(0,0)
    )

    ax.set_title(
        f"{sample_name}"
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

    sample_name="F200",

    mode="voltage_sweep"
):

    # =====================================================
    # LEITURA
    # =====================================================
    df = read_iv_file(file_like)

    # =====================================================
    # PRÉ PROCESSAMENTO
    # =====================================================
    df = preprocess_iv_data(

        df,

        mode=mode
    )

    # =====================================================
    # REGIÃO LINEAR
    # =====================================================
    X_lin, Y_lin = detect_linear_region(

        df,

        mode=mode
    )

    # =====================================================
    # AJUSTE
    # =====================================================
    (
        slope,
        intercept,
        R2,
        Y_fit
    ) = robust_linear_fit(

        X_lin,

        Y_lin
    )

    # =====================================================
    # RESISTÊNCIA
    # =====================================================
    if mode == "voltage_sweep":

        # I = V/R
        resistance = (
            1 / slope
            if slope != 0
            else np.nan
        )

    else:

        # V = RI
        resistance = slope

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
    features = extract_surface_features(

        df,

        mode=mode
    )

    # =====================================================
    # ASSIMETRIA
    # =====================================================
    if mode == "voltage_sweep":

        positive_signal = np.mean(
            df[df["V"] > 0]["I_smooth"]
        )

        negative_signal = np.mean(
            np.abs(
                df[df["V"] < 0]["I_smooth"]
            )
        )

    else:

        positive_signal = np.mean(
            df[df["I"] > 0]["V_smooth"]
        )

        negative_signal = np.mean(
            np.abs(
                df[df["I"] < 0]["V_smooth"]
            )
        )

    if negative_signal == 0:

        asymmetry_index = 1

    else:

        asymmetry_index = (
            positive_signal /
            negative_signal
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
    # FÍSICA SUPERFICIAL
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
                "derivative_std"
            ],

        asymmetry_index=
            asymmetry_index
    )

    # =====================================================
    # FIGURA
    # =====================================================
    fig = generate_publication_plot(

        df=df,

        X_fit=X_lin,

        Y_fit=Y_fit,

        R2=R2,

        sample_name=sample_name,

        mode=mode
    )

    # =====================================================
    # SUMMARY
    # =====================================================
    summary = {

        "Sample":
            sample_name,

        "Measurement_Mode":
            mode,

        "Resistance_Ohm":
            resistance,

        "Resistivity_Ohm_m":
            rho,

        "Conductivity_S_m":
            sigma,

        "Sheet_Resistance_Ohm_sq":
            sheet_resistance,

        "Slope":
            slope,

        "Intercept":
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
