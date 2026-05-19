# =========================================================
# resistividade_processing.py
# SurfaceXLab — Electrical Physics Pipeline
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

    df = pd.read_csv(

        file_like,

        sep=";",

        engine="python"
    )

    df.columns = [

        c.strip().lower()

        for c in df.columns
    ]

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

    if voltage_col is None:

        raise ValueError(
            "Coluna de tensão não encontrada."
        )

    if current_col is None:

        raise ValueError(
            "Coluna de corrente não encontrada."
        )

    selected_cols = [

        current_col,

        voltage_col
    ]

    if resistance_col is not None:

        selected_cols.append(
            resistance_col
        )

    df = df[selected_cols]

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

            .str.replace(",", ".")
        )

        df[col] = pd.to_numeric(

            df[col],

            errors="coerce"
        )

    df = df.dropna()

    df = df[np.isfinite(df["I"])]
    df = df[np.isfinite(df["V"])]

    return df


# =========================================================
# PRÉ PROCESSAMENTO
# =========================================================
def preprocess_iv_data(

    df,

    mode="voltage_sweep"
):

    df = df.copy()

    if mode == "voltage_sweep":

        df = df.sort_values("V")

    else:

        df = df.sort_values("I")

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
# REGIÃO LINEAR
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

    lower = np.percentile(X, 20)

    upper = np.percentile(X, 80)

    mask_lin = (

        (X >= lower) &

        (X <= upper)
    )

    return X[mask_lin], Y[mask_lin]


# =========================================================
# AJUSTE LINEAR
# =========================================================
def robust_linear_fit(

    X,

    Y
):

    result = linregress(X, Y)

    slope = result.slope

    intercept = result.intercept

    R2 = result.rvalue ** 2

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

    return (

        correction_factor *

        resistance *

        thickness
    )


# =========================================================
# PROCESSAMENTO PRINCIPAL
# =========================================================
def process_resistivity(

    file_like,

    thickness_m,

    sample_name="F200",

    mode="voltage_sweep"
):

    df = read_iv_file(file_like)

    df = preprocess_iv_data(

        df,

        mode=mode
    )

    X_lin, Y_lin = detect_linear_region(

        df,

        mode=mode
    )

    slope, intercept, R2, Y_fit = robust_linear_fit(

        X_lin,

        Y_lin
    )

    if mode == "voltage_sweep":

        resistance = 1 / slope

    else:

        resistance = slope

    rho = calculate_resistivity_4p(

        resistance,

        thickness_m
    )

    sigma = 1 / rho

    summary = {

        "Sample": sample_name,

        "Resistance_Ohm": resistance,

        "Resistivity_Ohm_m": rho,

        "Conductivity_S_m": sigma,

        "R_squared": R2
    }

    return {

        "dataframe": df,

        "summary": summary
    }


# =========================================================
# PLOT MULTI CURVAS
# =========================================================
def generate_group_plot(

    samples_dict,

    group_name="Acid",

    mode="voltage_sweep"
):

    fig, ax = plt.subplots(

        figsize=(7,5),

        dpi=300
    )

    for sample_name, df in samples_dict.items():

        if mode == "voltage_sweep":

            ax.plot(

                df["V"],

                df["I_smooth"],

                linewidth=2,

                label=sample_name
            )

        else:

            ax.plot(

                df["I"],

                df["V_smooth"],

                linewidth=2,

                label=sample_name
            )

    if mode == "voltage_sweep":

        ax.set_xlabel("Tensão (V)")

        ax.set_ylabel("Corrente (A)")

    else:

        ax.set_xlabel("Corrente (A)")

        ax.set_ylabel("Tensão (V)")

    ax.ticklabel_format(

        style='sci',

        axis='both',

        scilimits=(0,0)
    )

    ax.axhline(

        y=0,

        linestyle="--",

        linewidth=0.8,

        alpha=0.4
    )

    ax.axvline(

        x=0,

        linestyle="--",

        linewidth=0.8,

        alpha=0.4
    )

    if group_name.lower() == "acid":

        ax.set_title(
            "Curvas I×V — Grupo Ácido"
        )

    else:

        ax.set_title(
            "Curvas I×V — Grupo Alcalino"
        )

    ax.grid(alpha=0.25)

    ax.legend(

        frameon=False,

        fontsize=10
    )

    plt.tight_layout()

    return fig


# =========================================================
# FIGURA 28
# =========================================================
def build_acid_group_plot(

    processed_samples
):

    acid_samples = {

        k: v["dataframe"]

        for k, v in processed_samples.items()

        if k.upper().startswith("A")
    }

    return generate_group_plot(

        acid_samples,

        group_name="Acid"
    )


# =========================================================
# FIGURA 29
# =========================================================
def build_alkaline_group_plot(

    processed_samples
):

    alkaline_samples = {

        k: v["dataframe"]

        for k, v in processed_samples.items()

        if k.upper().startswith("B")
    }

    return generate_group_plot(

        alkaline_samples,

        group_name="Alkaline"
    )
