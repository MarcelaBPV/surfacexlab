# =========================================================
# resistividade_processing.py
# SurfaceXLab — Electrical Physics Pipeline
# VERSÃO FINAL — DISSERTAÇÃO / ARTIGO
# =========================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.stats import linregress
from scipy.signal import savgol_filter

# =========================================================
# LEITURA UNIVERSAL
# =========================================================
def read_iv_file(file_like):

    try:

        df = pd.read_csv(

            file_like,

            sep=";",

            engine="python"
        )

    except:

        df = pd.read_csv(file_like)

    # =====================================================
    # NORMALIZA COLUNAS
    # =====================================================
    df.columns = [

        c.strip().lower()

        for c in df.columns
    ]

    voltage_col = None
    current_col = None
    resistance_col = None

    # =====================================================
    # BUSCA AUTOMÁTICA
    # =====================================================
    for col in df.columns:

        if (

            "voltage" in col or
            "volt" in col

        ):

            voltage_col = col

        if (

            "current" in col or
            "curr" in col or
            "amp" in col

        ):

            current_col = col

        if "resistance" in col:

            resistance_col = col

    # =====================================================
    # FALLBACK
    # =====================================================
    if voltage_col is None:

        voltage_col = df.columns[0]

    if current_col is None:

        current_col = df.columns[1]

    # =====================================================
    # SELECIONA
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
    # CONVERSÃO NUMÉRICA
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

            .str.replace(",", ".")

            .str.strip()
        )

        df[col] = pd.to_numeric(

            df[col],

            errors="coerce"
        )

    df = df.dropna()

    # =====================================================
    # REMOVE INF
    # =====================================================
    df = df[np.isfinite(df["I"])]
    df = df[np.isfinite(df["V"])]

    return df


# =========================================================
# SUAVIZAÇÃO
# =========================================================
def preprocess_iv_data(

    df,

    mode="voltage_sweep"
):

    df = df.copy()

    # =====================================================
    # ORDENA
    # =====================================================
    if mode == "voltage_sweep":

        df = df.sort_values("V")

    else:

        df = df.sort_values("I")

    # =====================================================
    # SAVITZKY GOLAY
    # =====================================================
    if len(df) > 15:

        if mode == "voltage_sweep":

            df["I_smooth"] = savgol_filter(

                df["I"],

                window_length=15,

                polyorder=3
            )

        else:

            df["V_smooth"] = savgol_filter(

                df["V"],

                window_length=15,

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
# RESISTIVIDADE 4P
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

    # =====================================================
    # LEITURA
    # =====================================================
    df = read_iv_file(file_like)

    # =====================================================
    # SUAVIZA
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
    slope, intercept, R2, Y_fit = robust_linear_fit(

        X_lin,

        Y_lin
    )

    # =====================================================
    # RESISTÊNCIA
    # =====================================================
    if mode == "voltage_sweep":

        resistance = 1 / slope

    else:

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
    sigma = 1 / rho

    # =====================================================
    # HISTERESE
    # =====================================================
    hysteresis = (

        np.max(df["I_smooth"]) -

        np.min(df["I_smooth"])
    )

    # =====================================================
    # RESUMO
    # =====================================================
    summary = {

        "Sample": sample_name,

        "Resistance_Ohm": resistance,

        "Resistivity_Ohm_m": rho,

        "Conductivity_S_m": sigma,

        "R_squared": R2,

        "Hysteresis": hysteresis
    }

    return {

        "dataframe": df,

        "summary": summary
    }


# =========================================================
# PLOT CIENTÍFICO
# =========================================================
def generate_group_plot(

    samples_dict,

    group_name="Acid",

    mode="voltage_sweep"
):

    fig, ax = plt.subplots(

        figsize=(7,5),

        dpi=600
    )

    # =====================================================
    # CORES
    # =====================================================
    colors = [

        "black",

        "red",

        "blue",

        "green",

        "purple"
    ]

    # =====================================================
    # LOOP
    # =====================================================
    for idx, (

        sample_name,

        df

    ) in enumerate(samples_dict.items()):

        # =================================================
        # VOLTAGE SWEEP
        # =================================================
        if mode == "voltage_sweep":

            ax.plot(

                df["V"],

                df["I_smooth"],

                linewidth=2.2,

                label=sample_name,

                color=colors[idx]
            )

        # =================================================
        # CURRENT SWEEP
        # =================================================
        else:

            ax.plot(

                df["I"],

                df["V_smooth"],

                linewidth=2.2,

                label=sample_name,

                color=colors[idx]
            )

    # =====================================================
    # LABELS
    # =====================================================
    if mode == "voltage_sweep":

        ax.set_xlabel(

            "Voltagem (V)",

            fontsize=14
        )

        ax.set_ylabel(

            "Corrente (A)",

            fontsize=14
        )

    else:

        ax.set_xlabel(

            "Corrente (A)",

            fontsize=14
        )

        ax.set_ylabel(

            "Voltagem (V)",

            fontsize=14
        )

    # =====================================================
    # FORMATAÇÃO CIENTÍFICA
    # =====================================================
    ax.ticklabel_format(

        style='sci',

        axis='y',

        scilimits=(0,0)
    )

    ax.tick_params(

        axis='both',

        labelsize=12
    )

    # =====================================================
    # REMOVE GRID
    # =====================================================
    ax.grid(False)

    # =====================================================
    # REMOVE BORDAS
    # =====================================================
    ax.spines['top'].set_visible(False)

    ax.spines['right'].set_visible(False)

    # =====================================================
    # EIXOS
    # =====================================================
    ax.axhline(

        y=0,

        linewidth=0.8
    )

    ax.axvline(

        x=0,

        linewidth=0.8
    )

    # =====================================================
    # LEGENDA
    # =====================================================
    ax.legend(

        frameon=False,

        fontsize=11
    )

    # =====================================================
    # AJUSTE
    # =====================================================
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

        if (

            k.upper().startswith("A")

            and

            "_D" in k.upper()
        )
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

        if (

            k.upper().startswith("B")

            and

            "_D" in k.upper()
        )
    }

    return generate_group_plot(

        alkaline_samples,

        group_name="Alkaline"
    )
