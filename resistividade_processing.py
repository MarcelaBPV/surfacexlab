# =========================================================
# PROCESSAMENTO ELÉTRICO — SURFACEXLAB
# F200 SCIENTIFIC PIPELINE
# =========================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.stats import linregress
from scipy.signal import savgol_filter

from resistividade_database import classify_resistivity


# =========================================================
# LEITURA UNIVERSAL
# =========================================================
def read_iv_file(file_like):

    try:
        df = pd.read_csv(file_like, sep=None, engine="python")

    except:
        df = pd.read_table(file_like)

    df.columns = [c.lower().strip() for c in df.columns]

    # ---------------------------------------------
    # DETECÇÃO AUTOMÁTICA
    # ---------------------------------------------
    current_cols = [
        c for c in df.columns
        if ("i" in c or "current" in c)
    ]

    voltage_cols = [
        c for c in df.columns
        if ("v" in c or "voltage" in c)
    ]

    if not current_cols or not voltage_cols:
        raise ValueError(
            "Colunas de corrente/tensão não encontradas."
        )

    df = df[[current_cols[0], voltage_cols[0]]]

    df.columns = ["I", "V"]

    # ---------------------------------------------
    # CONVERSÃO
    # ---------------------------------------------
    df["I"] = pd.to_numeric(df["I"], errors="coerce")
    df["V"] = pd.to_numeric(df["V"], errors="coerce")

    df = df.dropna()

    # remove infinitos
    df = df[np.isfinite(df["I"])]
    df = df[np.isfinite(df["V"])]

    return df


# =========================================================
# PRÉ-PROCESSAMENTO
# =========================================================
def preprocess_iv_data(df):

    df = df.copy()

    # remove duplicados
    df = df.drop_duplicates()

    # ordena tensão
    df = df.sort_values("V")

    # suavização científica
    if len(df) > 11:

        df["I_smooth"] = savgol_filter(
            df["I"],
            window_length=11,
            polyorder=3
        )

    else:
        df["I_smooth"] = df["I"]

    return df


# =========================================================
# DETECÇÃO REGIÃO ÔHMICA
# =========================================================
def detect_linear_region(df):

    V = df["V"].values
    I = df["I_smooth"].values

    # região central física
    lower = np.percentile(V, 20)
    upper = np.percentile(V, 80)

    mask = (V >= lower) & (V <= upper)

    return V[mask], I[mask]


# =========================================================
# AJUSTE ROBUSTO
# =========================================================
def robust_linear_fit(V, I):

    result = linregress(V, I)

    slope = result.slope
    intercept = result.intercept
    r_value = result.rvalue

    R2 = r_value ** 2

    I_fit = slope * V + intercept

    return slope, intercept, R2, I_fit


# =========================================================
# RESISTIVIDADE 4 PONTAS
# =========================================================
def calculate_resistivity_4p(R, thickness):

    correction_factor = np.pi / np.log(2)

    rho = correction_factor * R * thickness

    return rho


# =========================================================
# FEATURES SUPERFICIAIS
# =========================================================
def extract_surface_features(df):

    V = df["V"].values
    I = df["I_smooth"].values

    dI_dV = np.gradient(I, V)

    features = {

        "I_max": np.max(I),

        "I_min": np.min(I),

        "I_mean": np.mean(I),

        "I_std": np.std(I),

        "dI_dV_mean": np.mean(dI_dV),

        "dI_dV_std": np.std(dI_dV),

        "nonlinearity_index": (
            np.std(dI_dV) /
            np.mean(np.abs(dI_dV))
        )
    }

    return features


# =========================================================
# PLOT PAPER
# =========================================================
def generate_publication_plot(
    df,
    V_fit,
    I_fit,
    R2,
    sample_name
):

    fig, ax = plt.subplots(
        figsize=(6,4),
        dpi=300
    )

    ax.scatter(
        df["V"],
        df["I"],
        s=14,
        alpha=0.8,
        label="Experimental"
    )

    ax.plot(
        V_fit,
        I_fit,
        linewidth=2,
        label=f"Linear Fit (R²={R2:.4f})"
    )

    ax.set_xlabel("Voltage (V)")
    ax.set_ylabel("Current (A)")

    ax.set_title(
        f"I–V Curve — {sample_name}"
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
    V_lin, I_lin = detect_linear_region(df)

    # =====================================================
    # AJUSTE
    # =====================================================
    slope, intercept, R2, I_fit = robust_linear_fit(
        V_lin,
        I_lin
    )

    # =====================================================
    # RESISTÊNCIA
    # =====================================================
    R = np.nan

    if slope != 0:
        R = 1 / slope

    # =====================================================
    # RESISTIVIDADE
    # =====================================================
    rho = calculate_resistivity_4p(
        R,
        thickness_m
    )

    sigma = 1 / rho if rho > 0 else np.nan

    sheet_resistance = (
        rho / thickness_m
    )

    # =====================================================
    # FEATURES
    # =====================================================
    features = extract_surface_features(df)

    # =====================================================
    # CLASSIFICAÇÃO FÍSICA
    # =====================================================
    classification = classify_resistivity(
        resistivity=rho,
        r_squared=R2,
        slope=slope
    )

    # =====================================================
    # PLOT
    # =====================================================
    fig = generate_publication_plot(
        df=df,
        V_fit=V_lin,
        I_fit=I_fit,
        R2=R2,
        sample_name=sample_name
    )

    # =====================================================
    # SUMMARY
    # =====================================================
    summary = {

        "Sample": sample_name,

        "Resistance_Ohm": R,

        "Resistivity_Ohm_m": rho,

        "Conductivity_S_m": sigma,

        "Sheet_Resistance_Ohm_sq":
            sheet_resistance,

        "Slope_A_V": slope,

        "Intercept_A": intercept,

        "R_squared": R2,

        **classification,

        **features
    }

    return {

        "dataframe": df,

        "summary": summary,

        "figure": fig,

        "features": features
    }
