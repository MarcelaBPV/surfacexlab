# =========================================================
# perfilometria_processing.py
# SurfaceXLab — Perfilometria Científica
# =========================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from scipy.signal import savgol_filter


# =========================================================
# LEITURA UNIVERSAL
# =========================================================
def read_profilometry_file(file_like):

    name = file_like.name.lower()

    # =====================================================
    # EXCEL
    # =====================================================
    if name.endswith((".xlsx", ".xls")):

        df = pd.read_excel(file_like)

    else:

        df = pd.read_csv(file_like)

    # normaliza colunas
    df.columns = [
        str(c).strip().lower()
        for c in df.columns
    ]

    # =====================================================
    # DETECÇÃO AUTOMÁTICA
    # =====================================================
    z_col = None

    for c in df.columns:

        if (
            "z" in c or
            "height" in c or
            "rough" in c or
            "profile" in c
        ):
            z_col = c
            break

    # fallback
    if z_col is None:
        z_col = df.columns[-1]

    z = pd.to_numeric(
        df[z_col],
        errors="coerce"
    )

    z = z.dropna()

    return z.reset_index(drop=True)


# =========================================================
# CÁLCULO DE RUGOSIDADE
# =========================================================
def calculate_roughness(z):

    # remove inclinação
    z_centered = z - np.mean(z)

    # suavização
    if len(z_centered) > 11:

        z_smooth = savgol_filter(
            z_centered,
            11,
            3
        )

    else:

        z_smooth = z_centered

    # =====================================================
    # PARÂMETROS
    # =====================================================
    Ra = np.mean(np.abs(z_smooth))

    Rq = np.sqrt(
        np.mean(z_smooth ** 2)
    )

    Rz = np.max(z_smooth) - np.min(z_smooth)

    Rp = np.max(z_smooth)

    Rv = np.min(z_smooth)

    return {

        "Ra": Ra,

        "Rq": Rq,

        "Rz": Rz,

        "Rp": Rp,

        "Rv": Rv,

        "profile": z_smooth
    }


# =========================================================
# CLASSIFICAÇÃO SUPERFICIAL
# =========================================================
def classify_surface(Ra):

    if Ra < 0.5:
        return "Superfície lisa"

    elif Ra < 2:
        return "Baixa rugosidade"

    elif Ra < 5:
        return "Rugosidade moderada"

    else:
        return "Alta rugosidade"


# =========================================================
# PLOT PERFIL
# =========================================================
def generate_profile_plot(profile, sample_name):

    fig, ax = plt.subplots(
        figsize=(7,4),
        dpi=300
    )

    ax.plot(
        profile,
        linewidth=1.5
    )

    ax.set_xlabel("Posição")

    ax.set_ylabel("Altura (µm)")

    ax.set_title(
        f"Perfil Topográfico — {sample_name}"
    )

    ax.grid(alpha=0.3)

    plt.tight_layout()

    return fig


# =========================================================
# PIPELINE PRINCIPAL
# =========================================================
def process_profilometry(file_like):

    # =====================================================
    # LEITURA
    # =====================================================
    z = read_profilometry_file(file_like)

    # =====================================================
    # RUGOSIDADE
    # =====================================================
    roughness = calculate_roughness(z)

    # =====================================================
    # CLASSIFICAÇÃO
    # =====================================================
    surface_class = classify_surface(
        roughness["Ra"]
    )

    # =====================================================
    # FIGURA
    # =====================================================
    fig = generate_profile_plot(
        roughness["profile"],
        file_like.name
    )

    # =====================================================
    # SUMMARY
    # =====================================================
    summary = {

        "Amostra":
            file_like.name,

        "Ra (µm)":
            round(roughness["Ra"], 3),

        "Rq (µm)":
            round(roughness["Rq"], 3),

        "Rz (µm)":
            round(roughness["Rz"], 3),

        "Rp (µm)":
            round(roughness["Rp"], 3),

        "Rv (µm)":
            round(roughness["Rv"], 3),

        "Classe":
            surface_class
    }

    return {

        "summary": summary,

        "figure": fig,

        "profile": roughness["profile"]
    }
