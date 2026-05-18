# =========================================================
# pca_processing.py
# SurfaceXLab — PCA Científico Publication Grade
# =========================================================

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


# =========================================================
# PCA MULTIMODAL
# =========================================================
def run_pca_analysis(df):

    # =====================================================
    # VALIDAÇÃO DAS COLUNAS
    # =====================================================
    required_cols = [

        "Amostra",
        "Rrms",
        "ID_IG",
        "I2D_IG",
        "Theta"

    ]

    missing = [

        c for c in required_cols
        if c not in df.columns

    ]

    if missing:

        raise ValueError(
            f"Colunas ausentes: {missing}"
        )

    # =====================================================
    # PREPARAÇÃO
    # =====================================================
    labels = df["Amostra"].astype(str)

    X = df[[
        "Rrms",
        "ID_IG",
        "I2D_IG",
        "Theta"
    ]].copy()

    # =====================================================
    # LIMPEZA DOS DADOS
    # =====================================================

    # remove espaços vazios
    X = X.replace(
        r'^\s*$',
        np.nan,
        regex=True
    )

    # remove símbolo °
    X = X.replace(
        "°",
        "",
        regex=True
    )

    # converte numérico
    X = X.apply(
        pd.to_numeric,
        errors="coerce"
    )

    # remove linhas totalmente vazias
    X = X.dropna(
        how="all"
    )

    # garante alinhamento labels
    labels = labels.loc[X.index]

    # substitui NaN restantes
    X = X.fillna(0)

    # =====================================================
    # VALIDAÇÃO FINAL
    # =====================================================
    if len(X) < 2:

        raise ValueError(
            "Número insuficiente de amostras válidas."
        )

    # =====================================================
    # NORMALIZAÇÃO
    # =====================================================
    scaler = StandardScaler()

    X_scaled = scaler.fit_transform(X)

    # =====================================================
    # PCA
    # =====================================================
    pca = PCA(
        n_components=2
    )

    scores = pca.fit_transform(
        X_scaled
    )

    loadings = pca.components_.T

    explained = (
        pca.explained_variance_ratio_ * 100
    )

    # =====================================================
    # FIGURA CIENTÍFICA
    # =====================================================
    fig, ax = plt.subplots(
        figsize=(7, 6),
        dpi=500
    )

    # =====================================================
    # SCORES
    # =====================================================
    ax.scatter(
        scores[:, 0],
        scores[:, 1],
        s=120,
        linewidth=1.2,
        alpha=0.9,
        zorder=3
    )

    # labels amostras
    for i, label in enumerate(labels):

        ax.text(
            scores[i, 0] + 0.05,
            scores[i, 1] + 0.05,
            label,
            fontsize=11
        )

    # =====================================================
    # LOADINGS
    # =====================================================
    scale = 2.4

    for i, var in enumerate(X.columns):

        ax.arrow(
            0,
            0,
            loadings[i, 0] * scale,
            loadings[i, 1] * scale,
            linewidth=1.8,
            head_width=0.06,
            length_includes_head=True,
            zorder=2
        )

        ax.text(
            loadings[i, 0] * scale * 1.15,
            loadings[i, 1] * scale * 1.15,
            var,
            fontsize=10,
            fontweight="bold"
        )

    # =====================================================
    # EIXOS
    # =====================================================
    ax.axhline(
        0,
        linewidth=0.8
    )

    ax.axvline(
        0,
        linewidth=0.8
    )

    # =====================================================
    # LABELS
    # =====================================================
    ax.set_xlabel(
        f"PC1 ({explained[0]:.1f}%)",
        fontsize=12
    )

    ax.set_ylabel(
        f"PC2 ({explained[1]:.1f}%)",
        fontsize=12
    )

    ax.set_title(
        "PCA Multimodal — Nanotubos de Carbono",
        fontsize=13,
        pad=15
    )

    # =====================================================
    # ESTILO PUBLICAÇÃO
    # =====================================================
    ax.spines["top"].set_visible(False)

    ax.spines["right"].set_visible(False)

    ax.grid(
        alpha=0.15,
        linestyle="--"
    )

    plt.tight_layout()

    # =====================================================
    # SAVE FIGURE
    # =====================================================
    fig.savefig(
        "pca_nanotubos.png",
        dpi=600,
        bbox_inches="tight"
    )

    # =====================================================
    # LOADINGS DATAFRAME
    # =====================================================
    loadings_df = pd.DataFrame({

        "Variável": X.columns,

        "PC1": np.round(
            loadings[:, 0],
            4
        ),

        "PC2": np.round(
            loadings[:, 1],
            4
        )

    })

    # =====================================================
    # RETURN
    # =====================================================
    return {

        "fig": fig,

        "pc1": explained[0],

        "pc2": explained[1],

        "loadings": loadings_df,

        "scores": scores,

        "explained_variance": explained
    }
