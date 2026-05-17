# =========================================================
# pca_processing.py
# SurfaceXLab — PCA Científico
# =========================================================

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


# =========================================================
# PCA
# =========================================================
def run_pca_analysis(df):

    # =====================================================
    # PREPARAÇÃO
    # =====================================================
    sample_col = df.columns[0]

    labels = df[sample_col]

    X = df.drop(columns=[sample_col])

    # remove °
    X = X.replace("°", "", regex=True)

    X = X.apply(pd.to_numeric)

    # =====================================================
    # NORMALIZAÇÃO
    # =====================================================
    scaler = StandardScaler()

    X_scaled = scaler.fit_transform(X)

    # =====================================================
    # PCA
    # =====================================================
    pca = PCA(n_components=2)

    scores = pca.fit_transform(X_scaled)

    loadings = pca.components_.T

    explained = (
        pca.explained_variance_ratio_ * 100
    )

    # =====================================================
    # FIGURA PUBLICAÇÃO
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
        s=90,
        linewidth=1.2
    )

    # labels
    for i, label in enumerate(labels):

        ax.text(
            scores[i, 0] + 0.04,
            scores[i, 1] + 0.04,
            label,
            fontsize=11
        )

    # =====================================================
    # LOADINGS
    # =====================================================
    scale = 2.2

    for i, var in enumerate(X.columns):

        ax.arrow(
            0,
            0,
            loadings[i, 0] * scale,
            loadings[i, 1] * scale,
            linewidth=1.6,
            head_width=0.05,
            length_includes_head=True
        )

        ax.text(
            loadings[i, 0] * scale * 1.12,
            loadings[i, 1] * scale * 1.12,
            var,
            fontsize=10,
            fontweight="bold"
        )

    # =====================================================
    # ESTILO CIENTÍFICO
    # =====================================================
    ax.axhline(
        0,
        linewidth=0.8
    )

    ax.axvline(
        0,
        linewidth=0.8
    )

    ax.set_xlabel(
        f"PC1 ({explained[0]:.1f}%)",
        fontsize=12
    )

    ax.set_ylabel(
        f"PC2 ({explained[1]:.1f}%)",
        fontsize=12
    )

    ax.set_title(
        "PCA Multimodal das Amostras Nanoestruturadas",
        fontsize=13,
        pad=15
    )

    # remove caixa
    ax.spines["top"].set_visible(False)

    ax.spines["right"].set_visible(False)

    # grid leve
    ax.grid(
        alpha=0.15,
        linestyle="--"
    )

    plt.tight_layout()

    # =====================================================
    # SAVE
    # =====================================================
    fig.savefig(
        "pca_publicacao.png",
        dpi=600,
        bbox_inches="tight"
    )

    # =====================================================
    # LOADINGS DF
    # =====================================================
    loadings_df = pd.DataFrame({

        "Variável": X.columns,
        "PC1": loadings[:, 0],
        "PC2": loadings[:, 1]

    })

    return {

        "fig": fig,

        "pc1": explained[0],

        "pc2": explained[1],

        "loadings": loadings_df
    }
