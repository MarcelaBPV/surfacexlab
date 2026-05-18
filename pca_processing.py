# =========================================================
# pca_processing.py
# SurfaceXLab — PCA Científico Publication Grade
# Compatível com:
# - Upload Manual
# - Integração Automática
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
    # VALIDAÇÃO INICIAL
    # =====================================================
    if df.empty:

        raise ValueError(
            "DataFrame vazio."
        )

    if "Amostra" not in df.columns:

        raise ValueError(
            "Coluna 'Amostra' não encontrada."
        )

    # =====================================================
    # PREPARAÇÃO
    # =====================================================
    labels = df["Amostra"].astype(str)

    X = df.drop(
        columns=["Amostra"]
    ).copy()

    # =====================================================
    # LIMPEZA DOS DADOS
    # =====================================================

    # remove espaços vazios
    X = X.replace(
        r'^\s*$',
        np.nan,
        regex=True
    )

    # remove °
    X = X.replace(
        "°",
        "",
        regex=True
    )

    # remove vírgula decimal
    X = X.replace(
        ",",
        ".",
        regex=True
    )

    # =====================================================
    # CONVERSÃO NUMÉRICA
    # =====================================================
    X = X.apply(
        pd.to_numeric,
        errors="coerce"
    )

    # =====================================================
    # REMOVE COLUNAS VAZIAS
    # =====================================================
    X = X.dropna(
        axis=1,
        how="all"
    )

    # =====================================================
    # REMOVE LINHAS VAZIAS
    # =====================================================
    valid_rows = ~X.isna().all(axis=1)

    X = X.loc[valid_rows]

    labels = labels.loc[valid_rows]

    # =====================================================
    # PREENCHIMENTO NaN
    # =====================================================
    X = X.fillna(0)

    # =====================================================
    # VALIDAÇÃO FINAL
    # =====================================================
    if X.shape[0] < 2:

        raise ValueError(
            "Número insuficiente de amostras."
        )

    if X.shape[1] < 2:

        raise ValueError(
            "Número insuficiente de variáveis."
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
    # FIGURA PUBLICATION GRADE
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

        linewidth=1.3,

        alpha=0.9,

        zorder=3
    )

    # =====================================================
    # LABELS DAS AMOSTRAS
    # =====================================================
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

        "PCA Multimodal",

        fontsize=13,

        pad=15
    )

    # =====================================================
    # ESTILO CIENTÍFICO
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

        "pca_multimodal.png",

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
    # SCORES DATAFRAME
    # =====================================================
    scores_df = pd.DataFrame({

        "Amostra": labels,

        "PC1": np.round(
            scores[:, 0],
            4
        ),

        "PC2": np.round(
            scores[:, 1],
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

        "scores": scores_df,

        "explained_variance": explained
    }
