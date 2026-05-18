# =========================================================
# pca_processing.py
# SurfaceXLab — PCA Científico
# Estilo Origin / Publication Grade
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

    # remove símbolo °
    X = X.replace(
        "°",
        "",
        regex=True
    )

    # troca vírgula decimal
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
    # FIGURA CIENTÍFICA — ESTILO ORIGIN
    # =====================================================
    fig, ax = plt.subplots(

        figsize=(6, 6),

        dpi=600
    )

    # fundo branco
    fig.patch.set_facecolor("white")

    ax.set_facecolor("white")

    # =====================================================
    # SCORES
    # =====================================================
    ax.scatter(

        scores[:, 0],

        scores[:, 1],

        s=55,

        linewidth=0.8,

        zorder=3
    )

    # =====================================================
    # LABELS DAS AMOSTRAS
    # =====================================================
    for i, label in enumerate(labels):

        ax.annotate(

            label,

            (

                scores[i, 0],

                scores[i, 1]

            ),

            textcoords="offset points",

            xytext=(6, 4),

            fontsize=6
        )

    # =====================================================
    # LOADINGS
    # =====================================================
    scale = 2.2

    for i, var in enumerate(X.columns):

        x = loadings[i, 0] * scale

        y = loadings[i, 1] * scale

        ax.arrow(

            0,
            0,

            x,
            y,

            linewidth=1.0,

            head_width=0.045,

            length_includes_head=True,

            zorder=2
        )

        # =================================================
        # LABELS DOS VETORES
        # =================================================
        ax.annotate(

            var,

            (x, y),

            textcoords="offset points",

            xytext=(8, 4),

            fontsize=6
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
    # LABELS EIXOS
    # =====================================================
    ax.set_xlabel(

        f"PC1 ({explained[0]:.1f}%)",

        fontsize=8
    )

    ax.set_ylabel(

        f"PC2 ({explained[1]:.1f}%)",

        fontsize=8
    )

    # =====================================================
    # ESTILO ORIGIN
    # =====================================================

    # remove topo/direita
    ax.spines["top"].set_visible(False)

    ax.spines["right"].set_visible(False)

    # espessura fina
    ax.spines["left"].set_linewidth(1)

    ax.spines["bottom"].set_linewidth(1)

    # ticks menores
    ax.tick_params(

        axis='both',

        which='major',

        labelsize=10,

        width=1,

        length=5
    )

    # remove grid
    ax.grid(False)

    # proporção quadrada
    ax.set_box_aspect(1)

    plt.tight_layout()

    # =====================================================
    # SAVE FIGURE
    # =====================================================
    fig.savefig(

        "pca_multimodal.tiff",

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
