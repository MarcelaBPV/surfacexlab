# =========================================================
# pca_processing.py
# SurfaceXLab — PCA Multimodal
# =========================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


# =========================================================
# CONSOLIDAÇÃO MULTIMODAL
# =========================================================
def build_multimodal_dataframe(

    raman_df,
    electrical_df,
    tensio_df,
    perfil_df

):

    # =====================================================
    # PADRONIZA NOMES
    # =====================================================
    raman_df = raman_df.copy()
    electrical_df = electrical_df.copy()
    tensio_df = tensio_df.copy()
    perfil_df = perfil_df.copy()

    # índice = amostra
    raman_df = raman_df.set_index("Amostra")
    electrical_df = electrical_df.set_index("Amostra")
    tensio_df = tensio_df.set_index("Amostra")
    perfil_df = perfil_df.set_index("Amostra")

    # =====================================================
    # CONCATENA
    # =====================================================
    df = pd.concat(

        [
            raman_df,
            electrical_df,
            tensio_df,
            perfil_df
        ],

        axis=1
    )

    # remove duplicadas
    df = df.loc[:, ~df.columns.duplicated()]

    # apenas numéricos
    df = df.select_dtypes(include=[np.number])

    # remove NaN
    df = df.fillna(0)

    return df


# =========================================================
# PCA
# =========================================================
def run_multimodal_pca(df):

    # =====================================================
    # NORMALIZAÇÃO
    # =====================================================
    scaler = StandardScaler()

    X = scaler.fit_transform(df.values)

    # =====================================================
    # PCA
    # =====================================================
    pca = PCA(n_components=2)

    scores = pca.fit_transform(X)

    loadings = pca.components_.T

    explained = (
        pca.explained_variance_ratio_ * 100
    )

    # =====================================================
    # DATAFRAMES
    # =====================================================
    score_df = pd.DataFrame(

        scores,

        columns=["PC1", "PC2"],

        index=df.index
    )

    loading_df = pd.DataFrame(

        loadings,

        columns=["PC1", "PC2"],

        index=df.columns
    )

    return {

        "scores": score_df,

        "loadings": loading_df,

        "explained": explained
    }


# =========================================================
# PLOT PCA
# =========================================================
def generate_pca_plot(

    scores,
    loadings,
    explained

):

    fig, ax = plt.subplots(

        figsize=(7,7),
        dpi=300
    )

    # =====================================================
    # SCORES
    # =====================================================
    ax.scatter(

        scores["PC1"],
        scores["PC2"],
        s=100
    )

    # labels
    for idx in scores.index:

        ax.text(

            scores.loc[idx, "PC1"],
            scores.loc[idx, "PC2"],
            idx,
            fontsize=9
        )

    # =====================================================
    # LOADINGS
    # =====================================================
    scale = np.max(np.abs(scores.values)) * 0.7

    for idx in loadings.index:

        x = loadings.loc[idx, "PC1"] * scale
        y = loadings.loc[idx, "PC2"] * scale

        ax.arrow(

            0,
            0,
            x,
            y,

            head_width=0.05,
            length_includes_head=True
        )

        ax.text(

            x * 1.1,
            y * 1.1,
            idx,
            fontsize=8
        )

    # =====================================================
    # ESTILO
    # =====================================================
    ax.axhline(0, linewidth=0.8)

    ax.axvline(0, linewidth=0.8)

    ax.set_xlabel(
        f"PC1 ({explained[0]:.1f}%)"
    )

    ax.set_ylabel(
        f"PC2 ({explained[1]:.1f}%)"
    )

    ax.set_title(
        "PCA Multimodal — SurfaceXLab"
    )

    ax.grid(alpha=0.3)

    plt.tight_layout()

    return fig
