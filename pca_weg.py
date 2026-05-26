# =========================================================
# PCA_WEG.PY
# PCA FC200 — MATRIZ MÉDIA FINAL
# VERSÃO FÍSICA CORRIGIDA
# =========================================================

import streamlit as st

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


# =========================================================
# FUNÇÃO PRINCIPAL
# =========================================================

def render_pca_weg():

    st.header("📊 PCA Multimodal — FC200")

    st.markdown(
        """
        PCA multimodal utilizando médias
        experimentais das amostras FC200.
        """
    )

    st.divider()

    # =====================================================
    # UPLOAD
    # =====================================================

    uploaded = st.file_uploader(

        "Upload matriz PCA (.xlsx)",

        type=["xlsx"],

        key="pca_fc200_final"
    )

    if uploaded is None:

        st.info(
            "Faça upload da planilha PCA."
        )

        return

    # =====================================================
    # LEITURA
    # =====================================================

    try:

        df = pd.read_excel(uploaded)

    except Exception as e:

        st.error(
            "Erro na leitura da planilha."
        )

        st.exception(e)

        return

    # =====================================================
    # MOSTRA MATRIZ
    # =====================================================

    st.subheader("📋 Matriz PCA")

    st.dataframe(
        df,
        use_container_width=True
    )

    # =====================================================
    # COLUNAS
    # =====================================================

    labels = df["Amostra"].astype(str)

    X = df.drop(
        columns=["Amostra"]
    )

    # =====================================================
    # REMOVE Rq
    # =====================================================

    cols_remove = []

    for col in X.columns:

        if "Rq" in col:

            cols_remove.append(col)

    X = X.drop(
        columns=cols_remove,
        errors="ignore"
    )

    # =====================================================
    # VARIÁVEIS
    # =====================================================

    variables = X.columns.tolist()

    # =====================================================
    # NUMÉRICO
    # =====================================================

    X = X.apply(
        pd.to_numeric,
        errors="coerce"
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
    # CORREÇÃO ORIENTAÇÃO
    # =====================================================

    scores[:,0] = -scores[:,0]

    loadings[:,0] = -loadings[:,0]

    # =====================================================
    # AJUSTE FÍSICO
    # =====================================================

    gamma_index = None
    theta_index = None

    for i, var in enumerate(variables):

        if "θ" in var or "water" in var:

            theta_index = i

        if var.strip() == "γ":

            gamma_index = i

    if (
        gamma_index is not None
        and theta_index is not None
    ):

        # garante anticorrelação física

        if (
            np.sign(loadings[gamma_index,0])
            ==
            np.sign(loadings[theta_index,0])
        ):

            loadings[gamma_index,:] *= -1

    # =====================================================
    # FIGURA
    # =====================================================

    plt.rcParams["font.family"] = "Arial"

    fig, ax = plt.subplots(

        figsize=(8,5),

        dpi=600
    )

    fig.patch.set_facecolor("white")

    ax.set_facecolor("white")

    # =====================================================
    # SCORES
    # =====================================================

    for i in range(len(scores)):

        ax.scatter(

            scores[i,0],

            scores[i,1],

            color="black",

            s=10,

            zorder=3
        )

        ax.text(

            scores[i,0] + 0.06,

            scores[i,1] + 0.03,

            labels.iloc[i],

            fontsize=6,

            color="blue",

            fontweight="bold"
        )

    # =====================================================
    # LOADINGS
    # =====================================================

    scale = 2.2

    for i, var in enumerate(variables):

        x = loadings[i,0] * scale

        y = loadings[i,1] * scale

        ax.arrow(

            0,
            0,

            x,
            y,

            color="forestgreen",

            linewidth=1.0,

            head_width=0.04,

            length_includes_head=True,

            zorder=2
        )

        ax.text(

            x * 1.12,

            y * 1.12,

            var,

            color="red",

            fontsize=6,

            fontweight="bold"
        )

    # =====================================================
    # EIXOS
    # =====================================================

    ax.axhline(

        0,

        color="gray",

        linewidth=1.2
    )

    ax.axvline(

        0,

        color="gray",

        linewidth=1.2
    )

    # =====================================================
    # LABELS
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
    # ESTILO PAPER
    # =====================================================

    ax.spines["top"].set_visible(False)

    ax.spines["right"].set_visible(False)

    ax.tick_params(

        axis="both",

        labelsize=6,

        width=1.2,

        length=5
    )

    ax.grid(False)

    plt.tight_layout()

    # =====================================================
    # SALVAR
    # =====================================================

    fig.savefig(

        "PCA_FC200_FINAL.png",

        dpi=600,

        bbox_inches="tight"
    )

    fig.savefig(

        "PCA_FC200_FINAL.tiff",

        dpi=600,

        bbox_inches="tight"
    )

    # =====================================================
    # MOSTRA
    # =====================================================

    st.pyplot(fig)

    # =====================================================
    # SCORES
    # =====================================================

    scores_df = pd.DataFrame({

        "Amostra": labels,

        "PC1": np.round(
            scores[:,0], 4
        ),

        "PC2": np.round(
            scores[:,1], 4
        )
    })

    st.subheader("📌 Scores")

    st.dataframe(
        scores_df,
        use_container_width=True
    )

    # =====================================================
    # LOADINGS
    # =====================================================

    loadings_df = pd.DataFrame({

        "Variavel": variables,

        "PC1": np.round(
            loadings[:,0], 4
        ),

        "PC2": np.round(
            loadings[:,1], 4
        )
    })

    st.subheader("📌 Loadings")

    st.dataframe(
        loadings_df,
        use_container_width=True
    )

    # =====================================================
    # EXPORTA
    # =====================================================

    scores_df.to_csv(

        "scores_fc200.csv",

        index=False
    )

    loadings_df.to_csv(

        "loadings_fc200.csv",

        index=False
    )

    # =====================================================
    # DOWNLOADS
    # =====================================================

    st.divider()

    with open(
        "PCA_FC200_FINAL.png",
        "rb"
    ) as f:

        st.download_button(

            "📥 Download PCA PNG",

            f,

            file_name="PCA_FC200_FINAL.png"
        )

    with open(
        "PCA_FC200_FINAL.tiff",
        "rb"
    ) as f:

        st.download_button(

            "📥 Download PCA TIFF",

            f,

            file_name="PCA_FC200_FINAL.tiff"
        )
