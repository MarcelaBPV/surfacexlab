# pca_upload_surface_style.py
# -*- coding: utf-8 -*-

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


# =========================================================
# UTIL — CONTRIBUIÇÃO QUALITATIVA
# =========================================================
def qualitative_contribution(value):
    if value >= 0.7:
        return "Alta"
    if value >= 0.4:
        return "Média"
    return "Baixa"


# =========================================================
# PCA COM UPLOAD — ESTILO ARTIGO
# =========================================================
def render_pca_upload():

    st.header("📊 PCA — Análise Multivariada de Superfícies")

    st.markdown("""
    **Formatos suportados**

    • Excel (.xlsx)  
    • CSV (.csv)  
    • TXT (.txt — delimitador automático)  

    **Formato esperado**

    ✔ Primeira coluna → Identificação da amostra  
    ✔ Demais colunas → Variáveis numéricas experimentais  
    """)

    uploaded_file = st.file_uploader(
        "Upload do arquivo de dados",
        type=["xlsx", "csv", "txt"]
    )

    if uploaded_file is None:
        st.info("Aguardando upload...")
        return

    # =====================================================
    # LEITURA AUTOMÁTICA
    # =====================================================
    try:

        if uploaded_file.name.endswith(".xlsx"):
            df = pd.read_excel(uploaded_file)

        elif uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)

        else:
            df = pd.read_csv(uploaded_file, sep=None, engine="python")

    except Exception as e:
        st.error(f"Erro ao importar arquivo: {e}")
        return

    st.subheader("Pré-visualização dos dados")
    st.dataframe(df, use_container_width=True)

    # =====================================================
    # SELEÇÃO DA COLUNA AMOSTRA
    # =====================================================
    sample_col = st.selectbox(
        "Coluna identificadora das amostras:",
        options=df.columns.tolist()
    )

    df = df.set_index(sample_col)

    # Conversão numérica
    df = df.apply(pd.to_numeric, errors="coerce")
    df = df.fillna(0)

    if df.shape[0] < 2:
        st.warning("Necessário no mínimo 2 amostras.")
        return

    if df.shape[1] < 2:
        st.warning("Necessário no mínimo 2 variáveis.")
        return

    st.success("Dados prontos para PCA")

    # =====================================================
    # CONFIGURAÇÃO PCA
    # =====================================================
    st.subheader("Configuração PCA")

    n_components = st.slider(
        "Número de Componentes Principais",
        min_value=2,
        max_value=min(10, df.shape[1]),
        value=2
    )

    X = df.values
    labels = df.index.values
    features = df.columns.values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    pca = PCA(n_components=n_components)
    scores = pca.fit_transform(X_scaled)

    loadings = pca.components_.T
    explained = pca.explained_variance_ratio_ * 100

    # =====================================================
    # BIPLOT — ESTILO ARTIGO CIENTÍFICO
    # =====================================================
    st.subheader("PCA — Biplot (Scores + Loadings)")

    fig, ax = plt.subplots(figsize=(6, 6), dpi=300)

    # Scatter das amostras
    ax.scatter(
        scores[:, 0],
        scores[:, 1],
        s=70,
        edgecolors="black",
        linewidths=0.6,
        zorder=3
    )

    # Labels das amostras
    for i, label in enumerate(labels):
        ax.text(
            scores[i, 0],
            scores[i, 1],
            label,
            fontsize=9,
            ha="left",
            va="bottom"
        )

    # Escala vetores
    scale = np.max(np.abs(scores)) * 0.9

    # Vetores das variáveis
    for i, var in enumerate(features):

        ax.arrow(
            0, 0,
            loadings[i, 0] * scale,
            loadings[i, 1] * scale,
            head_width=0.04,
            head_length=0.06,
            linewidth=1.1,
            length_includes_head=True,
            zorder=2
        )

        ax.text(
            loadings[i, 0] * scale * 1.05,
            loadings[i, 1] * scale * 1.05,
            var,
            fontsize=9,
            ha="center",
            va="center"
        )

    # Eixos centrais
    ax.axhline(0, linewidth=0.8)
    ax.axvline(0, linewidth=0.8)

    # Labels científicos
    ax.set_xlabel(f"PC1 ({explained[0]:.1f}%)", fontsize=11)
    ax.set_ylabel(f"PC2 ({explained[1]:.1f}%)", fontsize=11)

    # Remove margens e padding
    ax.margins(0)
    plt.tight_layout(pad=0)

    # Estilo journal
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.tick_params(
        direction="in",
        length=5,
        width=1
    )

    ax.set_aspect("equal", adjustable="box")

    ax.grid(alpha=0.15, linestyle="--")

    st.pyplot(fig)

    # =====================================================
    # VARIÂNCIA EXPLICADA
    # =====================================================
    st.subheader("Variância explicada")

    var_table = pd.DataFrame({
        "Componente": [f"PC{i+1}" for i in range(len(explained))],
        "Variância (%)": explained.round(2)
    })

    st.dataframe(var_table, use_container_width=True)

    # =====================================================
    # CONTRIBUIÇÃO QUALITATIVA
    # =====================================================
    st.subheader("Contribuição qualitativa das variáveis")

    contrib = np.abs(loadings)
    contrib_norm = contrib / contrib.max(axis=0)

    contrib_df = pd.DataFrame(
        contrib_norm,
        index=features,
        columns=[f"PC{i+1}" for i in range(n_components)]
    )

    for col in contrib_df.columns:
        contrib_df[col] = contrib_df[col].apply(qualitative_contribution)

    st.dataframe(contrib_df, use_container_width=True)

    # =====================================================
    # EXPORTAÇÃO
    # =====================================================
    st.subheader("Exportação dos resultados")

    scores_df = pd.DataFrame(
        scores,
        index=labels,
        columns=[f"PC{i+1}" for i in range(n_components)]
    )

    csv_scores = scores_df.to_csv().encode("utf-8")

    st.download_button(
        "⬇ Download Scores PCA (.csv)",
        csv_scores,
        file_name="pca_scores_surface.csv",
        mime="text/csv"
    )


# =========================================================
# EXECUÇÃO DIRETA
# =========================================================
if __name__ == "__main__":
    render_pca_upload()
