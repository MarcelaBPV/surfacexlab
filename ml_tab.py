# ml_tab.py
# -*- coding: utf-8 -*-

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


# =========================================================
# ABA OTIMIZAÇÃO — PCA MULTIVARIADO
# =========================================================
def render_ml_tab(supabase=None):

    st.header("🤖 Otimização — PCA Multivariado")

    st.markdown(
        """
        Esta seção permite realizar **Análise de Componentes Principais (PCA)**
        a partir de **tabelas experimentais consolidadas**, como:
        - Raman (fingerprints espectrais)
        - Tensiometria (ângulo de contato, energia de superfície, temperatura)
        - Ensaios elétricos ou físico-químicos

        Os arquivos devem estar em formato **.CSV, .TXT ou .XLS(X)**.
        """
    )

    # =====================================================
    # Upload
    # =====================================================
    uploaded_file = st.file_uploader(
        "Upload da tabela consolidada",
        type=["csv", "txt", "xls", "xlsx"]
    )

    if uploaded_file is None:
        st.info("Envie uma tabela para iniciar a PCA.")
        return

    # =====================================================
    # Leitura robusta
    # =====================================================
    try:
        if uploaded_file.name.lower().endswith((".xls", ".xlsx")):
            df = pd.read_excel(uploaded_file)
        else:
            df = pd.read_csv(uploaded_file, sep=None, engine="python")
    except Exception as e:
        st.error("❌ Erro ao ler o arquivo.")
        st.exception(e)
        return

    if df.empty:
        st.error("A tabela está vazia.")
        return

    st.subheader("Pré-visualização dos dados")
    st.dataframe(df)

    # =====================================================
    # Seleções
    # =====================================================
    st.subheader("Configuração da PCA")

    col1, col2 = st.columns(2)

    with col1:
        sample_col = st.selectbox(
            "Coluna identificadora da amostra",
            options=df.columns
        )

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    with col2:
        feature_cols = st.multiselect(
            "Variáveis numéricas (features)",
            options=numeric_cols,
            default=numeric_cols[:4]
        )

    if len(feature_cols) < 2:
        st.warning("Selecione ao menos **duas variáveis numéricas**.")
        return

    # =====================================================
    # Preparação dos dados
    # =====================================================
    X = df[feature_cols].values
    labels = df[sample_col].astype(str).values

    # remove linhas inválidas
    mask = np.all(np.isfinite(X), axis=1)
    X = X[mask]
    labels = labels[mask]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # =====================================================
    # PCA
    # =====================================================
    pca = PCA(n_components=2)
    scores = pca.fit_transform(X_scaled)
    loadings = pca.components_.T
    explained_var = pca.explained_variance_ratio_ * 100

    # =====================================================
    # BIPLOT
    # =====================================================
    st.subheader("PCA — Biplot (Scores + Loadings)")

    fig, ax = plt.subplots(figsize=(6, 6), dpi=300)

    ax.scatter(scores[:, 0], scores[:, 1], s=70, color="steelblue")

    for i, label in enumerate(labels):
        ax.text(
            scores[i, 0] + 0.03,
            scores[i, 1] + 0.03,
            label,
            fontsize=8
        )

    scale = 2.5
    for i, var in enumerate(feature_cols):
        ax.arrow(
            0, 0,
            loadings[i, 0] * scale,
            loadings[i, 1] * scale,
            color="black",
            width=0.01,
            head_width=0.08
        )
        ax.text(
            loadings[i, 0] * scale * 1.1,
            loadings[i, 1] * scale * 1.1,
            var,
            fontsize=9
        )

    ax.axhline(0, color="gray", lw=0.6)
    ax.axvline(0, color="gray", lw=0.6)

    ax.set_xlabel(f"PC1 ({explained_var[0]:.1f}%)")
    ax.set_ylabel(f"PC2 ({explained_var[1]:.1f}%)")
    ax.set_title("PCA — Análise Multivariada")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(alpha=0.3)

    st.pyplot(fig)

    # =====================================================
    # Variância explicada
    # =====================================================
    st.subheader("Variância explicada")

    var_df = pd.DataFrame({
        "Componente": ["PC1", "PC2"],
        "Variância explicada (%)": explained_var.round(2)
    })

    st.dataframe(var_df)

    # =====================================================
    # Loadings (importância das variáveis)
    # =====================================================
    st.subheader("Contribuição das variáveis (Loadings)")

    loadings_df = pd.DataFrame(
        loadings,
        index=feature_cols,
        columns=["PC1", "PC2"]
    )

    st.dataframe(loadings_df.round(3))
