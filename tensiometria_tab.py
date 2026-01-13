# tensiometria_tab.py
# -*- coding: utf-8 -*-

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


# =========================================================
# ABA PCA — TENSIOMETRIA
# =========================================================
def render_tensiometria_pca_tab():
    st.header("📊 PCA — Tensiometria (Energia de Superfície)")

    st.markdown(
        """
        Esta seção permite realizar **Análise de Componentes Principais (PCA)**
        a partir de **tabelas consolidadas de tensiometria**, geradas após o
        processamento de arquivos `.LOG` no módulo físico-mecânico.

        A PCA avalia a **influência da temperatura, ângulo de contato e
        componentes da energia livre de superfície** sobre o comportamento
        global das amostras.
        """
    )

    # =====================================================
    # Upload
    # =====================================================
    uploaded_file = st.file_uploader(
        "Upload da tabela de tensiometria",
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

    with col2:
        temp_col = st.selectbox(
            "Coluna da temperatura (°C)",
            options=df.columns
        )

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    feature_cols = st.multiselect(
        "Variáveis físico-químicas (features)",
        options=[c for c in numeric_cols if c != temp_col],
        default=[c for c in numeric_cols if c != temp_col][:4]
    )

    if len(feature_cols) < 2:
        st.warning("Selecione ao menos duas variáveis.")
        return

    # =====================================================
    # Preparação dos dados (BLINDADA)
    # =====================================================
    df_pca = df[[sample_col, temp_col] + feature_cols].copy()

    df_pca[temp_col] = pd.to_numeric(df_pca[temp_col], errors="coerce")
    df_pca = df_pca.dropna()

    if df_pca.shape[0] < 3:
        st.error("Número insuficiente de amostras válidas para PCA.")
        return

    X = df_pca[feature_cols].values
    labels = df_pca[sample_col].astype(str).values
    temperatures = df_pca[temp_col].values

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
    # BIPLOT (Scores + Loadings)
    # =====================================================
    st.subheader("PCA — Biplot (Tensiometria)")

    fig, ax = plt.subplots(figsize=(6, 6), dpi=300)

    scatter = ax.scatter(
        scores[:, 0],
        scores[:, 1],
        c=temperatures,
        cmap="plasma",
        s=80,
        edgecolor="black"
    )

    for i, label in enumerate(labels):
        ax.text(
            scores[i, 0] + 0.04,
            scores[i, 1] + 0.04,
            label,
            fontsize=9
        )

    # Escala dinâmica dos vetores
    scale = np.max(np.abs(scores)) * 0.9

    for i, var in enumerate(feature_cols):
        ax.arrow(
            0, 0,
            loadings[i, 0] * scale,
            loadings[i, 1] * scale,
            color="black",
            width=0.01,
            head_width=0.08,
            length_includes_head=True
        )
        ax.text(
            loadings[i, 0] * scale * 1.1,
            loadings[i, 1] * scale * 1.1,
            var,
            fontsize=9,
            ha="center",
            va="center"
        )

    ax.axhline(0, color="gray", lw=0.6)
    ax.axvline(0, color="gray", lw=0.6)

    ax.set_xlabel(f"PC1 ({explained_var[0]:.1f}%)")
    ax.set_ylabel(f"PC2 ({explained_var[1]:.1f}%)")
    ax.set_title("PCA — Energia de Superfície × Temperatura")
    ax.grid(alpha=0.3)

    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label("Temperatura (°C)")

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
