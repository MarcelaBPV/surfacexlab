import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


def render_ml_tab(supabase=None):
    st.header("🤖 Otimização — PCA Multivariado")

    st.markdown(
        """
        Esta seção permite realizar **Análise de Componentes Principais (PCA)**
        a partir de tabelas experimentais consolidadas (Raman, físico-químicas,
        mecânicas ou elétricas).
        """
    )

    # =====================================================
    # Upload
    # =====================================================
    uploaded_file = st.file_uploader(
        "Upload da tabela de dados",
        type=["csv", "txt", "xls", "xlsx"]
    )

    if not uploaded_file:
        st.info("Envie uma tabela para iniciar a análise PCA.")
        return

    # =====================================================
    # Leitura robusta
    # =====================================================
    try:
        if uploaded_file.name.endswith((".xls", ".xlsx")):
            df = pd.read_excel(uploaded_file)
        else:
            df = pd.read_csv(uploaded_file, sep=None, engine="python")
    except Exception as e:
        st.error("Erro ao ler o arquivo.")
        st.exception(e)
        return

    st.subheader("Pré-visualização dos dados")
    st.dataframe(df)

    # =====================================================
    # Seleções
    # =====================================================
    col1, col2 = st.columns(2)

    with col1:
        sample_col = st.selectbox(
            "Coluna das amostras",
            df.columns
        )

    with col2:
        feature_cols = st.multiselect(
            "Variáveis numéricas (features)",
            [c for c in df.columns if c != sample_col]
        )

    if not feature_cols:
        st.warning("Selecione ao menos uma variável.")
        return

    # =====================================================
    # PCA
    # =====================================================
    X = df[feature_cols].values
    labels = df[sample_col].astype(str).values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    pca = PCA(n_components=2)
    scores = pca.fit_transform(X_scaled)
    loadings = pca.components_.T

    # =====================================================
    # Plot PCA (biplot)
    # =====================================================
    fig, ax = plt.subplots(figsize=(6, 6), dpi=300)

    # Scores
    ax.scatter(scores[:, 0], scores[:, 1], color="blue")

    for i, label in enumerate(labels):
        ax.text(
            scores[i, 0] + 0.03,
            scores[i, 1] + 0.03,
            label,
            fontsize=9
        )

    # Loadings
    for i, var in enumerate(feature_cols):
        ax.arrow(
            0, 0,
            loadings[i, 0] * 2.5,
            loadings[i, 1] * 2.5,
            color="red",
            width=0.01,
            head_width=0.08
        )
        ax.text(
            loadings[i, 0] * 2.8,
            loadings[i, 1] * 2.8,
            var,
            color="red",
            fontsize=9
        )

    ax.axhline(0, color="gray", lw=0.5)
    ax.axvline(0, color="gray", lw=0.5)

    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
    ax.set_title("PCA — Biplot (Scores + Loadings)")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(alpha=0.3)

    st.pyplot(fig)

    # =====================================================
    # Métricas
    # =====================================================
    st.subheader("Variância explicada")
    st.write(
        {
            "PC1 (%)": pca.explained_variance_ratio_[0] * 100,
            "PC2 (%)": pca.explained_variance_ratio_[1] * 100,
        }
    )
