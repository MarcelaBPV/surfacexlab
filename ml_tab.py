# ml_tab.py
# -*- coding: utf-8 -*-

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


# =========================================================
# FUNÇÃO AUXILIAR — CLASSIFICAÇÃO QUALITATIVA
# =========================================================
def qualitative_contribution(value, thresholds):
    if abs(value) >= thresholds[1]:
        return "Alta"
    elif abs(value) >= thresholds[0]:
        return "Média"
    else:
        return "Baixa"


# =========================================================
# ABA OTIMIZAÇÃO — PCA INTEGRADO
# =========================================================
def render_ml_tab(supabase=None):

    st.header("🤖 Otimização — PCA Integrado Multivariado")

    st.markdown(
        """
        Este módulo realiza **Análise de Componentes Principais (PCA) integrada**
        combinando informações:

        - **Raman** → ID/IG, I2D/IG  
        - **Tensiometria** → Energia de superfície (OWRK)  
        
        A saída inclui:
        - **Biplot padronizado**
        - **Tabela automática de contribuição qualitativa**
        """
    )

    # =====================================================
    # VERIFICAÇÃO DOS DADOS
    # =====================================================
    if (
        "raman_features" not in st.session_state or
        "tensiometry_samples" not in st.session_state
    ):
        st.info(
            "⚠ Para executar o PCA integrado:\n"
            "- Processe amostras na aba **Raman**\n"
            "- Processe amostras na aba **Tensiometria**"
        )
        return

    df_raman = pd.DataFrame(st.session_state.raman_features)
    df_tens  = pd.DataFrame(st.session_state.tensiometry_samples)

    if df_raman.empty or df_tens.empty:
        st.warning("Dados insuficientes para PCA.")
        return

    # =====================================================
    # MERGE PELO NOME DA AMOSTRA
    # =====================================================
    df = pd.merge(df_raman, df_tens, on="Amostra", how="inner")

    if df.shape[0] < 2:
        st.warning("São necessárias pelo menos duas amostras comuns.")
        return

    st.subheader("Matriz integrada de entrada")
    st.dataframe(df, use_container_width=True)

    # =====================================================
    # SELEÇÃO DAS VARIÁVEIS
    # =====================================================
    feature_cols = st.multiselect(
        "Variáveis para PCA",
        options=[c for c in df.columns if c != "Amostra"],
        default=[
            "ID_IG",
            "I2D_IG",
            "Theta médio (°)",
            "gamma_total",
            "gamma_p",
            "gamma_d",
            "polar_fraction"
        ]
    )

    if len(feature_cols) < 2:
        st.warning("Selecione ao menos duas variáveis.")
        return

    # =====================================================
    # PCA
    # =====================================================
    X = df[feature_cols].values
    labels = df["Amostra"].values

    X_scaled = StandardScaler().fit_transform(X)

    pca = PCA(n_components=2)
    scores = pca.fit_transform(X_scaled)
    loadings = pca.components_.T
    explained = pca.explained_variance_ratio_ * 100

    # =====================================================
    # BIPLOT PADRONIZADO (IGUAL RAMAN/TENSIOMETRIA)
    # =====================================================
    st.subheader("PCA Integrado — Biplot")

    fig, ax = plt.subplots(figsize=(7, 7), dpi=300)

    ax.scatter(scores[:, 0], scores[:, 1], s=90, edgecolor="black")

    for i, label in enumerate(labels):
        ax.text(
            scores[i, 0] + 0.03,
            scores[i, 1] + 0.03,
            label,
            fontsize=9
        )

    scale = np.max(np.abs(scores)) * 0.85

    for i, var in enumerate(feature_cols):
        ax.arrow(
            0, 0,
            loadings[i, 0] * scale,
            loadings[i, 1] * scale,
            color="black",
            width=0.003,
            length_includes_head=True
        )
        ax.text(
            loadings[i, 0] * scale * 1.1,
            loadings[i, 1] * scale * 1.1,
            var,
            fontsize=9
        )

    ax.axhline(0, color="gray", lw=0.6)
    ax.axvline(0, color="gray", lw=0.6)
    ax.set_xlabel(f"PC1 ({explained[0]:.1f}%)")
    ax.set_ylabel(f"PC2 ({explained[1]:.1f}%)")
    ax.set_title("PCA Integrado — Raman + Energia de Superfície")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(alpha=0.3)

    st.pyplot(fig)

    # =====================================================
    # VARIÂNCIA EXPLICADA
    # =====================================================
    st.subheader("Variância explicada")

    st.dataframe(pd.DataFrame({
        "Componente": ["PC1", "PC2"],
        "Variância (%)": explained.round(2)
    }))

    # =====================================================
    # TABELA DE CONTRIBUIÇÃO QUALITATIVA
    # =====================================================
    st.subheader("Contribuição qualitativa das variáveis")

    loadings_df = pd.DataFrame(
        loadings,
        index=feature_cols,
        columns=["PC1", "PC2"]
    )

    # Limiares automáticos
    abs_vals = np.abs(loadings_df.values.flatten())
    t_low  = np.percentile(abs_vals, 33)
    t_high = np.percentile(abs_vals, 66)

    contrib_table = []

    for var in feature_cols:
        contrib_table.append({
            "Variável": var,
            "PC1": qualitative_contribution(loadings_df.loc[var, "PC1"], (t_low, t_high)),
            "PC2": qualitative_contribution(loadings_df.loc[var, "PC2"], (t_low, t_high)),
            "Sinal PC1": "Positivo" if loadings_df.loc[var, "PC1"] > 0 else "Negativo",
            "Sinal PC2": "Positivo" if loadings_df.loc[var, "PC2"] > 0 else "Negativo",
        })

    contrib_df = pd.DataFrame(contrib_table)

    st.dataframe(contrib_df, use_container_width=True)

    st.caption(
        "Classificação automática baseada na magnitude relativa dos loadings.\n"
        "Alta / Média / Baixa contribuição para cada componente principal."
    )
