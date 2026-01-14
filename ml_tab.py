# ml_tab.py
# -*- coding: utf-8 -*-

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


# =========================================================
# CLASSIFICAÇÃO QUALITATIVA DE CONTRIBUIÇÃO
# =========================================================
def qualitative_contribution(value):
    if value >= 0.7:
        return "Alta"
    if value >= 0.4:
        return "Média"
    return "Baixa"


# =========================================================
# ABA ML / OTIMIZAÇÃO
# =========================================================
def render_ml_tab(supabase=None):

    st.header("🤖 Otimizador — PCA Global & Contribuição Qualitativa")

    st.markdown(
        """
        Este módulo integra **Raman + Tensiometria + Resistividade**
        em uma **análise multivariada global**, permitindo identificar:

        • Variáveis dominantes por amostra  
        • Correlações cruzadas entre propriedades  
        • Contribuições físicas e químicas relevantes  
        """
    )

    # =====================================================
    # COLETA DOS DADOS (SESSION STATE)
    # =====================================================
    data_sources = []

    if "raman_fingerprint" in st.session_state:
        data_sources.append(st.session_state.raman_fingerprint)

    if "tensiometry_samples" in st.session_state:
        data_sources.append(pd.DataFrame(st.session_state.tensiometry_samples))

    if "electrical_samples" in st.session_state:
        data_sources.append(pd.DataFrame(st.session_state.electrical_samples))

    if not data_sources:
        st.info("Nenhum dado disponível ainda. Execute ao menos um módulo.")
        return

    # =====================================================
    # MERGE GLOBAL
    # =====================================================
    df_global = None

    for df in data_sources:
        if "Amostra" not in df.columns:
            continue

        if df_global is None:
            df_global = df.copy()
        else:
            df_global = pd.merge(
                df_global,
                df,
                on="Amostra",
                how="outer"
            )

    if df_global is None or len(df_global) < 2:
        st.warning("Dados insuficientes para PCA global.")
        return

    df_global = df_global.set_index("Amostra")
    df_global = df_global.apply(pd.to_numeric, errors="coerce")
    df_global = df_global.fillna(0.0)

    st.subheader("Matriz global integrada")
    st.dataframe(df_global, use_container_width=True)

    # =====================================================
    # PCA GLOBAL
    # =====================================================
    X = df_global.values
    labels = df_global.index.values
    features = df_global.columns.values

    X_scaled = StandardScaler().fit_transform(X)

    pca = PCA(n_components=2)
    scores = pca.fit_transform(X_scaled)
    loadings = pca.components_.T
    explained = pca.explained_variance_ratio_ * 100

    # =====================================================
    # BIPLOT GLOBAL
    # =====================================================
    st.subheader("📊 PCA Global — Raman + Tensiometria + Elétrica")

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
    for i, var in enumerate(features):
        ax.arrow(
            0, 0,
            loadings[i, 0] * scale,
            loadings[i, 1] * scale,
            color="black",
            alpha=0.6,
            head_width=0.06,
            length_includes_head=True
        )
        ax.text(
            loadings[i, 0] * scale * 1.1,
            loadings[i, 1] * scale * 1.1,
            var,
            fontsize=8
        )

    ax.axhline(0, color="gray", lw=0.6)
    ax.axvline(0, color="gray", lw=0.6)
    ax.set_xlabel(f"PC1 ({explained[0]:.1f}%)")
    ax.set_ylabel(f"PC2 ({explained[1]:.1f}%)")
    ax.set_title("PCA Global — SurfaceXLab")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(alpha=0.3)

    st.pyplot(fig)

    # =====================================================
    # CONTRIBUIÇÃO QUALITATIVA
    # =====================================================
    st.subheader("📋 Tabela de contribuição qualitativa")

    contrib = np.abs(loadings)
    contrib_norm = contrib / contrib.max(axis=0)

    table = pd.DataFrame(
        contrib_norm,
        index=features,
        columns=["PC1", "PC2"]
    )

    table["Contribuição PC1"] = table["PC1"].apply(qualitative_contribution)
    table["Contribuição PC2"] = table["PC2"].apply(qualitative_contribution)

    st.dataframe(
        table[["Contribuição PC1", "Contribuição PC2"]],
        use_container_width=True
    )

    # =====================================================
    # INTERPRETAÇÃO AUTOMÁTICA
    # =====================================================
    st.subheader("🧠 Interpretação automática")

    dominant_pc1 = table["PC1"].idxmax()
    dominant_pc2 = table["PC2"].idxmax()

    st.markdown(
        f"""
        • **PC1** é dominada principalmente por **{dominant_pc1}**, indicando que
        esta variável governa a separação principal das amostras.

        • **PC2** é dominada principalmente por **{dominant_pc2}**, refletindo
        um segundo mecanismo físico/químico independente.
        """
    )
