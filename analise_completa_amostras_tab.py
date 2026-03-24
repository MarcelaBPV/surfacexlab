# -*- coding: utf-8 -*-

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


# =========================================================
# PCA GLOBAL
# =========================================================
def run_pca_global(df):

    df = df.copy()

    if "Amostra" not in df.columns:
        st.error("Coluna 'Amostra' não encontrada.")
        return

    # apenas numéricas
    X = df.select_dtypes(include=[np.number])

    if X.shape[1] < 2:
        st.warning("Poucas variáveis numéricas para PCA.")
        return

    X_scaled = StandardScaler().fit_transform(X)

    pca = PCA(n_components=2)
    scores = pca.fit_transform(X_scaled)
    loadings = pca.components_.T
    explained = pca.explained_variance_ratio_ * 100

    fig, ax = plt.subplots(figsize=(7,7), dpi=300)

    ax.scatter(scores[:,0], scores[:,1], s=100, edgecolor="black")

    for i, label in enumerate(df["Amostra"]):
        ax.text(scores[i,0], scores[i,1], label)

    scale = np.max(np.abs(scores)) * 0.8

    for i, var in enumerate(X.columns):

        ax.arrow(
            0, 0,
            loadings[i,0]*scale,
            loadings[i,1]*scale,
            head_width=0.05,
            color="black"
        )

        ax.text(
            loadings[i,0]*scale*1.1,
            loadings[i,1]*scale*1.1,
            var,
            fontsize=9
        )

    ax.axhline(0, color="gray", lw=0.5)
    ax.axvline(0, color="gray", lw=0.5)

    ax.set_xlabel(f"PC1 ({explained[0]:.1f}%)")
    ax.set_ylabel(f"PC2 ({explained[1]:.1f}%)")

    ax.set_title("PCA Global — Integração Multimodal")

    ax.grid(alpha=0.3)

    return fig


# =========================================================
# INTEGRAÇÃO DOS MÓDULOS
# =========================================================
def build_global_dataset():

    dfs = []

    # =====================================================
    # TENSIOMETRIA
    # =====================================================
    if "tensiometry_samples" in st.session_state:

        df_tens = pd.DataFrame(
            st.session_state.tensiometry_samples.values()
        )

        df_tens["Amostra"] = list(st.session_state.tensiometry_samples.keys())

        dfs.append(df_tens)

    # =====================================================
    # RESISTIVIDADE
    # =====================================================
    if "electrical_features" in st.session_state:

        df_elec = st.session_state.electrical_features.copy()

        dfs.append(df_elec)

    # =====================================================
    # RAMAN (fingerprint)
    # =====================================================
    if "raman_peaks" in st.session_state:

        df_raman = (
            pd.DataFrame(st.session_state.raman_peaks)
            .T
            .fillna(0)
        )

        df_raman["Amostra"] = df_raman.index

        dfs.append(df_raman)

    # =====================================================
    # MERGE
    # =====================================================
    if len(dfs) < 2:
        return None

    df_merged = dfs[0]

    for df in dfs[1:]:
        df_merged = pd.merge(df_merged, df, on="Amostra", how="outer")

    return df_merged


# =========================================================
# STREAMLIT TAB
# =========================================================
def render_analise_completa_amostras_tab(supabase=None):

    st.header("🧠 Análise Completa de Amostras")

    st.markdown("""
    Integração automática de:

    - ⚡ Resistividade elétrica  
    - 💧 Tensiometria (OWRK)  
    - 🔬 Raman (fingerprint molecular)  

    + PCA global para análise científica
    """)

    # =====================================================
    # DATASET GLOBAL
    # =====================================================
    df_global = build_global_dataset()

    if df_global is None:
        st.warning("Execute pelo menos duas análises (Raman, Elétrica, Tensiometria).")
        return

    st.subheader("Dataset integrado")

    st.dataframe(df_global, use_container_width=True)

    # salvar para ML
    st.session_state.global_features = df_global

    # =====================================================
    # PCA GLOBAL
    # =====================================================
    st.subheader("PCA Global")

    if df_global.shape[0] < 2:
        st.info("Necessário ao menos duas amostras.")
        return

    fig = run_pca_global(df_global)

    if fig:
        st.pyplot(fig)

    # =====================================================
    # EXPORTAÇÃO
    # =====================================================
    st.subheader("Exportar")

    csv = df_global.to_csv(index=False).encode("utf-8")

    st.download_button(
        "📥 Baixar dataset completo",
        csv,
        "analise_completa_amostras.csv",
        "text/csv"
    )
