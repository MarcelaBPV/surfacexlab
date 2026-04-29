# =========================================================
# SurfaceXLab — Análise Integrada Científica
# =========================================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


# =========================================================
# DATASET GLOBAL (INTEGRAÇÃO REAL)
# =========================================================
def build_dataset():

    data = []

    # -------------------------
    # RAMAN
    # -------------------------
    raman = st.session_state.get("raman_peaks", {})

    # -------------------------
    # ELÉTRICO
    # -------------------------
    electrical = st.session_state.get("electrical_samples", {})

    # -------------------------
    # TENSIOMETRIA
    # -------------------------
    tensio = st.session_state.get("tensiometry_samples", {})

    # -------------------------
    # PERFILOMETRIA
    # -------------------------
    perfilo = st.session_state.get("perfilometria_samples", {})

    # -------------------------
    # UNIÃO POR NOME DA AMOSTRA
    # -------------------------
    samples = set(raman.keys()) | set(electrical.keys()) | set(tensio.keys()) | set(perfilo.keys())

    for s in samples:

        row = {"Amostra": s}

        # Raman
        if s in raman:
            row.update(raman[s])

        # Elétrico
        if s in electrical:
            row.update(electrical[s])

        # Tensiometria
        if s in tensio:
            row.update(tensio[s])

        # Perfilometria
        if s in perfilo:
            row.update(perfilo[s])

        data.append(row)

    if not data:
        return None

    df = pd.DataFrame(data)

    return df


# =========================================================
# PCA GLOBAL
# =========================================================
def run_pca(df):

    df = df.copy()

    numeric = df.select_dtypes(include=[np.number])

    if numeric.shape[1] < 2:
        st.warning("Poucas variáveis numéricas")
        return None

    X = numeric.fillna(0)

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

    for i, var in enumerate(numeric.columns):

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

    ax.set_xlabel(f"PC1 ({explained[0]:.1f}%)")
    ax.set_ylabel(f"PC2 ({explained[1]:.1f}%)")
    ax.set_title("PCA Integrada — Superfícies")

    ax.grid(alpha=0.3)

    return fig


# =========================================================
# CORRELAÇÃO FÍSICA
# =========================================================
def run_correlation(df):

    numeric = df.select_dtypes(include=[np.number]).fillna(0)

    corr = numeric.corr()

    fig, ax = plt.subplots(figsize=(6,5))

    im = ax.imshow(corr, aspect="auto")

    ax.set_xticks(range(len(corr.columns)))
    ax.set_yticks(range(len(corr.columns)))

    ax.set_xticklabels(corr.columns, rotation=90)
    ax.set_yticklabels(corr.columns)

    fig.colorbar(im)

    return fig


# =========================================================
# TAB PRINCIPAL
# =========================================================
def render_analise_completa_amostras_tab(supabase=None):

    st.header("🧠 Análise Integrada Científica")

    df = build_dataset()

    if df is None or df.empty:
        st.warning("Nenhuma amostra integrada ainda.")
        return

    st.subheader("Dataset integrado")
    st.dataframe(df, use_container_width=True)

    # =====================================================
    # SUBABAS
    # =====================================================
    subtabs = st.tabs([
        "📊 PCA Global",
        "🔗 Correlação Física"
    ])

    # =====================================================
    # PCA
    # =====================================================
    with subtabs[0]:

        if df.shape[0] < 2:
            st.info("Mínimo 2 amostras")
            return

        fig = run_pca(df)

        if fig:
            st.pyplot(fig)

    # =====================================================
    # CORRELAÇÃO
    # =====================================================
    with subtabs[1]:

        fig = run_correlation(df)
        st.pyplot(fig)

        st.markdown("### 🔬 Interpretação")

        st.write("""
        - Correlações positivas indicam propriedades acopladas  
        - Ex: energia superficial ↑ → condutividade ↑  
        - Raman (ID/IG) pode correlacionar com defeitos → transporte  
        - Rugosidade influencia molhabilidade e transporte elétrico  
        """)
