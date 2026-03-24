# -*- coding: utf-8 -*-

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# 🔬 MAPEAMENTO RAMAN (reuso direto do seu módulo)
from mapeamento_molecular_tab import (
    read_mapping,
    process_spectrum,
    plot_heatmap,
    run_pca as run_pca_raman,
    plot_raman_groups_annotated
)


# =========================================================
# PCA GLOBAL
# =========================================================
def run_pca_global(df):

    df = df.copy()

    if "Amostra" not in df.columns:
        df["Amostra"] = df.index.astype(str)

    X = df.select_dtypes(include=[np.number])

    if X.shape[1] < 2:
        st.warning("Poucas variáveis para PCA.")
        return None

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
# DATASET GLOBAL
# =========================================================
def build_global_dataset():

    dfs = []

    # TENSIOMETRIA
    if "tensiometry_samples" in st.session_state:
        df_t = pd.DataFrame(st.session_state.tensiometry_samples.values())
        df_t["Amostra"] = list(st.session_state.tensiometry_samples.keys())
        dfs.append(df_t)

    # RESISTIVIDADE
    if "electrical_features" in st.session_state:
        dfs.append(st.session_state.electrical_features)

    # RAMAN
    if "raman_peaks" in st.session_state:
        df_r = pd.DataFrame(st.session_state.raman_peaks).T.fillna(0)
        df_r["Amostra"] = df_r.index
        dfs.append(df_r)

    if len(dfs) < 2:
        return None

    df_merged = dfs[0]

    for df in dfs[1:]:
        df_merged = pd.merge(df_merged, df, on="Amostra", how="outer")

    return df_merged


# =========================================================
# TAB PRINCIPAL
# =========================================================
def render_analise_completa_amostras_tab(supabase=None):

    st.header("🧠 Análise Completa de Amostras")

    subtabs = st.tabs([
        "📊 Integração & PCA",
        "🧬 Mapeamento Raman"
    ])

# =========================================================
# 📊 PCA GLOBAL
# =========================================================
    with subtabs[0]:

        df_global = build_global_dataset()

        if df_global is None:
            st.warning("Execute pelo menos dois módulos (Raman, Elétrica, Tensiometria).")
            return

        st.subheader("Dataset Integrado")
        st.dataframe(df_global, use_container_width=True)

        st.subheader("PCA Global")

        if df_global.shape[0] >= 2:

            fig = run_pca_global(df_global)

            if fig:
                st.pyplot(fig)

# =========================================================
# 🧬 MAPEAMENTO RAMAN COMPLETO
# =========================================================
    with subtabs[1]:

        st.subheader("Upload do Mapeamento Raman")

        file = st.file_uploader(
            "Arquivo (CSV / TXT)",
            type=["csv","txt"]
        )

        if not file:
            st.info("Envie o arquivo de mapeamento.")
            return

        try:

            df = read_mapping(file)

            # =========================
            # HEATMAP
            # =========================
            st.subheader("Mapa Raman")
            st.pyplot(plot_heatmap(df))

            # =========================
            # PROCESSAMENTO ESPECTRAL
            # =========================
            spectra = []

            for y_val, group in df.groupby("y"):

                group = group.sort_values("wave")

                x, y = process_spectrum(
                    group["wave"].values,
                    group["intensity"].values
                )

                spectra.append({
                    "y": y_val,
                    "wave": x,
                    "intensity": y
                })

            # =========================
            # PCA RAMAN
            # =========================
            st.subheader("PCA Raman (mapping)")
            st.pyplot(run_pca_raman(spectra))

            # =========================
            # GRUPOS L1–L4 (NÍVEL ARTIGO)
            # =========================
            st.subheader("Espectros com identificação molecular")

            fig, tables = plot_raman_groups_annotated(spectra)

            st.pyplot(fig)

            st.subheader("Tabela de bandas por região")

            for g, t in tables.items():
                st.markdown(f"### {g}")
                st.dataframe(t, use_container_width=True)

        except Exception as e:
            st.error("Erro no processamento do mapeamento")
            st.exception(e)
