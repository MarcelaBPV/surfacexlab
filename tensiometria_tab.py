# tensiometria_tab.py
# -*- coding: utf-8 -*-

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


# =========================================================
# LEITURA ROBUSTA DO LOG DE TENSIOMETRIA
# =========================================================
def read_tensiometry_log(file):
    df = pd.read_csv(
        file,
        sep=None,
        engine="python",
        comment="#",
        skip_blank_lines=True
    )

    df.columns = [c.strip() for c in df.columns]

    rename_map = {
        "Time": "time_s",
        "Mean": "theta_mean",
        "Dev.": "theta_std",
        "Theta(L)": "theta_L",
        "Theta(R)": "theta_R",
        "Area": "area",
        "Volume": "volume",
        "Height": "height",
        "Width": "width",
    }

    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

    return df


# =========================================================
# CONSOLIDAÇÃO FÍSICA DO ENSAIO
# =========================================================
def summarize_tensiometry(df, sample_name):
    summary = {
        "Amostra": sample_name,
        "Theta_medio (°)": df["theta_mean"].mean(),
        "Theta_std (°)": df["theta_mean"].std(ddof=1),
        "Volume_medio": df["volume"].mean() if "volume" in df else np.nan,
        "Area_media": df["area"].mean() if "area" in df else np.nan,
        "Altura_media": df["height"].mean() if "height" in df else np.nan,
        "Largura_media": df["width"].mean() if "width" in df else np.nan,
        "N_pontos": len(df),
    }
    return summary


# =========================================================
# ABA PCA — TENSIOMETRIA
# =========================================================
def render_tensiometria_tab(supabase=None):
    st.header("💧 PCA — Tensiometria (LOG → Energia de Superfície)")

    st.markdown(
        """
        Esta aba realiza:
        1. **Leitura direta de arquivos `.LOG` de goniômetro**
        2. **Extração de parâmetros físicos médios**
        3. **Construção automática da matriz multivariada**
        4. **Análise de Componentes Principais (PCA)**
        """
    )

    # =====================================================
    # Upload múltiplo de LOGs
    # =====================================================
    uploaded_files = st.file_uploader(
        "Upload dos arquivos .LOG de tensiometria",
        type=["log", "txt", "csv"],
        accept_multiple_files=True
    )

    if not uploaded_files:
        st.info("Envie ao menos dois arquivos .LOG para executar a PCA.")
        return

    summaries = []

    for file in uploaded_files:
        try:
            df_log = read_tensiometry_log(file)

            if "theta_mean" not in df_log.columns:
                st.warning(f"{file.name} ignorado (sem coluna Mean).")
                continue

            summary = summarize_tensiometry(df_log, file.name)
            summaries.append(summary)

        except Exception as e:
            st.error(f"Erro ao processar {file.name}")
            st.exception(e)

    if len(summaries) < 2:
        st.error("São necessárias pelo menos duas amostras válidas para PCA.")
        return

    df_pca = pd.DataFrame(summaries)
    st.subheader("Tabela consolidada (entrada da PCA)")
    st.dataframe(df_pca)

    # =====================================================
    # Seleção das variáveis
    # =====================================================
    feature_cols = st.multiselect(
        "Variáveis para PCA",
        options=[c for c in df_pca.columns if c != "Amostra"],
        default=[c for c in df_pca.columns if c != "Amostra"][:4]
    )

    if len(feature_cols) < 2:
        st.warning("Selecione ao menos duas variáveis.")
        return

    # =====================================================
    # PCA
    # =====================================================
    X = df_pca[feature_cols].values
    labels = df_pca["Amostra"].values

    X_scaled = StandardScaler().fit_transform(X)

    pca = PCA(n_components=2)
    scores = pca.fit_transform(X_scaled)
    loadings = pca.components_.T
    explained = pca.explained_variance_ratio_ * 100

    # =====================================================
    # BIPLOT
    # =====================================================
    fig, ax = plt.subplots(figsize=(6, 6), dpi=300)

    ax.scatter(scores[:, 0], scores[:, 1], s=80)

    for i, label in enumerate(labels):
        ax.text(scores[i, 0] + 0.03, scores[i, 1] + 0.03, label, fontsize=9)

    scale = np.max(np.abs(scores)) * 0.8
    for i, var in enumerate(feature_cols):
        ax.arrow(
            0, 0,
            loadings[i, 0] * scale,
            loadings[i, 1] * scale,
            color="red",
            head_width=0.08
        )
        ax.text(
            loadings[i, 0] * scale * 1.1,
            loadings[i, 1] * scale * 1.1,
            var,
            color="red",
            fontsize=9
        )

    ax.axhline(0, color="gray", lw=0.6)
    ax.axvline(0, color="gray", lw=0.6)
    ax.set_xlabel(f"PC1 ({explained[0]:.1f}%)")
    ax.set_ylabel(f"PC2 ({explained[1]:.1f}%)")
    ax.set_title("PCA — Tensiometria")
    ax.grid(alpha=0.3)

    st.pyplot(fig)

    # =====================================================
    # Variância explicada
    # =====================================================
    st.subheader("Variância explicada")
    st.dataframe(pd.DataFrame({
        "Componente": ["PC1", "PC2"],
        "Variância (%)": explained.round(2)
    }))
