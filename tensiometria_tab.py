# tensiometria_tab.py
# -*- coding: utf-8 -*-

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from io import StringIO


# =========================================================
# LEITURA ROBUSTA DO LOG DE TENSIOMETRIA
# =========================================================
def read_tensiometry_log(file):
    """
    Leitura robusta de arquivos .LOG de goniômetro óptico.
    Detecta automaticamente o início da tabela e normaliza colunas.
    """

    file.seek(0)
    raw_text = file.read().decode("latin1", errors="ignore")
    lines = raw_text.splitlines()

    # ---------------------------------------------
    # Detecta linha de cabeçalho
    # ---------------------------------------------
    header_idx = None
    for i, line in enumerate(lines):
        if "Mean" in line and ("Theta" in line or "Time" in line):
            header_idx = i
            break

    if header_idx is None:
        return pd.DataFrame()

    table_text = "\n".join(lines[header_idx:])
    buffer = StringIO(table_text)

    # ---------------------------------------------
    # Tentativa robusta de leitura
    # ---------------------------------------------
    try:
        df = pd.read_csv(buffer, sep=";", engine="python", on_bad_lines="skip")
        if df.shape[1] < 2:
            buffer.seek(0)
            df = pd.read_csv(buffer, sep="\t", engine="python", on_bad_lines="skip")
    except Exception:
        return pd.DataFrame()

    df.columns = [c.strip() for c in df.columns]

    # ---------------------------------------------
    # Normalização de nomes
    # ---------------------------------------------
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

    # ---------------------------------------------
    # Conversão numérica segura
    # ---------------------------------------------
    for col in df.columns:
        df[col] = (
            df[col]
            .astype(str)
            .str.replace(",", ".", regex=False)
        )
        df[col] = pd.to_numeric(df[col], errors="coerce")

    if "theta_mean" not in df.columns:
        return pd.DataFrame()

    df = df.dropna(subset=["theta_mean"])
    df = df[(df["theta_mean"] > 0) & (df["theta_mean"] < 180)]

    return df.reset_index(drop=True)


# =========================================================
# CONSOLIDAÇÃO FÍSICA DO ENSAIO
# =========================================================
def summarize_tensiometry(df, sample_name):
    return {
        "Amostra": sample_name,
        "Theta_medio (°)": df["theta_mean"].mean(),
        "Theta_std (°)": df["theta_mean"].std(ddof=1),
        "Volume_medio": df["volume"].mean() if "volume" in df else np.nan,
        "Area_media": df["area"].mean() if "area" in df else np.nan,
        "Altura_media": df["height"].mean() if "height" in df else np.nan,
        "Largura_media": df["width"].mean() if "width" in df else np.nan,
        "N_pontos": int(len(df)),
    }


# =========================================================
# ABA PCA — TENSIOMETRIA
# =========================================================
def render_tensiometria_tab(supabase=None):

    st.header("💧 PCA — Tensiometria (arquivos .LOG)")

    st.markdown(
        """
        Este módulo executa **PCA diretamente sobre ensaios de tensiometria**,
        utilizando apenas os arquivos `.LOG` carregados nesta aba.

        Fluxo:
        1. Leitura robusta dos arquivos de goniômetro  
        2. Extração de parâmetros físicos médios  
        3. Construção automática da matriz multivariada  
        4. PCA com biplot padronizado  
        """
    )

    # =====================================================
    # Upload múltiplo
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
        df_log = read_tensiometry_log(file)

        if df_log.empty:
            st.warning(f"{file.name} ignorado (sem dados válidos).")
            continue

        summaries.append(summarize_tensiometry(df_log, file.name))

    if len(summaries) < 2:
        st.error("São necessárias pelo menos duas amostras válidas para PCA.")
        return

    df_pca = pd.DataFrame(summaries)

    st.subheader("Tabela consolidada (entrada da PCA)")
    st.dataframe(df_pca, use_container_width=True)

    # =====================================================
    # Seleção de variáveis
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
    # BIPLOT PADRONIZADO (IGUAL AO RAMAN)
    # =====================================================
    st.subheader("PCA — Biplot Tensiometria")

    fig, ax = plt.subplots(figsize=(6, 6), dpi=300)

    ax.scatter(
        scores[:, 0],
        scores[:, 1],
        s=80,
        color="#1f77b4",
        edgecolor="black"
    )

    for i, label in enumerate(labels):
        ax.text(
            scores[i, 0] + 0.03,
            scores[i, 1] + 0.03,
            label,
            fontsize=9
        )

    scale = np.max(np.abs(scores)) * 0.8

    for i, var in enumerate(feature_cols):
        ax.arrow(
            0, 0,
            loadings[i, 0] * scale,
            loadings[i, 1] * scale,
            color="black",
            alpha=0.7,
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
    ax.set_title("PCA — Tensiometria")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(alpha=0.3)

    st.pyplot(fig)

    # =====================================================
    # Variância explicada
    # =====================================================
    st.subheader("Variância explicada")

    st.dataframe(pd.DataFrame({
        "Componente": ["PC1", "PC2"],
        "Variância explicada (%)": explained.round(2)
    }))
