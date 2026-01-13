# ml_tab.py
# -*- coding: utf-8 -*-

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


# =========================================================
# LEITURA ROBUSTA DE LOG (TENSIOMETRIA)
# =========================================================
def read_tensiometry_log_safe(file):
    """
    Lê LOGs irregulares ignorando linhas malformadas.
    """
    df = pd.read_csv(
        file,
        sep=None,
        engine="python",
        comment="#",
        on_bad_lines="skip"
    )

    df.columns = [c.strip() for c in df.columns]

    rename_map = {
        "Mean": "theta_mean",
        "Dev.": "theta_std",
        "Time": "time_s",
        "Area": "area",
        "Volume": "volume",
        "Height": "height",
        "Width": "width",
    }

    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})
    return df


def summarize_log(df, sample_name):
    return {
        "Amostra": sample_name,
        "Theta_medio": df["theta_mean"].mean(),
        "Theta_std": df["theta_mean"].std(ddof=1),
        "Volume_medio": df["volume"].mean() if "volume" in df else np.nan,
        "Area_media": df["area"].mean() if "area" in df else np.nan,
        "Altura_media": df["height"].mean() if "height" in df else np.nan,
        "Largura_media": df["width"].mean() if "width" in df else np.nan,
    }


# =========================================================
# ABA OTIMIZAÇÃO — PCA
# =========================================================
def render_ml_tab(supabase=None):
    st.header("🤖 Otimização — PCA Multivariado")

    st.markdown(
        """
        Esta aba executa **PCA multivariado entre amostras** a partir de:
        - Tabelas consolidadas (`.csv`, `.xlsx`)
        - Arquivos `.LOG` de tensiometria (convertidos automaticamente)
        """
    )

    uploaded_files = st.file_uploader(
        "Upload dos arquivos (LOG ou tabelas)",
        type=["log", "csv", "txt", "xls", "xlsx"],
        accept_multiple_files=True
    )

    if not uploaded_files:
        st.info("Envie ao menos dois arquivos para PCA.")
        return

    rows = []

    for file in uploaded_files:
        try:
            if file.name.lower().endswith(".log"):
                df_log = read_tensiometry_log_safe(file)

                if "theta_mean" not in df_log.columns:
                    st.warning(f"{file.name} ignorado (sem coluna Mean).")
                    continue

                rows.append(summarize_log(df_log, file.name))

            else:
                df = pd.read_excel(file) if file.name.endswith(("xls", "xlsx")) else pd.read_csv(file)
                rows.extend(df.to_dict(orient="records"))

        except Exception as e:
            st.error(f"Erro ao processar {file.name}")
            st.exception(e)

    df_pca = pd.DataFrame(rows).dropna()

    if df_pca.shape[0] < 2:
        st.error("São necessárias pelo menos duas amostras válidas para PCA.")
        return

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

    ax.set_xlabel(f"PC1 ({explained[0]:.1f}%)")
    ax.set_ylabel(f"PC2 ({explained[1]:.1f}%)")
    ax.set_title("PCA — Otimização Multivariada")
    ax.grid(alpha=0.3)

    st.pyplot(fig)

    st.subheader("Variância explicada (%)")
    st.dataframe(pd.DataFrame({
        "Componente": ["PC1", "PC2"],
        "Variância (%)": explained.round(2)
    }))
