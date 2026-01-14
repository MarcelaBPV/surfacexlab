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

    file.seek(0)
    raw_text = file.read().decode("latin1", errors="ignore")
    lines = raw_text.splitlines()

    header_idx = None
    for i, line in enumerate(lines):
        if "Mean" in line and ("Theta" in line or "Time" in line):
            header_idx = i
            break

    if header_idx is None:
        return pd.DataFrame()

    table_text = "\n".join(lines[header_idx:])
    buffer = StringIO(table_text)

    try:
        df = pd.read_csv(buffer, sep=";", engine="python", on_bad_lines="skip")
        if df.shape[1] < 2:
            buffer.seek(0)
            df = pd.read_csv(buffer, sep="\t", engine="python", on_bad_lines="skip")
    except Exception:
        return pd.DataFrame()

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

    for col in df.columns:
        df[col] = (
            df[col].astype(str)
            .str.replace(",", ".", regex=False)
        )
        df[col] = pd.to_numeric(df[col], errors="coerce")

    if "theta_mean" not in df.columns:
        return pd.DataFrame()

    df = df.dropna(subset=["theta_mean"])
    df = df[(df["theta_mean"] > 0) & (df["theta_mean"] < 180)]

    return df.reset_index(drop=True)


# =========================================================
# CONSOLIDAÇÃO DA AMOSTRA
# =========================================================
def summarize_tensiometry(df, sample_name):

    return {
        "Amostra": sample_name,
        "Theta médio (°)": df["theta_mean"].mean(),
        "Theta std (°)": df["theta_mean"].std(ddof=1),
        "Volume médio": df["volume"].mean() if "volume" in df else np.nan,
        "Área média": df["area"].mean() if "area" in df else np.nan,
        "Altura média": df["height"].mean() if "height" in df else np.nan,
        "Largura média": df["width"].mean() if "width" in df else np.nan,
        "N pontos": int(len(df)),
    }


# =========================================================
# ABA TENSIOMETRIA
# =========================================================
def render_tensiometria_tab(supabase=None):

    st.header("💧 Físico-Mecânica — Tensiometria Óptica")

    st.markdown(
        """
        **Subaba 1**  
        Upload de arquivos `.LOG`, processamento físico e visualização  

        **Subaba 2**  
        PCA multivariada utilizando **somente os parâmetros calculados**
        """
    )

    if "tensiometry_samples" not in st.session_state:
        st.session_state.tensiometry_samples = []

    # =====================================================
    # SUBABAS
    # =====================================================
    subtabs = st.tabs([
        "📐 Upload & Processamento",
        "📊 PCA — Tensiometria"
    ])

    # =====================================================
    # SUBABA 1 — PROCESSAMENTO
    # =====================================================
    with subtabs[0]:

        uploaded_files = st.file_uploader(
            "Upload dos arquivos .LOG de tensiometria",
            type=["log", "txt", "csv"],
            accept_multiple_files=True
        )

        if uploaded_files:
            for file in uploaded_files:
                st.markdown(f"### 📄 Amostra: `{file.name}`")

                df_log = read_tensiometry_log(file)

                if df_log.empty:
                    st.warning("Arquivo ignorado (sem dados válidos).")
                    continue

                summary = summarize_tensiometry(df_log, file.name)
                st.session_state.tensiometry_samples.append(summary)

                # -----------------------------
                # GRÁFICO θ × tempo
                # -----------------------------
                fig, ax = plt.subplots(figsize=(6, 4), dpi=300)
                ax.plot(df_log["time_s"], df_log["theta_mean"], lw=1.5)
                ax.set_xlabel("Tempo (s)")
                ax.set_ylabel("Ângulo de contato (°)")
                ax.set_title("Evolução do ângulo de contato")
                ax.grid(alpha=0.3)
                st.pyplot(fig)

                st.success("✔ Amostra processada com sucesso")

        if st.session_state.tensiometry_samples:
            st.subheader("Resumo físico das amostras")
            st.dataframe(
                pd.DataFrame(st.session_state.tensiometry_samples),
                use_container_width=True
            )

    # =====================================================
    # SUBABA 2 — PCA
    # =====================================================
    with subtabs[1]:

        if len(st.session_state.tensiometry_samples) < 2:
            st.info("Carregue ao menos duas amostras na subaba de processamento.")
            return

        df_pca = pd.DataFrame(st.session_state.tensiometry_samples)

        st.subheader("Dados de entrada da PCA")
        st.dataframe(df_pca, use_container_width=True)

        feature_cols = st.multiselect(
            "Variáveis para PCA",
            options=[c for c in df_pca.columns if c != "Amostra"],
            default=[
                "Theta médio (°)",
                "Volume médio",
                "Área média",
                "Altura média",
            ]
        )

        if len(feature_cols) < 2:
            st.warning("Selecione ao menos duas variáveis.")
            return

        X = df_pca[feature_cols].values
        labels = df_pca["Amostra"].values

        X_scaled = StandardScaler().fit_transform(X)

        pca = PCA(n_components=2)
        scores = pca.fit_transform(X_scaled)
        loadings = pca.components_.T
        explained = pca.explained_variance_ratio_ * 100

        # ---------------------------
        # BIPLOT PADRONIZADO
        # ---------------------------
        fig, ax = plt.subplots(figsize=(6, 6), dpi=300)

        ax.scatter(scores[:, 0], scores[:, 1], s=80, edgecolor="black")

        for i, label in enumerate(labels):
            ax.text(
                scores[i, 0] + 0.03,
                scores[i, 1] + 0.03,
                label,
                fontsize=9
            )

        scale = np.max(np.abs(scores)) * 0.9
        for i, var in enumerate(feature_cols):
            ax.arrow(
                0, 0,
                loadings[i, 0] * scale,
                loadings[i, 1] * scale,
                color="black",
                head_width=0.08,
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
        ax.grid(alpha=0.3)

        st.pyplot(fig)

        st.subheader("Variância explicada")
        st.dataframe(pd.DataFrame({
            "Componente": ["PC1", "PC2"],
            "Variância (%)": explained.round(2)
        }))
