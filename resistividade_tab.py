# -*- coding: utf-8 -*-

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


# =========================================================
# LEITURA ROBUSTA DO ARQUIVO ELÉTRICO
# =========================================================
def read_electrical_file(file):
    if file.name.lower().endswith((".xls", ".xlsx")):
        df = pd.read_excel(file)
    else:
        df = pd.read_csv(file, sep=None, engine="python")

    df.columns = [c.strip() for c in df.columns]

    return df


# =========================================================
# PROCESSAMENTO DA AMOSTRA ELÉTRICA
# =========================================================
def process_electrical_sample(df, sample_name):
    """
    Consolida uma amostra elétrica em uma única linha
    """

    summary = {
        "Amostra": sample_name,
    }

    for col in df.columns:
        if df[col].dtype.kind in "if":
            summary[col] = df[col].mean()

    return summary


# =========================================================
# ABA RESISTIVIDADE (UPLOAD + PCA)
# =========================================================
def render_resistividade_tab(supabase=None):
    st.header("⚡ Propriedades Elétricas — Resistividade")

    st.markdown(
        """
        **Subaba 1**: Upload e processamento de arquivos elétricos  
        **Subaba 2**: PCA multivariada usando os dados processados
        """
    )

    if "electrical_samples" not in st.session_state:
        st.session_state.electrical_samples = []

    # =====================================================
    # SUBABAS
    # =====================================================
    subtabs = st.tabs([
        "📐 Upload & Processamento",
        "📊 PCA — Elétrica"
    ])

    # =====================================================
    # SUBABA 1 — UPLOAD
    # =====================================================
    with subtabs[0]:

        uploaded_files = st.file_uploader(
            "Upload dos arquivos elétricos (.csv, .xls, .txt)",
            type=["csv", "txt", "xls", "xlsx"],
            accept_multiple_files=True
        )

        if uploaded_files:
            for file in uploaded_files:
                try:
                    df_raw = read_electrical_file(file)

                    if df_raw.empty:
                        st.warning(f"{file.name} ignorado (arquivo vazio).")
                        continue

                    summary = process_electrical_sample(df_raw, file.name)
                    st.session_state.electrical_samples.append(summary)

                    st.success(f"{file.name} processado com sucesso.")

                except Exception as e:
                    st.error(f"Erro ao processar {file.name}")
                    st.exception(e)

        if st.session_state.electrical_samples:
            df_samples = pd.DataFrame(st.session_state.electrical_samples)
            st.subheader("Amostras elétricas consolidadas")
            st.dataframe(df_samples)

    # =====================================================
    # SUBABA 2 — PCA
    # =====================================================
    with subtabs[1]:

        if not st.session_state.electrical_samples:
            st.info("Nenhuma amostra processada ainda.")
            return

        df_pca = pd.DataFrame(st.session_state.electrical_samples)

        st.subheader("Dados de entrada da PCA")
        st.dataframe(df_pca)

        feature_cols = st.multiselect(
            "Variáveis elétricas para PCA",
            options=[c for c in df_pca.columns if c != "Amostra"],
            default=[c for c in df_pca.columns if c != "Amostra"][:3]
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
            ax.text(scores[i, 0] + 0.03, scores[i, 1] + 0.03, label, fontsize=9)

        scale = np.max(np.abs(scores)) * 0.9
        for i, var in enumerate(feature_cols):
            ax.arrow(
                0, 0,
                loadings[i, 0] * scale,
                loadings[i, 1] * scale,
                color="red",
                head_width=0.08,
                length_includes_head=True
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
        ax.set_title("PCA — Propriedades Elétricas")
        ax.grid(alpha=0.3)

        st.pyplot(fig)

        st.subheader("Variância explicada")
        st.dataframe(pd.DataFrame({
            "Componente": ["PC1", "PC2"],
            "Variância (%)": explained.round(2)
        }))
