# resistividade_tab.py
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
# IDENTIFICA COLUNAS I / V
# =========================================================
def find_IV_columns(df):

    col_I = None
    col_V = None

    for c in df.columns:
        cl = c.lower()
        if "current" in cl or cl in ["i", "corrente"]:
            col_I = c
        if "voltage" in cl or cl in ["v", "tensao", "tensão"]:
            col_V = c

    return col_I, col_V


# =========================================================
# PROCESSAMENTO DA AMOSTRA
# =========================================================
def process_electrical_sample(df, sample_name):

    col_I, col_V = find_IV_columns(df)

    if col_I is None or col_V is None:
        raise ValueError("Colunas de corrente (I) e tensão (V) não identificadas.")

    I = pd.to_numeric(df[col_I], errors="coerce").values
    V = pd.to_numeric(df[col_V], errors="coerce").values

    mask = np.isfinite(I) & np.isfinite(V)
    I, V = I[mask], V[mask]

    if len(I) < 3:
        raise ValueError("Dados insuficientes para ajuste elétrico.")

    # Ajuste linear V = R * I
    coeffs = np.polyfit(I, V, 1)
    resistance = coeffs[0]

    summary = {
        "Amostra": sample_name,
        "Resistência (Ω)": resistance,
        "Corrente média (A)": np.mean(I),
        "Tensão média (V)": np.mean(V),
        "Resistividade (Ω·cm)": resistance,  # placeholder (se não houver geometria)
        "N_pontos": len(I),
    }

    return summary, I, V


# =========================================================
# ABA RESISTIVIDADE
# =========================================================
def render_resistividade_tab(supabase=None):

    st.header("⚡ Propriedades Elétricas — Resistividade")

    st.markdown(
        """
        **Subaba 1**  
        Upload da amostra elétrica → gráfico **V × I** → cálculo da resistividade  

        **Subaba 2**  
        PCA multivariada usando **apenas os resultados calculados**
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
    # SUBABA 1 — PROCESSAMENTO
    # =====================================================
    with subtabs[0]:

        uploaded_files = st.file_uploader(
            "Upload dos arquivos elétricos (.csv, .txt, .xls, .xlsx)",
            type=["csv", "txt", "xls", "xlsx"],
            accept_multiple_files=True
        )

        if uploaded_files:
            for file in uploaded_files:
                st.markdown(f"### 📄 Amostra: `{file.name}`")

                try:
                    df_raw = read_electrical_file(file)

                    summary, I, V = process_electrical_sample(df_raw, file.name)
                    st.session_state.electrical_samples.append(summary)

                    # -----------------------------
                    # GRÁFICO V × I
                    # -----------------------------
                    fig, ax = plt.subplots(figsize=(5, 4), dpi=300)
                    ax.scatter(I, V, s=40, edgecolor="black")
                    ax.plot(I, np.polyval(
                        [summary["Resistência (Ω)"], 0], I
                    ), color="red", lw=1.5)

                    ax.set_xlabel("Corrente (A)")
                    ax.set_ylabel("Tensão (V)")
                    ax.set_title("Curva V × I")
                    ax.grid(alpha=0.3)

                    st.pyplot(fig)

                    st.success("✔ Amostra processada com sucesso")

                except Exception as e:
                    st.error(f"Erro ao processar {file.name}")
                    st.exception(e)

        if st.session_state.electrical_samples:
            st.subheader("Resumo elétrico das amostras")
            st.dataframe(pd.DataFrame(st.session_state.electrical_samples))

    # =====================================================
    # SUBABA 2 — PCA
    # =====================================================
    with subtabs[1]:

        if len(st.session_state.electrical_samples) < 2:
            st.info("Carregue ao menos duas amostras na subaba de processamento.")
            return

        df_pca = pd.DataFrame(st.session_state.electrical_samples)

        st.subheader("Dados de entrada da PCA")
        st.dataframe(df_pca)

        feature_cols = st.multiselect(
            "Variáveis elétricas para PCA",
            options=[c for c in df_pca.columns if c != "Amostra"],
            default=[
                "Resistência (Ω)",
                "Resistividade (Ω·cm)",
                "Corrente média (A)",
                "Tensão média (V)",
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
