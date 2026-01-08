# -*- coding: utf-8 -*-

"""
SurfaceXLab — Otimizador IA
PCA Multivariado + Machine Learning Supervisionado
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from ml_models import (
    random_forest_cv,
    temporal_pca,
)


# =========================================================
# UI — ABA ML / IA
# =========================================================
def render_ml_tab(supabase):

    st.header("🤖 Otimizador — Análise Multivariada e IA")

    st.markdown(
        """
        Este módulo integra **Análise de Componentes Principais (PCA)** e
        **Machine Learning supervisionado**, utilizando fingerprints
        experimentais previamente extraídos e armazenados no banco.
        """
    )

    # -----------------------------------------------------
    # 1️⃣ Carregar fingerprints
    # -----------------------------------------------------
    st.subheader("📦 Base de dados")

    try:
        res = supabase.table("raman_fingerprints").select("*").execute()
        data = res.data if res.data else []
    except Exception as e:
        st.error("Erro ao carregar fingerprints do banco.")
        st.exception(e)
        return

    if not data:
        st.warning("Nenhum fingerprint disponível no banco.")
        return

    df = pd.DataFrame(data)

    st.dataframe(
        df.head(50),
        use_container_width=True,
        key="ml_fingerprint_preview"
    )

    # -----------------------------------------------------
    # 2️⃣ Seleção da variável alvo
    # -----------------------------------------------------
    label_col = st.selectbox(
        "Variável alvo (classe ou resposta)",
        options=[c for c in df.columns if c not in ("id", "created_at")],
        key="ml_target_select"
    )

    X = df.drop(columns=[label_col, "id", "created_at"], errors="ignore")
    y = df[label_col]

    numeric_cols = X.select_dtypes(include="number").columns.tolist()

    if len(numeric_cols) < 2:
        st.warning("Número insuficiente de variáveis numéricas para PCA.")
        return

    # -----------------------------------------------------
    # 3️⃣ PCA MULTIVARIADO
    # -----------------------------------------------------
    st.divider()
    st.subheader("📉 Análise de Componentes Principais (PCA)")

    n_components = st.slider(
        "Número de componentes principais",
        min_value=2,
        max_value=min(6, len(numeric_cols)),
        value=2,
        key="ml_pca_n_components"
    )

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X[numeric_cols])

    pca = PCA(n_components=n_components)
    scores = pca.fit_transform(X_scaled)

    scores_df = pd.DataFrame(
        scores,
        columns=[f"PC{i+1}" for i in range(n_components)],
        index=df.index
    )

    explained = pca.explained_variance_ratio_ * 100

    st.markdown(
        f"""
        **Variância explicada:**
        - PC1: {explained[0]:.2f} %
        - PC2: {explained[1]:.2f} %
        """,
        key="ml_pca_explained"
    )

    st.dataframe(
        scores_df,
        use_container_width=True,
        key="ml_pca_scores"
    )

    # -----------------------------------------------------
    # 4️⃣ Loadings (interpretação física)
    # -----------------------------------------------------
    st.subheader("🧠 Loadings — contribuição das variáveis")

    loadings_df = pd.DataFrame(
        pca.components_.T,
        index=numeric_cols,
        columns=[f"PC{i+1}" for i in range(n_components)]
    )

    st.dataframe(
        loadings_df.sort_values("PC1", key=np.abs, ascending=False),
        key="ml_pca_loadings"
    )

    # -----------------------------------------------------
    # 5️⃣ PCA TEMPORAL (opcional)
    # -----------------------------------------------------
    if "created_at" in df.columns and "sample_code" in df.columns:

        st.divider()
        st.subheader("⏱ PCA Multi-Amostra Temporal")

        run_temporal = st.button(
            "Executar PCA Temporal",
            key="ml_temporal_pca_button"
        )

        if run_temporal:
            out = temporal_pca(
                df,
                feature_cols=numeric_cols,
                sample_col="sample_code",
                time_col="created_at"
            )

            df_pca_t = out["df_pca"]
            explained_t = out["explained_variance"] * 100

            fig, ax = plt.subplots(figsize=(7, 5))
            for sample in df_pca_t["sample_code"].unique():
                d = df_pca_t[df_pca_t["sample_code"] == sample].sort_values("created_at")
                ax.plot(d["PC1"], d["PC2"], marker="o", label=sample)

            ax.set_xlabel("PC1")
            ax.set_ylabel("PC2")
            ax.set_title("PCA Temporal — Evolução das Amostras")
            ax.legend()
            ax.grid(alpha=0.3)

            st.pyplot(fig, key="ml_temporal_pca_plot")

            st.markdown(
                f"""
                **Variância explicada (temporal):**
                - PC1: {explained_t[0]:.2f} %
                - PC2: {explained_t[1]:.2f} %
                """
            )

    # -----------------------------------------------------
    # 6️⃣ RANDOM FOREST + VALIDAÇÃO CRUZADA
    # -----------------------------------------------------
    st.divider()
    st.subheader("🌲 Random Forest com Validação Cruzada")

    task_type = st.selectbox(
        "Tipo de tarefa",
        ["classification", "regression"],
        key="ml_rf_task"
    )

    run_rf = st.button(
        "Executar Random Forest + CV",
        key="ml_rf_button"
    )

    if run_rf:
        X_rf = scores_df.dropna()
        y_rf = y.loc[X_rf.index]

        out = random_forest_cv(
            X_rf,
            y_rf,
            task=task_type,
            cv=5
        )

        st.success("✔ Random Forest executado com sucesso")

        st.markdown(
            f"""
            **Validação cruzada (5-fold):**
            - Média: {out['cv_mean']:.3f}
            - Desvio padrão: {out['cv_std']:.3f}
            """
        )

        st.subheader("📌 Importância das componentes (PCs)")
        st.bar_chart(
            out["feature_importance"],
            key="ml_rf_importance"
        )

    st.success("Pipeline PCA + IA pronto para uso científico.")
