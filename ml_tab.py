# -*- coding: utf-8 -*-

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

st.write("RAMAN:", "raman_fingerprint" in st.session_state)
st.write("TENSIO:", "tensiometry_samples" in st.session_state)
st.write("ELETR:", "electrical_samples" in st.session_state)
st.write("RF:", st.session_state.rf_model is not None)


from ml_engine import (
    train_random_forest_classifier,
    train_random_forest_regressor,
)

# =========================================================
# FUNÇÃO AUXILIAR — CONTRIBUIÇÃO QUALITATIVA
# =========================================================
def qualitative_contribution(value):
    if value >= 0.7:
        return "Alta"
    if value >= 0.4:
        return "Média"
    return "Baixa"


# =========================================================
# ABA ML — PCA GLOBAL + RANDOM FOREST
# =========================================================
def render_ml_tab(supabase=None):

    st.header("🤖 Machine Learning — PCA Global & Predição")

    st.markdown(
        """
        Este módulo integra **Raman + Tensiometria + Resistividade**
        em um **pipeline inteligente**, composto por:

        • PCA Global (estrutura multivariada)  
        • Random Forest supervisionado  
        • Predição de novas amostras  
        • Recomendação automática SurfaceXLab  
        """
    )

    # =====================================================
    # SESSION STATE
    # =====================================================
    for key in [
        "global_df",
        "pca_model",
        "scaler",
        "rf_model",
        "pca_scores",
        "pca_loadings",
        "feature_names",
    ]:
        if key not in st.session_state:
            st.session_state[key] = None

    subtabs = st.tabs([
        "1 PCA Global",
        "2 Treinamento ML",
        "3 Predizer nova amostra",
    ])

    # =====================================================
    # COLETA DOS DADOS DAS ABAS
    # =====================================================
    data_sources = []

    if "raman_fingerprint" in st.session_state:
        data_sources.append(st.session_state.raman_fingerprint)

    if "tensiometry_samples" in st.session_state:
        data_sources.append(pd.DataFrame(st.session_state.tensiometry_samples.values()))

    if "electrical_samples" in st.session_state:
        data_sources.append(pd.DataFrame(st.session_state.electrical_samples.values()))

    if not data_sources:
        st.info("Nenhum dado disponível. Execute ao menos um módulo experimental.")
        return

    # =====================================================
    # MERGE GLOBAL
    # =====================================================
    df_global = None

    for df in data_sources:
        if "Amostra" not in df.columns:
            continue

        if df_global is None:
            df_global = df.copy()
        else:
            df_global = pd.merge(df_global, df, on="Amostra", how="outer")

    if df_global is None or len(df_global) < 2:
        st.warning("Dados insuficientes para PCA Global.")
        return

    df_global = df_global.set_index("Amostra")
    df_global = df_global.apply(pd.to_numeric, errors="coerce").fillna(0.0)

    st.session_state.global_df = df_global

    # =====================================================
    # SUBABA 1 — PCA GLOBAL
    # =====================================================
    with subtabs[0]:

        st.subheader("Matriz global integrada")
        st.dataframe(df_global, use_container_width=True)

        X = df_global.values
        labels = df_global.index.values
        features = df_global.columns.values

        scaler = StandardScaler()
        Xs = scaler.fit_transform(X)

        pca = PCA(n_components=2)
        scores = pca.fit_transform(Xs)
        loadings = pca.components_.T
        explained = pca.explained_variance_ratio_ * 100

        st.session_state.scaler = scaler
        st.session_state.pca_model = pca
        st.session_state.pca_scores = scores
        st.session_state.pca_loadings = loadings
        st.session_state.feature_names = features

        # ---------------------------
        # BIPLOT
        # ---------------------------
        fig, ax = plt.subplots(figsize=(7, 7), dpi=300)

        ax.scatter(scores[:, 0], scores[:, 1], s=90, edgecolor="black")

        for i, label in enumerate(labels):
            ax.text(scores[i, 0] + 0.03, scores[i, 1] + 0.03, label, fontsize=9)

        scale = np.max(np.abs(scores)) * 0.85
        for i, var in enumerate(features):
            ax.arrow(
                0, 0,
                loadings[i, 0] * scale,
                loadings[i, 1] * scale,
                color="black",
                alpha=0.6,
                head_width=0.06,
                length_includes_head=True
            )
            ax.text(
                loadings[i, 0] * scale * 1.1,
                loadings[i, 1] * scale * 1.1,
                var,
                fontsize=8
            )

        ax.set_xlabel(f"PC1 ({explained[0]:.1f}%)")
        ax.set_ylabel(f"PC2 ({explained[1]:.1f}%)")
        ax.set_title("PCA Global — SurfaceXLab")
        ax.grid(alpha=0.3)
        ax.set_aspect("equal", adjustable="box")

        st.pyplot(fig)

        st.subheader("Variância explicada")
        st.dataframe(pd.DataFrame({
            "Componente": ["PC1", "PC2"],
            "Variância (%)": explained.round(2)
        }))

    # =====================================================
    # SUBABA 2 — TREINAMENTO ML
    # =====================================================
    with subtabs[1]:

        st.subheader("Treinamento supervisionado")

        task = st.selectbox(
            "Tipo de problema",
            ["Classificação", "Regressão"]
        )

        target_col = st.selectbox(
            "Variável alvo",
            options=df_global.columns
        )

        X_ml = pd.DataFrame(
            st.session_state.pca_scores,
            columns=["PC1", "PC2"],
            index=df_global.index
        )

        y_ml = df_global[target_col]

        if st.button("▶ Treinar modelo Random Forest"):

            if task == "Classificação":
                result = train_random_forest_classifier(X_ml, y_ml)
                st.metric("Accuracy", f"{result['accuracy']*100:.2f}%")

            else:
                result = train_random_forest_regressor(X_ml, y_ml)
                st.metric("R²", f"{result['r2']:.3f}")
                st.metric("MAE", f"{result['mae']:.3f}")

            st.session_state.rf_model = result["model"]

            st.subheader("Importância dos componentes")
            st.dataframe(result["feature_importance"])

            # ---------------------------
            # Recomendações
            # ---------------------------
            dominant_pc = result["feature_importance"].idxmax()

            st.subheader("Recomendação automática SurfaceXLab")
            st.markdown(
                f"""
                🔎 O modelo indica que **{dominant_pc}** é o principal
                componente responsável pela resposta do sistema.

                ➡ Recomenda-se priorizar ajustes experimentais que
                influenciem as variáveis fortemente correlacionadas a este
                componente principal.
                """
            )

    # =====================================================
    # SUBABA 3 — PREDIÇÃO DE NOVA AMOSTRA
    # =====================================================
    with subtabs[2]:

        if st.session_state.rf_model is None:
            st.info("Treine um modelo antes de realizar predições.")
            return

        st.subheader("Predição de nova amostra")

        new_sample = {}

        for col in df_global.columns:
            new_sample[col] = st.number_input(col, value=0.0)

        if st.button("Predizer"):

            new_df = pd.DataFrame([new_sample])

            X_new = st.session_state.scaler.transform(new_df)
            PCs_new = st.session_state.pca_model.transform(X_new)

            prediction = st.session_state.rf_model.predict(PCs_new)

            st.success(f"Resultado previsto: **{prediction[0]}**")
