# ml_tab.py
# -*- coding: utf-8 -*-

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    r2_score,
    mean_absolute_error
)


# =========================================================
# UTIL — CONTRIBUIÇÃO QUALITATIVA
# =========================================================
def qualitative_contribution(value):
    if value >= 0.7:
        return "Alta"
    if value >= 0.4:
        return "Média"
    return "Baixa"


# =========================================================
# ABA ML — SurfaceXLab
# =========================================================
def render_ml_tab(supabase=None):

    st.header("🤖 Inteligência Artificial — SurfaceXLab")

    st.markdown(
        """
        Este módulo executa:

        • Integração **Raman + Tensiometria + Elétrica**  
        • **PCA Global** (redução dimensional)  
        • Treinamento supervisionado (**Random Forest**)  
        • Predição automática de novas amostras  
        • Painel de recomendação inteligente  
        """
    )

    subtabs = st.tabs([
        "📊 PCA Global",
        "🧠 Treinar Modelo",
        "🔮 Predizer Nova Amostra"
    ])

    # =====================================================
    # COLETA DOS DADOS
    # =====================================================
    data_sources = []

    # Raman
    if "raman_peaks" in st.session_state:
        df_raman = (
            pd.DataFrame(st.session_state.raman_peaks)
            .T
            .fillna(0.0)
            .reset_index()
            .rename(columns={"index": "Amostra"})
        )
        data_sources.append(df_raman)

    # Tensiometria
    if "tensiometry_samples" in st.session_state:
        df_tensio = pd.DataFrame(st.session_state.tensiometry_samples.values())
        data_sources.append(df_tensio)

    # Elétrica
    if "electrical_samples" in st.session_state:
        df_eletric = pd.DataFrame(st.session_state.electrical_samples.values())
        data_sources.append(df_eletric)

    if not data_sources:
        st.info("Nenhum dado disponível ainda. Execute os módulos primeiro.")
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
            df_global = pd.merge(
                df_global,
                df,
                on="Amostra",
                how="outer"
            )

    if df_global is None or len(df_global) < 2:
        st.warning("Dados insuficientes para análise global.")
        return

    df_global = df_global.set_index("Amostra")
    df_global = df_global.apply(pd.to_numeric, errors="coerce")
    df_global = df_global.fillna(0.0)

    # =====================================================
    # SUBABA 1 — PCA GLOBAL
    # =====================================================
    with subtabs[0]:

        st.subheader("📊 Matriz global integrada")
        st.dataframe(df_global, use_container_width=True)

        X = df_global.values
        labels = df_global.index.values
        features = df_global.columns.values

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        pca = PCA(n_components=2)
        scores = pca.fit_transform(X_scaled)

        loadings = pca.components_.T
        explained = pca.explained_variance_ratio_ * 100

        # Salva para ML
        st.session_state.pca_scores = scores
        st.session_state.scaler_global = scaler
        st.session_state.pca_model = pca
        st.session_state.df_global_ml = df_global

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
                alpha=0.6,
                color="black",
                head_width=0.05,
                length_includes_head=True
            )

        ax.axhline(0, color="gray", lw=0.6)
        ax.axvline(0, color="gray", lw=0.6)

        ax.set_xlabel(f"PC1 ({explained[0]:.1f}%)")
        ax.set_ylabel(f"PC2 ({explained[1]:.1f}%)")
        ax.set_title("PCA Global — SurfaceXLab")
        ax.set_aspect("equal", adjustable="box")
        ax.grid(alpha=0.3)

        st.pyplot(fig)

        st.subheader("Variância explicada")
        st.dataframe(pd.DataFrame({
            "Componente": ["PC1", "PC2"],
            "Variância (%)": explained.round(2)
        }))

        # ---------------------------
        # Contribuição qualitativa
        # ---------------------------
        st.subheader("📋 Contribuição qualitativa das variáveis")

        contrib = np.abs(loadings)
        contrib_norm = contrib / contrib.max(axis=0)

        contrib_table = pd.DataFrame(
            contrib_norm,
            index=features,
            columns=["PC1", "PC2"]
        )

        contrib_table["PC1"] = contrib_table["PC1"].apply(qualitative_contribution)
        contrib_table["PC2"] = contrib_table["PC2"].apply(qualitative_contribution)

        st.dataframe(contrib_table, use_container_width=True)

    # =====================================================
    # SUBABA 2 — TREINAMENTO
    # =====================================================
    with subtabs[1]:

        if "pca_scores" not in st.session_state:
            st.warning("Execute primeiro a PCA Global.")
            return

        st.subheader("🧠 Treinamento supervisionado")

        df_ml = st.session_state.df_global_ml

        task_type = st.selectbox(
            "Tipo de problema",
            ["Regressão (predizer valor físico)", "Classificação (predizer classe)"]
        )

        target = st.selectbox(
            "Variável alvo (target)",
            options=df_ml.columns.tolist()
        )

        y = df_ml[target].values
        X_ml = st.session_state.pca_scores

        X_train, X_test, y_train, y_test = train_test_split(
            X_ml, y,
            test_size=0.25,
            random_state=42
        )

        if st.button("▶ Treinar Random Forest"):

            if task_type.startswith("Regressão"):

                model = RandomForestRegressor(
                    n_estimators=400,
                    random_state=42,
                    n_jobs=-1
                )

                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)

                r2 = r2_score(y_test, y_pred)
                mae = mean_absolute_error(y_test, y_pred)

                st.success(f"Modelo treinado — R² = {r2:.3f} | MAE = {mae:.3e}")

            else:

                model = RandomForestClassifier(
                    n_estimators=300,
                    random_state=42,
                    class_weight="balanced",
                    n_jobs=-1
                )

                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)

                acc = accuracy_score(y_test, y_pred)

                st.success(f"Modelo treinado — Accuracy = {acc:.3f}")
                st.json(classification_report(y_test, y_pred, output_dict=True))

            st.session_state.rf_model = model
            st.session_state.rf_task = task_type
            st.session_state.rf_target = target

    # =====================================================
    # SUBABA 3 — PREDIÇÃO
    # =====================================================
    with subtabs[2]:

        if "rf_model" not in st.session_state:
            st.info("Treine um modelo antes de realizar predições.")
            return

        st.subheader("🔮 Predição automática — SurfaceXLab")

        sample_name = st.selectbox(
            "Selecione uma amostra existente",
            options=df_global.index.tolist()
        )

        if st.button("▶ Predizer"):

            idx = list(df_global.index).index(sample_name)

            pc_vector = st.session_state.pca_scores[idx].reshape(1, -1)

            model = st.session_state.rf_model

            prediction = model.predict(pc_vector)[0]

            if st.session_state.rf_task.startswith("Classificação"):

                proba = model.predict_proba(pc_vector).max()

                st.success("Predição concluída")

                st.markdown(
                    f"""
                    ### ✅ Resultado SurfaceXLab

                    **Amostra:** {sample_name}  
                    **Classe prevista:** `{prediction}`  
                    **Confiança:** `{proba:.2%}`  
                    """
                )

            else:

                st.success("Predição concluída")

                st.markdown(
                    f"""
                    ### ✅ Resultado SurfaceXLab

                    **Amostra:** {sample_name}  
                    **{st.session_state.rf_target}:**

                    **{prediction:.5e}**
                    """
                )

            st.markdown(
                """
                ### 📌 Recomendação automática

                Utilize as variáveis dominantes da PCA Global
                para ajustar parâmetros de processamento
                e deslocar a amostra em direção à região ótima do espaço multivariado.
                """
            )
