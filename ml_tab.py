# ml_tab.py
# -*- coding: utf-8 -*-

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score


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
        • Treinamento **Random Forest supervisionado**  
        • Predição automática de novas amostras  
        • Painel de recomendação SurfaceXLab  
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
    if "tensiometry_features" in st.session_state:
        data_sources.append(st.session_state.tensiometry_features)

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
        st.session_state.pca_labels = labels
        st.session_state.pca_features = features
        st.session_state.scaler_global = scaler
        st.session_state.pca_model = pca

        # ---------------------------
        # BIPLOT
        # ---------------------------
        fig, ax = plt.subplots(figsize=(7, 7), dpi=300)

        ax.scatter(scores[:, 0], scores[:, 1], s=90, edgecolor="black")

        for i, label in enumerate(labels):
            ax.text(
                scores[i, 0] + 0.03,
                scores[i, 1] + 0.03,
                label,
                fontsize=9
            )

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
    # SUBABA 2 — TREINAMENTO RANDOM FOREST
    # =====================================================
    with subtabs[1]:

        if "pca_scores" not in st.session_state:
            st.warning("Execute primeiro a PCA Global.")
            return

        st.subheader("🧠 Treinamento supervisionado")

        # Simulação inicial: usuário escolhe classe alvo
        target = st.selectbox(
            "Escolha a variável alvo (target)",
            options=list(df_global.columns)
        )

        y = df_global[target].values
        X_ml = st.session_state.pca_scores

        X_train, X_test, y_train, y_test = train_test_split(
            X_ml, y,
            test_size=0.25,
            random_state=42
        )

        model = RandomForestClassifier(
            n_estimators=300,
            random_state=42,
            class_weight="balanced",
            n_jobs=-1
        )

        if st.button("▶ Treinar Random Forest"):

            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            acc = accuracy_score(y_test, y_pred)

            st.session_state.rf_model = model
            st.session_state.rf_target = target

            st.success(f"Modelo treinado com sucesso — Accuracy: {acc:.3f}")

            st.text("Relatório de classificação:")
            st.json(classification_report(y_test, y_pred, output_dict=True))

    # =====================================================
    # SUBABA 3 — PREDIÇÃO NOVA AMOSTRA
    # =====================================================
    with subtabs[2]:

        if "rf_model" not in st.session_state:
            st.info("Treine um modelo antes de usar a predição.")
            return

        st.subheader("🔮 Predição automática — SurfaceXLab")

        st.markdown(
            """
            Utilize os dados já carregados nos módulos físicos
            para realizar **predição automática** via PCA Global + Random Forest.
            """
        )

        sample_name = st.selectbox(
            "Escolha uma amostra existente",
            options=df_global.index.tolist()
        )

        if st.button("▶ Predizer amostra"):

            idx = list(df_global.index).index(sample_name)

            pc_vector = st.session_state.pca_scores[idx].reshape(1, -1)

            model = st.session_state.rf_model

            prediction = model.predict(pc_vector)[0]
            proba = model.predict_proba(pc_vector).max()

            st.success("Predição concluída")

            st.markdown(
                f"""
                ### ✅ Resultado SurfaceXLab

                **Amostra:** {sample_name}  
                **Classe prevista:** `{prediction}`  
                **Confiança do modelo:** `{proba:.2%}`  
                """
            )

            st.markdown(
                """
                ### 📌 Recomendação automática

                A plataforma sugere otimizar parâmetros de processamento
                associados aos componentes dominantes da PCA
                para deslocar a amostra em direção ao cluster alvo.
                """
            )
