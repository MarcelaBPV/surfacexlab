# ml_tab.py
# -*- coding: utf-8 -*-

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, accuracy_score, classification_report


# =====================================================
# UTIL — LEITURA DE ARQUIVO
# =====================================================
def load_file(uploaded_file):

    if uploaded_file.name.endswith(".xlsx"):
        df = pd.read_excel(uploaded_file)

    elif uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)

    else:
        df = pd.read_csv(uploaded_file, sep=None, engine="python")

    return df


# =====================================================
# ABA OTIMIZAÇÃO — PCA + ML
# =====================================================
def render_ml_tab():

    st.header("⚙ Otimização Inteligente — SurfaceXLab")

    st.markdown("""
    Este módulo permite:

    • Exploração multivariada (PCA)  
    • Integração de dados experimentais  
    • Modelagem preditiva (Machine Learning)  
    • Otimização orientada por dados  
    """)

    subtabs = st.tabs([
        "📊 PCA Exploratório",
        "🤖 Machine Learning"
    ])

    # =====================================================
    # SUBABA 1 — PCA
    # =====================================================
    with subtabs[0]:

        st.subheader("📊 PCA Exploratório")

        data_source = st.radio(
            "Fonte dos dados",
            ["Upload de arquivo", "Dados da plataforma"],
            horizontal=True
        )

        # ---------- UPLOAD ----------
        if data_source == "Upload de arquivo":

            uploaded_file = st.file_uploader(
                "Upload XLS, CSV ou TXT",
                type=["xlsx", "csv", "txt"]
            )

            if uploaded_file is None:
                st.info("Faça upload do arquivo para continuar.")
                return

            df = load_file(uploaded_file)

        # ---------- SISTEMA ----------
        else:

            if "df_global_ml" not in st.session_state:
                st.warning("Nenhum dado integrado disponível. Execute módulos de análise primeiro.")
                return

            df = st.session_state.df_global_ml.reset_index()

        # Preview
        st.subheader("Pré-visualização dos dados")
        st.dataframe(df, use_container_width=True)

        # Seleção da coluna amostra
        sample_col = st.selectbox(
            "Coluna identificadora da amostra",
            df.columns.tolist()
        )

        df = df.set_index(sample_col)

        # Conversão numérica
        df = df.apply(pd.to_numeric, errors="coerce")
        df = df.fillna(0)

        if df.shape[0] < 2:
            st.warning("Mínimo de 2 amostras necessário.")
            return

        if df.shape[1] < 2:
            st.warning("Mínimo de 2 variáveis necessário.")
            return

        # PCA config
        n_components = st.slider(
            "Número de Componentes Principais",
            2,
            min(10, df.shape[1]),
            2
        )

        X = df.values
        labels = df.index.values
        features = df.columns.values

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        pca = PCA(n_components=n_components)
        scores = pca.fit_transform(X_scaled)

        loadings = pca.components_.T
        explained = pca.explained_variance_ratio_ * 100

        # Guarda no estado global
        st.session_state.opt_scores = scores
        st.session_state.opt_labels = labels
        st.session_state.opt_features = features
        st.session_state.opt_scaler = scaler
        st.session_state.opt_pca = pca
        st.session_state.opt_df = df

        # =====================================================
        # BIPLOT ESTILO CIENTÍFICO
        # =====================================================
        st.subheader("PCA — Biplot")

        fig, ax = plt.subplots(figsize=(6, 6), dpi=300)

        ax.scatter(
            scores[:, 0],
            scores[:, 1],
            s=70,
            edgecolors="black",
            linewidths=0.6,
            zorder=3
        )

        for i, label in enumerate(labels):
            ax.text(
                scores[i, 0],
                scores[i, 1],
                label,
                fontsize=9,
                ha="left",
                va="bottom"
            )

        scale = np.max(np.abs(scores)) * 0.9

        for i, var in enumerate(features):

            ax.arrow(
                0, 0,
                loadings[i, 0] * scale,
                loadings[i, 1] * scale,
                head_width=0.04,
                head_length=0.06,
                linewidth=1.1,
                length_includes_head=True
            )

            ax.text(
                loadings[i, 0] * scale * 1.05,
                loadings[i, 1] * scale * 1.05,
                var,
                fontsize=9,
                ha="center",
                va="center"
            )

        ax.axhline(0, linewidth=0.8)
        ax.axvline(0, linewidth=0.8)

        ax.set_xlabel(f"PC1 ({explained[0]:.1f}%)")
        ax.set_ylabel(f"PC2 ({explained[1]:.1f}%)")

        ax.margins(0)
        plt.tight_layout(pad=0)

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        ax.tick_params(direction="in", length=5, width=1)
        ax.set_aspect("equal", adjustable="box")

        ax.grid(alpha=0.15, linestyle="--")

        st.pyplot(fig)

        # Variância
        st.subheader("Variância explicada")

        var_df = pd.DataFrame({
            "Componente": [f"PC{i+1}" for i in range(len(explained))],
            "Variância (%)": explained.round(2)
        })

        st.dataframe(var_df, use_container_width=True)

    # =====================================================
    # SUBABA 2 — MACHINE LEARNING
    # =====================================================
    with subtabs[1]:

        st.subheader("🤖 Machine Learning — Predição e Otimização")

        if "opt_scores" not in st.session_state:
            st.info("Execute a PCA antes de utilizar o módulo de IA.")
            return

        df_ml = st.session_state.opt_df

        task_type = st.selectbox(
            "Tipo de problema",
            ["Regressão (predição de valor físico)",
             "Classificação (predição de classe)"]
        )

        target = st.selectbox(
            "Variável alvo (target)",
            df_ml.columns.tolist()
        )

        y = df_ml[target].values
        X_ml = st.session_state.opt_scores

        X_train, X_test, y_train, y_test = train_test_split(
            X_ml,
            y,
            test_size=0.25,
            random_state=42
        )

        if st.button("▶ Treinar Modelo"):

            if task_type.startswith("Regressão"):

                model = RandomForestRegressor(
                    n_estimators=400,
                    random_state=42,
                    n_jobs=-1
                )

                model.fit(X_train, y_train)

                y_pred = model.predict(X_test)

                r2 = r2_score(y_test, y_pred)

                st.success(f"Modelo treinado — R² = {r2:.4f}")

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

                st.success(f"Modelo treinado — Accuracy = {acc:.4f}")

                st.json(classification_report(y_test, y_pred, output_dict=True))

            st.session_state.opt_model = model
            st.session_state.opt_target = target
            st.session_state.opt_task = task_type

        # =====================================================
        # PREDIÇÃO
        # =====================================================
        if "opt_model" in st.session_state:

            st.subheader("Predição automática")

            sample_select = st.selectbox(
                "Selecionar amostra",
                st.session_state.opt_labels
            )

            if st.button("▶ Predizer"):

                idx = list(st.session_state.opt_labels).index(sample_select)

                pc_vector = st.session_state.opt_scores[idx].reshape(1, -1)

                model = st.session_state.opt_model

                pred = model.predict(pc_vector)[0]

                st.success("Predição concluída")

                st.markdown(f"""
                ### Resultado SurfaceXLab

                **Amostra:** {sample_select}  
                **{st.session_state.opt_target}:** `{pred}`
                """)

                st.info("""
                Recomendação:
                Utilize as direções dominantes das componentes principais
                para ajustar parâmetros experimentais e deslocar o sistema
                em direção à região ótima do espaço multivariado.
                """)

