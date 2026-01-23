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
# SALVAR PCA NO SUPABASE
# =====================================================
def save_pca_supabase(supabase, labels, scores, explained):

    exp = supabase.table("experiments").insert({
        "experiment_name": "PCA Otimização",
        "module": "PCA"
    }).execute()

    exp_id = exp.data[0]["id"]

    rows = []

    for i, sample in enumerate(labels):
        rows.append({
            "experiment_id": exp_id,
            "sample_name": sample,
            "pc1": float(scores[i, 0]),
            "pc2": float(scores[i, 1]),
            "explained_pc1": float(explained[0]),
            "explained_pc2": float(explained[1])
        })

    supabase.table("pca_results").insert(rows).execute()

    return exp_id


# =====================================================
# SALVAR MODELO ML (VOCÊ JÁ POSSUI ml_models)
# =====================================================
def save_model_supabase(
    supabase,
    experiment_id,
    model_type,
    target,
    metric
):

    supabase.table("ml_models").insert({
        "experiment_id": experiment_id,
        "model_type": model_type,
        "target_variable": target,
        "metric_value": float(metric)
    }).execute()


# =====================================================
# INTERFACE STREAMLIT
# =====================================================
def render_ml_tab(supabase=None):

    st.header("🤖 Otimizador Inteligente — PCA + Machine Learning")

    subtabs = st.tabs([
        "📊 PCA Exploratório",
        "🧠 Machine Learning"
    ])

    # =====================================================
    # SUBABA PCA
    # =====================================================
    with subtabs[0]:

        st.subheader("📊 PCA Exploratório")

        source = st.radio(
            "Fonte dos dados",
            ["Upload de Arquivo", "Dados Integrados da Plataforma"],
            horizontal=True
        )

        # ---------------- UPLOAD ----------------
        if source == "Upload de Arquivo":

            uploaded_file = st.file_uploader(
                "Upload XLS, CSV ou TXT",
                type=["xlsx", "csv", "txt"]
            )

            if uploaded_file is None:
                st.info("Faça upload do arquivo.")
                return

            df = load_file(uploaded_file)

        # ---------------- SISTEMA ----------------
        else:

            if "df_global_ml" not in st.session_state:
                st.warning("Nenhum dado integrado disponível.")
                return

            df = st.session_state.df_global_ml.reset_index()

        st.dataframe(df, use_container_width=True)

        sample_col = st.selectbox(
            "Coluna identificadora da amostra",
            df.columns.tolist()
        )

        df = df.set_index(sample_col)
        df = df.apply(pd.to_numeric, errors="coerce").fillna(0)

        if df.shape[0] < 2 or df.shape[1] < 2:
            st.warning("Dados insuficientes para PCA.")
            return

        n_components = st.slider(
            "Número de componentes principais",
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

        # Guarda para ML
        st.session_state.opt_scores = scores
        st.session_state.opt_labels = labels
        st.session_state.opt_df = df
        st.session_state.opt_scaler = scaler
        st.session_state.opt_pca = pca

        # =====================================================
        # BIPLOT CIENTÍFICO
        # =====================================================
        fig, ax = plt.subplots(figsize=(6, 6), dpi=300)

        ax.scatter(
            scores[:, 0],
            scores[:, 1],
            s=70,
            edgecolors="black",
            linewidths=0.6
        )

        for i, label in enumerate(labels):
            ax.text(scores[i, 0], scores[i, 1], label, fontsize=9)

        scale = np.max(np.abs(scores)) * 0.9

        for i, var in enumerate(features):

            ax.arrow(
                0, 0,
                loadings[i, 0] * scale,
                loadings[i, 1] * scale,
                head_width=0.04,
                linewidth=1.1,
                length_includes_head=True
            )

            ax.text(
                loadings[i, 0] * scale * 1.05,
                loadings[i, 1] * scale * 1.05,
                var,
                fontsize=9
            )

        ax.axhline(0, lw=0.8)
        ax.axvline(0, lw=0.8)

        ax.set_xlabel(f"PC1 ({explained[0]:.1f}%)")
        ax.set_ylabel(f"PC2 ({explained[1]:.1f}%)")

        ax.margins(0)
        plt.tight_layout(pad=0)

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(alpha=0.2, linestyle="--")

        st.pyplot(fig)

        # =====================================================
        # SALVAR PCA
        # =====================================================
        if supabase and st.button("💾 Salvar PCA no Banco"):

            exp_id = save_pca_supabase(
                supabase,
                labels,
                scores,
                explained
            )

            st.success("PCA salvo com sucesso!")
            st.caption(f"Experimento ID: {exp_id}")

            st.session_state.current_experiment_id = exp_id

    # =====================================================
    # SUBABA MACHINE LEARNING
    # =====================================================
    with subtabs[1]:

        st.subheader("🧠 Machine Learning")

        if "opt_scores" not in st.session_state:
            st.info("Execute a PCA antes de treinar modelos.")
            return

        df_ml = st.session_state.opt_df

        task_type = st.selectbox(
            "Tipo de problema",
            [
                "Regressão (valor físico)",
                "Classificação (classe)"
            ]
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

            # ---------------- REGRESSÃO ----------------
            if task_type.startswith("Regressão"):

                model = RandomForestRegressor(
                    n_estimators=400,
                    random_state=42,
                    n_jobs=-1
                )

                model.fit(X_train, y_train)

                y_pred = model.predict(X_test)

                metric = r2_score(y_test, y_pred)

                st.success(f"Modelo treinado — R² = {metric:.4f}")

            # ---------------- CLASSIFICAÇÃO ----------------
            else:

                model = RandomForestClassifier(
                    n_estimators=300,
                    random_state=42,
                    class_weight="balanced",
                    n_jobs=-1
                )

                model.fit(X_train, y_train)

                y_pred = model.predict(X_test)

                metric = accuracy_score(y_test, y_pred)

                st.success(f"Modelo treinado — Accuracy = {metric:.4f}")

                st.json(classification_report(y_test, y_pred, output_dict=True))

            st.session_state.opt_model = model
            st.session_state.opt_metric = metric
            st.session_state.opt_target = target
            st.session_state.opt_task = task_type

            # =====================================================
            # SALVAR MODELO
            # =====================================================
            if supabase and "current_experiment_id" in st.session_state:

                save_model_supabase(
                    supabase,
                    st.session_state.current_experiment_id,
                    task_type,
                    target,
                    metric
                )

                st.success("Modelo salvo no histórico!")

        # =====================================================
        # PREDIÇÃO
        # =====================================================
        if "opt_model" in st.session_state:

            st.subheader("Predição Automática")

            sample_sel = st.selectbox(
                "Selecionar amostra",
                st.session_state.opt_labels
            )

            if st.button("▶ Predizer"):

                idx = list(st.session_state.opt_labels).index(sample_sel)

                pc_vector = st.session_state.opt_scores[idx].reshape(1, -1)

                pred = st.session_state.opt_model.predict(pc_vector)[0]

                st.success("Predição realizada")

                st.markdown(f"""
                ### Resultado SurfaceXLab

                **Amostra:** {sample_sel}  
                **{st.session_state.opt_target}:** `{pred}`
                """)

                st.info("""
                Recomendação:
                Utilize as direções dominantes das componentes principais
                para otimizar parâmetros experimentais e deslocar o sistema
                em direção à região ótima do espaço multivariado.
                """)
