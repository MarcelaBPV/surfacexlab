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
from sklearn.metrics import mean_absolute_error


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
# SALVAR MODELO
# =====================================================

def save_model_supabase(supabase, experiment_id, model_type, target, metric):

    supabase.table("ml_models").insert({
        "experiment_id": experiment_id,
        "model_type": model_type,
        "target_variable": target,
        "metric_value": float(metric)
    }).execute()


# =====================================================
# FEATURE IMPORTANCE BACK-PROJECTION
# =====================================================

def project_feature_importance(pca, rf_importance, feature_names):

    loadings = np.abs(pca.components_.T)

    weighted = loadings @ rf_importance

    importance_norm = weighted / weighted.sum()

    df_importance = pd.DataFrame({
        "Feature": feature_names,
        "Importance": importance_norm
    }).sort_values(by="Importance", ascending=False)

    return df_importance


# =====================================================
# INTERFACE STREAMLIT
# =====================================================

def render_ml_tab(supabase=None):

    st.header("🤖 Otimizador Inteligente — PCA + Machine Learning Científico")

    subtabs = st.tabs(["📊 PCA Exploratório", "🧠 Machine Learning"])

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

        if source == "Upload de Arquivo":

            uploaded_file = st.file_uploader(
                "Upload XLS, CSV ou TXT",
                type=["xlsx", "csv", "txt"]
            )

            if uploaded_file is None:
                st.info("Faça upload do arquivo.")
                return

            df = load_file(uploaded_file)

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

        if df.shape[0] < 3:
            st.warning("Dados insuficientes para PCA.")
            return

        X = df.values
        labels = df.index.values
        features = df.columns.values

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        n_components = min(6, X_scaled.shape[1])

        pca = PCA(n_components=n_components)
        scores = pca.fit_transform(X_scaled)

        explained = pca.explained_variance_ratio_ * 100

        st.caption(
            f"Variância explicada acumulada: "
            f"{np.sum(explained):.2f}%"
        )

        # Guardar PCA para visualização apenas
        st.session_state.opt_df = df

        # ===========================
        # BIPLOT
        # ===========================

        fig, ax = plt.subplots(figsize=(6, 6), dpi=300)

        ax.scatter(scores[:, 0], scores[:, 1],
                   s=70, edgecolors="black", linewidths=0.6)

        for i, label in enumerate(labels):
            ax.text(scores[i, 0], scores[i, 1], label, fontsize=8)

        loadings = pca.components_.T
        scale = np.max(np.abs(scores[:, :2])) * 0.85

        for i, var in enumerate(features):
            ax.arrow(0, 0,
                     loadings[i, 0] * scale,
                     loadings[i, 1] * scale,
                     head_width=0.04,
                     linewidth=1)

            ax.text(loadings[i, 0] * scale * 1.05,
                    loadings[i, 1] * scale * 1.05,
                    var,
                    fontsize=8)

        ax.set_xlabel(f"PC1 ({explained[0]:.1f}%)")
        ax.set_ylabel(f"PC2 ({explained[1]:.1f}%)")

        ax.axhline(0)
        ax.axvline(0)
        ax.grid(alpha=0.2, linestyle="--")
        ax.set_aspect("equal", adjustable="box")

        plt.tight_layout()
        st.pyplot(fig)

        # ===========================
        # Salvar PCA
        # ===========================

        if supabase and st.button("💾 Salvar PCA no Banco"):

            exp_id = save_pca_supabase(
                supabase,
                labels,
                scores,
                explained
            )

            st.session_state.current_experiment_id = exp_id

            st.success("PCA salvo com sucesso!")


    # =====================================================
    # SUBABA MACHINE LEARNING
    # =====================================================

    with subtabs[1]:

        st.subheader(" Machine Learning Científico")

        if "opt_df" not in st.session_state:
            st.info("Execute o PCA primeiro.")
            return

        df_ml = st.session_state.opt_df.copy()

        task_type = st.selectbox(
            "Tipo de problema",
            ["Regressão (valor físico)", "Classificação (classe)"]
        )

        target = st.selectbox(
            "Variável alvo (target)",
            df_ml.columns.tolist()
        )

        # ===========================
        # Separação correta
        # ===========================

        y = df_ml[target].values
        X = df_ml.drop(columns=[target])
        feature_names = X.columns.values
        X = X.values

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25, random_state=42
        )

        # ===========================
        # Pipeline Científico
        # ===========================

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        n_components = min(6, X_train_scaled.shape[1])

        pca = PCA(n_components=n_components)
        X_train_pca = pca.fit_transform(X_train_scaled)
        X_test_pca = pca.transform(X_test_scaled)

        if st.button("▶ Treinar Modelo Científico"):

            # -------- REGRESSÃO --------

            if task_type.startswith("Regressão"):

                model = RandomForestRegressor(
                    n_estimators=400,
                    random_state=42,
                    n_jobs=-1
                )

                model.fit(X_train_pca, y_train)

                y_pred = model.predict(X_test_pca)

                r2 = r2_score(y_test, y_pred)
                mae = mean_absolute_error(y_test, y_pred)

                st.metric("R²", f"{r2:.4f}")
                st.metric("MAE", f"{mae:.4f}")

                metric_main = r2

            # -------- CLASSIFICAÇÃO --------

            else:

                model = RandomForestClassifier(
                    n_estimators=300,
                    random_state=42,
                    class_weight="balanced",
                    n_jobs=-1
                )

                model.fit(X_train_pca, y_train)

                y_pred = model.predict(X_test_pca)

                acc = accuracy_score(y_test, y_pred)

                st.metric("Accuracy", f"{acc:.4f}")

                report = classification_report(
                    y_test, y_pred, output_dict=True
                )

                st.json(report)

                metric_main = acc

            # ===========================
            # FEATURE IMPORTANCE
            # ===========================

            rf_importance = model.feature_importances_

            df_importance = project_feature_importance(
                pca,
                rf_importance,
                feature_names
            )

            st.subheader("📊 Feature Importance Física")

            st.dataframe(df_importance, use_container_width=True)

            fig_imp, ax_imp = plt.subplots(figsize=(6, 4), dpi=300)

            ax_imp.barh(
                df_importance["Feature"],
                df_importance["Importance"]
            )

            ax_imp.invert_yaxis()
            ax_imp.set_xlabel("Importância normalizada")
            ax_imp.set_title("Importância das Variáveis Físicas")

            plt.tight_layout()
            st.pyplot(fig_imp)

            # ===========================
            # Armazenar pipeline
            # ===========================

            st.session_state.opt_model = model
            st.session_state.opt_scaler_ml = scaler
            st.session_state.opt_pca_ml = pca
            st.session_state.opt_target = target
            st.session_state.opt_task = task_type
            st.session_state.opt_metric = metric_main

            # ===========================
            # Salvar no banco
            # ===========================

            if supabase and "current_experiment_id" in st.session_state:

                save_model_supabase(
                    supabase,
                    st.session_state.current_experiment_id,
                    task_type,
                    target,
                    metric_main
                )

                st.success("Modelo científico salvo com sucesso!")


        # =====================================================
        # PREDIÇÃO
        # =====================================================

        if "opt_model" in st.session_state:

            st.subheader(" Predição Científica")

            sample_sel = st.selectbox(
                "Selecionar amostra",
                df_ml.index.tolist()
            )

            if st.button("▶ Predizer Amostra"):

                row = df_ml.loc[sample_sel]

                X_sample = row.drop(st.session_state.opt_target).values.reshape(1, -1)

                X_scaled = st.session_state.opt_scaler_ml.transform(X_sample)

                X_pca = st.session_state.opt_pca_ml.transform(X_scaled)

                pred = st.session_state.opt_model.predict(X_pca)[0]

                st.success("Predição realizada")

                st.markdown(f"""
                ### Resultado SurfaceXLab

                **Amostra:** {sample_sel}  
                **{st.session_state.opt_target}:** `{pred}`  
                """)

                st.info("""
                Utilize as variáveis físicas mais importantes
                para orientar a otimização experimental e deslocar
                o sistema em direção à região ótima do espaço PCA.
                """)
