# ml_tab.py
# -*- coding: utf-8 -*-

"""
SurfaceXLab — Módulo de Machine Learning (Raman)

Inclui:
- Leitura robusta de features Raman (Supabase)
- Seleção de features
- Treinamento real Random Forest (classificação ou regressão)
- Validação cruzada
- Gráfico automático de importância das features

⚠ Uso científico / exploratório. Não diagnóstico.
"""

import streamlit as st
import pandas as pd
import json
import numpy as np
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split, StratifiedKFold, KFold, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, r2_score, mean_absolute_error


# =========================================================
# LOAD FEATURES — ROBUSTO
# =========================================================
def load_ml_features(supabase) -> pd.DataFrame:
    try:
        res = (
            supabase
            .table("raman_features")
            .select(
                "id, raman_measurement_id, features, rules_triggered, model_version, created_at"
            )
            .order("created_at", desc=True)
            .execute()
        )
    except Exception as e:
        st.error("❌ Erro ao consultar a tabela raman_features.")
        st.exception(e)
        return pd.DataFrame()

    if not res.data:
        return pd.DataFrame()

    df = pd.DataFrame(res.data)

    def safe_parse(val):
        if isinstance(val, dict):
            return val
        if isinstance(val, str):
            try:
                return json.loads(val)
            except Exception:
                return {}
        return {}

    features_series = df["features"].apply(safe_parse)
    features_df = pd.json_normalize(features_series).add_prefix("feat_")

    df = pd.concat([df.drop(columns=["features"]), features_df], axis=1)
    return df


# =========================================================
# UI — ABA ML
# =========================================================
def render_ml_tab(supabase):
    st.header("Otimizador — Machine Learning (Raman)")

    st.markdown(
        """
        Treinamento **real** de modelos *Random Forest* a partir de
        fingerprints Raman armazenados no banco de dados.

        ⚠ Uso científico / exploratório.
        """
    )

    # -----------------------------------------------------
    # 1. Carregar dados
    # -----------------------------------------------------
    df = load_ml_features(supabase)

    if df.empty:
        st.info("Nenhuma feature Raman encontrada.")
        return

    st.subheader("📊 Dataset")
    st.write(f"Registros disponíveis: **{len(df)}**")

    # -----------------------------------------------------
    # 2. Seleção de features
    # -----------------------------------------------------
    numeric_cols = df.select_dtypes(include="number").columns.tolist()

    if not numeric_cols:
        st.warning("Nenhuma feature numérica disponível.")
        return

    selected_features = st.multiselect(
        "Selecione as features (X)",
        numeric_cols,
        default=numeric_cols,
    )

    if not selected_features:
        st.warning("Selecione ao menos uma feature.")
        return

    X = df[selected_features].copy()
    X = X.fillna(0.0)

    # -----------------------------------------------------
    # 3. Seleção da variável alvo (label)
    # -----------------------------------------------------
    st.subheader("Variável alvo")

    possible_targets = df.columns.tolist()
    target_col = st.selectbox(
        "Selecione a variável alvo (y)",
        possible_targets,
    )

    if target_col in selected_features:
        st.warning("A variável alvo não pode estar nas features.")
        return

    y = df[target_col]

    # -----------------------------------------------------
    # 4. Tipo de problema
    # -----------------------------------------------------
    is_classification = y.dtype == object or y.nunique() < 10

    problem_type = "Classificação" if is_classification else "Regressão"
    st.info(f"🔍 Tipo de problema detectado: **{problem_type}**")

    # -----------------------------------------------------
    # 5. Botão de treinamento
    # -----------------------------------------------------
    st.divider()
    if st.button("Treinar Random Forest"):
        with st.spinner("Treinando modelo..."):

            if is_classification:
                # -----------------------------
                # CLASSIFICAÇÃO
                # -----------------------------
                model = RandomForestClassifier(
                    n_estimators=300,
                    random_state=42,
                    class_weight="balanced",
                    n_jobs=-1,
                )

                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.25, random_state=42, stratify=y
                )

                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)

                acc = accuracy_score(y_test, y_pred)

                cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
                cv_scores = cross_val_score(model, X, y, cv=cv, scoring="accuracy")

                st.success(f"✅ Accuracy: **{acc:.3f}**")
                st.write("📊 Validação cruzada (accuracy):")
                st.write(f"Média = {cv_scores.mean():.3f} | Desvio = {cv_scores.std():.3f}")

                st.subheader("📄 Relatório de Classificação")
                st.json(classification_report(y_test, y_pred, output_dict=True))

            else:
                # -----------------------------
                # REGRESSÃO
                # -----------------------------
                model = RandomForestRegressor(
                    n_estimators=400,
                    random_state=42,
                    n_jobs=-1,
                )

                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.25, random_state=42
                )

                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)

                r2 = r2_score(y_test, y_pred)
                mae = mean_absolute_error(y_test, y_pred)

                cv = KFold(n_splits=5, shuffle=True, random_state=42)
                cv_scores = cross_val_score(model, X, y, cv=cv, scoring="r2")

                st.success(f"✅ R²: **{r2:.3f}** | MAE: **{mae:.3f}**")
                st.write("📊 Validação cruzada (R²):")
                st.write(f"Média = {cv_scores.mean():.3f} | Desvio = {cv_scores.std():.3f}")

            # -------------------------------------------------
            # 6. Importância das features
            # -------------------------------------------------
            st.divider()
            st.subheader("Importância das Features")

            importances = pd.Series(
                model.feature_importances_,
                index=X.columns
            ).sort_values(ascending=False)

            st.dataframe(importances.head(20))

            # Gráfico
            fig, ax = plt.subplots(figsize=(8, 5))
            importances.head(15).iloc[::-1].plot.barh(ax=ax)
            ax.set_xlabel("Importância")
            ax.set_title("Top 15 Features — Random Forest")
            ax.grid(alpha=0.3)

            st.pyplot(fig)

            st.success("Treinamento concluído com sucesso!")
