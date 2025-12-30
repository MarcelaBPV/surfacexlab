# ml_tab.py
# -*- coding: utf-8 -*-

"""
Aba 4 — Otimizador com IA (Machine Learning)
Random Forest para regressão e classificação
CRM científico: dados → modelo → métricas → insights
"""

# =========================================================
# IMPORTS
# =========================================================
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict
from datetime import datetime

# ---------------------------------------------------------
# Tentativa de importar scikit-learn (opcional)
# ---------------------------------------------------------
try:
    from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import (
        r2_score,
        mean_absolute_error,
        mean_squared_error,
        accuracy_score,
    )
    SKLEARN_AVAILABLE = True
except Exception:
    SKLEARN_AVAILABLE = False


# =========================================================
# SUPABASE HELPER
# =========================================================
def save_ml_result(
    supabase,
    model_name: str,
    problem_type: str,
    target: str,
    features: list,
    metrics: Dict,
    importances: Dict,
):
    supabase.table("results_ml_optimization").insert({
        "model_name": model_name,
        "parameters": {
            "problem_type": problem_type,
            "target": target,
            "features": features,
        },
        "scores": metrics | {"importances": importances},
        "date_run": datetime.utcnow().isoformat(),
    }).execute()


# =========================================================
# RENDER DA ABA
# =========================================================
def render_ml_tab(supabase, helpers):

    st.subheader("Otimizador com IA — Machine Learning")

    st.markdown(
        """
Este módulo permite explorar **relações entre parâmetros experimentais e propriedades de superfície**
utilizando **Random Forest** para **regressão** ou **classificação**.

👉 Recomenda-se utilizar **dados já agregados**, por exemplo:
- picos Raman
- resistividade / condutividade
- energia superficial
"""
    )

    # =====================================================
    # Verificação do ambiente
    # =====================================================
    if not SKLEARN_AVAILABLE:
        st.warning(
            """
⚠️ A biblioteca **scikit-learn** não está disponível neste ambiente.

A aba de IA está temporariamente desativada.
As demais abas da plataforma continuam funcionando normalmente.
"""
        )
        return

    # =====================================================
    # BLOCO 1 — UPLOAD DOS DADOS
    # =====================================================
    st.markdown("### 📂 Upload do dataset")

    file = st.file_uploader("Arquivo CSV com dados experimentais", type=["csv"])

    if file is None:
        st.info("Envie um arquivo CSV para iniciar a análise.")
        return

    try:
        df = pd.read_csv(file)
    except Exception as e:
        st.error(f"Erro ao ler o CSV: {e}")
        return

    if df.empty:
        st.error("O CSV está vazio.")
        return

    st.markdown("#### Pré-visualização")
    st.dataframe(df.head(), use_container_width=True)

    # =====================================================
    # BLOCO 2 — CONFIGURAÇÃO DO MODELO
    # =====================================================
    st.markdown("### Configuração do modelo")

    target_col = st.selectbox("Variável alvo (y)", df.columns)

    feature_cols = [c for c in df.columns if c != target_col]
    if not feature_cols:
        st.error("O dataset precisa ter ao menos 1 feature além da variável alvo.")
        return

    problem_choice = st.radio(
        "Tipo de problema",
        ["Detecção automática", "Regressão", "Classificação"]
    )

    # =====================================================
    # BLOCO 3 — PREPARAÇÃO DOS DADOS
    # =====================================================
    df_clean = df[feature_cols + [target_col]].dropna()
    if df_clean.empty:
        st.error("Após remover NaN, não restaram dados suficientes.")
        return

    X_raw = df_clean[feature_cols]
    y_raw = df_clean[target_col]

    X = pd.get_dummies(X_raw, drop_first=True)

    # Inferência automática
    if problem_choice == "Detecção automática":
        if pd.api.types.is_numeric_dtype(y_raw) and y_raw.nunique() > 10:
            problem_type = "regression"
        else:
            problem_type = "classification"
    else:
        problem_type = "regression" if problem_choice == "Regressão" else "classification"

    # Codificação para classificação
    class_labels = None
    y = y_raw.copy()
    if problem_type == "classification" and not pd.api.types.is_numeric_dtype(y_raw):
        y_codes, uniques = pd.factorize(y_raw)
        y = pd.Series(y_codes, index=y_raw.index)
        class_labels = {i: str(u) for i, u in enumerate(uniques)}

    # =====================================================
    # BLOCO 4 — SPLIT TREINO / TESTE
    # =====================================================
    test_size = st.slider("Proporção de teste", 0.1, 0.5, 0.2, 0.05)
    random_state = st.number_input("Random seed", 0, 9999, 42)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    # =====================================================
    # BLOCO 5 — HIPERPARÂMETROS
    # =====================================================
    st.markdown("### Hiperparâmetros")

    col1, col2 = st.columns(2)
    with col1:
        n_estimators = st.slider("Número de árvores", 50, 500, 200, 50)
    with col2:
        max_depth = st.slider("Profundidade máxima", 2, 20, 8)

    # =====================================================
    # BLOCO 6 — TREINO DO MODELO
    # =====================================================
    if not st.button("Treinar modelo"):
        return

    if problem_type == "regression":
        model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
            n_jobs=-1,
        )
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        metrics = {
            "r2": float(r2_score(y_test, y_pred)),
            "mae": float(mean_absolute_error(y_test, y_pred)),
            "rmse": float(mean_squared_error(y_test, y_pred, squared=False)),
        }

        st.markdown("### 📊 Métricas de regressão")
        st.json(metrics)

    else:
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
            n_jobs=-1,
        )
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        metrics = {
            "accuracy": float(accuracy_score(y_test, y_pred))
        }

        st.markdown("### 📊 Métricas de classificação")
        st.json(metrics)

        if class_labels:
            st.markdown("**Mapeamento de classes**")
            st.json(class_labels)

    # =====================================================
    # BLOCO 7 — IMPORTÂNCIA DAS FEATURES
    # =====================================================
    importances = dict(zip(X.columns, model.feature_importances_))
    imp_df = (
        pd.DataFrame(
            {"feature": importances.keys(), "importance": importances.values()}
        )
        .sort_values("importance", ascending=False)
    )

    st.markdown("### 🔍 Importância das variáveis")
    helpers["show_aggrid"](imp_df, height=260)

    if st.button("📌 Abrir importâncias no painel lateral"):
        helpers["open_side"](imp_df, "Importância das features")

    # Gráfico
    top_n = min(10, len(imp_df))
    fig, ax = plt.subplots(figsize=(8, 4))
    top = imp_df.head(top_n).iloc[::-1]
    ax.barh(top["feature"], top["importance"])
    ax.set_xlabel("Importância relativa")
    ax.set_title("Top variáveis do modelo")
    st.pyplot(fig)

    # =====================================================
    # BLOCO 8 — SALVAR NO SUPABASE
    # =====================================================
    if supabase and st.button("💾 Salvar resultado do modelo"):
        save_ml_result(
            supabase,
            model_name="RandomForest",
            problem_type=problem_type,
            target=target_col,
            features=list(X.columns),
            metrics=metrics,
            importances=importances,
        )
        st.success("Resultado do modelo salvo no Supabase.")
