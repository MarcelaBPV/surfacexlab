# ml_tab.py
# -*- coding: utf-8 -*-

"""
SurfaceXLab — Otimizador IA
PCA + Machine Learning supervisionado
"""

import streamlit as st
import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier


def render_ml_tab(supabase=None):

    st.header("🤖 Otimizador — Análise Multivariada e IA")

    st.markdown("""
    Esta aba integra **Análise de Componentes Principais (PCA)** e
    **modelos de aprendizado supervisionado**, utilizando fingerprints
    espectrais extraídos previamente.
    """)

    # -----------------------------------------------------
    # 1️⃣ Carregar fingerprints (exemplo: tabela fingerprints)
    # -----------------------------------------------------
    st.subheader("Base de dados")

    try:
        res = supabase.table("raman_fingerprints").select("*").execute()
        data = res.data if res.data else []
    except Exception:
        data = []

    if not data:
        st.warning("Nenhum fingerprint disponível no banco.")
        return

    df = pd.DataFrame(data)

    label_col = st.selectbox(
        "Variável alvo (classe)",
        options=[c for c in df.columns if c not in ("id", "created_at")]
    )

    X = df.drop(columns=[label_col, "id", "created_at"], errors="ignore")
    y = df[label_col]

    # -----------------------------------------------------
    # 2️⃣ PCA
    # -----------------------------------------------------
    st.subheader("Análise de Componentes Principais (PCA)")

    n_components = st.slider(
        "Número de componentes principais",
        min_value=2,
        max_value=min(6, X.shape[1]),
        value=2
    )

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    pca = PCA(n_components=n_components)
    scores = pca.fit_transform(X_scaled)

    scores_df = pd.DataFrame(
        scores,
        columns=[f"PC{i+1}" for i in range(n_components)]
    )

    st.write("📊 Variância explicada:")
    st.write(pca.explained_variance_ratio_)

    st.dataframe(scores_df)

    # -----------------------------------------------------
    # 3️⃣ Loadings
    # -----------------------------------------------------
    st.subheader("Loadings (contribuição das variáveis)")

    loadings_df = pd.DataFrame(
        pca.components_.T,
        index=X.columns,
        columns=[f"PC{i+1}" for i in range(n_components)]
    )

    st.dataframe(loadings_df)

    # -----------------------------------------------------
    # 4️⃣ ML supervisionado
    # -----------------------------------------------------
    st.subheader("Aprendizado supervisionado")

    X_train, X_test, y_train, y_test = train_test_split(
        scores, y, test_size=0.25, random_state=42
    )

    model = RandomForestClassifier(
        n_estimators=200,
        random_state=42
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)

    st.metric("Acurácia", f"{acc:.2f}")

    st.write("📉 Matriz de confusão:")
    st.dataframe(
        pd.DataFrame(
            confusion_matrix(y_test, y_pred),
            index=["Real 0", "Real 1"],
            columns=["Pred 0", "Pred 1"]
        )
    )
