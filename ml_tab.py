# ml_tab.py
# -*- coding: utf-8 -*-

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from datetime import datetime

# =========================================================
# HELPERS — BANCO
# =========================================================

def load_ml_features(supabase):
    res = (
        supabase
        .table("ml_features")
        .select("*")
        .order("created_at", desc=True)
        .execute()
    )
    return pd.DataFrame(res.data) if res.data else pd.DataFrame()


def save_ml_result(
    supabase,
    ml_feature_id,
    model_type,
    target,
    prediction,
    confidence,
    model_version="v1.0"
):
    supabase.table("ml_results").insert({
        "ml_feature_id": ml_feature_id,
        "model_type": model_type,
        "target": target,
        "prediction": float(prediction),
        "confidence": float(confidence),
        "model_version": model_version,
        "created_at": str(datetime.utcnow())
    }).execute()

# =========================================================
# UI — ML TAB
# =========================================================

def render_ml_tab(supabase):
    st.header("Otimizador — Machine Learning (Random Forest)")

    st.markdown(
        """
        Este módulo utiliza **Machine Learning supervisionado**
        para identificar padrões entre propriedades moleculares,
        elétricas e físico-mecânicas das superfícies.
        """
    )

    # -----------------------------------------------------
    # Carregar dados
    # -----------------------------------------------------
    df = load_ml_features(supabase)

    if df.empty:
        st.warning("Nenhum conjunto de features disponível para ML.")
        return

    st.subheader("📊 Dataset disponível")
    st.dataframe(df)

    # -----------------------------------------------------
    # Seleção de features e target
    # -----------------------------------------------------
    st.subheader("Configuração do Modelo")

    feature_columns = [
        "raman_peak_count",
        "raman_intensity_mean",
        "resistivity_mean",
        "surface_energy_total",
        "surface_energy_polar"
    ]

    feature_columns = [c for c in feature_columns if c in df.columns]

    target = st.selectbox(
        "Variável alvo (target)",
        options=feature_columns
    )

    X = df[feature_columns].drop(columns=[target])
    y = df[target]

    # -----------------------------------------------------
    # Parâmetros do modelo
    # -----------------------------------------------------
    st.subheader("Parâmetros do Random Forest")

    col1, col2, col3 = st.columns(3)
    with col1:
        n_estimators = st.number_input("n_estimators", 50, 500, 100)
    with col2:
        max_depth = st.number_input("max_depth", 2, 50, 10)
    with col3:
        test_size = st.slider("Proporção de teste", 0.1, 0.5, 0.2)

    # -----------------------------------------------------
    # Treinar modelo
    # -----------------------------------------------------
    if st.button("Treinar Modelo"):
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )

        model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=42
        )

        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))

        st.success("✔ Modelo treinado com sucesso")

        col1, col2 = st.columns(2)
        col1.metric("R²", f"{r2:.4f}")
        col2.metric("RMSE", f"{rmse:.4f}")

        # -------------------------------------------------
        # Importância das features
        # -------------------------------------------------
        st.subheader("Importância das Features")

        importances = pd.DataFrame({
            "feature": X.columns,
            "importance": model.feature_importances_
        }).sort_values("importance", ascending=False)

        st.bar_chart(importances.set_index("feature"))

        # -------------------------------------------------
        # Salvar resultados no banco
        # -------------------------------------------------
        for idx, row in df.iterrows():
            prediction = model.predict(
                row[X.columns].values.reshape(1, -1)
            )[0]

            save_ml_result(
                supabase=supabase,
                ml_feature_id=row["id"],
                model_type="RandomForest",
                target=target,
                prediction=prediction,
                confidence=r2
            )

        st.success("✔ Predições salvas no banco")

