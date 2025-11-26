
# ml_tab.py
# -*- coding: utf-8 -*-
"""Aba 4 — Otimização ML (Random Forest genérico)."""

import streamlit as st
import pandas as pd
import numpy as np
from typing import Optional

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import (
    r2_score,
    mean_absolute_error,
    accuracy_score,
)


def render_ml_tab(supabase: Optional[object] = None):
    st.header("4️⃣ Otimização ML — Modelos para superfícies de materiais")

    st.markdown(
        """Envie um CSV com qualquer conjunto de dados (Raman, tensiometria, resistividade, etc.).

- Escolha a **coluna alvo (y)**.
- Escolha as **features (X)**.
- O app decide automaticamente entre regressão e classificação (Random Forest).
        """
    )

    file = st.file_uploader("Arquivo CSV para modelagem", type=["csv"])
    if not file:
        st.info("Envie um CSV para começar.")
        return

    try:
        df = pd.read_csv(file)
    except Exception as e:
        st.error(f"Erro ao ler CSV: {e}")
        return

    st.subheader("Pré-visualização")
    st.dataframe(df.head())

    cols = list(df.columns)
    target_col = st.selectbox("Coluna alvo (y)", cols)
    feature_cols = st.multiselect(
        "Colunas de entrada (X)",
        [c for c in cols if c != target_col],
        default=[c for c in cols if c != target_col][:3],
    )

    if not feature_cols:
        st.warning("Selecione pelo menos uma feature.")
        return

    test_size = st.slider("Proporção de teste", 0.1, 0.5, 0.2, 0.05)
    random_state = st.number_input("Random state", min_value=0, value=42, step=1)

    if st.button("Treinar modelo"):
        try:
            X = df[feature_cols].values
            y = df[target_col].values

            is_regression = np.issubdtype(y.dtype, np.number)

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state
            )

            scaler = StandardScaler()
            X_train_s = scaler.fit_transform(X_train)
            X_test_s = scaler.transform(X_test)

            if is_regression:
                model = RandomForestRegressor(
                    n_estimators=200, random_state=random_state
                )
            else:
                model = RandomForestClassifier(
                    n_estimators=200, random_state=random_state
                )

            model.fit(X_train_s, y_train)
            y_pred = model.predict(X_test_s)

            st.subheader("Resultados")

            if is_regression:
                r2 = r2_score(y_test, y_pred)
                mae = mean_absolute_error(y_test, y_pred)
                st.markdown("**Tipo:** Regressão (RandomForestRegressor)")
                st.markdown(f"**R²:** `{r2:.4f}`")
                st.markdown(f"**MAE:** `{mae:.4f}`")
            else:
                acc = accuracy_score(y_test, y_pred)
                st.markdown("**Tipo:** Classificação (RandomForestClassifier)")
                st.markdown(f"**Acurácia:** `{acc:.4f}`")

            importances = model.feature_importances_
            imp_df = pd.DataFrame(
                {"feature": feature_cols, "importance": importances}
            ).sort_values("importance", ascending=False)

            st.subheader("Importância das Features")
            st.dataframe(imp_df)

        except Exception as e:
            st.error(f"Erro ao treinar modelo: {e}")
