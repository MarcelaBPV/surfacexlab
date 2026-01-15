# ml_tab.py
# -*- coding: utf-8 -*-

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


# =========================================================
# CRIA TARGET AUTOMÁTICO (EXEMPLO: MOLHABILIDADE)
# =========================================================
def create_target(df):
    """
    Classe funcional baseada em q*
    """
    if "q* (°)" not in df.columns:
        raise ValueError("Variável q* (°) não encontrada para classificação.")

    y = df["q* (°)"].apply(
        lambda x: "Hidrofóbica" if x >= 90 else "Hidrofílica"
    )

    return y


# =========================================================
# ABA ML
# =========================================================
def render_ml_tab(supabase=None):

    st.header("🤖 Machine Learning — Classificação Funcional de Superfícies")

    st.markdown(
        """
        Este módulo utiliza **Random Forest** para aprender relações entre:

        • Propriedades Raman  
        • Tensiometria  
        • Propriedades elétricas  

        e realizar **classificação funcional automática** das superfícies.
        """
    )

    # =====================================================
    # COLETA GLOBAL DOS DADOS
    # =====================================================
    data_sources = []

    if "tensiometry_samples" in st.session_state:
        data_sources.append(
            pd.DataFrame(st.session_state.tensiometry_samples.values())
        )

    if "electrical_samples" in st.session_state:
        data_sources.append(
            pd.DataFrame(st.session_state.electrical_samples.values())
        )

    if not data_sources:
        st.info("Execute os módulos físicos antes de usar o ML.")
        return

    # =====================================================
    # MERGE
    # =====================================================
    df_global = data_sources[0]

    for df in data_sources[1:]:
        df_global = pd.merge(
            df_global,
            df,
            on="Amostra",
            how="inner"
        )

    df_global = df_global.set_index("Amostra")
    df_global = df_global.apply(pd.to_numeric, errors="coerce")
    df_global = df_global.fillna(0)

    st.subheader("Dataset consolidado para ML")
    st.dataframe(df_global, use_container_width=True)

    # =====================================================
    # TARGET
    # =====================================================
    try:
        y = create_target(df_global)
    except Exception as e:
        st.error(str(e))
        return

    X = df_global.drop(columns=["q* (°)"])
    feature_names = X.columns

    st.subheader("Classe alvo (target)")
    st.dataframe(y.rename("Classe funcional"))

    # =====================================================
    # NORMALIZAÇÃO
    # =====================================================
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # =====================================================
    # TREINO / TESTE
    # =====================================================
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled,
        y,
        test_size=0.3,
        random_state=42,
        stratify=y
    )

    # =====================================================
    # RANDOM FOREST
    # =====================================================
    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        random_state=42
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)

    # =====================================================
    # RESULTADOS
    # =====================================================
    st.subheader("📊 Desempenho do modelo")

    st.metric("Acurácia", f"{acc*100:.2f} %")

    st.markdown("### Relatório de classificação")
    report = classification_report(y_test, y_pred, output_dict=True)
    st.dataframe(pd.DataFrame(report).T)

    # =====================================================
    # MATRIZ DE CONFUSÃO
    # =====================================================
    st.markdown("### Matriz de confusão")

    cm = confusion_matrix(y_test, y_pred)

    fig, ax = plt.subplots(figsize=(4, 4), dpi=300)
    im = ax.imshow(cm)

    ax.set_xticks(range(len(model.classes_)))
    ax.set_yticks(range(len(model.classes_)))

    ax.set_xticklabels(model.classes_)
    ax.set_yticklabels(model.classes_)

    ax.set_xlabel("Predito")
    ax.set_ylabel("Real")
    ax.set_title("Matriz de Confusão")

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, cm[i, j], ha="center", va="center")

    st.pyplot(fig)

    # =====================================================
    # FEATURE IMPORTANCE
    # =====================================================
    st.subheader("📈 Importância das variáveis")

    importances = model.feature_importances_

    imp_df = pd.DataFrame({
        "Variável": feature_names,
        "Importância": importances
    }).sort_values("Importância", ascending=False)

    st.dataframe(imp_df, use_container_width=True)

    # =====================================================
    # GRÁFICO IMPORTÂNCIA
    # =====================================================
    fig2, ax2 = plt.subplots(figsize=(6, 4), dpi=300)

    ax2.barh(
        imp_df["Variável"],
        imp_df["Importância"]
    )

    ax2.set_xlabel("Importância relativa")
    ax2.set_title("Importância das variáveis — Random Forest")
    ax2.invert_yaxis()

    st.pyplot(fig2)

    # =====================================================
    # INTERPRETAÇÃO AUTOMÁTICA
    # =====================================================
    top_var = imp_df.iloc[0]["Variável"]

    st.success(
        f"Variável mais relevante para a classificação: **{top_var}**"
    )
