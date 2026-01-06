# ml_tab.py
# -*- coding: utf-8 -*-

"""
SurfaceXLab — Módulo de Machine Learning (Raman)

Funções:
- Leitura segura das features Raman do Supabase
- Visualização das features (fingerprints)
- Preparação para treinamento de modelos ML (Random Forest)

⚠ Uso científico / exploratório. Não diagnóstico.
"""

import streamlit as st
import pandas as pd
import json


# =========================================================
# LOAD FEATURES (ROBUSTO)
# =========================================================
def load_ml_features(supabase) -> pd.DataFrame:
    """
    Carrega features Raman do Supabase de forma segura.
    """
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
        st.error("❌ Erro ao consultar tabela raman_features no Supabase.")
        st.exception(e)
        return pd.DataFrame()

    if not res.data:
        return pd.DataFrame()

    df = pd.DataFrame(res.data)

    # Expandir JSON de features em colunas
    try:
        features_expanded = df["features"].apply(
            lambda x: x if isinstance(x, dict) else json.loads(x)
        )
        features_df = pd.json_normalize(features_expanded)
        df = pd.concat([df.drop(columns=["features"]), features_df], axis=1)
    except Exception as e:
        st.warning("⚠ Não foi possível expandir o JSON de features.")
        st.exception(e)

    return df


# =========================================================
# UI — ABA ML
# =========================================================
def render_ml_tab(supabase):
    st.header("🤖 Otimizador — Machine Learning (Raman)")

    st.markdown(
        """
        Este módulo utiliza **features extraídas de espectros Raman**
        para análises exploratórias e treinamento de modelos de Machine Learning
        (ex.: Random Forest).

        ⚠ **Uso científico / exploratório — não diagnóstico clínico.**
        """
    )

    # -----------------------------------------------------
    # Carregar dados
    # -----------------------------------------------------
    df = load_ml_features(supabase)

    if df.empty:
        st.info(
            "Nenhuma feature Raman encontrada.\n\n"
            "➡ Execute análises Raman e gere features antes de usar o ML."
        )
        return

    # -----------------------------------------------------
    # Visão geral
    # -----------------------------------------------------
    st.subheader("📊 Visão geral do dataset")

    st.write(f"Total de registros: **{len(df)}**")

    st.dataframe(
        df.head(50),
        use_container_width=True,
    )

    # -----------------------------------------------------
    # Seleção de features numéricas
    # -----------------------------------------------------
    st.subheader("🔎 Seleção de Features")

    numeric_cols = df.select_dtypes(include="number").columns.tolist()

    if not numeric_cols:
        st.warning("Nenhuma feature numérica disponível para ML.")
        return

    selected_features = st.multiselect(
        "Selecione as features para o modelo",
        numeric_cols,
        default=numeric_cols,
    )

    if not selected_features:
        st.warning("Selecione ao menos uma feature.")
        return

    X = df[selected_features].copy()

    st.markdown("**Matriz de features (X):**")
    st.dataframe(X.head(), use_container_width=True)

    # -----------------------------------------------------
    # Placeholder para ML
    # -----------------------------------------------------
    st.divider()
    st.subheader("🚀 Treinamento de Modelo (em breve)")

    st.markdown(
        """
        Próximos passos previstos:
        -  Definição de variável alvo (label)
        -  Random Forest (classificação / regressão)
        -  Métricas: accuracy, ROC, importância das features
        -  Salvamento do modelo treinado
        """
    )

    st.info(
        "🔧 Este módulo já está **ML-ready**.\n\n"
        "O treinamento pode ser ativado assim que houver rótulos "
        "(ex.: condição experimental, classe clínica, tratamento)."
    )

    # -----------------------------------------------------
    # Regras exploratórias (opcional)
    # -----------------------------------------------------
    if "rules_triggered" in df.columns:
        st.divider()
        st.subheader("🧠 Regras exploratórias detectadas")

        rules_series = df["rules_triggered"].dropna()

        if not rules_series.empty:
            st.json(rules_series.iloc[0])
        else:
            st.info("Nenhuma regra exploratória registrada.")
