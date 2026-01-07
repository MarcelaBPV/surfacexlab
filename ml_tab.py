# ml_tab.py
# -*- coding: utf-8 -*-

"""
SurfaceXLab — Módulo de Machine Learning (Raman)

Funções:
- Leitura segura das features Raman do Supabase
- Expansão robusta de JSON (fingerprints)
- Visualização exploratória
- Preparação ML-ready (X)

⚠ Uso científico / exploratório. Não diagnóstico.
"""

import streamlit as st
import pandas as pd
import json
from typing import Any, Dict


# =========================================================
# LOAD FEATURES — ROBUSTO / À PROVA DE ERROS
# =========================================================
def load_ml_features(supabase) -> pd.DataFrame:
    """
    Carrega features Raman do Supabase de forma robusta.
    Nunca quebra o app (retorna DataFrame vazio em erro).
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
        st.error("❌ Erro ao consultar a tabela `raman_features` no Supabase.")
        st.exception(e)
        return pd.DataFrame()

    if not res.data:
        return pd.DataFrame()

    df = pd.DataFrame(res.data)

    # -----------------------------
    # Expandir JSON de features
    # -----------------------------
    def _safe_parse_features(val: Any) -> Dict:
        if val is None:
            return {}
        if isinstance(val, dict):
            return val
        if isinstance(val, str):
            try:
                return json.loads(val)
            except Exception:
                return {}
        return {}

    try:
        features_series = df["features"].apply(_safe_parse_features)
        features_df = pd.json_normalize(features_series)

        # Prefixo para evitar colisão de nomes
        features_df = features_df.add_prefix("feat_")

        df = pd.concat(
            [df.drop(columns=["features"]), features_df],
            axis=1
        )
    except Exception as e:
        st.warning("⚠ Falha ao expandir JSON de features.")
        st.exception(e)

    return df


# =========================================================
# UI — ABA ML
# =========================================================
def render_ml_tab(supabase):
    st.header("🤖 Otimizador — Machine Learning (Raman)")

    st.markdown(
        """
        Este módulo utiliza **fingerprints Raman** armazenados no banco de dados
        para análises exploratórias e preparação de modelos de *Machine Learning*.

        ⚠ **Uso científico / exploratório — não diagnóstico clínico.**
        """
    )

    # -----------------------------------------------------
    # 1. Carregar dados
    # -----------------------------------------------------
    df = load_ml_features(supabase)

    if df.empty:
        st.info(
            "Nenhuma feature Raman encontrada.\n\n"
            "➡ Execute análises Raman e gere *features* antes de usar o ML."
        )
        return

    # -----------------------------------------------------
    # 2. Visão geral
    # -----------------------------------------------------
    st.subheader("📊 Visão geral do dataset")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Registros", len(df))
    with col2:
        st.metric(
            "Features",
            len(df.select_dtypes(include="number").columns)
        )
    with col3:
        st.metric(
            "Versões de modelo",
            df["model_version"].nunique()
            if "model_version" in df.columns else 0
        )

    with st.expander("🔍 Visualizar dados brutos"):
        st.dataframe(df.head(100), use_container_width=True)

    # -----------------------------------------------------
    # 3. Seleção de features numéricas
    # -----------------------------------------------------
    st.subheader("🧬 Seleção de Features Raman")

    numeric_cols = (
        df
        .select_dtypes(include="number")
        .columns
        .tolist()
    )

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
    st.dataframe(X.head(20), use_container_width=True)

    # -----------------------------------------------------
    # 4. Diagnóstico rápido
    # -----------------------------------------------------
    st.subheader("🧪 Diagnóstico rápido das features")

    colA, colB = st.columns(2)

    with colA:
        st.markdown("**Valores ausentes (%):**")
        na_pct = (X.isna().mean() * 100).round(2)
        st.dataframe(na_pct[na_pct > 0], use_container_width=True)

    with colB:
        st.markdown("**Resumo estatístico:**")
        st.dataframe(X.describe().T, use_container_width=True)

    # -----------------------------------------------------
    # 5. Placeholder de ML (ativo no próximo passo)
    # -----------------------------------------------------
    st.divider()
    st.subheader("🚀 Treinamento de Modelo (Random Forest)")

    st.markdown(
        """
        **Pipeline já preparado para:**
        - Classificação (ex.: condição, tratamento)
        - Regressão (ex.: ângulo de contato, resistividade)
        - Validação cruzada
        - Importância das features
        - Salvamento do modelo treinado
        """
    )

    st.info(
        "🔧 O módulo está **100% ML-ready**.\n\n"
        "Assim que houver uma **variável alvo (label)** no banco, "
        "o treinamento pode ser ativado sem refatoração."
    )

    # -----------------------------------------------------
    # 6. Regras exploratórias (opcional)
    # -----------------------------------------------------
    if "rules_triggered" in df.columns:
        st.divider()
        st.subheader("🧠 Regras exploratórias detectadas")

        rules_series = df["rules_triggered"].dropna()

        if not rules_series.empty:
            try:
                example = rules_series.iloc[0]
                if isinstance(example, str):
                    example = json.loads(example)
                st.json(example)
            except Exception:
                st.write(rules_series.iloc[0])
        else:
            st.info("Nenhuma regra exploratória registrada.")
