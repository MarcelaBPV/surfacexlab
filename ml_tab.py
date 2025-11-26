# ml_tab.py
# -*- coding: utf-8 -*-
"""
Aba 4 — Otimização ML (Machine Learning) para a SurfaceXLab

Fluxo:
- Upload de CSV com dados experimentais (já tratados ou agregados)
- Escolha da coluna-alvo (y)
- Escolha do tipo de problema (regressão ou classificação)
- Treino rápido de Random Forest
- Exibição de métricas de desempenho
- Gráfico de importâncias das variáveis

Observação:
- A aba tenta importar scikit-learn. Se não estiver disponível no ambiente,
  ela apenas mostra um aviso amigável ao usuário.
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------------------------
# Tentativa de importar scikit-learn (opcional)
# ---------------------------------------------------
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


# ---------------------------------------------------
# Função principal da aba
# ---------------------------------------------------
def render_ml_tab(supabase):
    st.header("4️⃣ Otimização ML (Machine Learning)")

    st.markdown(
        """
Esta aba permite usar **modelos de Machine Learning** (Random Forest) para:

- Explorar relações entre parâmetros de processo e propriedades de superfície  
- Testar modelos de regressão ou classificação com seus próprios dados (CSV)  

**Dica:** use aqui dados já resumidos (por exemplo: média de ângulo de contato, RMS de rugosidade, área de pico Raman, etc.).
"""
    )

    # Se scikit-learn não está disponível, avisa e encerra
    if not SKLEARN_AVAILABLE:
        st.warning(
            """
⚠️ `scikit-learn` não está disponível neste ambiente (Python do Streamlit Cloud ainda não tem uma versão compatível).

A aba de ML está **temporariamente desativada**.

Você ainda pode usar todas as outras abas (Raman, Tensiometria, Resistividade) normalmente.
"""
        )
        return

    st.markdown("---")

    # -----------------------------------------------
    # Upload de dados
    # -----------------------------------------------
    st.subheader("📂 Upload de dados experimentais (CSV)")

    file = st.file_uploader(
        "Envie um arquivo .csv com suas variáveis (colunas) e observações (linhas)",
        type=["csv"],
    )

    if file is None:
        st.info("Envie um arquivo CSV para começar a configurar o modelo de ML.")
        return

    try:
        df = pd.read_csv(file)
    except Exception as e:
        st.error(f"Erro ao ler o CSV: {e}")
        return

    if df.empty:
        st.error("O arquivo CSV está vazio.")
        return

    st.markdown("#### Pré-visualização dos dados")
    st.dataframe(df.head())

    # -----------------------------------------------
    # Escolha da coluna alvo (y) e tipo do problema
    # -----------------------------------------------
    st.markdown("---")
    st.subheader("🎯 Configuração do modelo")

    target_col = st.selectbox(
        "Escolha a coluna alvo (variável que você quer prever):",
        df.columns,
    )

    feature_cols = [c for c in df.columns if c != target_col]
    if not feature_cols:
        st.error("O CSV precisa ter pelo menos 2 colunas (1 alvo + 1 feature).")
        return

    st.write("**Features (entradas do modelo):**", ", ".join(feature_cols))

    problem_type = st.radio(
        "Tipo de problema de ML:",
        ["Detecção automática", "Regressão (valor contínuo)", "Classificação (rótulos)"],
    )

    # -----------------------------------------------
    # Limpeza simples e preparação dos dados
    # -----------------------------------------------
    df_clean = df[feature_cols + [target_col]].dropna()
    if df_clean.empty:
        st.error("Após remover valores ausentes (NaN), não sobraram linhas suficientes.")
        return

    X_raw = df_clean[feature_cols]
    y_raw = df_clean[target_col]

    # one-hot encoding em colunas não numéricas de X
    X = pd.get_dummies(X_raw, drop_first=True)

    # decisão automática sobre o tipo de problema (se escolhido)
    auto_type = None
    if problem_type == "Detecção automática":
        # Heurística simples:
        # - se y for numérico e tiver muitos valores distintos -> regressão
        # - se y for texto ou tiver poucos valores distintos -> classificação
        if pd.api.types.is_numeric_dtype(y_raw):
            n_unique = y_raw.nunique()
            if n_unique <= max(10, len(y_raw) * 0.05):
                auto_type = "class"
            else:
                auto_type = "reg"
        else:
            auto_type = "class"
    elif problem_type.startswith("Regressão"):
        auto_type = "reg"
    else:
        auto_type = "class"

    # Para classificação: converter alvo em rótulos numéricos
    y = y_raw.copy()
    class_labels = None
    if auto_type == "class":
        if not pd.api.types.is_numeric_dtype(y_raw):
            y_codes, uniques = pd.factorize(y_raw)
            y = pd.Series(y_codes, index=y_raw.index)
            class_labels = {i: lab for i, lab in enumerate(uniques)}
        else:
            class_labels = {int(v): str(v) for v in sorted(y.unique())}

    # -----------------------------------------------
    # Splitting treino/teste
    # -----------------------------------------------
    test_size = st.slider(
        "Proporção para teste (test_size)",
        min_value=0.1,
        max_value=0.5,
        value=0.2,
        step=0.05,
    )

    random_state = st.number_input(
        "Random seed (para reprodutibilidade)",
        min_value=0,
        max_value=9999,
        value=42,
        step=1,
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    st.write(f"Número de amostras de treino: {len(X_train)}")
    st.write(f"Número de amostras de teste: {len(X_test)}")

    # -----------------------------------------------
    # Hiperparâmetros simples
    # -----------------------------------------------
    st.markdown("---")
    st.subheader("⚙️ Hiperparâmetros do modelo")

    col_m1, col_m2 = st.columns(2)

    with col_m1:
        n_estimators = st.slider(
            "Número de árvores (n_estimators)",
            min_value=50,
            max_value=500,
            value=200,
            step=50,
        )

    with col_m2:
        max_depth = st.slider(
            "Profundidade máxima das árvores (max_depth)",
            min_value=2,
            max_value=20,
            value=8,
            step=1,
        )

    # -----------------------------------------------
    # Treinar modelo
    # -----------------------------------------------
    if st.button("🚀 Treinar modelo"):
        if auto_type == "reg":
            st.info("Treinando **RandomForestRegressor** (regressão)...")
            model = RandomForestRegressor(
                n_estimators=n_estimators,
                max_depth=max_depth,
                random_state=random_state,
                n_jobs=-1,
            )
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            r2 = r2_score(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            rmse = mean_squared_error(y_test, y_pred, squared=False)

            st.markdown("### 📊 Métricas de Regressão")
            st.write(f"**R²:** {r2:.4f}")
            st.write(f"**MAE (erro médio absoluto):** {mae:.4f}")
            st.write(f"**RMSE:** {rmse:.4f}")

        else:
            st.info("Treinando **RandomForestClassifier** (classificação)...")
            model = RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                random_state=random_state,
                n_jobs=-1,
            )
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            acc = accuracy_score(y_test, y_pred)

            st.markdown("### 📊 Métricas de Classificação")
            st.write(f"**Acurácia:** {acc:.4f}")

            # Mapeia de volta os rótulos, se tivermos dicionário
            if class_labels is not None:
                st.markdown("**Mapeamento de classes:**")
                st.json(class_labels)

        # -------------------------------------------
        # Importâncias de features
        # -------------------------------------------
        try:
            importances = model.feature_importances_
            feat_names = X.columns

            st.markdown("---")
            st.markdown("### 🔍 Importância das variáveis (features)")

            imp_df = pd.DataFrame(
                {"feature": feat_names, "importance": importances}
            ).sort_values("importance", ascending=False)

            st.dataframe(imp_df)

            # Gráfico de barras das N principais
            top_n = st.slider(
                "Número de variáveis para mostrar no gráfico",
                min_value=3,
                max_value=min(20, len(imp_df)),
                value=min(10, len(imp_df)),
            )

            fig, ax = plt.subplots(figsize=(8, 4))
            top = imp_df.head(top_n).iloc[::-1]  # inverte para plotar de baixo pra cima
            ax.barh(top["feature"], top["importance"])
            ax.set_xlabel("Importância relativa")
            ax.set_ylabel("Feature")
            ax.set_title("Importância das variáveis no modelo")
            st.pyplot(fig)

        except Exception as e:
            st.warning(f"Não foi possível calcular/plotar importâncias de features: {e}")
