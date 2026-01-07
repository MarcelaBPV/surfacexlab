# ml_tab.py
# -*- coding: utf-8 -*-

"""
SurfaceXLab — Machine Learning & Análise Multivariada

Funcionalidades:
- Leitura robusta de features multimodais (Raman, Tensiometria, Elétrica)
- PCA (Análise de Componentes Principais) multimodal
- Visualização PC1 × PC2
- Base pronta para Random Forest (classificação ou regressão)

⚠ Uso científico / exploratório — não diagnóstico.
"""

import streamlit as st
import pandas as pd
import json
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


# =========================================================
# LOAD FEATURES — MULTIMODAL
# =========================================================
def load_ml_features(supabase) -> pd.DataFrame:
    """
    Carrega features Raman, Tensiometria e Elétrica do Supabase.
    Todas são consolidadas em um único DataFrame ML-ready.
    """

    dfs = []

    # ---------------- RAMAN ----------------
    try:
        res = (
            supabase
            .table("raman_features")
            .select("raman_measurement_id, features, created_at")
            .execute()
        )

        if res.data:
            df_r = pd.DataFrame(res.data)
            features = df_r["features"].apply(
                lambda x: x if isinstance(x, dict) else json.loads(x)
            )
            df_feat = pd.json_normalize(features)
            df_r = pd.concat([df_r[["raman_measurement_id"]], df_feat], axis=1)
            df_r["modality"] = "Raman"
            dfs.append(df_r)

    except Exception as e:
        st.warning("⚠ Falha ao carregar features Raman.")
        st.exception(e)

    # ---------------- TENSIOMETRIA ----------------
    try:
        res = (
            supabase
            .table("tensiometry_measurements")
            .select(
                "experiment_id, contact_angle_deg, contact_angle_std_deg"
            )
            .execute()
        )

        if res.data:
            df_t = pd.DataFrame(res.data)
            df_t["modality"] = "Tensiometria"
            dfs.append(df_t)

    except Exception as e:
        st.warning("⚠ Falha ao carregar dados de tensiometria.")
        st.exception(e)

    # ---------------- ELÉTRICA ----------------
    try:
        res = (
            supabase
            .table("electrical_measurements")
            .select(
                "experiment_id, sheet_resistance_ohm_sq, resistivity_ohm_cm, conductivity_s_cm"
            )
            .execute()
        )

        if res.data:
            df_e = pd.DataFrame(res.data)
            df_e["modality"] = "Elétrica"
            dfs.append(df_e)

    except Exception as e:
        st.warning("⚠ Falha ao carregar dados elétricos.")
        st.exception(e)

    if not dfs:
        return pd.DataFrame()

    df = pd.concat(dfs, axis=0, ignore_index=True)
    return df


# =========================================================
# UI — ABA ML + PCA
# =========================================================
def render_ml_tab(supabase):
    st.header("🤖 Análise Multivariada & Machine Learning")

    st.markdown(
        """
        Este módulo integra **dados multimodais** provenientes de:
        - Espectroscopia Raman
        - Tensiometria
        - Medições Elétricas

        Inclui **Análise de Componentes Principais (PCA)** como etapa
        exploratória e preparatória para modelos supervisionados.
        """
    )

    # -----------------------------------------------------
    # Carregar dados
    # -----------------------------------------------------
    df = load_ml_features(supabase)

    if df.empty:
        st.info("Nenhum dado multimodal disponível para análise.")
        return

    st.subheader("📊 Dataset consolidado")
    st.dataframe(df.head(50), use_container_width=True)

    # -----------------------------------------------------
    # Seleção de features numéricas
    # -----------------------------------------------------
    numeric_cols = df.select_dtypes(include="number").columns.tolist()

    if len(numeric_cols) < 2:
        st.warning("Dados numéricos insuficientes para PCA.")
        return

    selected_features = st.multiselect(
        "Selecione as variáveis para PCA",
        numeric_cols,
        default=numeric_cols,
    )

    if len(selected_features) < 2:
        st.warning("Selecione ao menos duas variáveis.")
        return

    X = df[selected_features].dropna()

    # -----------------------------------------------------
    # PCA
    # -----------------------------------------------------
    st.divider()
    st.subheader("📉 Análise de Componentes Principais (PCA)")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    pca = PCA(n_components=2)
    pcs = pca.fit_transform(X_scaled)

    df_pca = pd.DataFrame(
        pcs,
        columns=["PC1", "PC2"],
        index=X.index,
    )

    df_pca["modality"] = df.loc[X.index, "modality"].values

    explained = pca.explained_variance_ratio_ * 100

    st.markdown(
        f"""
        **Variância explicada:**
        - PC1: {explained[0]:.2f} %
        - PC2: {explained[1]:.2f} %
        """
    )

    # -----------------------------------------------------
    # Plot PC1 × PC2
    # -----------------------------------------------------
    fig, ax = plt.subplots(figsize=(7, 5))

    for mod in df_pca["modality"].unique():
        mask = df_pca["modality"] == mod
        ax.scatter(
            df_pca.loc[mask, "PC1"],
            df_pca.loc[mask, "PC2"],
            label=mod,
            alpha=0.7,
        )

    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title("PCA Multimodal — PC1 × PC2")
    ax.legend()
    ax.grid(alpha=0.3)

    st.pyplot(fig)

    # -----------------------------------------------------
    # Loadings (interpretação física)
    # -----------------------------------------------------
    st.subheader("🧠 Importância das variáveis (Loadings PCA)")

    loadings = pd.DataFrame(
        pca.components_.T,
        index=selected_features,
        columns=["PC1", "PC2"],
    )

    st.dataframe(loadings.sort_values("PC1", key=np.abs, ascending=False))

    # -----------------------------------------------------
    # ML-ready
    # -----------------------------------------------------
    st.divider()
    st.subheader("🚀 Pronto para Machine Learning")

    st.markdown(
        """
        O dataset já se encontra:
        - Normalizado
        - Reduzido (opcionalmente via PCA)
        - Multimodal

        ➡ Pode ser utilizado diretamente para:
        - Random Forest (classificação ou regressão)
        - Validação cruzada
        - Análise de importância das features
        """
    )

    st.success("Pipeline PCA + ML pronto para uso.")
