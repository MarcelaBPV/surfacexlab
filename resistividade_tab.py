# resistividade_tab.py
# -*- coding: utf-8 -*-

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import date

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


# =========================================================
# HELPERS — BANCO
# =========================================================
def get_samples(supabase):
    try:
        res = (
            supabase
            .table("samples")
            .select("id, sample_code")
            .order("created_at", desc=True)
            .execute()
        )
        return res.data if res.data else []
    except Exception:
        return []


def create_experiment(supabase, sample_id):
    res = supabase.table("experiments").insert({
        "sample_id": sample_id,
        "experiment_type": "Electrical",
        "experiment_date": str(date.today())
    }).execute()
    return res.data[0]["id"]


# =========================================================
# ABA RESISTIVIDADE (PROCESSAMENTO + PCA)
# =========================================================
def render_resistividade_tab(supabase):
    st.header("⚡ Propriedades Elétricas — Resistividade")

    st.markdown(
        """
        Este módulo permite:
        - Registro de **medições elétricas**
        - Armazenamento estruturado no banco
        - **Análise multivariada (PCA)** das propriedades elétricas
        """
    )

    # -----------------------------------------------------
    # Teste da tabela
    # -----------------------------------------------------
    try:
        supabase.table("electrical_measurements").select("id").limit(1).execute()
    except Exception:
        st.info("Módulo de resistividade ainda não inicializado no banco.")
        return

    # =====================================================
    # SUBABAS
    # =====================================================
    subtabs = st.tabs([
        "📐 Medição Elétrica",
        "📊 PCA — Elétrica"
    ])

    # =====================================================
    # SUBABA 1 — MEDIÇÃO
    # =====================================================
    with subtabs[0]:

        samples = get_samples(supabase)
        if not samples:
            st.warning("Nenhuma amostra cadastrada.")
            return

        sample_map = {s["sample_code"]: s["id"] for s in samples}
        sample_code = st.selectbox("Amostra", list(sample_map.keys()))
        sample_id = sample_map[sample_code]

        st.subheader("Parâmetros da Medição")

        col1, col2, col3 = st.columns(3)
        with col1:
            method = st.selectbox(
                "Método",
                ["Four-Point Probe", "Large-Area Electrodes"]
            )
        with col2:
            current_a = st.number_input("Corrente (A)", format="%.6e")
        with col3:
            voltage_v = st.number_input("Tensão (V)", format="%.6e")

        resistance_ohm = st.number_input("Resistência (Ω)", format="%.6e")
        resistivity_ohm_cm = st.number_input("Resistividade (Ω·cm)", format="%.6e")
        temperature_c = st.number_input("Temperatura (°C)", value=25.0)

        if st.button("Salvar Medição Elétrica"):
            experiment_id = create_experiment(supabase, sample_id)

            supabase.table("electrical_measurements").insert({
                "experiment_id": experiment_id,
                "method": method,
                "current_a": current_a,
                "voltage_v": voltage_v,
                "resistance_ohm": resistance_ohm,
                "resistivity_ohm_cm": resistivity_ohm_cm,
                "temperature_c": temperature_c
            }).execute()

            st.success("✔ Medição elétrica salva com sucesso!")

        st.subheader("Histórico")
        history = (
            supabase
            .table("electrical_measurements")
            .select(
                "created_at, method, resistance_ohm, "
                "resistivity_ohm_cm, temperature_c"
            )
            .order("created_at", desc=True)
            .execute()
        )

        if history.data:
            st.dataframe(pd.DataFrame(history.data))
        else:
            st.info("Nenhuma medição registrada.")

    # =====================================================
    # SUBABA 2 — PCA
    # =====================================================
    with subtabs[1]:

        st.subheader("PCA — Propriedades Elétricas")

        res = (
            supabase
            .table("electrical_measurements")
            .select(
                "resistance_ohm, resistivity_ohm_cm, "
                "current_a, voltage_v, temperature_c"
            )
            .execute()
        )

        if not res.data or len(res.data) < 3:
            st.warning("Dados insuficientes para PCA.")
            return

        df = pd.DataFrame(res.data)

        feature_cols = st.multiselect(
            "Variáveis para PCA",
            options=df.columns,
            default=[
                "resistivity_ohm_cm",
                "resistance_ohm",
                "temperature_c"
            ]
        )

        if len(feature_cols) < 2:
            st.warning("Selecione ao menos duas variáveis.")
            return

        X = df[feature_cols].values
        X_scaled = StandardScaler().fit_transform(X)

        pca = PCA(n_components=2)
        scores = pca.fit_transform(X_scaled)
        loadings = pca.components_.T
        explained = pca.explained_variance_ratio_ * 100

        # ---------------------------
        # BIPLOT PADRONIZADO
        # ---------------------------
        fig, ax = plt.subplots(figsize=(6, 6), dpi=300)

        ax.scatter(scores[:, 0], scores[:, 1], s=80, edgecolor="black")

        scale = np.max(np.abs(scores)) * 0.9
        for i, var in enumerate(feature_cols):
            ax.arrow(
                0, 0,
                loadings[i, 0] * scale,
                loadings[i, 1] * scale,
                color="red",
                head_width=0.08,
                length_includes_head=True
            )
            ax.text(
                loadings[i, 0] * scale * 1.1,
                loadings[i, 1] * scale * 1.1,
                var,
                color="red",
                fontsize=9
            )

        ax.axhline(0, color="gray", lw=0.6)
        ax.axvline(0, color="gray", lw=0.6)
        ax.set_xlabel(f"PC1 ({explained[0]:.1f}%)")
        ax.set_ylabel(f"PC2 ({explained[1]:.1f}%)")
        ax.set_title("PCA — Propriedades Elétricas")
        ax.grid(alpha=0.3)

        st.pyplot(fig)

        st.subheader("Variância explicada")
        st.dataframe(pd.DataFrame({
            "Componente": ["PC1", "PC2"],
            "Variância (%)": explained.round(2)
        }))
