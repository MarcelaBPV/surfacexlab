# tensiometria_tab.py
# -*- coding: utf-8 -*-

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from io import StringIO

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# importa o processamento físico correto
from tensiometria_processing import (
    read_contact_angle_log,
    clean_contact_angle,
    compute_rrms,
    compute_q_star,
)


# =========================================================
# ABA TENSIOMETRIA
# =========================================================
def render_tensiometria_tab(supabase=None):

    st.header("💧 Físico-Mecânica — Tensiometria Óptica")

    st.markdown(
        """
        **Subaba 1**  
        Upload do arquivo `.LOG` + cálculo automático de **Rrms\\*** e **q\\***  
        Inserção manual de **ID/IG** e **I2D/IG**

        **Subaba 2**  
        PCA multivariada baseada em  
        **Rrms\\* (°), ID/IG, I2D/IG e q\\* (°)**
        """
    )

    # =====================================================
    # SESSION STATE
    # =====================================================
    if "tensiometry_samples" not in st.session_state:
        st.session_state.tensiometry_samples = {}

    subtabs = st.tabs([
        "📐 Upload & Processamento",
        "📊 PCA — Tensiometria"
    ])

    # =====================================================
    # SUBABA 1 — PROCESSAMENTO
    # =====================================================
    with subtabs[0]:

        uploaded_files = st.file_uploader(
            "Upload dos arquivos .LOG de tensiometria",
            type=["log", "txt", "csv"],
            accept_multiple_files=True
        )

        st.markdown("### 🔬 Parâmetros complementares (Raman / Topografia)")

        col1, col2 = st.columns(2)
        with col1:
            id_ig = st.number_input("ID/IG", value=0.0, format="%.4f")
        with col2:
            i2d_ig = st.number_input("I2D/IG", value=0.0, format="%.4f")

        if uploaded_files:
            for file in uploaded_files:

                if file.name in st.session_state.tensiometry_samples:
                    st.warning(f"{file.name} já foi processado.")
                    continue

                st.markdown(f"---\n### 📄 Amostra: `{file.name}`")

                try:
                    # -----------------------------
                    # Leitura e limpeza
                    # -----------------------------
                    df_raw = read_contact_angle_log(file)
                    df = clean_contact_angle(df_raw)

                    if df.empty:
                        st.warning("Arquivo ignorado (sem dados válidos).")
                        continue

                    # -----------------------------
                    # Cálculos físicos
                    # -----------------------------
                    rrms = compute_rrms(df)
                    q_star = compute_q_star(df)

                    # -----------------------------
                    # Gráfico θ × tempo
                    # -----------------------------
                    fig, ax = plt.subplots(figsize=(7, 4), dpi=300)
                    ax.plot(df["time_s"], df["theta_mean"], lw=1.5)
                    ax.axhline(q_star, color="red", ls="--", label="q*")
                    ax.set_xlabel("Tempo (s)")
                    ax.set_ylabel("Ângulo de contato (°)")
                    ax.set_title("Evolução do ângulo de contato")
                    ax.legend()
                    ax.grid(alpha=0.3)
                    st.pyplot(fig)

                    # -----------------------------
                    # Summary CONSOLIDADO
                    # -----------------------------
                    summary = {
                        "Amostra": file.name,
                        "Rrms* (°)": rrms,
                        "ID/IG": id_ig,
                        "I2D/IG": i2d_ig,
                        "q* (°)": q_star,
                    }

                    st.session_state.tensiometry_samples[file.name] = summary

                    # -----------------------------
                    # Mostra variáveis calculadas
                    # -----------------------------
                    st.markdown("**Variáveis calculadas para a amostra:**")
                    st.dataframe(
                        pd.DataFrame([summary]).set_index("Amostra"),
                        use_container_width=True
                    )

                    st.success("✔ Amostra processada com sucesso")

                except Exception as e:
                    st.error("Erro ao processar a amostra")
                    st.exception(e)

        if st.session_state.tensiometry_samples:
            st.markdown("---")
            st.subheader("📋 Resumo físico consolidado (todas as amostras)")

            st.dataframe(
                pd.DataFrame(st.session_state.tensiometry_samples.values()),
                use_container_width=True
            )

            if st.button("🗑 Limpar amostras de tensiometria"):
                st.session_state.tensiometry_samples = {}
                st.experimental_rerun()

    # =====================================================
    # SUBABA 2 — PCA
    # =====================================================
    with subtabs[1]:

        if len(st.session_state.tensiometry_samples) < 2:
            st.info("Carregue ao menos duas amostras na subaba de processamento.")
            return

        df_pca = pd.DataFrame(st.session_state.tensiometry_samples.values())

        st.subheader("Dados de entrada da PCA")
        st.dataframe(df_pca, use_container_width=True)

        feature_cols = ["Rrms* (°)", "ID/IG", "I2D/IG", "q* (°)"]

        X = df_pca[feature_cols].values
        labels = df_pca["Amostra"].values

        X_scaled = StandardScaler().fit_transform(X)

        pca = PCA(n_components=2)
        scores = pca.fit_transform(X_scaled)
        loadings = pca.components_.T
        explained = pca.explained_variance_ratio_ * 100

        # ---------------------------
        # BIPLOT
        # ---------------------------
        fig, ax = plt.subplots(figsize=(7, 7), dpi=300)

        ax.scatter(scores[:, 0], scores[:, 1], s=90, edgecolor="black")

        for i, label in enumerate(labels):
            ax.text(scores[i, 0] + 0.03, scores[i, 1] + 0.03, label, fontsize=9)

        scale = np.max(np.abs(scores)) * 0.85
        for i, var in enumerate(feature_cols):
            ax.arrow(
                0, 0,
                loadings[i, 0] * scale,
                loadings[i, 1] * scale,
                color="black",
                head_width=0.08,
                length_includes_head=True
            )
            ax.text(
                loadings[i, 0] * scale * 1.1,
                loadings[i, 1] * scale * 1.1,
                var,
                fontsize=9
            )

        ax.axhline(0, color="gray", lw=0.6)
        ax.axvline(0, color="gray", lw=0.6)
        ax.set_xlabel(f"PC1 ({explained[0]:.1f}%)")
        ax.set_ylabel(f"PC2 ({explained[1]:.1f}%)")
        ax.set_title("PCA — Tensiometria")
        ax.set_aspect("equal", adjustable="box")
        ax.grid(alpha=0.3)

        st.pyplot(fig)

        st.subheader("Variância explicada")
        st.dataframe(pd.DataFrame({
            "Componente": ["PC1", "PC2"],
            "Variância (%)": explained.round(2)
        }))
