# tensiometria_tab.py
# -*- coding: utf-8 -*-

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from io import StringIO

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


# =========================================================
# LEITURA ROBUSTA DO LOG
# =========================================================
def read_tensiometry_log(file):

    file.seek(0)
    raw = file.read().decode("latin1", errors="ignore")
    lines = raw.splitlines()

    header_idx = None
    for i, line in enumerate(lines):
        if "Mean" in line and ("Theta" in line or "Time" in line):
            header_idx = i
            break

    if header_idx is None:
        return pd.DataFrame()

    buffer = StringIO("\n".join(lines[header_idx:]))

    try:
        df = pd.read_csv(buffer, sep=None, engine="python", on_bad_lines="skip")
    except Exception:
        return pd.DataFrame()

    df.columns = [c.strip() for c in df.columns]

    rename_map = {
        "Time": "time_s",
        "Mean": "theta_mean",
        "Theta(L)": "theta_L",
        "Theta(R)": "theta_R",
        "Messages": "messages",
    }

    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

    for col in df.columns:
        df[col] = (
            df[col]
            .astype(str)
            .str.replace(",", ".", regex=False)
        )
        df[col] = pd.to_numeric(df[col], errors="coerce")

    if "theta_mean" not in df.columns:
        return pd.DataFrame()

    df = df[
        (df["theta_mean"] > 0) &
        (df["theta_mean"] < 180)
    ]

    if "messages" in df.columns:
        df = df[~df["messages"].astype(str).str.contains("Error", na=False)]

    return df.reset_index(drop=True)


# =========================================================
# q* — ÂNGULO CARACTERÍSTICO
# =========================================================
def compute_q_star(df):
    """
    q* = ângulo médio estável
    Calculado nos últimos 30% da curva
    """
    n = len(df)
    if n < 10:
        return float(df["theta_mean"].mean())

    tail = df.iloc[int(0.7 * n):]
    return float(tail["theta_mean"].mean())


# =========================================================
# ABA TENSIOMETRIA
# =========================================================
def render_tensiometria_tab(supabase=None):

    st.header("💧 Físico-Mecânica — Tensiometria Óptica")

    st.markdown(
        """
        **Subaba 1**  
        Upload do arquivo `.LOG` + parâmetros físicos complementares  

        **Subaba 2**  
        PCA multivariada baseada em **Rrms, ID/IG, I2D/IG e q\\***
        """
    )

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

        col1, col2, col3 = st.columns(3)
        with col1:
            rrms = st.number_input("Rrms (mm)", value=0.0, format="%.4f")
        with col2:
            id_ig = st.number_input("ID/IG", value=0.0, format="%.4f")
        with col3:
            i2d_ig = st.number_input("I2D/IG", value=0.0, format="%.4f")

        if uploaded_files:
            for file in uploaded_files:

                if file.name in st.session_state.tensiometry_samples:
                    st.warning(f"{file.name} já processado.")
                    continue

                st.markdown(f"### 📄 Amostra: `{file.name}`")

                df = read_tensiometry_log(file)

                if df.empty:
                    st.warning("Arquivo ignorado (sem dados válidos).")
                    continue

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

                summary = {
                    "Amostra": file.name,
                    "Rrms (mm)": rrms,
                    "ID/IG": id_ig,
                    "I2D/IG": i2d_ig,
                    "q* (°)": q_star,
                }

                st.session_state.tensiometry_samples[file.name] = summary
                st.success("✔ Amostra processada com sucesso")

        if st.session_state.tensiometry_samples:
            st.subheader("Resumo físico das amostras")
            st.dataframe(
                pd.DataFrame(st.session_state.tensiometry_samples.values()),
                use_container_width=True
            )

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

        feature_cols = ["Rrms (mm)", "ID/IG", "I2D/IG", "q* (°)"]

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
