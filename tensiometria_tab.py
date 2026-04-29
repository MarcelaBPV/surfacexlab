# =========================================================
# TENSIOMETRIA — SurfaceXLab (FINAL ROBUSTO)
# =========================================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from tensiometria_processing import process_tensiometry


# =========================================================
# GRÁFICO COMPARATIVO
# =========================================================
def plot_comparison(df):

    fig, ax = plt.subplots(figsize=(7,4), dpi=300)

    x = np.arange(len(df.index))
    width = 0.35

    ax.bar(x - width/2, df["Energia superficial (mJ/m²)"], width, label="Energia")
    ax.bar(x + width/2, df["q* (°)"], width, label="q*")

    ax.set_xticks(x)
    ax.set_xticklabels(df.index)

    ax.set_xlabel("Amostras")
    ax.set_ylabel("Valores")

    ax.set_title("Energia de superfície vs Ângulo de contato")

    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    return fig


# =========================================================
# PCA
# =========================================================
def plot_pca(df):

    df_num = df.select_dtypes(include=[np.number]).fillna(0)

    X = StandardScaler().fit_transform(df_num.values)

    pca = PCA(n_components=2)
    scores = pca.fit_transform(X)
    loadings = pca.components_.T

    var_exp = pca.explained_variance_ratio_ * 100

    fig, ax = plt.subplots(figsize=(6,6), dpi=300)

    ax.scatter(scores[:,0], scores[:,1], s=90, edgecolor="black")

    for i, label in enumerate(df.index):
        ax.text(scores[i,0], scores[i,1], label)

    scale = np.max(np.abs(scores)) * 0.8

    for i, var in enumerate(df_num.columns):

        ax.arrow(
            0, 0,
            loadings[i,0]*scale,
            loadings[i,1]*scale,
            head_width=0.08,
            color="black"
        )

        ax.text(
            loadings[i,0]*scale*1.1,
            loadings[i,1]*scale*1.1,
            var
        )

    ax.axhline(0, color="gray")
    ax.axvline(0, color="gray")

    ax.set_xlabel(f"PC1 ({var_exp[0]:.1f}%)")
    ax.set_ylabel(f"PC2 ({var_exp[1]:.1f}%)")

    ax.set_title("PCA — Tensiometria")

    ax.grid(alpha=0.3)

    return fig


# =========================================================
# TAB PRINCIPAL
# =========================================================
def render_tensiometria_tab(supabase=None):

    st.header("💧 Tensiometria — Energia de Superfície")

    if "tensiometry_samples" not in st.session_state:
        st.session_state.tensiometry_samples = {}

    tabs = st.tabs([
        "📐 Upload & Análise",
        "📊 Comparação",
        "🧠 PCA"
    ])

    # =====================================================
    # SUBABA 1 — UPLOAD
    # =====================================================
    with tabs[0]:

        files = st.file_uploader(
            "Upload (.LOG, .xlsx, .csv)",
            type=["log", "txt", "xlsx", "csv"],
            accept_multiple_files=True
        )

        col1, col2, col3 = st.columns(3)

        with col1:
            water = st.number_input("Água", value=70.0)

        with col2:
            diiodo = st.number_input("Diiodometano", value=50.0)

        with col3:
            formamide = st.number_input("Formamida", value=60.0)

        theta = {
            "water": water,
            "diiodomethane": diiodo,
            "formamide": formamide
        }

        id_ig = st.number_input("ID/IG", value=0.5)
        i2d_ig = st.number_input("I2D/IG", value=0.3)

        if files:

            for f in files:

                if f.name in st.session_state.tensiometry_samples:
                    st.warning(f"{f.name} já processado")
                    continue

                st.markdown(f"---\n### 📄 {f.name}")

                try:
                    result = process_tensiometry(f, theta, id_ig, i2d_ig)
                except Exception as e:
                    st.error("Erro ao processar arquivo")
                    st.exception(e)
                    continue

                # gráfico
                st.pyplot(result["figure"])

                s = result["summary"]

                # diagnóstico
                st.markdown("### 🧠 Diagnóstico físico")
                st.write(f"**Molhabilidade:** {s['Molhabilidade']}")
                st.write(f"**Diagnóstico:** {s['Diagnóstico']}")

                # métricas
                st.markdown("### 📊 Parâmetros")

                c1, c2, c3 = st.columns(3)

                c1.metric("Energia (mJ/m²)", f"{s['Energia superficial (mJ/m²)']:.2f}")
                c2.metric("q* (°)", f"{s['q* (°)']:.2f}")
                c3.metric("Rrms", f"{s['Rrms (mm)']:.3f}")

                st.dataframe(pd.DataFrame([s]))

                st.session_state.tensiometry_samples[f.name] = s

        # dataset consolidado
        if st.session_state.tensiometry_samples:

            st.markdown("### 📋 Dataset consolidado")

            df_all = pd.DataFrame(st.session_state.tensiometry_samples).T

            st.dataframe(df_all)

            if st.button("🗑 Limpar dados"):
                st.session_state.tensiometry_samples = {}
                st.experimental_rerun()

    # =====================================================
    # SUBABA 2 — COMPARAÇÃO
    # =====================================================
    with tabs[1]:

        if len(st.session_state.tensiometry_samples) < 2:
            st.info("Carregue pelo menos duas amostras")
            return

        df = pd.DataFrame(st.session_state.tensiometry_samples).T

        st.dataframe(df)

        st.markdown("### 📊 Comparação")

        st.pyplot(plot_comparison(df))

    # =====================================================
    # SUBABA 3 — PCA
    # =====================================================
    with tabs[2]:

        if len(st.session_state.tensiometry_samples) < 2:
            st.info("Carregue pelo menos duas amostras")
            return

        df = pd.DataFrame(st.session_state.tensiometry_samples).T

        st.dataframe(df)

        st.pyplot(plot_pca(df))
