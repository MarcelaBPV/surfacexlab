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
# CONSTANTES — LÍQUIDOS (25 °C)
# =========================================================
LIQUIDS = {
    "Água": {"gamma_L": 72.8, "gamma_d": 21.8, "gamma_p": 51.0},
    "Diiodometano": {"gamma_L": 50.8, "gamma_d": 50.8, "gamma_p": 0.0},
    "Formamida": {"gamma_L": 58.0, "gamma_d": 39.0, "gamma_p": 19.0},
}


# =========================================================
# OWRK
# =========================================================
def owkr_surface_energy(theta_by_liquid):

    X, Y = [], []

    for liquid, theta in theta_by_liquid.items():
        props = LIQUIDS[liquid]

        cos_theta = np.cos(np.deg2rad(theta))
        y = props["gamma_L"] * (1 + cos_theta) / (2 * np.sqrt(props["gamma_d"]))
        x = np.sqrt(props["gamma_p"] / props["gamma_d"]) if props["gamma_d"] > 0 else 0

        X.append(x)
        Y.append(y)

    X = np.array(X)
    Y = np.array(Y)

    A = np.vstack([X, np.ones_like(X)]).T
    slope, intercept = np.linalg.lstsq(A, Y, rcond=None)[0]

    gamma_d = intercept ** 2
    gamma_p = slope ** 2

    return {
        "Energia superficial total (mJ/m²)": gamma_d + gamma_p,
        "Componente dispersiva (mJ/m²)": gamma_d,
        "Componente polar (mJ/m²)": gamma_p,
    }


# =========================================================
# LEITURA LOG
# =========================================================
def read_tensiometry_log(file):

    file.seek(0)
    raw = file.read().decode("latin1", errors="ignore")
    lines = raw.splitlines()

    header = next(
        (i for i, l in enumerate(lines) if "Mean" in l and ("Theta" in l or "Time" in l)),
        None
    )

    if header is None:
        return pd.DataFrame()

    buffer = StringIO("\n".join(lines[header:]))

    df = pd.read_csv(buffer, sep=";", engine="python", on_bad_lines="skip")
    df.columns = [c.strip() for c in df.columns]

    rename = {
        "Time": "time_s",
        "Mean": "theta_mean",
        "Area": "area",
        "Volume": "volume",
        "Height": "height",
        "Width": "width",
    }

    df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})

    for c in df.columns:
        df[c] = pd.to_numeric(df[c].astype(str).str.replace(",", "."), errors="coerce")

    df = df.dropna(subset=["theta_mean"])
    df = df[(df["theta_mean"] > 0) & (df["theta_mean"] < 180)]

    return df.reset_index(drop=True)


# =========================================================
# ABA TENSIOMETRIA
# =========================================================
def render_tensiometria_tab(supabase=None):

    st.header("💧 Tensiometria + Raman — PCA Integrado")

    if "samples" not in st.session_state:
        st.session_state.samples = []

    tabs = st.tabs(["📐 Processamento", "📊 PCA Integrado"])

    # =====================================================
    # SUBABA 1
    # =====================================================
    with tabs[0]:

        uploaded = st.file_uploader(
            "Upload dos arquivos .LOG",
            type=["log", "txt", "csv"],
            accept_multiple_files=True
        )

        if uploaded:
            for file in uploaded:

                st.markdown(f"### 📄 {file.name}")
                df = read_tensiometry_log(file)

                if df.empty:
                    st.warning("LOG inválido.")
                    continue

                liquid = st.selectbox(
                    "Líquido",
                    list(LIQUIDS.keys()),
                    key=f"liq_{file.name}"
                )

                id_ig = st.number_input(
                    "ID/IG (Raman)",
                    value=np.nan,
                    key=f"idig_{file.name}"
                )

                i2d_ig = st.number_input(
                    "I2D/IG (Raman)",
                    value=np.nan,
                    key=f"i2dig_{file.name}"
                )

                theta = df["theta_mean"].mean()

                sample = {
                    "Amostra": file.name,
                    "Theta médio (°)": theta,
                    "Volume médio": df["volume"].mean() if "volume" in df else np.nan,
                    "Área média": df["area"].mean() if "area" in df else np.nan,
                    "Altura média": df["height"].mean() if "height" in df else np.nan,
                    "Largura média": df["width"].mean() if "width" in df else np.nan,
                    "ID/IG": id_ig,
                    "I2D/IG": i2d_ig,
                    "Líquido": liquid,
                }

                st.session_state.samples.append(sample)
                st.success("✔ Processado")

        if st.session_state.samples:
            st.dataframe(pd.DataFrame(st.session_state.samples))

    # =====================================================
    # SUBABA 2 — PCA
    # =====================================================
    with tabs[1]:

        if len(st.session_state.samples) < 2:
            st.info("Envie pelo menos duas amostras.")
            return

        df = pd.DataFrame(st.session_state.samples)

        # ---------------- OWRK ----------------
        energy_rows = []
        for name, g in df.groupby("Amostra"):
            thetas = dict(zip(g["Líquido"], g["Theta médio (°)"]))
            if len(thetas) >= 2:
                owkr = owkr_surface_energy(thetas)
            else:
                owkr = {k: np.nan for k in [
                    "Energia superficial total (mJ/m²)",
                    "Componente polar (mJ/m²)",
                    "Componente dispersiva (mJ/m²)"
                ]}
            row = g.iloc[0].to_dict()
            row.update(owkr)
            energy_rows.append(row)

        df_pca = pd.DataFrame(energy_rows)

        features = st.multiselect(
            "Variáveis para PCA",
            options=[c for c in df_pca.columns if c not in ["Amostra", "Líquido"]],
            default=[
                "Energia superficial total (mJ/m²)",
                "Componente polar (mJ/m²)",
                "ID/IG",
                "I2D/IG",
            ]
        )

        X = StandardScaler().fit_transform(df_pca[features].values)
        labels = df_pca["Amostra"].values

        pca = PCA(n_components=2)
        scores = pca.fit_transform(X)
        loadings = pca.components_.T
        var = pca.explained_variance_ratio_ * 100

        fig, ax = plt.subplots(figsize=(6, 6), dpi=300)

        ax.scatter(scores[:, 0], scores[:, 1], s=80, edgecolor="black")

        for i, lab in enumerate(labels):
            ax.text(scores[i, 0]+0.03, scores[i, 1]+0.03, lab, fontsize=9)

        scale = np.max(np.abs(scores)) * 0.8
        for i, v in enumerate(features):
            ax.arrow(0, 0, loadings[i, 0]*scale, loadings[i, 1]*scale,
                     color="black", width=0.003)
            ax.text(loadings[i, 0]*scale*1.1, loadings[i, 1]*scale*1.1, v)

        ax.set_xlabel(f"PC1 ({var[0]:.1f}%)")
        ax.set_ylabel(f"PC2 ({var[1]:.1f}%)")
        ax.set_title("PCA Integrado — Raman + Tensiometria")
        ax.grid(alpha=0.3)

        st.pyplot(fig)

        st.dataframe(pd.DataFrame({
            "Componente": ["PC1", "PC2"],
            "Variância (%)": var.round(2)
        }))
