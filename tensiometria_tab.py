# tensiometria_tab.py
# -*- coding: utf-8 -*-

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


# ==========================================================
# PROPRIEDADES DOS LÍQUIDOS (OWRK)
# ==========================================================

LIQUIDS = {

"Agua":{
"gamma":72.8,
"gamma_d":21.8,
"gamma_p":51.0
},

"Diiodometano":{
"gamma":50.8,
"gamma_d":50.8,
"gamma_p":0.0
},

"Etilenoglicol":{
"gamma":48.0,
"gamma_d":29.0,
"gamma_p":19.0
},

"Glicerol":{
"gamma":63.4,
"gamma_d":37.0,
"gamma_p":26.4
}

}


# ==========================================================
# MÉTODO OWRK
# ==========================================================

def calculate_owrk(df_sample):

    X=[]
    Y=[]

    for _,row in df_sample.iterrows():

        liquid=row["Liquido"]
        theta=row["Angulo"]

        if liquid not in LIQUIDS:
            continue

        L=LIQUIDS[liquid]

        theta=np.radians(theta)

        y=(L["gamma"]*(1+np.cos(theta))) / (2*np.sqrt(L["gamma_d"]))
        x=np.sqrt(L["gamma_p"]/L["gamma_d"])

        X.append(x)
        Y.append(y)

    X=np.array(X)
    Y=np.array(Y)

    coef=np.polyfit(X,Y,1)

    slope=coef[0]
    intercept=coef[1]

    gamma_s_d = intercept**2
    gamma_s_p = slope**2
    gamma_total = gamma_s_d + gamma_s_p

    return gamma_total, gamma_s_d, gamma_s_p


# ==========================================================
# ABA TENSIOMETRIA
# ==========================================================

def render_tensiometria_tab():

    st.header("💧 Tensiometria Óptica — Energia de Superfície")

    subtabs = st.tabs([
        "📐 Energia de Superfície",
        "📊 PCA — Tensiometria"
    ])

    if "tensiometry_samples" not in st.session_state:
        st.session_state.tensiometry_samples = {}



# ==========================================================
# SUBABA 1 — CÁLCULO ENERGIA
# ==========================================================

    with subtabs[0]:

        uploaded_file = st.file_uploader(
        "Upload arquivo Excel de ângulo de contato",
        type=["xls","xlsx","ods"]
        )

        if uploaded_file:

            df = pd.read_excel(uploaded_file)

            st.subheader("Dados carregados")

            st.dataframe(df)

            samples=df["Amostra"].unique()

            results=[]

            for sample in samples:

                sub=df[df["Amostra"]==sample]

                gamma,gamma_d,gamma_p = calculate_owrk(sub)

                results.append({

                "Amostra":sample,
                "Energia_total":gamma,
                "Dispersiva":gamma_d,
                "Polar":gamma_p

                })

            df_results=pd.DataFrame(results)

            st.subheader("Energia de superfície calculada")

            st.dataframe(df_results)

            for _,row in df_results.iterrows():

                st.session_state.tensiometry_samples[row["Amostra"]]=row


# ==========================================================
# GRÁFICO POLAR vs DISPERSIVO
# ==========================================================

            st.subheader("Mapa polar vs dispersivo")

            fig,ax=plt.subplots(figsize=(6,6),dpi=300)

            ax.scatter(
            df_results["Dispersiva"],
            df_results["Polar"],
            s=90,
            edgecolor="black"
            )

            for i,row in df_results.iterrows():

                ax.text(
                row["Dispersiva"]+0.5,
                row["Polar"]+0.5,
                row["Amostra"]
                )

            ax.set_xlabel("Componente dispersiva γsᵈ (mN/m)")
            ax.set_ylabel("Componente polar γsᵖ (mN/m)")

            ax.set_title("Energia de superfície — método OWRK")

            ax.grid(alpha=0.3)

            st.pyplot(fig)



# ==========================================================
# SUBABA 2 — PCA
# ==========================================================

    with subtabs[1]:

        if len(st.session_state.tensiometry_samples) < 2:

            st.info("Carregue ao menos duas amostras.")
            return

        df_pca=pd.DataFrame(st.session_state.tensiometry_samples.values())

        st.subheader("Dados utilizados na PCA")

        st.dataframe(df_pca)

        features=[
        "Energia_total",
        "Dispersiva",
        "Polar"
        ]

        X=df_pca[features].values

        labels=df_pca["Amostra"]

        scaler=StandardScaler()

        X_scaled=scaler.fit_transform(X)

        pca=PCA(n_components=2)

        scores=pca.fit_transform(X_scaled)

        loadings=pca.components_.T

        explained=pca.explained_variance_ratio_*100


# ==========================================================
# BIPLOT PCA
# ==========================================================

        fig,ax=plt.subplots(figsize=(7,7),dpi=300)

        ax.scatter(scores[:,0],scores[:,1],s=90,edgecolor="black")

        for i,label in enumerate(labels):

            ax.text(scores[i,0]+0.05,scores[i,1]+0.05,label)

        scale=np.max(np.abs(scores))*0.8

        for i,var in enumerate(features):

            ax.arrow(
            0,0,
            loadings[i,0]*scale,
            loadings[i,1]*scale,
            head_width=0.08
            )

            ax.text(
            loadings[i,0]*scale*1.1,
            loadings[i,1]*scale*1.1,
            var
            )

        ax.axhline(0,color="gray")
        ax.axvline(0,color="gray")

        ax.set_xlabel(f"PC1 ({explained[0]:.1f}%)")
        ax.set_ylabel(f"PC2 ({explained[1]:.1f}%)")

        ax.set_title("PCA — Energia de superfície")

        ax.grid(alpha=0.3)

        st.pyplot(fig)


# ==========================================================
# VARIÂNCIA
# ==========================================================

        st.subheader("Variância explicada")

        st.dataframe(pd.DataFrame({

        "Componente":["PC1","PC2"],
        "Variância (%)":explained.round(2)

        }))
