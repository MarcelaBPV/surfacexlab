# =========================================================
# PERFILOMETRIA TAB — SurfaceXLab (VERSÃO AVANÇADA)
# =========================================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# =========================================================
# FUNÇÕES DE PROCESSAMENTO
# =========================================================

def remover_tendencia(x, z):
    """Remove inclinação linear (form removal)"""
    coef = np.polyfit(x, z, 1)
    tendencia = np.polyval(coef, x)
    return z - tendencia


def calcular_parametros(z):
    """Calcula parâmetros clássicos de rugosidade"""

    z_mean = np.mean(z)

    Ra = np.mean(np.abs(z - z_mean))
    Rq = np.sqrt(np.mean((z - z_mean)**2))

    Rp = np.max(z - z_mean)
    Rv = np.abs(np.min(z - z_mean))
    Rt = np.max(z) - np.min(z)

    # Rz simplificado (ISO aproximado)
    z_sorted = np.sort(z)
    top5 = np.mean(z_sorted[-5:])
    bottom5 = np.mean(z_sorted[:5])
    Rz = top5 - bottom5

    return {
        "Ra": Ra,
        "Rq": Rq,
        "Rz": Rz,
        "Rt": Rt,
        "Rp": Rp,
        "Rv": Rv
    }


# =========================================================
# PLOT CIENTÍFICO
# =========================================================

def plot_perfil(x, z, z_filtrado=None):
    fig, ax = plt.subplots(figsize=(10, 4))

    ax.plot(x, z, label="Perfil bruto", linewidth=1)

    if z_filtrado is not None:
        ax.plot(x, z_filtrado, label="Perfil sem forma", linewidth=1.5)

    ax.set_xlabel("Posição (µm)")
    ax.set_ylabel("Altura (µm)")
    ax.set_title("Perfil de Rugosidade")

    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.4)

    return fig


# =========================================================
# TAB PRINCIPAL
# =========================================================

def render_perfilometria_tab():

    st.subheader("📏 Perfilometria de Superfície")
    st.caption("Análise de rugosidade com tratamento de forma e parâmetros ISO")

    st.divider()

    # =====================================================
    # UPLOAD
    # =====================================================
    file = st.file_uploader(
        "📂 Envie arquivos de perfilometria",
        type=["csv", "txt", "xlsx"],
        accept_multiple_files=True
    )

    if not file:
        st.info("Envie um ou mais arquivos para iniciar.")
        return

    # =====================================================
    # LOOP DE ARQUIVOS
    # =====================================================
    resultados = []

    for f in file:

        st.markdown(f"### 📄 {f.name}")

        try:
            # ==============================
            # LEITURA INTELIGENTE
            # ==============================
            if f.name.endswith(".xlsx"):
                df = pd.read_excel(f)
            else:
                df = pd.read_csv(f)

            st.dataframe(df.head())

            # ==============================
            # SELEÇÃO DE COLUNAS
            # ==============================
            col_x = st.selectbox(
                f"Coluna X ({f.name})",
                df.columns,
                key=f"x_{f.name}"
            )

            col_z = st.selectbox(
                f"Coluna Z ({f.name})",
                df.columns,
                key=f"z_{f.name}"
            )

            x = df[col_x].values
            z = df[col_z].values

            # ==============================
            # TRATAMENTO
            # ==============================
            aplicar_remocao = st.checkbox(
                f"Remover tendência (forma) — {f.name}",
                value=True,
                key=f"trend_{f.name}"
            )

            if aplicar_remocao:
                z_processado = remover_tendencia(x, z)
            else:
                z_processado = z

            # ==============================
            # PARÂMETROS
            # ==============================
            params = calcular_parametros(z_processado)

            c1, c2, c3 = st.columns(3)
            c4, c5, c6 = st.columns(3)

            c1.metric("Ra", f"{params['Ra']:.4f}")
            c2.metric("Rq", f"{params['Rq']:.4f}")
            c3.metric("Rz", f"{params['Rz']:.4f}")

            c4.metric("Rt", f"{params['Rt']:.4f}")
            c5.metric("Rp", f"{params['Rp']:.4f}")
            c6.metric("Rv", f"{params['Rv']:.4f}")

            # ==============================
            # GRÁFICO
            # ==============================
            st.pyplot(plot_perfil(x, z, z_processado))

            # ==============================
            # ARMAZENAMENTO
            # ==============================
            resultados.append({
                "arquivo": f.name,
                **params
            })

        except Exception as e:
            st.error(f"Erro no arquivo {f.name}: {e}")

    # =====================================================
    # TABELA FINAL
    # =====================================================
    if resultados:
        st.divider()
        st.markdown("### 📊 Comparação entre amostras")

        df_resultados = pd.DataFrame(resultados)
        st.dataframe(df_resultados)

        # salvar no session_state
        if "perfilometria_samples" not in st.session_state:
            st.session_state["perfilometria_samples"] = {}

        for r in resultados:
            st.session_state["perfilometria_samples"][r["arquivo"]] = r
