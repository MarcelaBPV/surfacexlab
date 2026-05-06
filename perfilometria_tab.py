# =========================================================
# perfilometria_tab.py
# SurfaceXLab — Perfilometria
# =========================================================

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from perfilometria_processing import (
    process_profilometry
)


# =========================================================
# TAB PRINCIPAL
# =========================================================
def render_perfilometria_tab():

    st.header("📏 Perfilometria e Rugosidade")

    st.markdown("""
    Módulo para análise topográfica de superfícies,
    cálculo automático de parâmetros de rugosidade
    e comparação entre diferentes tratamentos
    superficiais.
    """)

    # =====================================================
    # UPLOAD
    # =====================================================
    files = st.file_uploader(
        "Upload arquivos perfilométricos",
        accept_multiple_files=True,
        type=["xlsx", "xls", "csv", "txt"]
    )

    if not files:

        st.info(
            "Faça upload das amostras perfilométricas."
        )

        return

    results = []

    # =====================================================
    # PROCESSAMENTO INDIVIDUAL
    # =====================================================
    for f in files:

        st.divider()

        st.subheader(f"🧪 {f.name}")

        try:

            result = process_profilometry(f)

            results.append(
                result["summary"]
            )

            # =============================================
            # PERFIL
            # =============================================
            st.pyplot(
                result["figure"]
            )

            # =============================================
            # MÉTRICAS
            # =============================================
            col1, col2, col3 = st.columns(3)

            col1.metric(
                "Ra (µm)",
                result["summary"]["Ra (µm)"]
            )

            col2.metric(
                "Rq (µm)",
                result["summary"]["Rq (µm)"]
            )

            col3.metric(
                "Classe",
                result["summary"]["Classe"]
            )

        except Exception as e:

            st.error(
                f"Erro no processamento de {f.name}"
            )

            st.exception(e)

    # =====================================================
    # DASHBOARD COMPARATIVO
    # =====================================================
    if results:

        st.divider()

        st.subheader(
            "📊 Comparação de Rugosidade"
        )

        df_summary = pd.DataFrame(results)

        st.dataframe(
            df_summary,
            use_container_width=True
        )

        # =============================================
        # GRÁFICO Ra / Rq
        # =============================================
        fig, ax = plt.subplots(
            figsize=(8,5),
            dpi=300
        )

        x = range(len(df_summary))

        width = 0.4

        ax.bar(
            [i - width/2 for i in x],
            df_summary["Ra (µm)"],
            width,
            label="Ra"
        )

        ax.bar(
            [i + width/2 for i in x],
            df_summary["Rq (µm)"],
            width,
            label="Rq"
        )

        ax.set_xticks(x)

        ax.set_xticklabels(
            [
                s.replace(".xlsx","")
                for s in df_summary["Amostra"]
            ],
            rotation=15
        )

        ax.set_ylabel("Rugosidade (µm)")

        ax.set_xlabel("Amostras")

        ax.set_title(
            "Comparação dos parâmetros Ra e Rq"
        )

        ax.legend()

        ax.grid(alpha=0.3)

        plt.tight_layout()

        st.pyplot(fig)

        # =============================================
        # EXPORTAÇÃO
        # =============================================
        csv = df_summary.to_csv(
            index=False
        )

        st.download_button(
            label="⬇ Exportar resultados",
            data=csv,
            file_name="perfilometria_summary.csv",
            mime="text/csv"
        )

        # =============================================
        # SESSION STATE
        # =============================================
        st.session_state[
            "perfilometria_samples"
        ] = df_summary
