# =========================================================
# tensiometria_tab.py
# SurfaceXLab — Tensiometria
# =========================================================

import streamlit as st
import pandas as pd

from tensiometria_processing import process_tensiometry


# =========================================================
# TAB PRINCIPAL
# =========================================================
def render_tensiometria_tab():

    st.header("💧 Molhabilidade e Energia Superficial")

    subtabs = st.tabs([
        "📐 Ângulo de Contato",
        "⚡ Energia Superficial"
    ])

    # =====================================================
    # UPLOAD
    # =====================================================
    files = st.file_uploader(
        "Upload arquivos tensiométricos (.LOG, .CSV, .XLSX)",
        accept_multiple_files=True
    )

    if not files:
        st.info("Faça upload das amostras")
        return

    results = []

    # =====================================================
    # PROCESSAMENTO
    # =====================================================
    for f in files:

        st.divider()

        st.subheader(f"🧪 {f.name}")

        try:

            result = process_tensiometry(f)

            results.append(result["summary"])

            # =============================================
            # SUBABA 1
            # =============================================
            with subtabs[0]:

                st.pyplot(result["fig_theta"])

                col1, col2, col3 = st.columns(3)

                col1.metric(
                    "θ Final",
                    f'{result["summary"]["Theta final (°)"]:.2f}°'
                )

                col2.metric(
                    "Histerese",
                    f'{result["summary"]["Histerese"]:.2f}°'
                )

                col3.metric(
                    "Classe",
                    result["summary"]["Classe"]
                )

            # =============================================
            # SUBABA 2
            # =============================================
            with subtabs[1]:

                st.pyplot(result["fig_energy"])

                col1, col2, col3 = st.columns(3)

                col1.metric(
                    "γ Total",
                    f'{result["summary"]["Energia superficial"]:.2f}'
                )

                col2.metric(
                    "γ Dispersiva",
                    f'{result["summary"]["Componente dispersiva"]:.2f}'
                )

                col3.metric(
                    "γ Polar",
                    f'{result["summary"]["Componente polar"]:.2f}'
                )

        except Exception as e:

            st.error(f"Erro em {f.name}")

            st.exception(e)

    # =====================================================
    # TABELA FINAL
    # =====================================================
    if results:

        st.divider()

        st.subheader("📊 Resumo Comparativo")

        df_summary = pd.DataFrame(results)

        st.dataframe(
            df_summary,
            use_container_width=True
        )

        st.session_state["tensiometria_samples"] = df_summary
