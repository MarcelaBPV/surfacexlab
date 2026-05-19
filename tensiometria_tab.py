# =========================================================
# tensiometria_tab.py
# SurfaceXLab — Módulo Físico-Químico de Superfícies
# =========================================================

import streamlit as st
import pandas as pd
import plotly.express as px

from tensiometria_processing import process_tensiometry


# =========================================================
# TAB PRINCIPAL
# =========================================================
def render_tensiometria_tab():

    st.header("💧 Módulo Físico-Químico de Superfícies")

    st.markdown("""
    Plataforma para análise interfacial de superfícies FC200,
    incluindo molhabilidade, energia superficial e análise estatística.
    """)

    # =====================================================
    # SUBABAS
    # =====================================================
    subtabs = st.tabs([
        "💧 Molhabilidade",
        "⚡ Energia Superficial",
        "📊 PCA e Correlações"
    ])

    # =====================================================
    # UPLOAD
    # =====================================================
    files = st.file_uploader(
        "Upload arquivos tensiométricos (.LOG, .CSV, .XLSX)",
        accept_multiple_files=True
    )

    if not files:

        st.info("Faça upload das amostras FC200")

        return

    results = []

    # =====================================================
    # PROCESSAMENTO DAS AMOSTRAS
    # =====================================================
    for f in files:

        st.divider()

        st.subheader(f"🧪 {f.name}")

        try:

            result = process_tensiometry(f)

            summary = result["summary"]

            results.append(summary)

            # =================================================
            # SUBABA 1 — MOLHABILIDADE
            # =================================================
            with subtabs[0]:

                st.pyplot(result["fig_theta"])

                col1, col2, col3, col4 = st.columns(4)

                col1.metric(
                    "θ Médio",
                    f'{summary["Theta final (°)"]:.2f}°'
                )

                col2.metric(
                    "Desvio padrão",
                    f'{summary["Desvio padrão"]:.2f}°'
                )

                col3.metric(
                    "Histerese",
                    f'{summary["Histerese"]:.2f}°'
                )

                col4.metric(
                    "Classe",
                    summary["Classe"]
                )

                st.markdown(
                    f"**Diagnóstico:** {summary['Diagnóstico']}"
                )

            # =================================================
            # SUBABA 2 — ENERGIA SUPERFICIAL
            # =================================================
            with subtabs[1]:

                st.pyplot(result["fig_energy"])

                col1, col2, col3, col4 = st.columns(4)

                col1.metric(
                    "γ Total",
                    f'{summary["Energia superficial"]:.2f} mJ/m²'
                )

                col2.metric(
                    "γ Dispersiva",
                    f'{summary["Componente dispersiva"]:.2f}'
                )

                col3.metric(
                    "γ Polar",
                    f'{summary["Componente polar"]:.2f}'
                )

                col4.metric(
                    "R² Ajuste",
                    f'{summary["R²"]:.3f}'
                )

        except Exception as e:

            st.error(f"Erro no processamento: {f.name}")

            st.exception(e)

    # =====================================================
    # RESUMO COMPARATIVO
    # =====================================================
    if results:

        st.divider()

        st.subheader("📊 Comparação entre Amostras FC200")

        df_summary = pd.DataFrame(results)

        # =================================================
        # IDENTIFICAÇÃO DOS GRUPOS
        # =================================================
        def identify_group(name):

            name = name.lower()

            if "a1.5" in name:
                return "FC200 A1.5"

            elif "a2.5" in name:
                return "FC200 A2.5"

            elif "b1.5" in name:
                return "FC200 B1.5"

            elif "b2.5" in name:
                return "FC200 B2.5"

            return "Outros"

        df_summary["Grupo"] = df_summary["Amostra"].apply(
            identify_group
        )

        # =================================================
        # TABELA
        # =================================================
        st.dataframe(
            df_summary,
            use_container_width=True
        )

        # =================================================
        # BOXPLOT — θ
        # =================================================
        fig_theta_box = px.box(
            df_summary,
            x="Grupo",
            y="Theta final (°)",
            points="all",
            title="Comparação do Ângulo de Contato"
        )

        st.plotly_chart(
            fig_theta_box,
            use_container_width=True
        )

        # =================================================
        # BOXPLOT — γ
        # =================================================
        fig_gamma_box = px.box(
            df_summary,
            x="Grupo",
            y="Energia superficial",
            points="all",
            title="Energia Superficial das Amostras"
        )

        st.plotly_chart(
            fig_gamma_box,
            use_container_width=True
        )

        # =================================================
        # DISPERSIVA vs POLAR
        # =================================================
        fig_scatter = px.scatter(
            df_summary,
            x="Componente dispersiva",
            y="Componente polar",
            color="Grupo",
            size="Energia superficial",
            hover_name="Amostra",
            title="Componentes da Energia Superficial"
        )

        st.plotly_chart(
            fig_scatter,
            use_container_width=True
        )

        # =================================================
        # SUBABA PCA
        # =================================================
        with subtabs[2]:

            st.subheader("📊 PCA Multimodal")

            st.info("""
            Integração futura entre:
            - Molhabilidade
            - Resistividade elétrica
            - Raman
            - Rugosidade
            """)

            pca_cols = [
                "Theta final (°)",
                "Histerese",
                "Energia superficial",
                "Componente dispersiva",
                "Componente polar"
            ]

            st.dataframe(
                df_summary[pca_cols],
                use_container_width=True
            )

        # =================================================
        # EXPORTAÇÃO
        # =================================================
        csv = df_summary.to_csv(index=False).encode("utf-8")

        st.download_button(
            "⬇️ Exportar Resultados",
            csv,
            file_name="surfaceXlab_tensiometria_fc200.csv",
            mime="text/csv"
        )

        # =================================================
        # SESSION STATE
        # =================================================
        st.session_state["tensiometria_samples"] = df_summary
