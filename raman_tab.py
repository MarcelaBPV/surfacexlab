# -*- coding: utf-8 -*-

import streamlit as st
import pandas as pd
from datetime import date

from raman_processing import process_raman_spectrum_with_groups


# =========================================================
# UI — ABA RAMAN (OBRIGATÓRIA)
# =========================================================
def render_raman_tab(supabase):

    st.header("🔬 Análises Moleculares — Espectroscopia Raman")

    # -----------------------------------------------------
    # Teste visual básico
    # -----------------------------------------------------
    st.info("Aba Raman carregada com sucesso.")

    # -----------------------------------------------------
    # Upload simples (teste)
    # -----------------------------------------------------
    uploaded_file = st.file_uploader(
        "Upload do espectro Raman",
        type=["csv", "txt", "xls", "xlsx"],
        key="raman_upload"
    )

    if uploaded_file:
        st.success(f"Arquivo carregado: {uploaded_file.name}")

        result = process_raman_spectrum_with_groups(
            file_like=uploaded_file,
            peak_prominence=0.02
        )

        st.pyplot(result["figure"])
        st.dataframe(result["peaks_df"])
