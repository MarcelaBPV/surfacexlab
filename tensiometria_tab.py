# tensiometria_tab.py

import streamlit as st
import pandas as pd

from tensiometria_processing import process_tensiometry


def render_tensiometria_tab(supabase=None):

    st.header("💧 Análises Fisico-Mecênicas")

    if "tensiometry_samples" not in st.session_state:
        st.session_state.tensiometry_samples = {}

    files = st.file_uploader(
        "Upload (.LOG)",
        type=["log", "txt"],
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

            st.markdown(f"### {f.name}")

            result = process_tensiometry(f, theta, id_ig, i2d_ig)

            st.pyplot(result["figure"])

            s = result["summary"]

            st.markdown("### 🧠 Diagnóstico físico")
            st.write(f"Molhabilidade: {s['Molhabilidade']}")
            st.write(f"Diagnóstico: {s['Diagnóstico']}")

            st.markdown("### 📊 Parâmetros")

            col1, col2, col3 = st.columns(3)

            col1.metric("Energia (mJ/m²)", f"{s['Energia superficial (mJ/m²)']:.2f}")
            col2.metric("q* (°)", f"{s['q* (°)']:.2f}")
            col3.metric("Rrms", f"{s['Rrms (mm)']:.3f}")

            st.dataframe(pd.DataFrame([s]))

            st.session_state.tensiometry_samples[f.name] = s
