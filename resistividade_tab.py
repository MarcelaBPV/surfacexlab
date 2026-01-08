# -*- coding: utf-8 -*-

import streamlit as st
import pandas as pd
from datetime import date


# =========================================================
# HELPERS — BANCO DE DADOS
# =========================================================

def get_samples(supabase):
    rtry:
    res = (
        supabase
        .table("alguma_tabela")
        .select("*")
        .execute()
    )
    data = res.data if res.data else []
except Exception as e:
    st.warning("⚠ Módulo de resistividade ainda não configurado no banco.")
    st.stop()

    return res.data if res.data else []


def create_experiment(supabase, sample_id):
    res = supabase.table("experiments").insert({
        "sample_id": sample_id,
        "experiment_type": "Electrical",
        "experiment_date": str(date.today())
    }).execute()
    return res.data[0]["id"]


# =========================================================
# UI — ABA RESISTIVIDADE
# =========================================================

def render_resistividade_tab(supabase):
    st.header("⚡ Resistividade Elétrica")

    try:
        res = supabase.table("resistivity_measurements").select("*").limit(1).execute()
    except Exception:
        st.info("Módulo de resistividade ainda não inicializado no banco.")
        return

    st.success("Tabela de resistividade encontrada.")

    # -----------------------------------------------------
    # 1️⃣ Seleção da amostra
    # -----------------------------------------------------
    samples = get_samples(supabase)

    if not samples:
        st.warning("Nenhuma amostra cadastrada.")
        return

    sample_map = {s["sample_code"]: s["id"] for s in samples}

    sample_code = st.selectbox(
        "Amostra",
        list(sample_map.keys()),
        key="res_sample_select"
    )
    sample_id = sample_map[sample_code]

    # -----------------------------------------------------
    # 2️⃣ Parâmetros do experimento
    # -----------------------------------------------------
    st.subheader("Parâmetros da Medição")

    col1, col2, col3 = st.columns(3)
    with col1:
        method = st.selectbox(
            "Método",
            ["Four-Point Probe", "Large-Area Electrodes"],
            key="res_method"
        )
    with col2:
        current_a = st.number_input(
            "Corrente (A)",
            value=0.0,
            format="%.6f",
            key="res_current"
        )
    with col3:
        voltage_v = st.number_input(
            "Tensão (V)",
            value=0.0,
            format="%.6f",
            key="res_voltage"
        )

    resistance_ohm = st.number_input(
        "Resistência (Ω)",
        value=0.0,
        format="%.6f",
        key="res_resistance"
    )

    resistivity_ohm_cm = st.number_input(
        "Resistividade (Ω·cm)",
        value=0.0,
        format="%.6f",
        key="res_resistivity"
    )

    temperature_c = st.number_input(
        "Temperatura (°C)",
        value=25.0,
        key="res_temperature"
    )

    # -----------------------------------------------------
    # 3️⃣ Salvar no banco
    # -----------------------------------------------------
    save_clicked = st.button(
        "Salvar Medição Elétrica",
        key="res_save_button"
    )

    if save_clicked:
        experiment_id = create_experiment(supabase, sample_id)

        supabase.table("electrical_measurements").insert({
            "experiment_id": experiment_id,
            "method": method,
            "current_a": current_a,
            "voltage_v": voltage_v,
            "resistance_ohm": resistance_ohm,
            "resistivity_ohm_cm": resistivity_ohm_cm,
            "temperature_c": temperature_c
        }).execute()

        st.success("✔ Medição elétrica salva com sucesso!")

    # -----------------------------------------------------
    # 4️⃣ Histórico
    # -----------------------------------------------------
    st.subheader("Histórico de Medições Elétricas")

    history = (
        supabase
        .table("electrical_measurements")
        .select(
            "created_at, method, resistance_ohm, resistivity_ohm_cm, temperature_c"
        )
        .order("created_at", desc=True)
        .execute()
    )

    if history.data:
        df = pd.DataFrame(history.data)
        st.dataframe(
            df,
            key="res_history_table"
        )
    else:
        st.info("Nenhuma medição elétrica registrada.")
