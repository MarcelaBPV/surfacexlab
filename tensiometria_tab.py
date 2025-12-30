# tensiometria_tab.py
# -*- coding: utf-8 -*-

import streamlit as st
import pandas as pd
import numpy as np
from datetime import date

# =========================================================
# CONSTANTES — LÍQUIDOS PADRÃO
# =========================================================
LIQUIDS = {
    "Water": {"gamma_total": 72.8, "gamma_d": 21.8, "gamma_p": 51.0},
    "Diiodomethane": {"gamma_total": 50.8, "gamma_d": 50.8, "gamma_p": 0.0},
    "Ethylene Glycol": {"gamma_total": 47.7, "gamma_d": 29.0, "gamma_p": 18.7},
}

# =========================================================
# OWRK — MODELO
# =========================================================
def owkr_fit(contact_angles, liquids):
    """
    Owens–Wendt–Rabel–Kaelble (OWRK)
    Retorna: gamma_total, gamma_d, gamma_p, R²
    """
    y = []
    x = []

    for angle, liquid in zip(contact_angles, liquids):
        theta = np.deg2rad(angle)
        gamma_l = LIQUIDS[liquid]["gamma_total"]
        gamma_ld = LIQUIDS[liquid]["gamma_d"]
        gamma_lp = LIQUIDS[liquid]["gamma_p"]

        y.append((gamma_l * (1 + np.cos(theta))) / (2 * np.sqrt(gamma_ld)))
        x.append(np.sqrt(gamma_lp / gamma_ld))

    x = np.array(x)
    y = np.array(y)

    coeffs = np.polyfit(x, y, 1)
    gamma_sd = coeffs[1] ** 2
    gamma_sp = coeffs[0] ** 2
    gamma_total = gamma_sd + gamma_sp

    y_pred = coeffs[0] * x + coeffs[1]
    r2 = 1 - np.sum((y - y_pred) ** 2) / np.sum((y - np.mean(y)) ** 2)

    return gamma_total, gamma_sd, gamma_sp, r2

# =========================================================
# HELPERS — BANCO
# =========================================================
def get_samples(supabase):
    res = (
        supabase
        .table("samples")
        .select("id, sample_code")
        .order("created_at", desc=True)
        .execute()
    )
    return res.data if res.data else []


def create_experiment(supabase, sample_id):
    res = supabase.table("experiments").insert({
        "sample_id": sample_id,
        "experiment_type": "SurfaceEnergy",
        "experiment_date": str(date.today())
    }).execute()
    return res.data[0]["id"]

# =========================================================
# UI — TENSIOMETRIA
# =========================================================
def render_tensiometria_tab(supabase):
    st.header("🧲 Físico-Mecânica — Energia Livre de Superfície (OWRK)")

    # -----------------------------------------------------
    # Seleção da amostra
    # -----------------------------------------------------
    samples = get_samples(supabase)
    if not samples:
        st.warning("Nenhuma amostra cadastrada.")
        return

    sample_map = {s["sample_code"]: s["id"] for s in samples}
    sample_code = st.selectbox("Amostra", list(sample_map.keys()))
    sample_id = sample_map[sample_code]

    # -----------------------------------------------------
    # Ângulos de contato
    # -----------------------------------------------------
    st.subheader("Ângulos de Contato (graus)")

    angles = {}
    cols = st.columns(len(LIQUIDS))
    for col, liquid in zip(cols, LIQUIDS.keys()):
        with col:
            angles[liquid] = st.number_input(
                f"{liquid}",
                min_value=0.0,
                max_value=180.0,
                value=60.0
            )

    # -----------------------------------------------------
    # Cálculo OWRK
    # -----------------------------------------------------
    if st.button("Calcular Energia de Superfície"):
        contact_angles = list(angles.values())
        liquids = list(angles.keys())

        gamma_total, gamma_d, gamma_p, r2 = owkr_fit(contact_angles, liquids)

        st.success("✔ Cálculo realizado com sucesso")

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("γ Total (mJ/m²)", f"{gamma_total:.2f}")
        col2.metric("γ Dispersiva", f"{gamma_d:.2f}")
        col3.metric("γ Polar", f"{gamma_p:.2f}")
        col4.metric("R²", f"{r2:.4f}")

        # -------------------------------------------------
        # Salvar no banco
        # -------------------------------------------------
        experiment_id = create_experiment(supabase, sample_id)

        for liquid, angle in angles.items():
            supabase.table("surface_energy_measurements").insert({
                "experiment_id": experiment_id,
                "liquid": liquid,
                "contact_angle_deg": angle,
                "surface_energy_total": gamma_total,
                "surface_energy_dispersive": gamma_d,
                "surface_energy_polar": gamma_p,
                "model": "OWRK",
                "r2_fit": r2
            }).execute()

        st.success("✔ Resultados salvos no banco")

    # -----------------------------------------------------
    # Histórico
    # -----------------------------------------------------
    st.subheader("Histórico de Energia Superficial")

    history = (
        supabase
        .table("surface_energy_measurements")
        .select(
            "created_at, liquid, contact_angle_deg, surface_energy_total, surface_energy_dispersive, surface_energy_polar, r2_fit"
        )
        .order("created_at", desc=True)
        .execute()
    )

    if history.data:
        df = pd.DataFrame(history.data)
        st.dataframe(df)
