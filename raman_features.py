# -*- coding: utf-8 -*-

"""
SurfaceXLab — Otimização, PCA e Machine Learning

- PCA estatístico (biplot: scores + loadings)
- PCA temporal multi-amostra
- Random Forest com validação cruzada
- Fingerprints Raman derivados automaticamente (sem tabela fixa)
"""

import streamlit as st
import pandas as pd

from raman_features import extract_raman_features
from ml_models import (
    pca_biplot,
    random_forest_cv,
    temporal_pca
)

# =========================================================
# UI — ABA ML / OTIMIZAÇÃO
# =========================================================
def render_ml_tab(supabase):

    st.header("🤖 Otimização — PCA & Machine Learning")

    st.markdown(
        """
        Este módulo integra **análise estatística multivariada (PCA)** e
        **modelos supervisionados (Random Forest)** a partir de
        **fingerprints espectrais Raman derivados automaticamente**.
        """
    )

    # =====================================================
    # 1️⃣ CARREGAR DADOS RAMAN DO BANCO
    # =====================================================
    try:
        res_meas = (
            supabase
            .table("raman_measurements")
            .select("id, created_at")
            .execute()
        )

        res_peaks = (
            supabase
            .table("raman_peaks")
            .select("""
                raman_measurement_id,
                peak_position_cm,
                peak_intensity,
                peak_fwhm
            """)
            .execute()
        )

    except Exception as e:
        st.error("Erro ao carregar dados Raman do banco.")
        st.exception(e)
        return

    if not res_meas.data or not res_peaks.data:
        st.warning("Dados Raman insuficientes para análise.")
        return

    df_meas = pd.DataFrame(res_meas.data)
    df_peaks = pd.DataFrame(res_peaks.data)

    # =====================================================
    # 2️⃣ GERAR FINGERPRINTS RAMAN (DINÂMICO)
    # =====================================================
    fingerprints = []

    for mid in df_meas["id"]:
        peaks_df = df_peaks[
            df_peaks["raman_measurement_id"] == mid
        ].copy()

        if peaks_df.empty:
            continue

        # Normalização de nomes esperados pelo módulo
        peaks_df = peaks_df.rename(columns={
            "peak_position_cm": "peak_cm1",
            "peak_intensity": "intensity_norm",
            "peak_fwhm": "width",
        })

        features_out = extract_raman_features(
            spectrum_df=None,  # áreas podem ser desativadas se necessário
            peaks_df=peaks_df
        )

        fingerprint = features_out["fingerprint"]
        fingerprint["measurement_id"] = mid
        fingerprint["created_at"] = df_meas.loc[
            df_meas["id"] == mid, "created_at"
        ].values[0]

        fingerprints.append(fingerprint)

    if not fingerprints:
        st.warning("Fingerprints Raman não puderam ser gerados.")
        return

    df_fp = pd.DataFrame(fingerprints).fillna(0.0)

    st.subheader("📊 Fingerprints Raman (ML-ready)")
    st.dataframe(df_fp.head(20), use_container_width=True)

    feature_cols = df_fp.drop(
        columns=["measurement_id", "created_at"],
        errors="ignore"
    ).columns.tolist()

    # =====================================================
    # 3️⃣ PCA ESTATÍSTICO — BIPLOT (PAPER-LEVEL)
    # =====================================================
    st.divider()
    st.subheader("PCA Estatístico — Biplot")

    pca_out = pca_biplot(
        X=df_fp[feature_cols],
        n_components=2
    )

    st.pyplot(pca_out["figure"])

    st.markdown(
        f"""
        **Variância explicada:**
        - PC1: {pca_out['explained_variance'][0]*100:.2f} %
        - PC2: {pca_out['explained_variance'][1]*100:.2f} %
        """
    )

    # =====================================================
    # 4️⃣ PCA MULTI-AMOSTRA TEMPORAL
    # =====================================================
    st.divider()
    st.subheader("⏱ PCA Temporal — Evolução das Amostras")

    df_fp_time = df_fp.copy()
    df_fp_time["created_at"] = pd.to_datetime(df_fp_time["created_at"])
    df_fp_time["sample_code"] = "Raman"

    temp_out = temporal_pca(
        df=df_fp_time,
        feature_cols=feature_cols,
        sample_col="sample_code",
        time_col="created_at",
        n_components=2
    )

    st.dataframe(temp_out["df_pca"].head(20), use_container_width=True)

    # =====================================================
    # 5️⃣ RANDOM FOREST + VALIDAÇÃO CRUZADA
    # =====================================================
    st.divider()
    st.subheader("Random Forest com Validação Cruzada")

    target = st.selectbox(
        "Selecione a variável alvo (fingerprint)",
        options=feature_cols
    )

    X = df_fp[feature_cols].drop(columns=[target])
    y = df_fp[target]

    rf_out = random_forest_cv(
        X=X,
        y=y,
        task="regression",
        cv=5
    )

    col1, col2 = st.columns(2)
    col1.metric("R² médio (CV)", f"{rf_out['cv_mean']:.3f}")
    col2.metric("Desvio padrão", f"{rf_out['cv_std']:.3f}")

    st.subheader("Importância das variáveis")
    st.dataframe(
        rf_out["feature_importance"].head(10),
        use_container_width=True
    )

    st.success("Pipeline PCA + Random Forest executado com sucesso.")
