# raman_features.py
# -*- coding: utf-8 -*-

"""
SurfaceXLab — Raman Feature Engineering & Chemometrics

Gera:
- Mapeamento molecular automático
- Áreas integradas por grupo molecular
- Razões espectrais relevantes
- Fingerprint vetorial ML-ready
- PCA quimiométrico (exploratório)
- Regras exploratórias científicas (não diagnósticas)

© 2025 Marcela Veiga
"""

from dataclasses import dataclass
from typing import List, Dict, Optional, Any

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


# =========================================================
# MAPA MOLECULAR (BIO / GERAL)
# =========================================================
MOLECULAR_MAP = [
    # ==============================
    # ÁCIDOS NUCLEICOS / BIO
    # ==============================
    {"range": (720, 735), "group": "Adenina / nucleotídeos (DNA/RNA)"},
    {"range": (780, 790), "group": "DNA/RNA – ligações fosfato"},

    # ==============================
    # HEME / PORFIRINAS
    # ==============================
    {"range": (730, 750), "group": "Hemoglobina / porfirinas"},
    {"range": (748, 760), "group": "Citocromo c / heme"},

    # ==============================
    # PROTEÍNAS / AMINOÁCIDOS
    # ==============================
    {"range": (935, 955), "group": "Proteínas – esqueleto α-hélice"},
    {"range": (1000, 1008), "group": "Fenilalanina"},
    {"range": (1240, 1280), "group": "Amida III (proteínas)"},
    {"range": (1535, 1560), "group": "Amida II (proteínas)"},
    {"range": (1650, 1680), "group": "Amida I / C=C (proteínas e NR)"},

    # ==============================
    # LIPÍDIOS / FOSFOLIPÍDIOS
    # ==============================
    {"range": (1120, 1135), "group": "Lipídios – C–C estiramento"},
    {"range": (1295, 1315), "group": "Lipídios – CH2 torção"},
    {"range": (1440, 1475), "group": "Lipídios – CH2 deformação"},
    {"range": (2850, 2885), "group": "Lipídios – CH2 simétrico"},
    {"range": (2920, 2960), "group": "Lipídios / proteínas – CH3"},

    # ==============================
    # BORRACHA NATURAL (NR)
    # ==============================
    {"range": (1660, 1685), "group": "NR – C=C cis-1,4-poliisopreno"},
    {"range": (2820, 3030), "group": "NR – C–H stretching (CH2/CH3)"},

    # ==============================
    # FOSFATOS / BIOATIVIDADE (CaP / SBF)
    # ==============================
    {"range": (940, 960), "group": "Fosfato PO4³⁻ ν1 (CaP amorfo)"},
    {"range": (980, 1000), "group": "P–O stretching (CaP / DCPD)"},
    {"range": (1000, 1070), "group": "Fosfatos secundários / Mg-fosfatos"},
]



# =========================================================
# REGRAS EXPLORATÓRIAS (NÃO DIAGNÓSTICAS)
# =========================================================
DISEASE_RULES = [
    {
        "name": "Alteração hemoglobina",
        "description": "Possíveis alterações estruturais no grupo heme/porfirinas.",
        "groups_required": [
            "Hemoglobina / porfirinas",
            "Citocromo c / heme",
        ],
    },
    {
        "name": "Alteração proteica",
        "description": "Alterações conformacionais em proteínas.",
        "groups_required": [
            "Amida I (proteínas, C=O)",
            "Amida II",
            "Amida III (proteínas)",
        ],
    },
    {
        "name": "Alteração lipídica de membrana",
        "description": "Modificações estruturais em lipídios de membrana.",
        "groups_required": [
            "Lipídios – CH2 deformação",
            "Lipídios – CH2 torção",
            "Lipídios – C–C estiramento",
        ],
    },
]


# =========================================================
# DATACLASS DE PICO
# =========================================================
@dataclass
class Peak:
    position_cm1: float
    intensity: float
    width: Optional[float] = None
    molecular_group: Optional[str] = None
    fit_params: Optional[Dict[str, Any]] = None


# =========================================================
# MAPEAMENTO MOLECULAR
# =========================================================
def assign_molecular_group(peak_cm1: float) -> Optional[str]:
    for item in MOLECULAR_MAP:
        lo, hi = item["range"]
        if lo <= peak_cm1 <= hi:
            return item["group"]
    return None


def map_peaks_to_groups(peaks_df: pd.DataFrame) -> pd.DataFrame:
    df = peaks_df.copy()
    df["molecular_group"] = df["peak_cm1"].apply(assign_molecular_group)
    return df


# =========================================================
# ÁREAS INTEGRADAS POR GRUPO
# =========================================================
def compute_group_areas(
    spectrum_df: pd.DataFrame,
    peaks_df: pd.DataFrame,
    window_cm1: float = 10.0,
) -> Dict[str, float]:
    """
    Integra área espectral normalizada em torno dos picos
    associados a cada grupo molecular.
    """
    areas: Dict[str, float] = {}

    x = spectrum_df["shift"].values
    y = spectrum_df["intensity_norm"].values

    for _, row in peaks_df.iterrows():
        group = row.get("molecular_group")
        if not group:
            continue

        cen = row["peak_cm1"]
        mask = (x >= cen - window_cm1 / 2) & (x <= cen + window_cm1 / 2)

        if mask.sum() < 3:
            continue

        area = np.trapz(y[mask], x[mask])
        areas[group] = areas.get(group, 0.0) + float(area)

    return areas


# =========================================================
# RAZÕES ESPECTRAIS
# =========================================================
def compute_peak_ratios(peaks_df: pd.DataFrame) -> Dict[str, float]:
    ratios: Dict[str, float] = {}

    def _mean(group_name: str) -> float:
        vals = peaks_df.loc[
            peaks_df["molecular_group"] == group_name, "intensity_norm"
        ]
        return float(vals.mean()) if not vals.empty else np.nan

    I_phenyl = _mean("Fenilalanina")
    I_amide_I = _mean("Amida I (proteínas, C=O)")
    I_amide_III = _mean("Amida III (proteínas)")
    I_lipid = _mean("Lipídios – CH2 deformação")
    I_heme = _mean("Hemoglobina / porfirinas")

    if np.isfinite(I_phenyl) and np.isfinite(I_amide_I):
        ratios["phenylalanine_amideI"] = I_phenyl / I_amide_I

    if np.isfinite(I_amide_I) and np.isfinite(I_amide_III):
        ratios["amideI_amideIII"] = I_amide_I / I_amide_III

    if np.isfinite(I_lipid) and np.isfinite(I_amide_I):
        ratios["lipid_protein"] = I_lipid / I_amide_I

    if np.isfinite(I_heme) and np.isfinite(I_phenyl):
        ratios["heme_phenylalanine"] = I_heme / I_phenyl

    return ratios


# =========================================================
# FINGERPRINT ML-READY
# =========================================================
def build_fingerprint(
    group_areas: Dict[str, float],
    peak_ratios: Dict[str, float],
) -> Dict[str, float]:
    features: Dict[str, float] = {}

    for g in sorted(group_areas.keys()):
        key = f"area_{g.lower().replace(' ', '_').replace('/', '_').replace('–', '_')}"
        features[key] = group_areas[g]

    for r, v in peak_ratios.items():
        features[f"ratio_{r}"] = v

    return features


# =========================================================
# REGRAS EXPLORATÓRIAS
# =========================================================
def apply_exploratory_rules(peaks_df: pd.DataFrame) -> List[Dict[str, str]]:
    triggered = []

    present_groups = set(
        peaks_df["molecular_group"].dropna().unique()
    )

    for rule in DISEASE_RULES:
        if all(g in present_groups for g in rule["groups_required"]):
            triggered.append({
                "rule": rule["name"],
                "description": rule["description"],
            })

    return triggered


# =========================================================
# PCA QUIMIOMÉTRICO (EXPLORATÓRIO)
# =========================================================
def run_pca_on_fingerprints(
    fingerprints: List[Dict[str, float]],
    n_components: int = 3,
) -> Dict[str, Any]:
    """
    Executa PCA sobre fingerprints Raman consolidados.
    """
    df = pd.DataFrame(fingerprints).fillna(0.0)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df.values)

    pca = PCA(n_components=min(n_components, X_scaled.shape[1]))
    scores = pca.fit_transform(X_scaled)

    loadings = pd.DataFrame(
        pca.components_.T,
        index=df.columns,
        columns=[f"PC{i+1}" for i in range(scores.shape[1])]
    )

    return {
        "scores": scores,
        "loadings": loadings,
        "explained_variance": pca.explained_variance_ratio_,
        "features": df.columns.tolist(),
    }


# =========================================================
# PIPELINE COMPLETO DE FEATURES
# =========================================================
def extract_raman_features(
    spectrum_df: pd.DataFrame,
    peaks_df: pd.DataFrame,
) -> Dict[str, Any]:
    """
    Pipeline principal consumido pelo Streamlit / Supabase / ML.
    """
    peaks_mapped = map_peaks_to_groups(peaks_df)

    group_areas = compute_group_areas(spectrum_df, peaks_mapped)
    peak_ratios = compute_peak_ratios(peaks_mapped)
    fingerprint = build_fingerprint(group_areas, peak_ratios)
    rules = apply_exploratory_rules(peaks_mapped)

    return {
        "group_areas": group_areas,
        "peak_ratios": peak_ratios,
        "fingerprint": fingerprint,
        "exploratory_rules": rules,
        "peaks_annotated": peaks_mapped,
    }
