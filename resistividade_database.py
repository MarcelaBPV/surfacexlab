# =========================================================
# resistividade_database.py
# SurfaceXLab — Banco físico elétrico
# =========================================================

import numpy as np


# =========================================================
# CLASSES ELÉTRICAS
# =========================================================
ELECTRICAL_CLASSES = {

    "high_conductor": {
        "label": "Condutor Elevado",
        "rho_max": 1e-5,
        "description": (
            "Alta mobilidade eletrônica associada "
            "à elevada conectividade superficial."
        )
    },

    "moderate_conductor": {
        "label": "Condutor Moderado",
        "rho_max": 1e-3,
        "description": (
            "Regime condutivo intermediário com "
            "possível influência interfacial."
        )
    },

    "semiconductor": {
        "label": "Semicondutor",
        "rho_max": 1,
        "description": (
            "Condução parcialmente limitada "
            "por barreiras eletrônicas superficiais."
        )
    },

    "insulator": {
        "label": "Isolante",
        "rho_max": np.inf,
        "description": (
            "Elevada resistividade associada "
            "à passivação superficial."
        )
    }
}


# =========================================================
# CLASSIFICAÇÃO FÍSICA
# =========================================================
def classify_resistivity(
    resistivity,
    r_squared,
    slope
):

    # -----------------------------------------------------
    # CLASSE ELÉTRICA
    # -----------------------------------------------------
    electrical_class = "Indefinido"
    description = ""

    for key, item in ELECTRICAL_CLASSES.items():

        if resistivity <= item["rho_max"]:

            electrical_class = item["label"]
            description = item["description"]
            break

    # -----------------------------------------------------
    # REGIME INTERFACIAL
    # -----------------------------------------------------
    if resistivity < 1e-4:

        regime = "Ativação superficial"

    elif resistivity > 1e-1:

        regime = "Passivação superficial"

    else:

        regime = "Regime intermediário"

    # -----------------------------------------------------
    # QUALIDADE DO AJUSTE
    # -----------------------------------------------------
    if r_squared >= 0.99:

        fit_quality = "Excelente"

    elif r_squared >= 0.95:

        fit_quality = "Bom"

    else:

        fit_quality = "Baixa linearidade"

    # -----------------------------------------------------
    # RESPOSTA FINAL
    # -----------------------------------------------------
    return {

        "Classe": electrical_class,

        "Descrição": description,

        "Regime": regime,

        "Qualidade Ajuste": fit_quality,

        "Slope": slope,

        "R²": r_squared
    }
# =========================================================
# INTERPRETAÇÃO FÍSICO-QUÍMICA
# =========================================================
def infer_surface_physics(
    resistivity,
    r_squared,
    slope,
    nonlinearity_index,
    dI_dV_std,
    asymmetry_index=1.0
):

    # -----------------------------------------------------
    # HOMOGENEIDADE
    # -----------------------------------------------------
    homogeneity = max(
        0,
        1 - nonlinearity_index
    )

    # -----------------------------------------------------
    # ÓXIDO
    # -----------------------------------------------------
    oxide_index = (
        nonlinearity_index *
        resistivity *
        10
    )

    oxide_index = np.clip(
        oxide_index,
        0,
        1
    )

    # -----------------------------------------------------
    # DENSIDADE DE DEFEITOS
    # -----------------------------------------------------
    defect_density = (
        dI_dV_std *
        nonlinearity_index
    )

    # -----------------------------------------------------
    # REGIME DE CONDUÇÃO
    # -----------------------------------------------------
    if r_squared >= 0.995:

        conduction_regime = (
            "Ôhmico metálico"
        )

    elif nonlinearity_index > 0.25:

        conduction_regime = (
            "Barreira interfacial"
        )

    else:

        conduction_regime = (
            "Semicondutor superficial"
        )

    # -----------------------------------------------------
    # COMPATIBILIDADE COATING
    # -----------------------------------------------------
    coating_score = (
        homogeneity * 0.4 +
        (1 - oxide_index) * 0.3 +
        r_squared * 0.3
    ) * 10

    # -----------------------------------------------------
    # INTERFACE
    # -----------------------------------------------------
    if oxide_index > 0.6:

        interface_state = (
            "Superfície oxidada"
        )

    elif oxide_index > 0.3:

        interface_state = (
            "Oxidação parcial"
        )

    else:

        interface_state = (
            "Superfície ativa"
        )

    return {

        "Oxide_Index":
            oxide_index,

        "Surface_Homogeneity":
            homogeneity,

        "Defect_Density":
            defect_density,

        "Conduction_Regime":
            conduction_regime,

        "Interface_State":
            interface_state,

        "Coating_Compatibility":
            coating_score,

        "Asymmetry_Index":
            asymmetry_index
    }
