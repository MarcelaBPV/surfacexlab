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
