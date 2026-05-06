# =========================================================
# Raman Database — SurfaceXLab
# Biblioteca Molecular Raman
# =========================================================

RAMAN_DATABASE = {

    # =====================================================
    # CARBONO / NANOTUBOS
    # =====================================================
    "carbon_materials": {

        "D Band": {
            "range": (1280, 1380),
            "description": "Defeitos estruturais em carbono sp2"
        },

        "G Band": {
            "range": (1560, 1610),
            "description": "Vibração grafítica sp2"
        },

        "2D Band": {
            "range": (2600, 2750),
            "description": "Empilhamento grafítico"
        }
    },

    # =====================================================
    # SANGUE / BIOFLUIDOS
    # =====================================================
    "blood_biomolecules": {

        "Hemoglobin": {
            "range": (750, 770),
            "description": "Grupo heme"
        },

        "Phenylalanine": {
            "range": (995, 1005),
            "description": "Aminoácido aromático"
        },

        "Proteins": {
            "range": (930, 980),
            "description": "Estruturas proteicas"
        },

        "Amide III": {
            "range": (1230, 1300),
            "description": "Estrutura secundária"
        },

        "CH2/CH3": {
            "range": (1430, 1470),
            "description": "Lipídios e proteínas"
        },

        "Amide I": {
            "range": (1600, 1700),
            "description": "Estrutura proteica"
        }
    },

    # =====================================================
    # METAIS / FC200
    # =====================================================
    "metal_interfaces": {

        "Iron Oxide": {
            "range": (650, 720),
            "description": "Óxidos de ferro"
        },

        "Hematite": {
            "range": (220, 300),
            "description": "Fe2O3"
        },

        "Magnetite": {
            "range": (640, 690),
            "description": "Fe3O4"
        },

        "Graphitic Carbon": {
            "range": (1550, 1610),
            "description": "Carbono grafítico"
        }
    }
}


# =========================================================
# CLASSIFICAÇÃO
# =========================================================
def classify_peak(position):

    matches = []

    for category, groups in RAMAN_DATABASE.items():

        for label, info in groups.items():

            low, high = info["range"]

            if low <= position <= high:

                matches.append({

                    "group": label,

                    "category": category,

                    "description": info["description"]
                })

    return matches
