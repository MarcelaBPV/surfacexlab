# =========================================================
# pca_database.py
# SurfaceXLab — Banco Multimodal
# =========================================================


# =========================================================
# INTERPRETAÇÕES FÍSICO-QUÍMICAS
# =========================================================
PCA_INTERPRETATIONS = {

    "ID_IG": {

        "label": "Desordem Estrutural",

        "description": (
            "Associado à presença de defeitos "
            "estruturais, reorganização superficial "
            "e aumento da desordem em domínios grafíticos."
        ),

        "correlation": (
            "Valores elevados tendem a indicar "
            "maior densidade de defeitos estruturais."
        )
    },

    "I2D_IG": {

        "label": "Organização Grafítica",

        "description": (
            "Associado ao grau de organização "
            "estrutural e empilhamento grafítico."
        ),

        "correlation": (
            "Valores elevados indicam maior "
            "ordenação estrutural."
        )
    },

    "Resistivity": {

        "label": "Barreiras Eletrônicas",

        "description": (
            "Associado à mobilidade eletrônica "
            "e aos estados interfaciais de transporte."
        ),

        "correlation": (
            "Altos valores indicam aumento das "
            "barreiras de condução elétrica."
        )
    },

    "Conductivity": {

        "label": "Transporte Eletrônico",

        "description": (
            "Associado à conectividade elétrica "
            "e eficiência do transporte de carga."
        ),

        "correlation": (
            "Altos valores indicam maior "
            "mobilidade eletrônica."
        )
    },

    "Angle": {

        "label": "Molhabilidade",

        "description": (
            "Associado à energia superficial "
            "e interação líquido-superfície."
        ),

        "correlation": (
            "Ângulos elevados indicam "
            "comportamento hidrofóbico."
        )
    },

    "Surface_Energy": {

        "label": "Energia Superficial",

        "description": (
            "Associado à afinidade interfacial "
            "e à energia livre da superfície."
        ),

        "correlation": (
            "Valores elevados indicam maior "
            "interação superficial."
        )
    },

    "Ra": {

        "label": "Rugosidade Média",

        "description": (
            "Associado à amplitude média "
            "das irregularidades topográficas."
        ),

        "correlation": (
            "Valores elevados indicam maior "
            "rugosidade global."
        )
    },

    "Rq": {

        "label": "Heterogeneidade Topográfica",

        "description": (
            "Associado às variações locais "
            "e dispersão topográfica."
        ),

        "correlation": (
            "Valores elevados indicam maior "
            "heterogeneidade superficial."
        )
    },

    "Rz": {

        "label": "Amplitude Topográfica",

        "description": (
            "Associado à distância entre "
            "picos e vales topográficos."
        ),

        "correlation": (
            "Altos valores indicam superfícies "
            "com maior amplitude topográfica."
        )
    }
}


# =========================================================
# INTERPRETAÇÃO DOS LOADINGS
# =========================================================
def interpret_loading(variable):

    """
    Retorna interpretação físico-química
    da variável utilizada no PCA.
    """

    if variable in PCA_INTERPRETATIONS:

        return PCA_INTERPRETATIONS[variable]

    return {

        "label": "Parâmetro Experimental",

        "description": (
            "Variável experimental integrada "
            "ao espaço multivariado."
        ),

        "correlation": (
            "Sem interpretação específica cadastrada."
        )
    }


# =========================================================
# INTERPRETAÇÃO GLOBAL PCA
# =========================================================
def interpret_pca_behavior(
    pc1_dominant,
    pc2_dominant
):

    interpretation = []

    # =====================================================
    # PC1
    # =====================================================
    if pc1_dominant in PCA_INTERPRETATIONS:

        interpretation.append(

            f"""
            A PC1 é predominantemente influenciada por
            {PCA_INTERPRETATIONS[pc1_dominant]['label']},
            indicando que a principal variabilidade do
            sistema está associada a alterações
            relacionadas a este parâmetro.
            """
        )

    # =====================================================
    # PC2
    # =====================================================
    if pc2_dominant in PCA_INTERPRETATIONS:

        interpretation.append(

            f"""
            A PC2 apresenta maior contribuição associada a
            {PCA_INTERPRETATIONS[pc2_dominant]['label']},
            refletindo mecanismos secundários de
            variabilidade físico-química.
            """
        )

    return "\n".join(interpretation)


# =========================================================
# CLASSIFICAÇÃO MULTIMODAL
# =========================================================
def classify_multimodal_surface(

    resistivity,
    angle,
    id_ig,
    rq

):

    classification = []

    # -----------------------------------------------------
    # ELÉTRICO
    # -----------------------------------------------------
    if resistivity > 1e-2:

        classification.append(
            "Superfície passivada"
        )

    else:

        classification.append(
            "Superfície condutiva"
        )

    # -----------------------------------------------------
    # MOLHABILIDADE
    # -----------------------------------------------------
    if angle > 90:

        classification.append(
            "Hidrofóbica"
        )

    else:

        classification.append(
            "Hidrofílica"
        )

    # -----------------------------------------------------
    # ESTRUTURA
    # -----------------------------------------------------
    if id_ig > 1:

        classification.append(
            "Alta desordem estrutural"
        )

    else:

        classification.append(
            "Estrutura organizada"
        )

    # -----------------------------------------------------
    # TOPOGRAFIA
    # -----------------------------------------------------
    if rq > 2:

        classification.append(
            "Alta heterogeneidade topográfica"
        )

    else:

        classification.append(
            "Topografia homogênea"
        )

    return classification
