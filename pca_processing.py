# =========================================================
# PCA MULTIMODAL — REPRODUÇÃO PAPER
# Surface and Interfaces (2022)
# CORRIGIDO PARA TRIPLICATAS REAIS
# =========================================================

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from google.colab import files


# =========================================================
# UPLOAD DO ARQUIVO
# =========================================================

uploaded = files.upload()

filename = list(uploaded.keys())[0]


# =========================================================
# LEITURA AUTOMÁTICA
# =========================================================

if filename.endswith('.xlsx'):

    df_raw = pd.read_excel(filename)

elif filename.endswith('.xls'):

    df_raw = pd.read_excel(filename)

elif filename.endswith('.ods'):

    df_raw = pd.read_excel(
        filename,
        engine='odf'
    )

else:

    raise ValueError(
        'Formato não suportado'
    )


# =========================================================
# VISUALIZAÇÃO INICIAL
# =========================================================

print(df_raw.head())


# =========================================================
# RENOMEIA PRIMEIRA COLUNA
# =========================================================

col0 = df_raw.columns[0]

df_raw = df_raw.rename(
    columns={col0: 'Variavel'}
)


# =========================================================
# REMOVE LINHAS INVÁLIDAS
# =========================================================

df_raw = df_raw[
    ~df_raw['Variavel']
    .astype(str)
    .str.contains('Temp', case=False)
]


# =========================================================
# FUNÇÃO LIMPEZA
# =========================================================

def extract_mean(value):

    if pd.isna(value):

        return np.nan

    value = str(value)

    value = value.replace(',', '.')

    value = value.replace('−', '-')

    value = value.replace('±', '+-')

    value = value.replace(' ', '')

    try:

        return float(
            value.split('+-')[0]
        )

    except:

        return np.nan


# =========================================================
# IDENTIFICA COLUNAS REAIS
# =========================================================

print('\nCOLUNAS ENCONTRADAS:\n')

print(df_raw.columns.tolist())


# =========================================================
# AJUSTE DAS COLUNAS
# =========================================================
# ALTERE AQUI CASO O NOME
# ESTEJA DIFERENTE NO ODS
# =========================================================

sample_columns = [

    'ST',
    'ST.1',
    'ST.2',

    'T1',
    'T1.1',
    'T1.2',

    'T2',
    'T2.1',
    'T2.2',

    'T3',
    'T3.1',
    'T3.2'
]


# =========================================================
# LABELS
# =========================================================

labels = [

    'ST',
    'ST',
    'ST',

    'T1',
    'T1',
    'T1',

    'T2',
    'T2',
    'T2',

    'T3',
    'T3',
    'T3'
]


# =========================================================
# EXTRAÇÃO DOS DADOS
# =========================================================

variables = []

matrix = []


for _, row in df_raw.iterrows():

    var_name = str(
        row['Variavel']
    )

    variables.append(var_name)

    values = []

    for col in sample_columns:

        val = extract_mean(
            row[col]
        )

        values.append(val)

    matrix.append(values)


# =========================================================
# MATRIZ FINAL
# =========================================================

X = np.array(matrix).T


# =========================================================
# VALIDAÇÃO
# =========================================================

print('\nFORMATO MATRIZ:\n')

print(X.shape)

# esperado:
# (12, número_variáveis)


# =========================================================
# REMOVE NAN
# =========================================================

X = np.nan_to_num(X)


# =========================================================
# NORMALIZAÇÃO
# =========================================================

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)


# =========================================================
# PCA
# =========================================================

pca = PCA(
    n_components=2
)

scores = pca.fit_transform(
    X_scaled
)

loadings = pca.components_.T

explained = (
    pca.explained_variance_ratio_ * 100
)


# =========================================================
# LOADINGS SCALE
# =========================================================

scale = 2.5


# =========================================================
# FIGURA
# =========================================================

fig, ax = plt.subplots(

    figsize=(8, 5),

    dpi=600
)

fig.patch.set_facecolor('white')

ax.set_facecolor('white')


# =========================================================
# SCORES
# =========================================================

for i in range(len(scores)):

    ax.scatter(

        scores[i, 0],

        scores[i, 1],

        color='black',

        s=40,

        zorder=3
    )

    ax.text(

        scores[i, 0] + 0.05,

        scores[i, 1] + 0.03,

        labels[i],

        fontsize=10,

        color='blue',

        fontweight='bold'
    )


# =========================================================
# LOADINGS
# =========================================================

for i, var in enumerate(variables):

    x = loadings[i, 0] * scale

    y = loadings[i, 1] * scale

    ax.arrow(

        0,
        0,

        x,
        y,

        color='forestgreen',

        linewidth=1.6,

        head_width=0.08,

        length_includes_head=True,

        zorder=2
    )

    ax.text(

        x * 1.08,

        y * 1.08,

        var,

        color='red',

        fontsize=11,

        fontweight='bold'
    )


# =========================================================
# EIXOS CENTRAIS
# =========================================================

ax.axhline(

    0,

    color='gray',

    linewidth=1
)

ax.axvline(

    0,

    color='gray',

    linewidth=1
)


# =========================================================
# LABELS DOS EIXOS
# =========================================================

ax.set_xlabel(

    f'Component 1 ({explained[0]:.1f}%)',

    fontsize=12
)

ax.set_ylabel(

    f'Component 2 ({explained[1]:.1f}%)',

    fontsize=12
)


# =========================================================
# ESTILO PAPER
# =========================================================

ax.spines['top'].set_visible(False)

ax.spines['right'].set_visible(False)

ax.tick_params(

    axis='both',

    labelsize=10
)

ax.grid(False)


# =========================================================
# LIMITES
# =========================================================

margin = 0.7

ax.set_xlim(

    scores[:,0].min() - margin,

    scores[:,0].max() + margin
)

ax.set_ylim(

    scores[:,1].min() - margin,

    scores[:,1].max() + margin
)


# =========================================================
# SALVAR FIGURAS
# =========================================================

plt.tight_layout()

plt.savefig(

    'PCA_Surface_Interfaces.tiff',

    dpi=600,

    bbox_inches='tight'
)

plt.savefig(

    'PCA_Surface_Interfaces.png',

    dpi=600,

    bbox_inches='tight'
)


# =========================================================
# MOSTRAR
# =========================================================

plt.show()


# =========================================================
# SCORES DATAFRAME
# =========================================================

scores_df = pd.DataFrame({

    'Amostra': labels,

    'PC1': np.round(
        scores[:,0], 4
    ),

    'PC2': np.round(
        scores[:,1], 4
    )
})

print('\nSCORES:\n')

print(scores_df)


# =========================================================
# LOADINGS DATAFRAME
# =========================================================

loadings_df = pd.DataFrame({

    'Variavel': variables,

    'PC1': np.round(
        loadings[:,0], 4
    ),

    'PC2': np.round(
        loadings[:,1], 4
    )
})

print('\nLOADINGS:\n')

print(loadings_df)


# =========================================================
# DOWNLOAD AUTOMÁTICO
# =========================================================

files.download(
    'PCA_Surface_Interfaces.tiff'
)

files.download(
    'PCA_Surface_Interfaces.png'
)
