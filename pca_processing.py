# =========================================================
# PCA MULTIMODAL — REPRODUÇÃO PAPER
# Surface and Interfaces (2022)
# CORRIGIDO DEFINITIVAMENTE PARA TRIPLICATAS
# =========================================================

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from google.colab import files


# =========================================================
# UPLOAD
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
# NORMALIZAÇÃO DOS NOMES DAS COLUNAS
# EVITA ERRO ST.1 / T1.1 / DUPLICATAS
# =========================================================

new_cols = []

counter = {}

for col in df_raw.columns:

    col = str(col).strip()

    col = col.replace(' ', '')

    if col in counter:

        counter[col] += 1

        col = f"{col}_{counter[col]}"

    else:

        counter[col] = 0

    new_cols.append(col)

df_raw.columns = new_cols


# =========================================================
# VISUALIZAÇÃO INICIAL
# =========================================================

print('\nCOLUNAS DETECTADAS:\n')

print(df_raw.columns.tolist())

print('\nHEAD:\n')

print(df_raw.head())


# =========================================================
# PRIMEIRA COLUNA = VARIÁVEL
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
    .str.contains(
        'Temp',
        case=False,
        na=False
    )
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
# IDENTIFICA AUTOMATICAMENTE AS COLUNAS
# =========================================================

sample_columns = []

for col in df_raw.columns:

    if col == 'Variavel':

        continue

    sample_columns.append(col)


print('\nCOLUNAS USADAS:\n')

print(sample_columns)


# =========================================================
# LABELS AUTOMÁTICOS
# =========================================================

labels = []

for col in sample_columns:

    base = col.split('_')[0]

    labels.append(base)


print('\nLABELS:\n')

print(labels)


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

            row.get(col, np.nan)

        )

        values.append(val)

    matrix.append(values)


# =========================================================
# MATRIZ FINAL
# =========================================================

X = np.array(matrix).T


# =========================================================
# REMOVE NAN
# =========================================================

X = np.nan_to_num(X)


# =========================================================
# VALIDAÇÃO
# =========================================================

print('\nFORMATO MATRIZ:\n')

print(X.shape)


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
# ESCALA DOS LOADINGS
# =========================================================

scale = 2.5


# =========================================================
# FIGURA
# =========================================================

fig, ax = plt.subplots(

    figsize=(8,5),

    dpi=600
)

fig.patch.set_facecolor('white')

ax.set_facecolor('white')


# =========================================================
# SCORES
# =========================================================

for i in range(len(scores)):

    ax.scatter(

        scores[i,0],

        scores[i,1],

        color='black',

        s=40,

        zorder=3
    )

    ax.text(

        scores[i,0] + 0.05,

        scores[i,1] + 0.03,

        labels[i],

        fontsize=6,

        color='blue',

        fontweight='bold'
    )


# =========================================================
# LOADINGS
# =========================================================

for i, var in enumerate(variables):

    x = loadings[i,0] * scale

    y = loadings[i,1] * scale

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

        fontsize=6,

        fontweight='bold'
    )


# =========================================================
# EIXOS
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
# LABELS
# =========================================================

ax.set_xlabel(

    f'PC1 ({explained[0]:.1f}%)',

    fontsize=6
)

ax.set_ylabel(

    f'PC2 ({explained[1]:.1f}%)',

    fontsize=6
)


# =========================================================
# ESTILO
# =========================================================

ax.spines['top'].set_visible(False)

ax.spines['right'].set_visible(False)

ax.tick_params(

    axis='both',

    labelsize=6
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
# SALVAR
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
# SCORES
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
# LOADINGS
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
