import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_california_housing
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split

dados = fetch_california_housing()

df = pd.DataFrame(dados.data, columns=dados.feature_names)

df['MedHouseVal'] = dados.target

print(df.info())
print(df.shape)
print(df.describe())
print(df.head())
print(df.isnull().sum())

print(df.isnull().sum())

df_faltantes = df.copy()

df_faltantes.loc[
    df_faltantes.sample(frac=0.05, random_state=42).index,
    'MedInc'
] = np.nan

df_faltantes.loc[
    df_faltantes.sample(frac=0.05, random_state=42).index,
    'HouseAge'
] = np.nan

print(df_faltantes.isnull().sum())

media_medinc = df_faltantes['MedInc'].mean()

df_faltantes['MedInc'] = df_faltantes['MedInc'].fillna(media_medinc)

mediana_houseage = df_faltantes['HouseAge'].median()

df_faltantes['HouseAge'] = df_faltantes['HouseAge'].fillna(
    mediana_houseage
)

print(df_faltantes.isnull().sum())

x = df_faltantes.drop(columns=['MedHouseVal'])
y = df_faltantes['MedHouseVal']

x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    test_size=0.2,
    random_state=42
)

scaler_mm = MinMaxScaler()

x_train_mm = scaler_mm.fit_transform(x_train)
x_test_mm = scaler_mm.transform(x_test)

scaler_std = StandardScaler()

x_train_std = scaler_std.fit_transform(x_train)
x_test_std = scaler_std.transform(x_test)

print("Min-Max - primeiras 3 linhas:")
print(x_train_mm[:3])

print("\nStandardScaler - primeiras 3 linhas:")
print(x_train_std[:3])

for coluna in df_faltantes.columns:

    plt.figure(figsize=(8, 5))

    sns.histplot(
        df_faltantes[coluna],
        kde=True
    )

    plt.title(f'Distribuição de {coluna}')
    plt.xlabel(coluna)
    plt.ylabel('Frequência')

    plt.savefig(
        f'hist_{coluna}.png',
        dpi=150
    )

    plt.show()

colunas_boxplot = [
    'MedInc',
    'HouseAge',
    'AveRooms',
    'MedHouseVal'
]

for coluna in colunas_boxplot:

    plt.figure(figsize=(8, 5))

    sns.boxplot(
        x=df_faltantes[coluna]
    )

    plt.title(f'Boxplot de {coluna}')
    plt.xlabel(coluna)

    plt.savefig(
        f'boxplot_{coluna}.png',
        dpi=150
    )

    plt.show()

correlacao = df_faltantes.corr()

print("Matriz de correlação:")
print(correlacao)

plt.figure(figsize=(10, 8))

sns.heatmap(
    correlacao,
    annot=True,
    cmap='coolwarm',
    fmt='.2f'
)

plt.title('Mapa de Correlação - California Housing')
plt.show()

correlacao_alvo = (
    correlacao['MedHouseVal']
    .drop('MedHouseVal')
    .sort_values(ascending=False)
)

print("\nCorrelação das features com MedHouseVal:")
print(correlacao_alvo)

df_faltantes['FaixaRenda'] = pd.cut(
    df_faltantes['MedInc'],
    bins=3,
    labels=['Baixa', 'Média', 'Alta']
)

plt.figure(figsize=(8, 5))

sns.scatterplot(
    data=df_faltantes,
    x='MedInc',
    y='MedHouseVal',
    hue='FaixaRenda'
)

plt.title('Renda Média x Valor Médio dos Imóveis')
plt.xlabel('Renda Média (MedInc)')
plt.ylabel('Valor Médio do Imóvel (MedHouseVal)')
plt.show()

colunas_pairplot = [
    'MedInc',
    'HouseAge',
    'AveRooms',
    'AveOccup',
    'MedHouseVal'
]

sns.pairplot(
    df_faltantes[colunas_pairplot],
    diag_kind='hist'
)

plt.show()