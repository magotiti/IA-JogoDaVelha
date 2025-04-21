# ANALISE E LIMPEZA DO DATASET
import pandas as pd

# definir nomes das colunas
colunas = [
    'top-left', 'top-middle', 'top-right',
    'middle-left', 'middle-middle', 'middle-right',
    'bottom-left', 'bottom-middle', 'bottom-right',
    'class'
]

# carregar o dataset
# --> certifique-se que o arquivo esta na mesma pasta do script
df = pd.read_csv('tic-tac-toe.data', names=colunas)

# exibir as primeiras linhas para inspecionar
print("Primeiras linhas do dataset:")
print(df.head())
print("\n")

# verificar valores únicos em cada coluna
print("Valores únicos por coluna:")
for col in df.columns:
    print(f"{col}: {df[col].unique()}")
print("\n")

# verificar se ha valores nulos
print("Verificando valores nulos:")
print(df.isnull().sum())
print("\n")

# verificar se todas as celulas possuem apenas 'x', 'o' ou 'b'
print("Validando se todas as células do tabuleiro têm valores válidos ('x', 'o', 'b'):")
valid_states = ['x', 'o', 'b']
tabuleiro_valido = df.drop(columns=['class']).apply(lambda col: col.isin(valid_states)).all()
print(tabuleiro_valido)
print("\n")

# verificar se ha duplicatas
print("Quantidade de linhas duplicadas:")
print(df.duplicated().sum())
print("\n")

# remover duplicatas
df = df.drop_duplicates()

# resultado final
print(f"Dataset final possui {df.shape[0]} linhas e {df.shape[1]} colunas.")
