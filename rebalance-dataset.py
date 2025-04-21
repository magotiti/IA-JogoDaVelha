# REBALANCE DO DATASET 
# 400 amostragens de cada classe, embaralhamento dos dados, conversao para valores numericos

import pandas as pd

# carregar o dataset original
colunas = [
    'top-left', 'top-middle', 'top-right',
    'middle-left', 'middle-middle', 'middle-right',
    'bottom-left', 'bottom-middle', 'bottom-right',
    'class'
]

df = pd.read_csv('tic-tac-toe.data', names=colunas)

# verificar quantidade de exemplos por classe
print("Contagem de exemplos por classe:")
print(df['class'].value_counts())
print("\n")

# balancear o dataset: selecionar 400 exemplos de cada classe (jogo terminou/nao terminou)
positive_samples = df[df['class'] == 'positive'].sample(332, random_state=42)
negative_samples = df[df['class'] == 'negative'].sample(332, random_state=42)


df_balanced = pd.concat([positive_samples, negative_samples])

# embaralhar o dataset
df_balanced = df_balanced.sample(frac=1, random_state=42).reset_index(drop=True)

# transformar valores de 'x', 'o' e 'b' para números
transformacao = {'x': 1, 'o': -1, 'b': 0}
df_balanced.replace(transformacao, inplace=True)

# transformar 'class' para numerico
# 1 = jogo terminado, 0 = jogo nao terminado)
df_balanced['class'] = df_balanced['class'].map({'positive': 1, 'negative': 0})

# conferir o resultado final
print("Primeiras linhas do dataset preparado:")
print(df_balanced.head())
print("\n")
print("Contagem de classes após balanceamento e transformação:")
print(df_balanced['class'].value_counts())

# salvar o dataset pronto para modelagem
df_balanced.to_csv('tic_tac_toe_balanced.csv', index=False)
print("\nDataset salvo como 'tic_tac_toe_balanced.csv'!")
