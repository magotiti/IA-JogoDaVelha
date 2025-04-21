import pandas as pd
from sklearn.model_selection import train_test_split

# carregar o dataset limpo
df = pd.read_csv('tic_tac_toe_balanced.csv')

# separar as variáveis independentes (features) e a variável dependente (target)
X = df.drop('class', axis=1)  # Features
y = df['class']  # Target

# dividir o dataset em treino (60%), validação (20%) e teste (20%), mantendo a estratificação
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, stratify=y, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42)

# juntando features e target para cada conjunto
train_set = pd.concat([X_train, y_train], axis=1)
val_set = pd.concat([X_val, y_val], axis=1)
test_set = pd.concat([X_test, y_test], axis=1)

# salvando em arquivos CSV
train_set.to_csv('tic_tac_toe_train.csv', index=False)
val_set.to_csv('tic_tac_toe_val.csv', index=False)
test_set.to_csv('tic_tac_toe_test.csv', index=False)

# verificando a divisão
print(f"Tamanho do conjunto de treino: {len(X_train)}")
print(f"Tamanho do conjunto de validação: {len(X_val)}")
print(f"Tamanho do conjunto de teste: {len(X_test)}")

# exemplo de como verificar as proporções de classes nos conjuntos
print("\nDistribuição das classes no conjunto de treino:")
print(y_train.value_counts(normalize=True))

print("\nDistribuição das classes no conjunto de validação:")
print(y_val.value_counts(normalize=True))

print("\nDistribuição das classes no conjunto de teste:")
print(y_test.value_counts(normalize=True))

print("\nArquivos 'tic_tac_toe_train.csv', 'tic_tac_toe_val.csv' e 'tic_tac_toe_test.csv' foram salvos com sucesso!")
