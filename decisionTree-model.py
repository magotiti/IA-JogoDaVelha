from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler
import pandas as pd

# carregar os datasets separados
df_train = pd.read_csv('tic_tac_toe_train.csv')
df_val = pd.read_csv('tic_tac_toe_val.csv')
df_test = pd.read_csv('tic_tac_toe_test.csv')

# separar as variáveis independentes e a variavel dependente para cada conjunto de dados
X_train = df_train.drop('class', axis=1)  # features
y_train = df_train['class']  # target

X_val = df_val.drop('class', axis=1)  # features
y_val = df_val['class']  # target

X_test = df_test.drop('class', axis=1)  # features
y_test = df_test['class']  # target

# normalizar os dados
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# validar que os dados estao sendo carregados corretamente
print(f"Forma dos conjuntos de dados:")
print(f"Treinamento: X_train: {X_train_scaled.shape}, y_train: {y_train.shape}")
print(f"Validação: X_val: {X_val_scaled.shape}, y_val: {y_val.shape}")
print(f"Teste: X_test: {X_test_scaled.shape}, y_test: {y_test.shape}")

# criar o modelo de arvore de decisao
dt = DecisionTreeClassifier(random_state=42)

# treinar o modelo
dt.fit(X_train_scaled, y_train)

# realizar previsoes
y_val_pred = dt.predict(X_val_scaled)
y_test_pred = dt.predict(X_test_scaled)

# avaliar o modelo no conjunto de validacao
print("\nResultados no conjunto de Validação:")
print(f"Acurácia: {accuracy_score(y_val, y_val_pred):.4f}")
print(f"Precisão: {precision_score(y_val, y_val_pred):.4f}")
print(f"Recall: {recall_score(y_val, y_val_pred):.4f}")
print(f"F1-Score: {f1_score(y_val, y_val_pred):.4f}")

# avaliar o modelo no conjunto de teste
print("\nResultados no conjunto de Teste:")
print(f"Acurácia: {accuracy_score(y_test, y_test_pred):.4f}")
print(f"Precisão: {precision_score(y_test, y_test_pred):.4f}")
print(f"Recall: {recall_score(y_test, y_test_pred):.4f}")
print(f"F1-Score: {f1_score(y_test, y_test_pred):.4f}")
