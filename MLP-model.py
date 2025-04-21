from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler
import pandas as pd

# carregar o dataset
df = pd.read_csv('tic_tac_toe_balanced.csv')

# separar as variaveis independentes e a variavel dependente 
X = df.drop('class', axis=1)  # features
y = df['class']  # target

# normalizar os dados 
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# dividir o dataset em treino (60%), validacao (20%) e teste (20%)
from sklearn.model_selection import train_test_split
X_train, X_temp, y_train, y_temp = train_test_split(X_scaled, y, test_size=0.4, stratify=y, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42)

# criar o modelo MLP
mlp = MLPClassifier(hidden_layer_sizes=(100,), learning_rate='adaptive', max_iter=1000, activation='relu', alpha=0.0001, early_stopping=True)

# treinar o modelo
mlp.fit(X_train, y_train)

# realizar previsoes
y_val_pred = mlp.predict(X_val)
y_test_pred = mlp.predict(X_test)

# avaliar o modelo
print("Resultados no conjunto de Validação:")
print(f"Acurácia: {accuracy_score(y_val, y_val_pred):.4f}")
print(f"Precisão: {precision_score(y_val, y_val_pred):.4f}")
print(f"Recall: {recall_score(y_val, y_val_pred):.4f}")
print(f"F1-Score: {f1_score(y_val, y_val_pred):.4f}")

print("\nResultados no conjunto de Teste:")
print(f"Acurácia: {accuracy_score(y_test, y_test_pred):.4f}")
print(f"Precisão: {precision_score(y_test, y_test_pred):.4f}")
print(f"Recall: {recall_score(y_test, y_test_pred):.4f}")
print(f"F1-Score: {f1_score(y_test, y_test_pred):.4f}")
