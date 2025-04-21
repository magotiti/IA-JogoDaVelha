import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# carregar os datasets separados
train_set = pd.read_csv('tic_tac_toe_train.csv')
val_set = pd.read_csv('tic_tac_toe_val.csv')
test_set = pd.read_csv('tic_tac_toe_test.csv')

# separar features e target
X_train = train_set.drop('class', axis=1)
y_train = train_set['class']

X_val = val_set.drop('class', axis=1)
y_val = val_set['class']

X_test = test_set.drop('class', axis=1)
y_test = test_set['class']

# criar o modelo k-NN
knn = KNeighborsClassifier(n_neighbors=11)

# treinar o modelo
knn.fit(X_train, y_train)

# prever no conjunto de validação
y_val_pred = knn.predict(X_val)

# prever no conjunto de teste
y_test_pred = knn.predict(X_test)

# avaliar no conjunto de validação
print("\nResultados no conjunto de Validação:")
print(f"Acurácia: {accuracy_score(y_val, y_val_pred):.4f}")
print(f"Precisão: {precision_score(y_val, y_val_pred):.4f}")
print(f"Recall: {recall_score(y_val, y_val_pred):.4f}")
print(f"F1-Score: {f1_score(y_val, y_val_pred):.4f}")

# avaliar no conjunto de teste
print("\nResultados no conjunto de Teste:")
print(f"Acurácia: {accuracy_score(y_test, y_test_pred):.4f}")
print(f"Precisão: {precision_score(y_test, y_test_pred):.4f}")
print(f"Recall: {recall_score(y_test, y_test_pred):.4f}")
print(f"F1-Score: {f1_score(y_test, y_test_pred):.4f}")
