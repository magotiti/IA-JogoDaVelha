#!/usr/bin/env python3
"""
Gera tabela única com Accuracy / Precision / Recall / F1 no conjunto de teste
para k‑NN, MLP, Decision Tree e Random Forest.
"""
import pandas as pd, joblib
from pathlib import Path
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

MODELS = {
    'kNN':           'knn.joblib',
    'MLP':           'mlp.joblib',          # salvo como dict {'scaler', 'model'}
    'DecisionTree':  'decision_tree.joblib',
    'RandomForest':  'rf.joblib'
}

TEST = pd.read_csv('tic_tac_toe_test.csv')
X, y = TEST.drop(columns='class'), TEST['class']

rows = []
for name, path in MODELS.items():
    if not Path(path).exists():
        print(f'⚠️  Arquivo {path} não encontrado; pulando {name}')
        continue
    mdl = joblib.load(path)
    if name == 'MLP':               # MLP foi salvo num dicionário
        mdl = mdl['model']
    yp = mdl.predict(X)
    rows.append([
        name,
        accuracy_score(y, yp),
        precision_score(y, yp, average='macro', zero_division=0),
        recall_score(y, yp, average='macro', zero_division=0),
        f1_score(y, yp, average='macro', zero_division=0)
    ])

df = pd.DataFrame(rows, columns=['Model', 'Acc', 'Prec', 'Rec', 'F1']).round(4)
print(df.to_markdown(index=False))
