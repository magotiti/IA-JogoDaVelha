#!/usr/bin/env python3
"""
MLP-model.py
Treina MLP (1 hidden‑layer) com normalização.
"""
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib

TRAIN = 'tic_tac_toe_train.csv'
VAL   = 'tic_tac_toe_val.csv'
TEST  = 'tic_tac_toe_test.csv'
MODEL = 'mlp.joblib'
RS = 42

def metrics(y_true, y_pred):
    return (
        accuracy_score(y_true, y_pred),
        precision_score(y_true, y_pred, average='macro', zero_division=0),
        recall_score(y_true, y_pred, average='macro', zero_division=0),
        f1_score(y_true, y_pred, average='macro', zero_division=0)
    )

def main() -> None:
    for f in (TRAIN, VAL, TEST):
        if not Path(f).exists():
            raise FileNotFoundError(f)

    train, val, test = (pd.read_csv(f) for f in (TRAIN, VAL, TEST))
    Xtr, ytr = train.drop('class', axis=1), train['class']
    Xv , yv  = val.drop('class', axis=1),   val['class']
    Xts, yts = test.drop('class', axis=1),  test['class']

    scaler = StandardScaler().fit(Xtr)
    Xtr_s, Xv_s, Xts_s = scaler.transform(Xtr), scaler.transform(Xv), scaler.transform(Xts)

    clf = MLPClassifier(hidden_layer_sizes=(100,),
                        learning_rate='adaptive',
                        max_iter=1000,
                        activation='relu',
                        alpha=1e-4,
                        early_stopping=True,
                        random_state=RS).fit(Xtr_s, ytr)

    joblib.dump({'scaler': scaler, 'model': clf}, MODEL)
    print(f'✓ MLP + scaler salvos → {MODEL}')

    for name, X, y in [('Val', Xv_s, yv), ('Test', Xts_s, yts)]:
        acc, prec, rec, f1 = metrics(y, clf.predict(X))
        print(f'{name}: acc={acc:.4f}  prec={prec:.4f}  rec={rec:.4f}  f1={f1:.4f}')

if __name__ == '__main__':
    main()
