#!/usr/bin/env python3
"""
kNN-Model.py  ·  k = 11
Treina, avalia e salva modelo k‑NN.
"""
import pandas as pd
from pathlib import Path
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib

TRAIN = 'tic_tac_toe_train.csv'
VAL   = 'tic_tac_toe_val.csv'
TEST  = 'tic_tac_toe_test.csv'
MODEL = 'knn.joblib'

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

    clf = KNeighborsClassifier(n_neighbors=11).fit(Xtr, ytr)
    joblib.dump(clf, MODEL)
    print(f'✓ Modelo salvo → {MODEL}')

    for name, X, y in [('Val', Xv, yv), ('Test', Xts, yts)]:
        acc, prec, rec, f1 = metrics(y, clf.predict(X))
        print(f'{name}: acc={acc:.4f}  prec={prec:.4f}  rec={rec:.4f}  f1={f1:.4f}')

if __name__ == '__main__':
    main()
