#!/usr/bin/env python3
# rf-Model.py  —  Random Forest para o Tic‑Tac‑Toe

import pandas as pd, joblib
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

TRAIN = 'tic_tac_toe_train.csv'
VAL   = 'tic_tac_toe_val.csv'
TEST  = 'tic_tac_toe_test.csv'
MODEL = 'rf.joblib'
RS    = 42


def metrics(y_true, y_pred):
    return (
        accuracy_score(y_true, y_pred),
        precision_score(y_true, y_pred, average='macro', zero_division=0),
        recall_score(y_true, y_pred, average='macro', zero_division=0),
        f1_score(y_true, y_pred, average='macro', zero_division=0),
    )


def main() -> None:
    # garante que os três arquivos CSV existem
    for f in (TRAIN, VAL, TEST):
        if not Path(f).exists():
            raise FileNotFoundError(f)

    tr, vl, ts = (pd.read_csv(f) for f in (TRAIN, VAL, TEST))

    # ---- separa features e alvo (uso do pandas 3+ → drop(columns='class')) ----
    Xtr, ytr = tr.drop(columns='class'), tr['class']
    Xv,  yv  = vl.drop(columns='class'), vl['class']
    Xts, yts = ts.drop(columns='class'), ts['class']

    # ---- treina Random Forest ----
    rf = RandomForestClassifier(
        n_estimators=200,
        random_state=RS,
        n_jobs=-1
    ).fit(Xtr, ytr)

    joblib.dump(rf, MODEL)
    print(f'✓ Modelo salvo → {MODEL}')

    # ---- métricas ----
    for name, X, y in [('Val', Xv, yv), ('Test', Xts, yts)]:
        acc, prec, rec, f1 = metrics(y, rf.predict(X))
        print(f'{name}: acc={acc:.4f}  prec={prec:.4f}  rec={rec:.4f}  f1={f1:.4f}')


if __name__ == '__main__':
    main()
