#!/usr/bin/env python3
"""
cluster_dataset.py
Divide tic_tac_toe_balanced.csv em:
    80 % treino · 10 % validação · 10 % teste (estratificado)
Gera: tic_tac_toe_train.csv · tic_tac_toe_val.csv · tic_tac_toe_test.csv
"""
import pandas as pd
from sklearn.model_selection import train_test_split
from pathlib import Path

INPUT_CSV = 'tic_tac_toe_balanced.csv'
RANDOM_STATE = 42

def main() -> None:
    if not Path(INPUT_CSV).exists():
        raise FileNotFoundError(f'Arquivo {INPUT_CSV} não encontrado.')

    df = pd.read_csv(INPUT_CSV)

    train_df, temp_df = train_test_split(
        df, test_size=0.2, stratify=df['class'], random_state=RANDOM_STATE)

    val_df, test_df = train_test_split(
        temp_df, test_size=0.5, stratify=temp_df['class'], random_state=RANDOM_STATE)

    train_df.to_csv('tic_tac_toe_train.csv', index=False)
    val_df.to_csv('tic_tac_toe_val.csv',   index=False)
    test_df.to_csv('tic_tac_toe_test.csv', index=False)

    print('✓ train/val/test gerados:')
    for name, d in [('train', train_df), ('val', val_df), ('test', test_df)]:
        print(name, d['class'].value_counts().to_dict())

if __name__ == '__main__':
    main()
