#!/usr/bin/env python3
"""
analysis-dataset.py  ·  Converte o tic‑tac‑toe.data para 4 classes:
    0 → ongoing   (ainda há casas vazias e ninguém venceu)
    1 → x_win     (X venceu)
    2 → o_win     (O venceu)
    3 → draw      (tabuleiro cheio, ninguém venceu)
Salva: data/tic_tac_toe_clean.csv
"""

from pathlib import Path
import pandas as pd

RAW = Path('tic-tac-toe.data')      # arquivo original da UCI
OUT = Path('tic_tac_toe_clean.csv')
RAW.parent.mkdir(exist_ok=True)

# ---- leitura sem cabeçalho ----
df = pd.read_csv(RAW, header=None)

# renomeia colunas 0‑8 para c0…c8 e a última para 'orig_class'
cols = {i: f'c{i}' for i in range(9)}
cols[9] = 'orig_class'
df = df.rename(columns=cols)

# ---- converte símbolos em números ----
sym_map = {'x': 1, 'o': -1, 'b': 0}
for c in df.columns[:9]:
    df[c] = df[c].map(sym_map)

# ---- calcula nova label de 4 classes ----
LINES = [(0,1,2),(3,4,5),(6,7,8),
         (0,3,6),(1,4,7),(2,5,8),
         (0,4,8),(2,4,6)]

def classify_row(row):
    board = row.values[:9]
    # X ou O venceu?
    for a,b,c in LINES:
        if board[a] == board[b] == board[c] != 0:
            return 1 if board[a] == 1 else 2   # x_win ou o_win
    # empatou?
    if 0 not in board:
        return 3
    return 0                                    # ongoing

df['class'] = df.apply(classify_row, axis=1)

# guarda apenas colunas numéricas + nova classe
out_df = df[[f'c{i}' for i in range(9)] + ['class']]
out_df.to_csv(OUT, index=False)
print('✓', OUT, 'salvo')
print(out_df['class'].value_counts().sort_index())
