#!/usr/bin/env python3
"""
rebalance-dataset.py
Equilibra o dataset em 4 classes pegando o mesmo nº de amostras de cada uma
Entrada : data/tic_tac_toe_clean.csv
Saída   : data/tic_tac_toe_balanced.csv
"""

import pandas as pd
from pathlib import Path
import numpy as np

IN  = Path('tic_tac_toe_clean.csv')
OUT = Path('tic_tac_toe_balanced.csv')
RS  = 42

df = pd.read_csv(IN)
n_min = df['class'].value_counts().min()

balanced = (
    df.groupby('class', group_keys=False)
      .apply(lambda x: x.sample(n_min, random_state=RS))
      .reset_index(drop=True)
      .sample(frac=1, random_state=RS)   # embaralha
)

balanced.to_csv(OUT, index=False)
print('✓', OUT, 'salvo')
print(balanced['class'].value_counts().sort_index())
