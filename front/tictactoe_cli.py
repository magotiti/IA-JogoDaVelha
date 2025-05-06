#!/usr/bin/env python3
"""
tictactoe_cli.py  ·  Avaliação de IA em partida Jogador × Jogador

Uso:
    python tictactoe_cli.py              # usa Random Forest (padrão)
    python tictactoe_cli.py --model knn  # usa k‑NN
    python tictactoe_cli.py --model mlp  # usa MLP
    python tictactoe_cli.py --model dt   # usa Decision Tree

• Dois humanos alternam jogadas (X = 1, O = −1) digitando posições 1‑9.
• Após cada lance, o classificador selecionado prevê se a partida JÁ terminou.
• Ao final, mostra quem venceu/empate e a acurácia global da IA.
"""

import argparse
import os
import joblib
import numpy as np

# ------------------------------------------------------------------ #
# 1. Escolha do modelo via linha de comando
# ------------------------------------------------------------------ #
MODELS = {
    'rf':  'models/rf.joblib',
    'knn': 'models/knn.joblib',
    'mlp': 'models/mlp.joblib',
    'dt':  'models/decision_tree.joblib'
}

ap = argparse.ArgumentParser()
ap.add_argument('--model', choices=MODELS.keys(), default='rf',
                help='Classificador a usar (rf, knn, mlp, dt)')
args = ap.parse_args()

model_path = MODELS[args.model]
model = joblib.load(model_path)
print(f'🔍  IA carregada: {args.model}  ({model_path})')

# ------------------------------------------------------------------ #
# 2. Funções utilitárias
# ------------------------------------------------------------------ #
SYM   = {1: 'X', -1: 'O', 0: ' '}
LINES = [(0,1,2),(3,4,5),(6,7,8),(0,3,6),(1,4,7),(2,5,8),(0,4,8),(2,4,6)]

def clear():
    os.system('cls' if os.name == 'nt' else 'clear')

def print_board(b):
    clear()
    for i in range(0, 9, 3):
        print(' | '.join(SYM[b[i+j]] for j in range(3)))
        if i < 6:
            print('--+---+--')

def winner(b):
    for a, b_, c in LINES:
        if b[a] == b[b_] == b[c] != 0:
            return b[a]     # 1 (X) ou −1 (O)
    if 0 not in b:
        return 2            # empate
    return 0                # continua

# ------------------------------------------------------------------ #
# 3. Loop principal da partida
# ------------------------------------------------------------------ #
board   = [0] * 9
current = 1                 # X começa
acc = tot = 0               # métricas da IA

while True:
    print_board(board)
    try:
        pos = int(input(f'Jogador {SYM[current]} → posição (1‑9): ')) - 1
        if not 0 <= pos <= 8 or board[pos] != 0:
            raise ValueError
    except ValueError:
        input('Posição inválida. <Enter> para tentar de novo…')
        continue

    board[pos] = current

    # ----- Avaliação do classificador -----
    real = 1 if winner(board) else 0
    pred = model.predict(np.array(board).reshape(1, -1))[0]
    acc += (real == pred)
    tot += 1

    if real:                # partida acabou
        break
    current *= -1           # troca jogador

# ------------------------------------------------------------------ #
# 4. Encerramento e métrica final
# ------------------------------------------------------------------ #
print_board(board)
msg = {1: 'Vitória do X!', -1: 'Vitória do O!', 2: 'Empate!'}
print(msg[winner(board)])
print(f'Acurácia da IA durante a partida: {acc / tot:.3f}  ({acc}/{tot} lances)')
