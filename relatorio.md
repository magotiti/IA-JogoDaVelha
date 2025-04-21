# 1. Tratamento de dados

## 1.1. Análise Inicial

O dataset "Tic-Tac-Toe Endgame Data Set", obtido do repositório UCI Machine Learning Repository, contém registros de estados finais de tabuleiros de jogo da velha. Cada amostra descreve a situação de todas as casas do tabuleiro, além de uma classificação (`positive` ou `negative`) indicando se o jogo terminou com vitória (`positive`) ou não (`negative`).

Utilizando o script analysis-dataset.py filtramos os seguintes dados:

- **Valores únicos:** Cada posição do tabuleiro apresenta apenas três valores válidos: `'x'`, `'o'` e `'b'` (representando casas vazias — "blank").
- **Valores nulos:** Nenhuma célula do dataset apresenta valores nulos.
- **Classes:** A coluna `class` possui apenas dois valores válidos: `positive` e `negative`.

**Conclusão:** O dataset, estruturalmente, é **coerente e completo** para a finalidade do trabalho, que é validar o estado de um jogo.

---

## 1.2. Problemas encontrados e ajustes realizados:

| Problema Identificado                                                              | Ajuste Realizado                                                                                    | Justificativa                                                                                                                                                                                                                |
| :--------------------------------------------------------------------------------- | :-------------------------------------------------------------------------------------------------- | :--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Desequilíbrio nas classes** (`positive`: 626 amostras, `negative`: 332 amostras) | Limitamos o número de exemplos de cada classe para 332 amostras.                                    | Para garantir um **dataset balanceado**, essencial para que o modelo não seja enviesado para a classe majoritária (`positive`).                                                                                              |
| **Instrução para uso de 400 amostras por classe**                                  | Não foi possível atingir 400 amostras para `negative`, pois só existiam 332 registros dessa classe. | Seguindo a orientação de trabalhar apenas com dados representativos e balanceados, optou-se por 332 exemplos de cada classe.                                                                                                 |
| **Possível excesso de instâncias muito similares**                                 | Uso de **amostragem aleatória** (`random_state=42`) durante a seleção.                              | A amostragem aleatória ajuda a diversificar os exemplos selecionados, evitando um dataset final com padrões muito repetitivos.                                                                                               |
| **Formato textual dos dados ('x', 'o', 'b')**                                      | Mantivemos o formato original na primeira etapa.                                                    | Como a IA pedida inicialmente é para reconhecimento do estado do jogo e não para prever movimentos, a transformação dos dados (por exemplo, para valores numéricos) será avaliada posteriormente, no momento do treinamento. |

---

s ajustes foram feitos a partir do script rebalance-dataset.py. Foram atribuídos valores numéricos aos atributos do dataset para uma melhor adesão do modelo desenvolvido.

## 1.3. Resultado final do tratamento:

- Total de 664 amostras balanceadas (332 `positive`, 332 `negative`).
- Cada amostra representa um estado do tabuleiro.
- Nenhum valor nulo ou inválido.
- Dataset pronto para ser utilizado no treinamento do modelo de IA.

# 2. Análise dos Resultados do Modelo k-NN

O objetivo dos testes com diferentes valores de **k** no modelo **k-NN** foi avaliar como a escolha do parâmetro **k** afeta a performance do modelo. A seguir, estão os resultados obtidos para diferentes valores de **k**:

## 2.1. Resultados

### 2.1.1. k = 5

- **Validação**:
  - Acurácia: 96,99%
  - Precisão: 98,46%
  - Recall: 95,52%
  - F1-Score: 96,97%
- **Teste**:
  - Acurácia: 97,74%
  - Precisão: 97,01%
  - Recall: 98,48%
  - F1-Score: 97,74%

**Análise**: O modelo com **k = 5** apresentou um bom desempenho geral, com uma boa combinação de **precisão** e **recall**, indicando que o modelo foi bem ajustado tanto para o conjunto de treino quanto para o de teste.

---

### 2.1.2. k = 3 (possível overfitting)

- **Validação**:
  - Acurácia: 100%
  - Precisão: 100%
  - Recall: 100%
  - F1-Score: 100%
- **Teste**:
  - Acurácia: 100%
  - Precisão: 100%
  - Recall: 100%
  - F1-Score: 100%

**Análise**: O modelo com **k = 3** apresentou um desempenho perfeito, mas isso sugere **overfitting**, pois o modelo está altamente ajustado ao conjunto de treino, sem conseguir generalizar bem para novos dados.

---

### 2.1.3. k = 7

- **Validação**:
  - Acurácia: 93,23%
  - Precisão: 96,77%
  - Recall: 89,55%
  - F1-Score: 93,02%
- **Teste**:
  - Acurácia: 96,24%
  - Precisão: 95,52%
  - Recall: 96,97%
  - F1-Score: 96,24%

**Análise**: O modelo com **k = 7** apresentou um desempenho equilibrado, mas com um recall mais baixo no conjunto de validação, indicando que o modelo pode estar deixando passar alguns exemplos positivos.

---

### 2.1.4. k = 9

- **Validação**:
  - Acurácia: 95,49%
  - Precisão: 100%
  - Recall: 91,04%
  - F1-Score: 95,31%
- **Teste**:
  - Acurácia: 94,74%
  - Precisão: 95,38%
  - Recall: 93,94%
  - F1-Score: 94,66%

**Análise**: Com **k = 9**, a precisão foi muito alta no conjunto de validação (100%), mas o recall foi mais baixo, indicando que o modelo é conservador na identificação de instâncias positivas. No conjunto de teste, a performance foi equilibrada.

---

### 2.1.5. k = 11

- **Validação**:
  - Acurácia: 96,24%
  - Precisão: 98,44%
  - Recall: 94,03%
  - F1-Score: 96,18%
- **Teste**:
  - Acurácia: 93,98%
  - Precisão: 90,28%
  - Recall: 98,48%
  - F1-Score: 94,20%

**Análise**: O modelo com **k = 11** teve um aumento no recall no conjunto de teste, mas a precisão caiu significativamente, o que pode indicar que o modelo está classificando muitos exemplos negativos como positivos.

---

## 2.2. Comparação de Desempenho

| **k** | **Acurácia (Teste)** | **Precisão (Teste)** | **Recall (Teste)** | **F1-Score (Teste)** |
| ----- | -------------------- | -------------------- | ------------------ | -------------------- |
| 3     | 100%                 | 100%                 | 100%               | 100%                 |
| 5     | 97,74%               | 97,01%               | 98,48%             | 97,74%               |
| 7     | 96,24%               | 95,52%               | 96,97%             | 96,24%               |
| 9     | 94,74%               | 95,38%               | 93,94%             | 94,66%               |
| 11    | 93,98%               | 90,28%               | 98,48%             | 94,20%               |

## 2.3. Conclusão

- O **melhor desempenho** foi observado com **k = 3**, que teve **100%** em todas as métricas. No entanto, esse resultado sugere **overfitting**, já que o modelo não está generalizando bem.
- O modelo com **k = 5** teve um desempenho equilibrado e foi o mais adequado, com uma boa **acurácia** e um bom equilíbrio entre **precisão** e **recall**.
- **k = 7** e **k = 9** também apresentaram bons resultados, mas **k = 5** foi o mais robusto e o mais confiável para generalização.
- O modelo com **k = 11** teve um recall muito bom, mas houve um custo em precisão.

Com base nesses resultados, **k = 5** é o valor recomendado para o modelo, pois oferece uma boa combinação de desempenho e generalização.

# 3. Análise dos Resultados do Modelo MLP

## 3.1. Introdução

O modelo Multi-Layer Perceptron (MLP) foi testado com diferentes combinações de parâmetros para analisar seu desempenho e verificar como a configuração dos hiperparâmetros influencia os resultados. Foram realizados cinco testes com variações nas configurações de camadas ocultas, taxa de aprendizado, iteração máxima, função de ativação, regularização e utilização de early stopping. A seguir, os detalhes de cada teste.

## 3.2.1 Teste 1: `hidden_layer_sizes=(100,)`, `learning_rate='adaptive'`, `max_iter=1000`, `activation='relu'`, `alpha=0.0001`

### Escolha dos parâmetros

- **`hidden_layer_sizes=(100,)`**: Foi escolhida uma única camada oculta com 100 neurônios, o que oferece uma estrutura simples e eficiente para começar. A quantidade de neurônios na camada oculta é um dos parâmetros cruciais para a capacidade de aprendizagem do modelo, e o valor de 100 é um ponto de partida comum.
- **`learning_rate='adaptive'`**: A escolha do `adaptive` faz com que a taxa de aprendizado seja ajustada durante o treinamento, ajudando a evitar problemas de convergência ou grandes oscilações nos pesos, especialmente em redes mais profundas.
- **`max_iter=1000`**: O número de iterações foi ajustado para 1000, oferecendo tempo suficiente para o modelo aprender e se ajustar aos dados sem um risco elevado de overfitting.
- **`activation='relu'`**: A função ReLU (Rectified Linear Unit) foi escolhida devido à sua popularidade em redes neurais profundas, principalmente porque ajuda a acelerar o treinamento e a combater o problema de vanishing gradient.
- **`alpha=0.0001`**: Um valor pequeno de regularização foi escolhido para evitar overfitting, sem penalizar excessivamente o modelo e impedir o aprendizado de padrões significativos.

### Resultados

- **Validação:**
  - Acurácia: 0.9925
  - Precisão: 0.9853
  - Recall: 1.0000
  - F1-Score: 0.9926
- **Teste:**
  - Acurácia: 0.9925
  - Precisão: 0.9851
  - Recall: 1.0000
  - F1-Score: 0.9925

### Análise

- O modelo teve um desempenho excelente, com alta precisão e recall tanto na validação quanto no teste.
- **Possível overfitting:** A semelhança entre os resultados de validação e teste sugere que o modelo pode estar sobreajustado aos dados. Uma validação adicional com dados diferentes seria prudente.

---

## 3.2.2 Teste 2: `hidden_layer_sizes=(50, 50)`, `learning_rate='constant'`, `max_iter=500`, `activation='tanh'`, `alpha=0.001`

### Escolha dos parâmetros

- **`hidden_layer_sizes=(50, 50)`**: Neste teste, foi escolhida uma rede neural com duas camadas ocultas de 50 neurônios cada, o que cria uma estrutura mais complexa em comparação ao Teste 1. Isso foi feito para explorar se uma arquitetura mais profunda poderia melhorar a capacidade de aprendizado do modelo.
- **`learning_rate='constant'`**: A taxa de aprendizado constante foi escolhida para fornecer uma taxa fixa durante todo o treinamento, buscando estabilidade no aprendizado. Embora essa configuração possa levar a uma convergência mais lenta, ela ajuda a evitar ajustes excessivos.
- **`max_iter=500`**: A quantidade de iterações foi reduzida para 500, forçando o modelo a aprender mais rapidamente, mas também pode fazer com que o treinamento termine prematuramente.
- **`activation='tanh'`**: A função de ativação `tanh` foi escolhida para verificar como a rede se comportaria com uma função que mapeia os valores para um intervalo entre -1 e 1, em contraste com o `relu`.
- **`alpha=0.001`**: A regularização foi aumentada um pouco para prevenir overfitting, mas sem ser excessivamente rigorosa.

### Resultados

- **Validação:**
  - Acurácia: 0.9850
  - Precisão: 0.9710
  - Recall: 1.0000
  - F1-Score: 0.9853
- **Teste:**
  - Acurácia: 0.9774
  - Precisão: 0.9565
  - Recall: 1.0000
  - F1-Score: 0.9778

### Análise

- O desempenho deste teste foi ligeiramente inferior ao Teste 1, com uma queda na acurácia e F1-Score no conjunto de teste.
- A precisão e o recall continuam sendo muito bons, mas o modelo não obteve os mesmos resultados extremos do primeiro teste.
- **Estrutura mais complexa:** A adição de camadas e o uso de `tanh` não aumentaram a performance em comparação ao primeiro teste.

---

## 3.2.3 Teste 3: `hidden_layer_sizes=(200, 100)`, `learning_rate='adaptive'`, `max_iter=1500`, `activation='logistic'`, `alpha=0.01`

### Escolha dos parâmetros

- **`hidden_layer_sizes=(200, 100)`**: A escolha de uma rede mais profunda com camadas de 200 e 100 neurônios visa aumentar a capacidade do modelo de aprender padrões mais complexos.
- **`learning_rate='adaptive'`**: A taxa de aprendizado adaptativa foi novamente escolhida, permitindo ao modelo ajustar a taxa de aprendizado durante o treinamento para evitar problemas de convergência.
- **`max_iter=1500`**: A quantidade de iterações foi aumentada para 1500, proporcionando mais tempo de treinamento para a rede, o que pode ajudar a melhorar a performance em modelos mais profundos.
- **`activation='logistic'`**: A função de ativação `logistic` (sigmoide) foi escolhida para observar como o modelo se comportaria com uma função que mapeia os valores para o intervalo entre 0 e 1.
- **`alpha=0.01`**: O valor de regularização foi aumentado para 0.01, buscando controlar o overfitting em uma rede com maior capacidade.

### Resultados

- **Validação:**
  - Acurácia: 0.9850
  - Precisão: 0.9710
  - Recall: 1.0000
  - F1-Score: 0.9853
- **Teste:**
  - Acurácia: 0.9774
  - Precisão: 0.9565
  - Recall: 1.0000
  - F1-Score: 0.9778

### Análise

- O desempenho foi semelhante ao Teste 2, com bons resultados, mas sem melhorias significativas.
- **Arquitetura maior** não trouxe um grande impacto na performance, e a função logística não superou a ReLU em termos de desempenho.

---

## 3.2.4 Teste 4: `hidden_layer_sizes=(100,)`, `learning_rate='constant'`, `max_iter=2000`, `activation='relu'`, `alpha=0.1`

### Escolha dos parâmetros

- **`hidden_layer_sizes=(100,)`**: Optou-se por uma única camada oculta de 100 neurônios, uma vez que já foi observado bom desempenho com essa configuração no Teste 1.
- **`learning_rate='constant'`**: A escolha da taxa de aprendizado constante visa garantir que o modelo aprenda de forma estável e não tenha variações abruptas durante o treinamento.
- **`max_iter=2000`**: O número de iterações foi aumentado para 2000, proporcionando mais tempo de treinamento para permitir que o modelo aprenda melhor.
- **`activation='relu'`**: A função ReLU foi escolhida novamente, pois mostrou um bom desempenho nas iterações anteriores.
- **`alpha=0.1`**: A regularização foi aumentada significativamente para verificar se uma maior penalização ajudaria a controlar o overfitting.

### Resultados

- **Validação:**
  - Acurácia: 1.0000
  - Precisão: 1.0000
  - Recall: 1.0000
  - F1-Score: 1.0000
- **Teste:**
  - Acurácia: 0.9850
  - Precisão: 0.9706
  - Recall: 1.0000
  - F1-Score: 0.9851

### Análise

- Embora o modelo tenha mostrado acurácia perfeita na validação, houve uma queda significativa no desempenho no conjunto de teste.
- **Possível overfitting:** A discrepância entre os resultados de validação e teste sugere que o modelo pode estar sobreajustado aos dados de validação e não generalizou bem para o conjunto de teste.

---

## 3.2.5. Teste 5: `hidden_layer_sizes=(100,)`, `learning_rate='adaptive'`, `max_iter=1000`, `activation='relu'`, `alpha=0.0001`, `early_stopping=True`

### Escolha dos parâmetros

- **`early_stopping=True`**: O early stopping foi introduzido para interromper o treinamento quando o modelo começar a mostrar sinais de overfitting, ou seja, quando a performance no conjunto de validação começar a piorar.
- **Outros parâmetros**: Foram mantidos os mesmos parâmetros do Teste 1, com a adição do early stopping para melhorar a generalização.

### Resultados

- **Validação:**
  - Acurácia: 0.9925
  - Precisão: 0.9853
  - Recall: 1.0000
  - F1-Score: 0.9926
- **Teste:**
  - Acurácia: 0.9925
  - Precisão: 0.9851
  - Recall: 1.0000
  - F1-Score: 0.9925

### 3.3. Análise

- O early stopping ajudou a melhorar a generalização do modelo, pois os resultados no conjunto de teste e validação estavam muito próximos, sem sinais de overfitting.

---

## 3.4. Conclusão

Os cinco testes realizados demonstraram que uma configuração de rede neural bem ajustada pode melhorar significativamente a performance do modelo. A introdução de early stopping e a escolha cuidadosa de funções de ativação, taxas de aprendizado e regularização são fundamentais para a obtenção de um bom desempenho sem overfitting.

# 4. Análise dos Resultados do Modelo de Árvore de Decisão

## 4.1. Introdução

O modelo de árvore de decisão foi testado com diferentes combinações de parâmetros para analisar seu desempenho e verificar como as configurações influenciam os resultados. Foram realizados cinco testes com variações nas configurações de profundidade da árvore, critério de divisão, número mínimo de amostras para dividir um nó, e uso de randomização nas amostras. A seguir, os detalhes de cada teste.

## 4.2.1 Teste 1: `max_depth=5`, `criterion='gini'`, `min_samples_split=2`, `random_state=42`

### Escolha dos parâmetros

- **`max_depth=5`**: A profundidade máxima foi definida como 5 para evitar que a árvore ficasse muito profunda, o que poderia levar a overfitting. Essa configuração é comumente usada em problemas simples e evita que o modelo se ajuste excessivamente aos dados.
- **`criterion='gini'`**: O critério Gini foi escolhido para medir a impureza dos nós. Ele é amplamente utilizado em árvores de decisão por ser eficiente e fácil de interpretar.
- **`min_samples_split=2`**: Definir 2 como o número mínimo de amostras para dividir um nó permite que a árvore explore completamente os dados, embora isso possa aumentar o risco de overfitting se não controlado.
- **`random_state=42`**: O valor do random_state foi fixado para garantir a reprodutibilidade dos resultados.

### Resultados

- **Validação:**
  - Acurácia: 0.88
  - Precisão: 0.85
  - Recall: 0.92
  - F1-Score: 0.88
- **Teste:**
  - Acurácia: 0.86
  - Precisão: 0.82
  - Recall: 0.91
  - F1-Score: 0.86

### Análise

- O modelo obteve bons resultados, com alta acurácia e recall tanto na validação quanto no teste.
- **Possível overfitting:** A diferença de desempenho entre os conjuntos de validação e teste sugere que o modelo pode estar se ajustando excessivamente aos dados de treino. Isso pode ser mitigado ajustando o parâmetro `max_depth` ou utilizando técnicas de regularização.

---

## 4.2.2 Teste 2: `max_depth=10`, `criterion='entropy'`, `min_samples_split=10`, `random_state=42`

### Escolha dos parâmetros

- **`max_depth=10`**: Neste teste, aumentamos a profundidade da árvore para 10, o que permite uma maior complexidade no modelo. Isso pode melhorar a capacidade de captura de padrões, mas também aumenta o risco de overfitting.
- **`criterion='entropy'`**: O critério de entropia foi escolhido para calcular a impureza dos nós. A entropia pode ser mais eficaz do que o Gini em certos contextos, especialmente quando se deseja uma maior distinção entre as classes.
- **`min_samples_split=10`**: Um valor maior para `min_samples_split` foi escolhido para evitar divisões excessivas em pequenos subconjuntos de dados, ajudando a prevenir o overfitting.
- **`random_state=42`**: O valor do random_state foi mantido para garantir a reprodutibilidade dos resultados.

### Resultados

- **Validação:**
  - Acurácia: 0.85
  - Precisão: 0.83
  - Recall: 0.88
  - F1-Score: 0.85
- **Teste:**
  - Acurácia: 0.84
  - Precisão: 0.81
  - Recall: 0.87
  - F1-Score: 0.84

### Análise

- Os resultados são semelhantes aos do Teste 1, mas a maior profundidade e o critério de entropia não trouxeram melhorias significativas no desempenho.
- **Aumento da complexidade não necessariamente melhora a performance**: A maior profundidade da árvore não resultou em ganhos consideráveis, sugerindo que o modelo pode ser excessivamente complexo para o problema em questão.

---

## 4.2.3 Teste 3: `max_depth=15`, `criterion='gini'`, `min_samples_split=5`, `random_state=42`

### Escolha dos parâmetros

- **`max_depth=15`**: A profundidade foi aumentada novamente para 15, permitindo que a árvore capture mais informações dos dados, embora com risco de overfitting.
- **`criterion='gini'`**: O critério Gini foi mantido, pois já havia mostrado um bom desempenho no Teste 1.
- **`min_samples_split=5`**: O número de amostras mínimas para dividir um nó foi ajustado para 5, um valor intermediário que pode balancear a complexidade da árvore e a necessidade de capturar padrões mais sutis.
- **`random_state=42`**: O valor do random_state foi mantido.

### Resultados

- **Validação:**
  - Acurácia: 0.90
  - Precisão: 0.87
  - Recall: 0.94
  - F1-Score: 0.90
- **Teste:**
  - Acurácia: 0.89
  - Precisão: 0.85
  - Recall: 0.92
  - F1-Score: 0.88

### Análise

- O desempenho melhorou ligeiramente em comparação aos testes anteriores, principalmente no conjunto de validação.
- **Possível overfitting:** Embora o modelo tenha apresentado bons resultados, a diferença entre os dados de validação e teste sugere que ele pode estar se ajustando demais aos dados de treino. A profundidade da árvore pode ser um fator decisivo nesse comportamento.

---

## 4.2.4 Teste 4: `max_depth=3`, `criterion='entropy'`, `min_samples_split=2`, `random_state=42`

### Escolha dos parâmetros

- **`max_depth=3`**: Neste teste, a profundidade foi reduzida para 3, com o objetivo de evitar o overfitting e melhorar a generalização do modelo.
- **`criterion='entropy'`**: O critério de entropia foi novamente utilizado, já que ele permite uma diferenciação mais clara entre as classes.
- **`min_samples_split=2`**: O valor de `min_samples_split` foi mantido em 2, permitindo que a árvore divida nós de forma mais granular.
- **`random_state=42`**: O valor do random_state foi fixado para garantir a reprodutibilidade dos resultados.

### Resultados

- **Validação:**
  - Acurácia: 0.82
  - Precisão: 0.79
  - Recall: 0.85
  - F1-Score: 0.81
- **Teste:**
  - Acurácia: 0.80
  - Precisão: 0.76
  - Recall: 0.83
  - F1-Score: 0.79

### Análise

- Embora o modelo com profundidade reduzida tenha apresentado uma queda na acurácia em comparação aos testes anteriores, ele foi capaz de generalizar melhor para o conjunto de teste.
- **Melhor generalização:** A redução da profundidade ajudou a evitar o overfitting, mas à custa de uma ligeira queda no desempenho geral.

---

## 4.2.5 Teste 5: `max_depth=5`, `criterion='gini'`, `min_samples_split=10`, `random_state=42`

### Escolha dos parâmetros

- **`max_depth=5`**: A profundidade foi mantida em 5, similar ao Teste 1, para balancear a complexidade e evitar overfitting.
- **`criterion='gini'`**: O critério Gini foi mantido, pois já havia mostrado bom desempenho nos testes anteriores.
- **`min_samples_split=10`**: O número de amostras mínimas para dividir um nó foi aumentado para 10, para reduzir ainda mais o risco de overfitting.
- **`random_state=42`**: O valor do random_state foi fixado para garantir a reprodutibilidade dos resultados.

### Resultados

- **Validação:**
  - Acurácia: 0.86
  - Precisão: 0.84
  - Recall: 0.89
  - F1-Score: 0.86
- **Teste:**
  - Acurácia: 0.85
  - Precisão: 0.82
  - Recall: 0.88
  - F1-Score: 0.85

### Análise

- Este teste apresentou resultados semelhantes ao Teste 1, com a vantagem de uma melhor capacidade de generalização, graças ao aumento do valor de `min_samples_split`.

---

## 4.3. Conclusão

Os testes realizados com o modelo de árvore de decisão demonstraram que a escolha dos parâmetros, como a profundidade da árvore e o critério de divisão, impacta significativamente o desempenho do modelo. Reduzir a profundidade da árvore e aumentar o número de amostras mínimas para dividir um nó ajuda a melhorar a generalização e a evitar o overfitting, embora possa causar uma leve queda na acurácia. A combinação de parâmetros ideal depende das características do conjunto de dados e da necessidade de balancear precisão e generalização.
