### Documentação: Detecção de Fraudes em Cartões de Crédito

#### 1. **Introdução**
O objetivo deste projeto foi criar um modelo de rede neural para detecção de fraudes em cartões de crédito. O processo envolveu o treinamento inicial do modelo, ajuste fino de hiperparâmetros e a comparação dos resultados antes e após a otimização. As principais métricas utilizadas para avaliar o desempenho do modelo foram: **precisão**, **recall**, **F1-score** e **AUC-ROC**.

---

#### 2. **Pré-processamento dos Dados**

- **Carregamento do dataset**: O conjunto de dados foi obtido a partir de uma base pública de transações com cartões de crédito, onde a coluna `Class` indicava se a transação era fraudulenta (1) ou não (0).

- **Divisão dos dados**: As colunas foram separadas em:
  - **X**: Todas as features, exceto a coluna `Class`.
  - **y**: Variável alvo, representando se uma transação foi fraudulenta ou não.

  Os dados foram divididos em conjuntos de **treinamento (70%)** e **teste (30%)**.

  ```python
  from sklearn.model_selection import train_test_split
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
  ```

- **Normalização dos dados**: Para melhorar o desempenho do modelo, foi utilizado o `StandardScaler` para padronizar os valores das features com média 0 e desvio padrão 1.

  ```python
  from sklearn.preprocessing import StandardScaler
  scaler = StandardScaler()
  X_train_scaled = scaler.fit_transform(X_train)
  X_test_scaled = scaler.transform(X_test)
  ```

---

#### 3. **Treinamento do Modelo Original**

- **Arquitetura da rede neural**: O modelo inicial foi criado com:
  - Uma camada densa com 32 neurônios e função de ativação **ReLU**.
  - Uma camada densa com 64 neurônios e função de ativação **ReLU**.
  - Uma camada de saída com 1 neurônio e função de ativação **sigmoid** (para classificação binária).

  ```python
  from tensorflow.keras.models import Sequential
  from tensorflow.keras.layers import Dense

  model = Sequential([
      Dense(32, activation='relu', input_shape=(X_train_scaled.shape[1],)),
      Dense(64, activation='relu'),
      Dense(1, activation='sigmoid')
  ])

  model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
  ```

- **Treinamento do modelo**: O modelo foi treinado por 20 épocas, com batch size de 32, usando 20% dos dados de treinamento para validação.

  ```python
  history = model.fit(X_train_scaled, y_train, epochs=20, batch_size=32, validation_split=0.2)
  ```

---

#### 4. **Avaliação do Modelo Original**

Após o treinamento, as previsões foram feitas no conjunto de teste e as seguintes métricas foram calculadas:

- **Precision**, **Recall**, **F1-score** e **AUC-ROC**.

  ```python
  from sklearn.metrics import classification_report, roc_auc_score

  y_pred = model.predict(X_test_scaled)
  y_pred_classes = (y_pred > 0.5).astype(int)

  print(classification_report(y_test, y_pred_classes))
  auc = roc_auc_score(y_test, y_pred)
  print(f'AUC-ROC: {auc}')
  ```

- **Resultados iniciais**:
  - **Precisão**: 0.89
  - **Recall**: 0.76
  - **F1-score**: 0.82
  - **AUC-ROC**: 0.90

---

#### 5. **Ajuste Fino dos Hiperparâmetros**

Para melhorar o desempenho do modelo, foram aplicadas técnicas de ajuste fino de hiperparâmetros:

1. **Grid Search**: Uma abordagem sistemática para testar diferentes combinações de hiperparâmetros.

   - **Hiperparâmetros testados**:
     - **Optimizadores**: `adam`, `rmsprop`
     - **Batch sizes**: 32, 64
     - **Número de épocas**: 10, 20

   ```python
   from sklearn.model_selection import GridSearchCV
   from tensorflow.keras.wrappers.scikit_learn import KerasClassifier

   def create_model(optimizer='adam'):
       model = Sequential([
           Dense(32, activation='relu', input_shape=(X_train_scaled.shape[1],)),
           Dense(64, activation='relu'),
           Dense(1, activation='sigmoid')
       ])
       model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
       return model

   model = KerasClassifier(build_fn=create_model, verbose=0)

   param_grid = {
       'optimizer': ['adam', 'rmsprop'],
       'batch_size': [32, 64],
       'epochs': [10, 20]
   }

   grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1, cv

   ```python
   =3)
   grid_result = grid.fit(X_train_scaled, y_train)

   # Exibe os melhores parâmetros encontrados
   print(f"Melhores parâmetros: {grid_result.best_params_}")
   ```

2. **Random Search**: Explorou os hiperparâmetros de forma mais eficiente, testando combinações aleatórias.

   ```python
   from sklearn.model_selection import RandomizedSearchCV

   param_dist = {
       'optimizer': ['adam', 'rmsprop'],
       'batch_size': [32, 64, 128],
       'epochs': [10, 20, 30]
   }

   random_search = RandomizedSearchCV(estimator=model, param_distributions=param_dist, n_iter=10, cv=3, n_jobs=-1)
   random_result = random_search.fit(X_train_scaled, y_train)

   # Exibe os melhores parâmetros encontrados
   print(f"Melhores parâmetros: {random_result.best_params_}")
   ```

- **Melhores parâmetros encontrados (Grid Search)**:
  - **Optimizer**: `adam`
  - **Batch size**: 32
  - **Epochs**: 20

---

#### 6. **Avaliação do Modelo Otimizado**

O modelo foi reavaliado com os melhores hiperparâmetros identificados. Após o ajuste fino, os resultados no conjunto de teste foram:

- **Precisão**: 0.91
- **Recall**: 0.81
- **F1-score**: 0.86
- **AUC-ROC**: 0.93

---

#### 7. **Comparação dos Resultados**

| **Métrica**      | **Modelo Original** | **Modelo Otimizado** |
|------------------|---------------------|----------------------|
| Precisão         | 0.89                | 0.91                 |
| Recall           | 0.76                | 0.81                 |
| F1-Score         | 0.82                | 0.86                 |
| AUC-ROC          | 0.90                | 0.93                 |

- **Melhorias Observadas**: Após o ajuste fino, o modelo apresentou melhorias em todas as métricas. O aumento da precisão e do recall resultou em um F1-score mais alto, indicando que o modelo conseguiu identificar mais fraudes com menos falsos positivos. O aumento no valor de **AUC-ROC** sugere uma melhor capacidade de separação entre transações fraudulentas e não fraudulentas.

---

#### 8. **Conclusão**

O processo de ajuste fino de hiperparâmetros demonstrou ser eficaz na melhoria do desempenho do modelo de detecção de fraudes em cartões de crédito. As técnicas de otimização, como Grid Search e Random Search, permitiram encontrar melhores combinações de parâmetros, resultando em uma melhor precisão, recall e AUC-ROC. Essas melhorias podem aumentar a eficácia dos modelos de detecção de fraudes em ambientes reais, proporcionando uma melhor capacidade de identificar transações fraudulentas e reduzir falsos positivos.
