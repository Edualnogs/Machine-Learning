'''
Usando algoritmo de M.A para prever notas de corte do curso de química medicinal da ufcspa
'''

# Importando as bibliotecas necessárias
import numpy as np
from sklearn.linear_model import LinearRegression

# Dados de treinamento
X = np.array([[2018], [2019], [2020], [2021], [2022], [2023]])  # Variável independente
y = np.array([638.18, 656.22, 642.68, 646.62, 643.72, 618.72])  # Variável dependente

# Criando o modelo e treinando
model = LinearRegression()
model.fit(X, y)

# Dados de teste
X_test = np.array([[2024], [2025], [2026]])

# Realizando as previsões
predictions = model.predict(X_test)

# Exibindo as previsões
for i in range(len(X_test)):
    print(f"X_test: {X_test[i]}, Previsão: {predictions[i]}")
