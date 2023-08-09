# Importando as bibliotecas necessárias
import numpy as np
from sklearn.linear_model import LinearRegression

# Dados de treinamento
X = np.array([[1], [2], [3], [4], [5]])  # Variável independente
y = np.array([2, 4, 6, 8, 10])           # Variável dependente

# Criando o modelo e treinando
model = LinearRegression()
model.fit(X, y)

# Dados de teste
X_test = np.array([[6], [7], [8]])

# Realizando as previsões
predictions = model.predict(X_test)

# Exibindo as previsões
for i in range(len(X_test)):
    print(f"X_test: {X_test[i]}, Previsão: {predictions[i]}")
