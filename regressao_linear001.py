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


'''''
Neste exemplo, estamos utilizando um modelo de regressão linear simples para prever valores com base em uma única variável 
independente (X). O modelo é treinado com um conjunto de dados de treinamento (X e y). Em seguida, realizamos previsões 
utilizando um conjunto de dados de teste (X_test) e exibimos os resultados.

É importante ressaltar que este é um exemplo básico e que existem muitas outras técnicas e bibliotecas disponíveis para 
diferentes tipos de problemas de machine learning em Python. O scikit-learn é uma biblioteca popular e abrangente para 
aprendizado de máquina em Python, mas também existem outras bibliotecas poderosas, como TensorFlow e PyTorch, que fornecem 
recursos avançados para tarefas mais complexas.
'''''
