'''


Algoritmo Support Vector Machines (SVM), gerei para esses dados de treinamento: 
X = ([ 2018, 2019, 2020, 2021, 2022, 2023]) # Variável independente

y = ([638.18, 656.22, 642.68, 646.62, 643.72, 618.72]) #Variável dependente

Para esses dados de treinamento, podemos usar o Support Vector Regression (SVR) para criar um modelo de regressão que faça 
previsões para os valores de y correspondentes a novos valores de X. Vamos usar o kernel RBF (Radial Basis Function) para 
este exemplo.

Aqui está o código Python para gerar o modelo SVR e fazer previsões:

'''


import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVR

# Dados de treinamento
X_train = np.array([2018, 2019, 2020, 2021, 2022, 2023]).reshape(-1, 1)
y_train = np.array([638.18, 656.22, 642.68, 646.62, 643.72, 618.72])

# Criar o modelo SVR com o kernel RBF
svm_model = SVR(kernel='rbf')

# Treinar o modelo usando os dados de treinamento
svm_model.fit(X_train, y_train)

# Prever os valores de y para novos valores de X
X_new = np.array([2024, 2025, 2026]).reshape(-1, 1)
y_pred = svm_model.predict(X_new)

# Exibir as previsões
print("Previsões para os anos 2024, 2025 e 2026:")
print(y_pred)

# Plotar o gráfico com os dados de treinamento e as previsões
plt.scatter(X_train, y_train, label='Dados de Treinamento')
plt.plot(X_new, y_pred, 'r-', label='Previsões')
plt.xlabel('Ano')
plt.ylabel('Valor de y')
plt.title('Previsões usando SVR')
plt.legend()
plt.show()

'''
O resultado irá plotar um gráfico e as previsões
'''
