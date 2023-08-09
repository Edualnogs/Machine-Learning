'''
Aqui está um esqueleto de código para criar um modelo SVM para classificação usando a biblioteca scikit-learn:

Algoritmo Support Vector Machines (SVM)
'''

# Importar as bibliotecas necessárias
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Carregar um conjunto de dados de exemplo
dataset = datasets.load_iris()
X = dataset.data
y = dataset.target

# Dividir os dados em conjuntos de treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Criar o modelo SVM
svm_model = SVC(kernel='linear')  # Você pode experimentar com diferentes kernels (linear, poly, rbf, etc.)

# Treinar o modelo usando os dados de treinamento
svm_model.fit(X_train, y_train)

# Realizar previsões nos dados de teste
y_pred = svm_model.predict(X_test)

# Avaliar o desempenho do modelo
accuracy = accuracy_score(y_test, y_pred)
print(f"Acurácia do modelo: {accuracy:.2f}")

# Exibir outras métricas de avaliação
print(classification_report(y_test, y_pred))

# Exibir matriz de confusão
conf_matrix = confusion_matrix(y_test, y_pred)
print("Matriz de Confusão:")
print(conf_matrix)
