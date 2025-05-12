import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import numpy as np

# Cargar los conjuntos de datos
train_data = pd.read_csv('train_data.csv')
val_data = pd.read_csv('val_data.csv')
test_data = pd.read_csv('test_data.csv')

# Crear una columna objetivo artificial para pruebas (si no tienes una)
np.random.seed(42)  # Para reproducibilidad
train_data['target'] = np.random.choice([0, 1], size=len(train_data))
val_data['target'] = np.random.choice([0, 1], size=len(val_data))
test_data['target'] = np.random.choice([0, 1], size=len(test_data))

# Verificar las columnas del DataFrame cargado
print("Columnas en train_data:", train_data.columns)
print("Columnas en val_data:", val_data.columns)
print("Columnas en test_data:", test_data.columns)

# Usar 'target' como la variable objetivo
X_train = train_data.drop(columns=['target'])
y_train = train_data['target']

X_val = val_data.drop(columns=['target'])
y_val = val_data['target']

X_test = test_data.drop(columns=['target'])
y_test = test_data['target']

# Definir el modelo y los hiperparámetros a buscar
model = LogisticRegression(max_iter=1000)






#Hiperparámetros
param_grid = {
    'C': [0.01, 0.1, 1, 10],  # Hiperparámetro de regularización
    'solver': ['lbfgs', 'liblinear']  # Solvers posibles
}

# Usar GridSearchCV para encontrar la mejor combinación de parámetros
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, n_jobs=-1)
grid_search.fit(X_train, y_train)

# Ver los mejores parámetros encontrados
print("Mejores parámetros encontrados:", grid_search.best_params_)

# Usar el mejor modelo encontrado
best_model = grid_search.best_estimator_

# Evaluar el modelo ajustado en el conjunto de prueba
y_test_pred = best_model.predict(X_test)
test_accuracy = accuracy_score(y_test, y_test_pred)
print(f"Precisión en conjunto de prueba (ajustado): {test_accuracy * 100:.2f}%")

# Imprimir un reporte completo de clasificación para el conjunto de prueba
print("\nReporte de clasificación para conjunto de prueba (ajustado):")
print(classification_report(y_test, y_test_pred))





# Gráficos
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, confusion_matrix, precision_recall_curve

# Curva ROC y AUC
fpr, tpr, thresholds = roc_curve(y_test, best_model.predict_proba(X_test)[:, 1])
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(10, 6))
plt.plot(fpr, tpr, color='blue', label=f'Curva ROC (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Tasa de Falsos Positivos (FPR)')
plt.ylabel('Tasa de Verdaderos Positivos (TPR)')
plt.title('Curva ROC')
plt.legend(loc='lower right')
plt.show()

# Matriz de confusión
cm = confusion_matrix(y_test, y_test_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Clase 0', 'Clase 1'], yticklabels=['Clase 0', 'Clase 1'])
plt.xlabel('Predicción')
plt.ylabel('Verdadera')
plt.title('Matriz de Confusión')
plt.show()

# Curva Precision-Recall
precision, recall, _ = precision_recall_curve(y_test, best_model.predict_proba(X_test)[:, 1])

plt.figure(figsize=(10, 6))
plt.plot(recall, precision, color='green')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Curva Precision-Recall')
plt.show()

# Histograma de las predicciones (probabilidades)
plt.figure(figsize=(10, 6))
plt.hist(best_model.predict_proba(X_test)[:, 1], bins=10, color='orange', alpha=0.7)
plt.title('Histograma de Probabilidades Predichas')
plt.xlabel('Probabilidad Predicha para Clase 1')
plt.ylabel('Frecuencia')
plt.show()
