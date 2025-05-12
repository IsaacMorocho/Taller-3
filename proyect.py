import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

# 1. Leer los tres archivos CSV
# Suponiendo que los archivos están en el mismo directorio
df1 = pd.read_csv('train_data.csv')
df2 = pd.read_csv('test_data.csv')
df3 = pd.read_csv('val_data.csv')

#Concatenar los DataFrames
df = pd.concat([df1, df2, df3], ignore_index=True)

#Preprocesar los datos
# Eliminar valores nulos (si es necesario)
df = df.dropna()

#Filtrar los precios menores a 10,000
df = df[df['Precio Consulta'] < 10000]

#Dividir en características (X) y objetivo (y)
#Asumimos que 'precio' es la columna objetivo y el resto son características
X = df.drop('Precio Consulta', axis=1)
y = df['Precio Consulta']

#Dividir en entrenamiento (70%), validación (15%) y prueba (15%)
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_valid, X_test, y_valid, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

#Crear el modelo de regresión lineal
model = LinearRegression()
#Entrenar el modelo
model.fit(X_train, y_train)

#Hacer predicciones en el conjunto de validación
y_valid_pred = model.predict(X_valid)

# Evaluar el modelo en el conjunto de validación
mse_valid = mean_squared_error(y_valid, y_valid_pred)
rmse_valid = np.sqrt(mse_valid)
r2_valid = r2_score(y_valid, y_valid_pred)

print("Evaluación en el conjunto de validación:")
print(f"RMSE: {rmse_valid}")
print(f"R^2: {r2_valid}")

# hacer predicciones en el conjunto de prueba
y_test_pred = model.predict(X_test)

# Evaluar el modelo en el conjunto de prueba
mse_test = mean_squared_error(y_test, y_test_pred)
rmse_test = np.sqrt(mse_test)
r2_test = r2_score(y_test, y_test_pred)

print("\nEvaluación en el conjunto de prueba:")
print(f"RMSE: {rmse_test}") # El RMSE mide la diferencia promedio entre los valores reales y los predichos, pero con más peso en los errores grandes porque está basado en el cuadrado de las diferencias.
print(f"R^2: {r2_test}")  
#R² cercano a 1: El modelo se ajusta muy bien a los datos.
#R² cercano a 0: El modelo no explica casi nada de la variabilidad en los datos.

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score

# Aquí asumimos que ya tienes el modelo entrenado y las predicciones (y_valid_pred, y_test_pred)
# Vamos a usar las predicciones en el conjunto de validación y prueba

# Calcular el RMSE para los conjuntos de validación y prueba
rmse_valid = np.sqrt(mean_squared_error(y_valid, y_valid_pred))
rmse_test = np.sqrt(mean_squared_error(y_test, y_test_pred))

# Calcular el R² para los conjuntos de validación y prueba
r2_valid = r2_score(y_valid, y_valid_pred)
r2_test = r2_score(y_test, y_test_pred)

# Crear listas para los valores de RMSE y R²
rmse_values = [rmse_valid, rmse_test]
r2_values = [r2_valid, r2_test]

# Crear etiquetas para los conjuntos
labels = ['Validación', 'Prueba']

# Configurar el gráfico
fig, axes = plt.subplots(1, 2, figsize=(12, 6))

# Graficar el RMSE
sns.barplot(x=labels, y=rmse_values, ax=axes[0], palette='Blues')
axes[0].set_title('RMSE para Validación y Prueba')
axes[0].set_ylabel('RMSE')

# Graficar el R²
sns.barplot(x=labels, y=r2_values, ax=axes[1], palette='Greens')
axes[1].set_title('R² para Validación y Prueba')
axes[1].set_ylabel('R²')

# Mostrar el gráfico
plt.tight_layout()
plt.show()
