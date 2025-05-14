import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Cargar el dataset
df = pd.read_csv('hospital_data.csv')

# Dividir en Entrenamiento (70%) y el resto (30%)
train_data, temp_data = train_test_split(df, test_size=0.30, random_state=42)

# Dividir el resto (30%) en Validación (15%) y Prueba (15%)
val_data, test_data = train_test_split(temp_data, test_size=0.50, random_state=42)

# Guardar los conjuntos en archivos CSV
train_data.to_csv('train_data.csv', index=False)
val_data.to_csv('val_data.csv', index=False)
test_data.to_csv('test_data.csv', index=False)

print("Datos divididos y guardados.")

# Ahora, definimos las características y el objetivo
features = ['Edad', 'Consultas Previas', 'Medicamentos Prescritos'] # Características
target = 'Precio Consulta' # Objetivo

X_train = train_data[features]
y_train = train_data[target]

X_val = val_data[features]
y_val = val_data[target]

X_test = test_data[features]
y_test = test_data[target]

# Escalar las características numéricas
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# Crear y entrenar el modelo de regresión lineal
modelo = LinearRegression()
modelo.fit(X_train_scaled, y_train)

# Realizar las predicciones en el conjunto de prueba
y_pred = modelo.predict(X_test_scaled)

# Evaluar el modelo con el error cuadrático medio (MSE)
mse = mean_squared_error(y_test, y_pred)
print(f"Error cuadrático medio en el conjunto de prueba: {mse:.2f}")

# Mostrar las predicciones y valores reales para ver cómo se ajusta el modelo
for real, pred in zip(y_test, y_pred):
print(f"Real: {real}, Predicción: {pred:.2f}")

plt.figure(figsize=(10, 6))

# Graficar precios reales (rojo)
plt.plot(y_test.values, color='red', label='Precio Real', linestyle='--', marker='o')

# Graficar precios predichos (azul)
plt.plot(y_pred, color='blue', label='Precio Predicho', linestyle='-', marker='x')

# Agregar etiquetas y título
plt.title("Comparación entre Precios Reales y Predicciones")
plt.xlabel("Índice de Paciente")
plt.ylabel("Precio de la Consulta")
plt.legend()

# Mostrar la gráfica
plt.show()