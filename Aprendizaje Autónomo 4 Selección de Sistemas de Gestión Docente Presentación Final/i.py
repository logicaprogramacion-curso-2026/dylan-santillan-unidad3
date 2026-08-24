import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from opo import NeuralNetwork


df = pd.read_csv(r"data\winequality-white.csv", sep=";")


X = df.drop(columns=["quality"])
Y = df["quality"]


X = X / X.max()
Y = Y / 10.0


n_train = 500
n_test = 200


X_train = X[:n_train].values
Y_train = Y[:n_train].values.reshape(-1, 1)


X_test = X[n_train : n_train + n_test].values
Y_test = Y[n_train : n_train + n_test].values.reshape(-1, 1)


nn = NeuralNetwork()
nn.add_layer(num_neurons=8, inputs_size=X.shape[1])  
nn.add_layer(num_neurons=4, inputs_size=8) 
nn.add_layer(num_neurons=1, inputs_size=4)  

print("--- Iniciando Entrenamiento con Dataset de Vinos ---")

nn.train(X_train, Y_train, epochs=1000, learning_rate=0.1)


nn.save("wine_quality.json")


Y_pred_normalizado = nn.predict(X_test)


Y_test_real = np.round(Y_test * 10)
Y_pred_real = np.round(Y_pred_normalizado * 10)

print("\n--- Control de Dimensiones ---")
print(f"Forma de X_train (Entradas): {X_train.shape}")
print(f"Forma de Y_train (Salidas):  {Y_train.shape}")
print(f"Forma de X_test (Prueba):     {X_test.shape}")
print(f"Forma de Y_test (Prueba Y):   {Y_test.shape}")


error = abs(Y_test_real - Y_pred_real).mean()
print(f"\nError Absoluto Medio en Test: {error:.2f} puntos de calificación")

print("\n--- Muestra de Comparación (Primeros 3 vinos) ---")
for i in range(3):
    print(
        f"Vino #{i+1} -> Real: {Y_test_real[i][0]} | Predicho: {Y_pred_real[i][0]}"
    )


plt.figure(figsize=(8, 4))
plt.plot(nn.loss_list, color="red", linewidth=2, label="Pérdida (MSE)")
plt.title("Evolución del Error durante el Entrenamiento")
plt.xlabel("Épocas")
plt.ylabel("Valor de Pérdida")
plt.grid(True)
plt.legend()
plt.show()


plt.figure(figsize=(10, 4))
muestras_a_graficar = 50  
plt.plot(
    Y_test_real[:muestras_a_graficar], label="Calidad Real", color="blue", marker="o"
)
plt.plot(
    Y_pred_real[:muestras_a_graficar],
    label="Predicción IA",
    color="orange",
    linestyle="--",
    marker="x",
)
plt.title(
    f"Comparativa: Calidad Real vs Predicción (Primeros {muestras_a_graficar} vinos)"
)
plt.xlabel("Índice del Vino")
plt.ylabel("Calificación (0 - 10)")
plt.grid(True)
plt.legend()
plt.show()

print("\n--- Dataset de Vinos Cargado, Entrenado y Guardado con Éxito ---")