import json
import os
import matplotlib.pyplot as plt
import numpy as np
from opo import NeuralNetwork  # <--- Usamos TU motor de IA


# ========================================================
# LOAD DATA FROM DATA1
# ========================================================
def load_data():
    # Verificamos que los archivos que acabas de generar existan en data1
    assert os.path.exists(
        "data1/train.bin"
    ), "❌ Can't find 'data1/train.bin'. Did you run step1?"

    # Cargamos el vocabulario de letras
    with open("data1/vocabulary.json", "r", encoding="utf-8") as f:
        vocab = json.load(f)

    char_to_idx = vocab["char_to_idx"]
    idx_to_char = {int(k): v for k, v in vocab["idx_to_char"].items()}
    vocab_size = vocab["vocab_size"]

    # Cargamos los datos binarios del Quijote
    train_data = np.fromfile("data1/train.bin", dtype=np.uint16)
    val_data = np.fromfile("data1/val.bin", dtype=np.uint16)

    print(f"\n📦 DATA LOADED FROM DATA1:")
    print(f"  Vocabulary size : {vocab_size} unique characters")
    print(f"  Train tokens    : {len(train_data):,} letters")
    print(f"  Validation      : {len(val_data):,} letters")

    return train_data, val_data, char_to_idx, idx_to_char, vocab_size


# ========================================================
# RUN TRAINING
# ========================================================
if __name__ == "__main__":
    # 1. Traemos los datos procesados del Quijote
    train_data, val_data, char_to_idx, idx_to_char, vocab_size = load_data()

    # 2. Creamos muestras de entrenamiento (X: Contexto -> Y: Siguiente Letra)
    context_size = 5  # La IA lee 5 letras seguidas para adivinar la 6ta
    X_list = []
    Y_list = []

    # Tomamos los primeros 2,000 fragmentos para no saturar la memoria
    for i in range(min(2000, len(train_data) - context_size - 1)):
        X_list.append(train_data[i : i + context_size])
        Y_list.append(train_data[i + context_size])

    X_train = np.array(X_list)
    Y_train = np.array(Y_list).reshape(-1, 1)

    # Normalizamos dividiendo por el vocabulario para que la sigmoide funcione (0 a 1)
    X_train_norm = X_train / float(vocab_size)
    Y_train_norm = Y_train / float(vocab_size)

    # 3. Configuración de tu arquitectura de red (opo.py)
    nn = NeuralNetwork()
    nn.add_layer(num_neurons=16, inputs_size=context_size)  # Capa Oculta 1
    nn.add_layer(num_neurons=8, inputs_size=16)  # Capa Oculta 2
    nn.add_layer(num_neurons=1, inputs_size=8)  # Capa de Salida (Predicción)

    # 4. Iniciar el entrenamiento
    print("\n--- Iniciando Entrenamiento del Modelo de Texto ---")
    nn.train(X_train_norm, Y_train_norm, epochs=1500, learning_rate=0.1)

    # 5. Guardar el cerebro de texto calibrado
    nn.save("text_generator.json")
    print("\n✅ ¡Modelo de texto guardado como 'text_generator.json'!")

    # 6. Mostrar gráfica de pérdida
    plt.figure(figsize=(8, 4))
    plt.plot(nn.loss_list, color="blue", linewidth=2, label="Loss (MSE)")
    plt.title("Evolución del Aprendizaje (Texto)")
    plt.xlabel("Épocas")
    plt.ylabel("Error")
    plt.grid(True)
    plt.legend()
    plt.show()