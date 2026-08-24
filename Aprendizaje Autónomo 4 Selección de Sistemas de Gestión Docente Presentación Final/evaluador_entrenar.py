import hashlib
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from evaluador_datos import (
    transformar_respuesta_a_vector,
    validar_id_pregunta,
)
from generar_dataset import ruta_dataset_pregunta
from opo import NeuralNetwork

BASE_DIR = Path(__file__).resolve().parent
MODELOS_DIR = BASE_DIR / "modelos"
GRAFICAS_DIR = BASE_DIR / "graficas"
MODELOS_DIR.mkdir(parents=True, exist_ok=True)
GRAFICAS_DIR.mkdir(parents=True, exist_ok=True)

PROVEEDORES_PERMITIDOS = [
    "Gemini",
    "Cohere",
    "NVIDIA",
    "Vocabulario local",
]


def ruta_modelo_pregunta(id_pregunta):
    codigo = validar_id_pregunta(id_pregunta)
    return MODELOS_DIR / f"cerebro_pregunta_{codigo}.json"


def ruta_metadata_pregunta(id_pregunta):
    codigo = validar_id_pregunta(id_pregunta)
    return MODELOS_DIR / f"metadata_pregunta_{codigo}.json"


def localizar_modelo_pregunta(id_pregunta):
    """Localiza el modelo nuevo o uno legado guardado en la raíz."""
    ruta_nueva = ruta_modelo_pregunta(id_pregunta)
    if ruta_nueva.exists():
        return ruta_nueva

    ruta_legada = BASE_DIR / f"cerebro_pregunta_{validar_id_pregunta(id_pregunta)}.json"
    if ruta_legada.exists():
        return ruta_legada
    return ruta_nueva


def cargar_metadata_modelo(id_pregunta):
    ruta = ruta_metadata_pregunta(id_pregunta)
    if not ruta.exists():
        raise FileNotFoundError(
            f"No existe {ruta.name}. Debes volver a entrenar esta pregunta "
            "para registrar el proveedor de embeddings."
        )
    with ruta.open("r", encoding="utf-8") as archivo:
        metadata = json.load(archivo)

    proveedor = metadata.get("proveedor_embedding")
    if proveedor not in PROVEEDORES_PERMITIDOS:
        raise ValueError("La metadata contiene un proveedor de embedding inválido")
    return metadata


def graficar_loss(modelo, id_pregunta):
    """Guarda la curva de aprendizaje en la carpeta ``graficas``."""
    codigo = validar_id_pregunta(id_pregunta)
    plt.figure(figsize=(10, 5))
    plt.plot(modelo.loss_list, linewidth=1.5, label="Train Loss")
    if modelo.val_loss_list:
        plt.plot(
            modelo.val_loss_list,
            linewidth=1.5,
            label="Validation Loss",
            linestyle="--",
        )
    plt.title(f"Curva de aprendizaje - Pregunta {codigo}")
    plt.xlabel("Épocas")
    plt.ylabel("Loss (MSE)")
    plt.legend()
    plt.grid(True, linestyle=":", alpha=0.6)
    ruta = GRAFICAS_DIR / f"loss_pregunta_{codigo}.png"
    plt.savefig(ruta, bbox_inches="tight")
    plt.close()
    print(f"📊 Gráfica guardada: {ruta}")
    return str(ruta)


def _seleccionar_proveedor(texto_prueba):
    errores = []
    for proveedor in PROVEEDORES_PERMITIDOS:
        try:
            tensor, nombre = transformar_respuesta_a_vector(
                texto_prueba,
                proveedor_preferido=proveedor,
                permitir_fallback=False,
            )
            if tensor.shape != (1, 768):
                raise ValueError(f"dimensión inesperada: {tuple(tensor.shape)}")
            print(f"✅ Proveedor seleccionado para todo el modelo: {nombre}")
            return nombre
        except Exception as exc:
            errores.append(f"{proveedor}: {str(exc)[:120]}")
            print(f"⚠️ {proveedor} no disponible para entrenar: {str(exc)[:100]}")

    raise RuntimeError(
        "No existe un proveedor de embeddings disponible. " + " | ".join(errores)
    )


def _generar_embedding_fijo(texto, proveedor, reintentos=2):
    ultimo_error = None
    for intento in range(1, reintentos + 1):
        try:
            tensor, nombre = transformar_respuesta_a_vector(
                texto,
                proveedor_preferido=proveedor,
                permitir_fallback=False,
            )
            return tensor.squeeze(0).detach().cpu().numpy(), nombre
        except Exception as exc:
            ultimo_error = exc
            print(
                f"      ⚠️ Intento {intento}/{reintentos} con {proveedor}: "
                f"{str(exc)[:100]}"
            )
    raise RuntimeError(
        f"El proveedor {proveedor} falló durante el entrenamiento: {ultimo_error}"
    )


def cargar_dataset(id_pregunta, proveedor=None):
    """Carga el CSV y crea todos sus embeddings con un único proveedor."""
    archivo = ruta_dataset_pregunta(id_pregunta)
    if not archivo.exists():
        raise FileNotFoundError(
            f"No existe {archivo}. Genera primero el dataset de la pregunta."
        )

    df = pd.read_csv(archivo)
    columnas = {str(c).strip().lower() for c in df.columns}
    if not {"respuesta", "nota"}.issubset(columnas):
        raise ValueError("El CSV debe contener las columnas respuesta y nota")

    df = df[["respuesta", "nota"]].copy()
    df["respuesta"] = df["respuesta"].fillna("").astype(str).str.strip()
    df["nota"] = pd.to_numeric(df["nota"], errors="coerce")
    df = df[(df["respuesta"].str.len() >= 3) & df["nota"].between(0, 10)]
    df = df.drop_duplicates(subset=["respuesta"]).reset_index(drop=True)

    if len(df) < 20:
        raise ValueError(
            f"El dataset solo tiene {len(df)} registros válidos; se requieren 20."
        )

    proveedor_elegido = proveedor or _seleccionar_proveedor(df.iloc[0]["respuesta"])
    X_list, Y_list = [], []

    for indice, row in df.iterrows():
        print(f"   [{indice + 1}/{len(df)}] Embedding con {proveedor_elegido}...")
        embedding, proveedor_real = _generar_embedding_fijo(
            row["respuesta"], proveedor_elegido
        )
        if proveedor_real != proveedor_elegido:
            raise RuntimeError("El proveedor cambió durante el entrenamiento")
        X_list.append(embedding)
        Y_list.append([float(row["nota"]) / 10.0])

    X = np.asarray(X_list, dtype=np.float32)
    Y = np.asarray(Y_list, dtype=np.float32)
    if X.ndim != 2 or X.shape[1] != 768:
        raise ValueError(f"Matriz de embeddings inválida: {X.shape}")
    return X, Y, proveedor_elegido, archivo


def dividir_dataset(X, Y, test_ratio=0.15, seed=42):
    if len(X) != len(Y) or len(X) < 20:
        raise ValueError("Dataset insuficiente o desalineado")

    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(X))
    n_test = max(3, int(round(len(X) * test_ratio)))
    test_idx = indices[:n_test]
    train_idx = indices[n_test:]
    return X[train_idx], Y[train_idx], X[test_idx], Y[test_idx]


def calcular_metricas(modelo, X, Y, umbral=0.6):
    """Calcula métricas de regresión sobre datos no usados al entrenar."""
    pred = np.asarray(modelo.predict(X), dtype=np.float32).reshape(-1)
    real = np.asarray(Y, dtype=np.float32).reshape(-1)

    errores = real - pred
    mse = float(np.mean(errores**2))
    mae = float(np.mean(np.abs(errores)))
    rmse = float(math.sqrt(mse))
    denominador = float(np.sum((real - np.mean(real)) ** 2))
    r2 = 1.0 - float(np.sum(errores**2)) / denominador if denominador > 0 else 0.0
    dentro_un_punto = float(np.mean(np.abs(errores) <= 0.1) * 100)
    exactitud_aprobacion = float(
        np.mean((pred >= umbral) == (real >= umbral)) * 100
    )

    metricas = {
        "mse": round(mse, 6),
        "mae_puntos": round(mae * 10, 4),
        "rmse_puntos": round(rmse * 10, 4),
        "r2": round(r2, 4),
        "dentro_de_1_punto_pct": round(dentro_un_punto, 2),
        "exactitud_aprobacion_pct": round(exactitud_aprobacion, 2),
        "muestras_prueba": int(len(real)),
    }

    print("\n" + "=" * 56)
    print("       MÉTRICAS EN CONJUNTO DE PRUEBA")
    print("=" * 56)
    print(f"  MAE                  : {metricas['mae_puntos']:.4f} puntos")
    print(f"  RMSE                 : {metricas['rmse_puntos']:.4f} puntos")
    print(f"  R²                   : {metricas['r2']:.4f}")
    print(f"  Dentro de ±1 punto   : {metricas['dentro_de_1_punto_pct']:.2f}%")
    print(f"  Exactitud aprobación : {metricas['exactitud_aprobacion_pct']:.2f}%")
    print("=" * 56)
    return metricas



def _ruta_portable(ruta):
    ruta = Path(ruta).resolve()
    try:
        return str(ruta.relative_to(BASE_DIR.resolve()))
    except ValueError:
        return str(ruta)

def _hash_archivo(ruta):
    digest = hashlib.sha256()
    with Path(ruta).open("rb") as archivo:
        for bloque in iter(lambda: archivo.read(65536), b""):
            digest.update(bloque)
    return digest.hexdigest()


def entrenar_evaluador(
    id_pregunta,
    epochs=1200,
    lr=0.001,
    proveedor=None,
    seed=42,
):
    """Entrena un evaluador y guarda modelo, proveedor y métricas."""
    codigo = validar_id_pregunta(id_pregunta)
    np.random.seed(seed)
    torch.manual_seed(seed)

    print("\n" + "=" * 60)
    print(f"  🚀 Entrenando IA semántica para pregunta {codigo}")
    print("=" * 60)

    X, Y, proveedor_elegido, archivo_dataset = cargar_dataset(
        codigo, proveedor=proveedor
    )
    X_train, Y_train, X_test, Y_test = dividir_dataset(X, Y, seed=seed)

    modelo = NeuralNetwork()
    modelo.add_layer(256, inputs_size=X.shape[1], activation="relu", dropout_rate=0.2)
    modelo.add_layer(128, activation="relu", dropout_rate=0.2)
    modelo.add_layer(64, activation="relu", dropout_rate=0.1)
    modelo.add_layer(1, activation="sigmoid")

    batch_size = max(2, min(8, len(X_train)))
    modelo.train_model(
        X_train,
        Y_train,
        learning_rate=lr,
        epochs=epochs,
        patience=min(200, max(40, epochs // 5)),
        lr_factor=0.5,
        lr_patience=min(60, max(15, epochs // 12)),
        batch_size=batch_size,
        val_split=0.18,
        clip_norm=1.0,
    )

    metricas = calcular_metricas(modelo, X_test, Y_test)
    ruta_modelo = ruta_modelo_pregunta(codigo)
    modelo.save(str(ruta_modelo))
    ruta_grafica = graficar_loss(modelo, codigo)

    metadata = {
        "id_pregunta": codigo,
        "proveedor_embedding": proveedor_elegido,
        "dimensiones": int(X.shape[1]),
        "muestras_totales": int(len(X)),
        "muestras_entrenamiento_validacion": int(len(X_train)),
        "muestras_prueba": int(len(X_test)),
        "dataset": _ruta_portable(archivo_dataset),
        "dataset_sha256": _hash_archivo(archivo_dataset),
        "modelo": _ruta_portable(ruta_modelo),
        "grafica": _ruta_portable(ruta_grafica),
        "metricas_prueba": metricas,
        "fecha_entrenamiento_utc": datetime.now(timezone.utc).isoformat(),
        "seed": seed,
    }

    ruta_metadata = ruta_metadata_pregunta(codigo)
    temporal = ruta_metadata.with_suffix(".json.tmp")
    with temporal.open("w", encoding="utf-8") as archivo:
        json.dump(metadata, archivo, ensure_ascii=False, indent=4)
    temporal.replace(ruta_metadata)

    print(f"✅ Modelo guardado: {ruta_modelo}")
    print(f"✅ Metadata guardada: {ruta_metadata}")
    return modelo, metadata


if __name__ == "__main__":
    from evaluador_datos import cargar_banco_preguntas

    banco = cargar_banco_preguntas()
    for id_p in banco:
        try:
            entrenar_evaluador(id_p)
        except Exception as exc:
            print(f"❌ Pregunta {id_p}: {exc}")
