import csv
import json
import os
from pathlib import Path
from typing import Iterable

from dotenv import load_dotenv
from evaluador_datos import (
    cargar_banco_preguntas,
    normalizar_criterios,
    validar_id_pregunta,
)

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)


def ruta_dataset_pregunta(id_pregunta):
    codigo = validar_id_pregunta(id_pregunta)
    return DATA_DIR / f"dataset_pregunta_{codigo}.csv"


def _obtener_cliente_groq():
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError(
            "Falta GROQ_API_KEY en el archivo .env. No se puede generar "
            "el dataset sintético."
        )
    try:
        from groq import Groq
    except ImportError as exc:
        raise RuntimeError("Falta instalar groq: pip install groq") from exc
    return Groq(api_key=api_key)


def _construir_prompt(pregunta, criterios, cantidad):
    criterios_texto = "\n".join(
        f"- Grupo {indice + 1} (2 puntos): {', '.join(grupo)}"
        for indice, grupo in enumerate(criterios)
    )
    return f"""
Eres un generador de datos académicos para entrenar un prototipo de evaluación.
Genera exactamente {cantidad} respuestas distintas de estudiantes para la
pregunta indicada y asigna una nota de 0 a 10 según los cinco grupos de
criterios. Cada grupo vale 2 puntos.

PREGUNTA:
{pregunta}

CRITERIOS:
{criterios_texto}

REQUISITOS:
- Incluye respuestas excelentes, buenas, regulares, malas y muy malas.
- La nota debe corresponder a los criterios realmente presentes.
- Escribe respuestas naturales y variadas; algunas pueden contener errores.
- No repitas respuestas ni cambies la pregunta.
- La nota debe ser numérica y estar entre 0 y 10.

Responde únicamente con JSON válido:
{{
  "dataset": [
    {{"respuesta": "texto", "nota": 8.5}}
  ]
}}
""".strip()


def _validar_registros(registros: Iterable[dict]):
    resultado = []
    vistos = set()

    if not isinstance(registros, list):
        raise ValueError("La respuesta de Groq no contiene una lista 'dataset'")

    for item in registros:
        if not isinstance(item, dict):
            continue

        respuesta = " ".join(str(item.get("respuesta", "")).strip().split())
        try:
            nota = float(item.get("nota"))
        except (TypeError, ValueError):
            continue

        if len(respuesta) < 3 or not 0 <= nota <= 10:
            continue

        clave = respuesta.casefold()
        if clave in vistos:
            continue
        vistos.add(clave)
        resultado.append({"respuesta": respuesta, "nota": round(nota, 2)})

    return resultado


def generar_respuestas_con_groq(pregunta, criterios, n_respuestas=50):
    """Genera y valida un conjunto de respuestas sintéticas."""
    pregunta = " ".join(str(pregunta).strip().split())
    if len(pregunta) < 10:
        raise ValueError("La pregunta debe tener al menos 10 caracteres")

    criterios_limpios = normalizar_criterios(criterios, exigir_cinco=True)
    if n_respuestas < 20:
        raise ValueError("Se requieren al menos 20 respuestas para entrenar")

    cliente = _obtener_cliente_groq()
    acumulados = []
    vistos = set()

    for intento in range(1, 4):
        faltantes = n_respuestas - len(acumulados)
        if faltantes <= 0:
            break

        prompt = _construir_prompt(pregunta, criterios_limpios, faltantes)
        completion = cliente.chat.completions.create(
            model="openai/gpt-oss-120b",
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0.8,
        )

        try:
            contenido = completion.choices[0].message.content
            payload = json.loads(contenido)
        except (AttributeError, IndexError, json.JSONDecodeError) as exc:
            raise RuntimeError(
                "Groq devolvió una respuesta que no pudo interpretarse como JSON"
            ) from exc

        nuevos = _validar_registros(payload.get("dataset", []))
        for registro in nuevos:
            clave = registro["respuesta"].casefold()
            if clave not in vistos:
                vistos.add(clave)
                acumulados.append(registro)
                if len(acumulados) == n_respuestas:
                    break

        print(
            f"🧪 Generación {intento}/3: {len(acumulados)}/{n_respuestas} "
            "respuestas válidas"
        )

    if len(acumulados) < max(20, int(n_respuestas * 0.8)):
        raise RuntimeError(
            f"Solo se obtuvieron {len(acumulados)} respuestas válidas de "
            f"{n_respuestas}. No se entrenará con un dataset insuficiente."
        )

    return acumulados[:n_respuestas]


def guardar_dataset_csv(id_pregunta, dataset):
    """Guarda el CSV de forma atómica para evitar archivos incompletos."""
    archivo = ruta_dataset_pregunta(id_pregunta)
    registros = _validar_registros(list(dataset))
    if len(registros) < 20:
        raise ValueError("El dataset debe contener al menos 20 registros válidos")

    temporal = archivo.with_suffix(".csv.tmp")
    with temporal.open("w", newline="", encoding="utf-8") as salida:
        writer = csv.DictWriter(salida, fieldnames=["respuesta", "nota"])
        writer.writeheader()
        writer.writerows(registros)
    temporal.replace(archivo)

    print(f"✅ Dataset guardado en {archivo} con {len(registros)} ejemplos")
    return str(archivo)


def generar_dataset_pregunta(
    id_pregunta, pregunta, criterios, n_respuestas=50
):
    """Genera y guarda el dataset de una sola pregunta."""
    codigo = validar_id_pregunta(id_pregunta)
    dataset = generar_respuestas_con_groq(
        pregunta=pregunta,
        criterios=criterios,
        n_respuestas=n_respuestas,
    )
    return guardar_dataset_csv(codigo, dataset)


def generar_datasets_todos(n_respuestas=50):
    """Genera datasets para todas las preguntas que tengan cinco criterios."""
    banco = cargar_banco_preguntas()
    generados = []

    for id_p, info in banco.items():
        try:
            criterios = normalizar_criterios(
                info.get("criterios", []), exigir_cinco=True
            )
            archivo = generar_dataset_pregunta(
                id_p,
                info.get("pregunta", ""),
                criterios,
                n_respuestas=n_respuestas,
            )
            generados.append(archivo)
        except Exception as exc:
            print(f"❌ No se generó el dataset de la pregunta {id_p}: {exc}")

    print(f"\n✅ Datasets generados correctamente: {len(generados)}")
    return generados


if __name__ == "__main__":
    generar_datasets_todos()
