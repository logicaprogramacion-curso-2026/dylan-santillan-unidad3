import os
import json
import time
import numpy as np
import sqlite3
import torch
import torch.optim as optim
from dotenv import load_dotenv
from opo import NeuralNetwork
from evaluador_datos import (
    transformar_respuesta_a_vector,
    guardar_registro_estudiante,
    verificar_plagio,
    agregar_pregunta_dinamica,
    cargar_banco_preguntas,
    conectar_db,
    DB_FILE,
    normalizar_criterios,
    pregunta_existe,
    validar_id_pregunta,
    eliminar_pregunta_dinamica
)

load_dotenv()
API_KEY_GEMINI = os.environ.get("GEMINI_API_KEY")
API_KEY_GROQ = os.environ.get("GROQ_API_KEY")
API_KEY_COHERE = os.environ.get("API_KEY_COHERE")
API_KEY_NVIDIA = os.environ.get("API_KEY_NVIDIA")



def generar_retroalimentacion_ia(
    pregunta, respuesta_alumno, nota_predicha, criterios=None
):
    """Genera feedback y analiza los criterios reales de la pregunta."""
    criterios_limpios = normalizar_criterios(criterios or [])
    criterios_texto = "\n".join(
        f"Grupo {i + 1}: {', '.join(grupo)}"
        for i, grupo in enumerate(criterios_limpios)
    ) or "No se proporcionaron criterios específicos."

    prompt = f"""
Actúa como evaluador académico. Analiza la respuesta sin modificar la nota
producida por la red neuronal.

PREGUNTA: {pregunta}
RESPUESTA DEL ESTUDIANTE: {respuesta_alumno}
NOTA DE LA RED: {nota_predicha:.2f}/10
CRITERIOS:
{criterios_texto}

Devuelve únicamente JSON válido con esta estructura:
{{
  "analisis_error": "fortalezas y debilidades concretas",
  "retroalimentacion_alumno": "mensaje constructivo de máximo 3 líneas",
  "criterios": [
    {{
      "grupo": "nombre o resumen del grupo",
      "cumplido": true,
      "evidencia": "evidencia breve encontrada en la respuesta"
    }}
  ]
}}
Incluye un elemento por cada grupo de criterios recibido. Si no hay evidencia,
usa cumplido=false y explica brevemente por qué.
""".strip()

    errores = []
    if API_KEY_GEMINI:
        for intento in range(2):
            try:
                from google import genai
                client_gemini = genai.Client(api_key=API_KEY_GEMINI)
                response = client_gemini.models.generate_content(
                    model="gemini-2.5-flash",
                    contents=prompt,
                    config={"response_mime_type": "application/json"},
                )
                resultado = json.loads(response.text.strip())
                if not isinstance(resultado, dict):
                    raise ValueError("Gemini no devolvió un objeto JSON")
                resultado.setdefault("criterios", [])
                return resultado
            except Exception as exc:
                errores.append(f"Gemini: {str(exc)[:100]}")
                if "503" in str(exc) and intento == 0:
                    time.sleep(3)
                    continue
                break

    if API_KEY_GROQ:
        try:
            from groq import Groq

            response = Groq(api_key=API_KEY_GROQ).chat.completions.create(
                model="openai/gpt-oss-120b",
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
            )
            resultado = json.loads(response.choices[0].message.content)
            if not isinstance(resultado, dict):
                raise ValueError("Groq no devolvió un objeto JSON")
            resultado.setdefault("criterios", [])
            return resultado
        except Exception as exc:
            errores.append(f"Groq: {str(exc)[:100]}")

    return {
        "analisis_error": "No se pudo generar el análisis cualitativo.",
        "retroalimentacion_alumno": (
            "La nota fue calculada, pero el servicio de retroalimentación "
            "no estuvo disponible."
        ),
        "criterios": [
            {
                "grupo": f"Grupo {i + 1}: {', '.join(grupo)}",
                "cumplido": None,
                "evidencia": "Análisis no disponible",
            }
            for i, grupo in enumerate(criterios_limpios)
        ],
        "errores_tecnicos": errores,
    }


def menu_docente():
    print("\n--- PANEL DEL DOCENTE ---")
    id_p = None
    artefactos = []
    try:
        id_p = validar_id_pregunta(
            input("Asigne un código a la nueva pregunta: ").strip()
        )
        if pregunta_existe(id_p):
            raise ValueError(f"Ya existe una pregunta con el código {id_p}")

        preg_texto = input("Escriba la pregunta del examen:\n> ").strip()
        criterios = []
        for i in range(1, 6):
            entrada = input(
                f"Grupo {i} de criterios, separados por comas:\n> "
            ).strip()
            criterios.append(entrada.split(","))
        criterios = normalizar_criterios(criterios, exigir_cinco=True)

        from generar_dataset import generar_dataset_pregunta, ruta_dataset_pregunta
        from evaluador_entrenar import (
            entrenar_evaluador, ruta_modelo_pregunta,
            ruta_metadata_pregunta, GRAFICAS_DIR
        )
        from pathlib import Path

        artefactos = [
            Path(ruta_dataset_pregunta(id_p)),
            Path(ruta_modelo_pregunta(id_p)),
            Path(ruta_metadata_pregunta(id_p)),
            Path(GRAFICAS_DIR) / f"loss_pregunta_{id_p}.png",
        ]
        if any(ruta.exists() for ruta in artefactos):
            raise ValueError(
                "Existen archivos antiguos con ese código. Elimínalos o usa otro código."
            )

        print("\n🧪 Generando dataset de entrenamiento...")
        generar_dataset_pregunta(id_p, preg_texto, criterios, n_respuestas=50)

        print("\n🧠 Entrenando red neuronal...")
        _, metadata = entrenar_evaluador(id_p)

        agregar_pregunta_dinamica(id_p, preg_texto, criterios)
        print(
            "✅ Pregunta lista. Proveedor de embeddings: "
            f"{metadata['proveedor_embedding']}"
        )
    except Exception as exc:
        for ruta in artefactos:
            for candidata in (ruta, ruta.with_suffix(ruta.suffix + ".tmp"), ruta.with_suffix(".tmp")):
                try:
                    candidata.unlink(missing_ok=True)
                except OSError:
                    pass
        if id_p:
            try:
                eliminar_pregunta_dinamica(id_p)
            except Exception:
                pass
        print(f"❌ No se pudo crear la pregunta: {exc}")


def ver_historial():
    try:
        conn, cursor = conectar_db()
        cursor.execute("SELECT id, estudiante, id_pregunta, puntuacion_ia, alerta_plagio, fecha FROM historial_evaluaciones")
        registros = cursor.fetchall()
        conn.close()

        if not registros:
            print("❌ No hay evaluaciones guardadas aún.")
            return

        print("\n" + "="*75)
        print("                    HISTORIAL DE EVALUACIONES                    ")
        print("="*75)
        print(f"{'#':<4} {'Estudiante':<20} {'P.':<4} {'Nota':<12} {'Plagio':<10} {'Fecha'}")
        print("-"*75)

        for r in registros:
            print(f"{r[0]:<4} {r[1]:<20} {r[2]:<4} {r[3]:<12} {r[4]:<10} {r[5]}")

        print("="*75)
        print(f"Total de evaluaciones: {len(registros)}")

    except Exception as e:
        print(f"❌ Error al leer historial: {e}")


def buscar_historial():
    print("\n--- BUSCAR EN HISTORIAL ---")
    print("[1] Buscar por nombre de estudiante")
    print("[2] Buscar por número de pregunta")
    op = input("Seleccione: ").strip()

    try:
        conn, cursor = conectar_db()

        if op == "1":
            nombre = input("Ingrese el nombre del estudiante: ").strip()
            cursor.execute("""
                SELECT id, estudiante, id_pregunta, puntuacion_ia, alerta_plagio, fecha 
                FROM historial_evaluaciones 
                WHERE LOWER(estudiante) LIKE LOWER(?)
            """, (f"%{nombre}%",))

        elif op == "2":
            id_p = input("Ingrese el número de pregunta: ").strip()
            cursor.execute("""
                SELECT id, estudiante, id_pregunta, puntuacion_ia, alerta_plagio, fecha 
                FROM historial_evaluaciones 
                WHERE id_pregunta = ?
            """, (id_p,))
        else:
            print("❌ Opción inválida.")
            conn.close()
            return

        registros = cursor.fetchall()
        conn.close()

        if not registros:
            print("❌ No se encontraron resultados.")
            return

        print("\n" + "="*75)
        print(f"{'#':<4} {'Estudiante':<20} {'P.':<4} {'Nota':<12} {'Plagio':<10} {'Fecha'}")
        print("-"*75)
        for r in registros:
            print(f"{r[0]:<4} {r[1]:<20} {r[2]:<4} {r[3]:<12} {r[4]:<10} {r[5]}")
        print("="*75)
        print(f"Resultados encontrados: {len(registros)}")

    except Exception as e:
        print(f"❌ Error al buscar: {e}")


def estadisticas_por_pregunta():
    try:
        conn, cursor = conectar_db()
        cursor.execute("SELECT DISTINCT id_pregunta FROM historial_evaluaciones")
        preguntas = cursor.fetchall()

        if not preguntas:
            print("❌ No hay evaluaciones guardadas aún.")
            conn.close()
            return

        print("\n" + "="*60)
        print("           ESTADÍSTICAS POR PREGUNTA           ")
        print("="*60)

        for (id_p,) in preguntas:
            cursor.execute("""
                SELECT puntuacion_ia FROM historial_evaluaciones WHERE id_pregunta = ?
            """, (id_p,))
            notas_raw = cursor.fetchall()

            notes = []
            for (nota_str,) in notas_raw:
                try:
                    nota = float(nota_str.split("/")[0].strip())
                    notes.append(nota)
                except:
                    pass

            if not notes:
                continue

            promedio = sum(notes) / len(notes)
            aprobados = sum(1 for n in notes if n >= 6)
            reprobados = len(notes) - aprobados

            print(f"\n📋 Pregunta {id_p}:")
            print(f"   Total evaluados : {len(notes)}")
            print(f"   Promedio        : {promedio:.2f} / 10.00")
            print(f"   ✅ Aprobados    : {aprobados}")
            print(f"   ❌ Reprobados   : {reprobados}")

        conn.close()
        print("="*60)

    except Exception as e:
        print(f"❌ Error al calcular estadísticas: {e}")


def eliminar_historial():
    confirm = input("\n⚠️ ¿Estás seguro de ELIMINAR TODAS las evaluaciones? (SI para confirmar): ").strip()
    if confirm.upper() != "SI":
        print("❌ Operación cancelada.")
        return

    try:
        conn, cursor = conectar_db()
        cursor.execute("DELETE FROM historial_evaluaciones")
        conn.commit()
        conn.close()
        print("✅ Historial eliminado correctamente.")
    except Exception as e:
        print(f"❌ Error al eliminar historial: {e}")


def exportar_historial_excel():
    try:
        import pandas as pd
        conn, cursor = conectar_db()
        df = pd.read_sql_query("SELECT * FROM historial_evaluaciones", conn)
        conn.close()

        if df.empty:
            print("❌ La base de datos está vacía.")
            return

        nombre_archivo = "historial_evaluaciones.xlsx"
        df.to_excel(nombre_archivo, index=False)
        print(f"✅ Historial exportado a {nombre_archivo}")
    except Exception as e:
        print(f"❌ Error al exportar: {e}")


def corregir_evaluacion():
    print("\n--- CORRECCIÓN DE EVALUACIÓN POR DOCENTE ---")

    try:
        conn, cursor = conectar_db()
        cursor.execute("SELECT id, estudiante, id_pregunta, puntuacion_ia, fecha FROM historial_evaluaciones")
        registros = cursor.fetchall()
        conn.close()

        if not registros:
            print("❌ No hay evaluaciones para corregir.")
            return

        print("\n" + "="*75)
        print(f"{'#':<4} {'Estudiante':<20} {'P.':<4} {'Nota actual':<15} {'Fecha'}")
        print("-"*75)
        for r in registros:
            print(f"{r[0]:<4} {r[1]:<20} {r[2]:<4} {r[3]:<15} {r[4]}")
        print("="*75)

    except Exception as e:
        print(f"❌ Error: {e}")
        return

    try:
        id_registro = int(input("\nIngrese el # de la evaluación a corregir: ").strip())
        nota_correcta = float(input("Ingrese la nota correcta (0-10): ").strip())

        if not (0 <= nota_correcta <= 10):
            print("❌ La nota debe estar entre 0 y 10.")
            return

    except ValueError:
        print("❌ Valor inválido.")
        return

    try:
        conn, cursor = conectar_db()
        cursor.execute("""
            SELECT id_pregunta, vector_neurona FROM historial_evaluaciones WHERE id = ?
        """, (id_registro,))
        resultado = cursor.fetchone()
        conn.close()

        if not resultado:
            print("❌ No se encontró esa evaluación.")
            return

        id_pregunta = resultado[0]
        # El vector es semántico (768 dims, truncado al guardar)
        print("⚠️ La corrección manual usa el vector almacenado (primeros 5 valores)")

        # Actualizar solo la nota en la BD
        conn, cursor = conectar_db()
        cursor.execute("""
            UPDATE historial_evaluaciones SET puntuacion_ia = ? WHERE id = ?
        """, (f"{nota_correcta:.2f} / 10.00", id_registro))
        conn.commit()
        conn.close()

        print(f"✅ Nota actualizada a {nota_correcta:.2f} / 10.00")

    except Exception as e:
        print(f"❌ Error al corregir: {e}")


def evaluar_con_api():
    """Evalúa con el mismo proveedor utilizado durante el entrenamiento."""
    from evaluador_entrenar import (
        cargar_metadata_modelo,
        localizar_modelo_pregunta,
    )

    banco_completo = cargar_banco_preguntas()
    preguntas_actuales = {}
    for codigo, info in banco_completo.items():
        try:
            cargar_metadata_modelo(codigo)
            if localizar_modelo_pregunta(codigo).exists():
                preguntas_actuales[codigo] = info
        except Exception:
            continue

    if not preguntas_actuales:
        print("❌ No existen preguntas entrenadas y disponibles.")
        return

    nombre_alumno = input("Ingrese el nombre completo del estudiante: ").strip()
    nombre_alumno = nombre_alumno or "Estudiante Anónimo"

    print("\nPreguntas disponibles:")
    for id_p, info in preguntas_actuales.items():
        print(f" [{id_p}] - {info['pregunta']}")

    id_elegido = input("\nSeleccione la pregunta: ").strip()
    if id_elegido not in preguntas_actuales:
        print("❌ Pregunta no válida.")
        return

    info_pregunta = preguntas_actuales[id_elegido]
    try:
        metadata = cargar_metadata_modelo(id_elegido)
        proveedor_modelo = metadata["proveedor_embedding"]
        ruta_modelo = localizar_modelo_pregunta(id_elegido)
        nn = NeuralNetwork()
        nn.load(str(ruta_modelo))
    except Exception as exc:
        print(f"❌ Modelo no disponible o desactualizado: {exc}")
        return

    respuesta_alumno = input("Escriba la respuesta del examen: ").strip()
    if not respuesta_alumno:
        print("❌ La respuesta no puede estar vacía.")
        return

    es_plagio, porcentaje, sospechoso_de, detalle = verificar_plagio(
        respuesta_alumno, id_elegido
    )
    plagio_string = "Ninguno"
    if es_plagio:
        plagio_string = (
            f"Sospecha alta ({porcentaje:.1f}% con {sospechoso_de}) — "
            f"vocabulario: {detalle.get('score_vocabulario', 0)}%, "
            f"frases: {detalle.get('score_frases', 0)}%"
        )

    try:
        vector_entradas, proveedor_embedding = transformar_respuesta_a_vector(
            respuesta_alumno,
            proveedor_preferido=proveedor_modelo,
            permitir_fallback=False,
        )
    except Exception as exc:
        print(f"❌ No se pudo analizar la respuesta: {exc}")
        return

    nota_final = float(nn.predict(vector_entradas).reshape(-1)[0]) * 10
    nota_final = max(0.0, min(10.0, nota_final))
    feedback = generar_retroalimentacion_ia(
        info_pregunta["pregunta"],
        respuesta_alumno,
        nota_final,
        info_pregunta.get("criterios", []),
    )

    print("\n" + "=" * 60)
    print(f"ESTUDIANTE: {nombre_alumno}")
    print(f"NOTA: {nota_final:.2f}/10")
    print(f"PROVEEDOR: {proveedor_embedding}")
    print(f"PLAGIO: {plagio_string}")
    print(f"ANÁLISIS: {feedback.get('analisis_error', 'N/A')}")
    print(f"FEEDBACK: {feedback.get('retroalimentacion_alumno', 'N/A')}")
    for criterio in feedback.get("criterios", []):
        estado = criterio.get("cumplido")
        etiqueta = "Sí" if estado is True else "No" if estado is False else "Sin analizar"
        print(f"- {criterio.get('grupo', 'Criterio')}: {etiqueta}")
    print("=" * 60)

    guardar_registro_estudiante(
        nombre_estudiante=nombre_alumno,
        id_pregunta=id_elegido,
        pregunta_texto=info_pregunta["pregunta"],
        respuesta_alumno=respuesta_alumno,
        vector_generado=vector_entradas,
        puntuacion_ia=f"{nota_final:.2f} / 10.00",
        feedback_ia=feedback.get("retroalimentacion_alumno", ""),
        plagio_info=plagio_string,
        proveedor_embedding=proveedor_embedding,
    )


def panel_principal():
    while True:
        print("\n" + "="*50)
        print("   MENÚ PRINCIPAL - SISTEMA DE EVALUACIÓN IA   ")
        print("="*50)
        print("[1] Panel del Docente (Agregar preguntas)")
        print("[2] Evaluar Respuesta (Red Neuronal + LLM)")
        print("[3] Ver Historial de Evaluaciones")
        print("[4] Buscar en Historial")
        print("[5] Estadísticas por Pregunta")
        print("[6] Exportar Historial a Excel")
        print("[7] Eliminar Historial")
        print("[8] Corregir Evaluación")
        print("[9] Salir")
        print("="*50)
        
        op = input("Seleccione una opción: ").strip()

        if op == "1":
            menu_docente()
        elif op == "2":
            evaluar_con_api()
        elif op == "3":
            ver_historial()
        elif op == "4":
            buscar_historial()
        elif op == "5":
            estadisticas_por_pregunta()
        elif op == "6":
            exportar_historial_excel()
        elif op == "7":
            eliminar_historial()
        elif op == "8":
            corregir_evaluacion()
        elif op == "9":
            print("\n👋 ¡Hasta luego!")
            break
        else:
            print("❌ Opción inválida.")


if __name__ == "__main__":
    panel_principal()