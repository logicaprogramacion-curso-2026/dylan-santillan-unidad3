import sqlite3
import numpy as np
from evaluador_datos import transformar_respuesta_a_vector, guardar_registro_estudiante

def probar_sistema():
    print("=" * 60)
    print("🚀 INICIANDO PRUEBA DE EMBEDDINGS Y SQLITE")
    print("=" * 60)

    
    criterios_examen = [
        ["bucle", "ciclo", "repeticion"],
        ["for", "while"],
        ["condicion", "limite"]
    ]

    
    respuesta_alumno = "Se ejecuta una estructura iterativa usando un while hasta que se cumpla el parametro establecido."
    
    print(f"\n📝 Respuesta del alumno a evaluar:\n\"{respuesta_alumno}\"")
    print("\n📋 Criterios que busca el docente:", criterios_examen)

    
    vector_resultado = transformar_respuesta_a_vector(respuesta_alumno, criterios_examen, umbral_semantico=0.65)
    print(f"\n🎯 VECTOR GENERADO POR EMEDDINGS: {vector_resultado}")
    
  
    if vector_resultado[0] == 1:
        print("✅ ¡ÉXITO SEMÁNTICO! La IA entendió que 'estructura iterativa' equivale a 'bucle/ciclo'.")
    else:
        print("❌ El umbral es muy alto o no se asociaron los conceptos.")

   
    print("\n" + "-"*40)
    print("💾 PROBANDO ALMACENAMIENTO EN SQLITE...")
    print("-"*40)
    
    guardar_registro_estudiante(
        nombre_estudiante="Lucas Modificado",
        id_pregunta="2",
        pregunta_texto="¿Cómo se repite un bloque de código en Python?",
        respuesta_alumno=respuesta_alumno,
        vector_generado=vector_resultado,
        puntuacion_ia="8.50 / 10.00",
        feedback_ia="Buen uso de estructuras de control iterativas.",
        plagio_info="Ninguno"
    )


    print("\n📖 VERIFICANDO LECTURA DIRECTA DESDE LA BASE DE DATOS...")
    try:
        conn = sqlite3.connect("sistema_evaluacion.db")
        cursor = conn.cursor()
        cursor.execute("SELECT id, estudiante, vector_neurona, puntuacion_ia FROM historial_evaluaciones ORDER BY id DESC LIMIT 1")
        ultimo_registro = cursor.fetchone()
        conn.close()

        if ultimo_registro:
            print("\n==================================================")
            print("       ÚLTIMO REGISTRO ENCONTRADO EN SQLITE       ")
            print("==================================================")
            print(f"ID en DB:      {ultimo_registro[0]}")
            print(f"Estudiante:    {ultimo_registro[1]}")
            print(f"Vector Guardado:{ultimo_registro[2]} (Formato Texto)")
            print(f"Nota Asignada:  {ultimo_registro[3]}")
            print("==================================================")
            print("\n🎉 ¡La base de datos SQLite funciona al 100%! Ya no dependes de JSON.")
        else:
            print("❌ No se encontraron datos en la tabla.")
    except Exception as e:
        print(f"❌ Error al leer de SQLite: {e}")

if __name__ == "__main__":
    probar_sistema()