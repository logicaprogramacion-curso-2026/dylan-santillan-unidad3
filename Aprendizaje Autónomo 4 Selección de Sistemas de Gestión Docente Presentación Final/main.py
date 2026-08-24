import numpy as np
from opo import NeuralNetwork  
from evaluador_datos import BANCO_PREGUNTAS, transformar_respuesta_a_vector

def evaluar_examen_alumno():
    print("==================================================")
    print("      SISTEMA IA: EVALUADOR DE RESPUESTAS         ")
    print("==================================================")
    
    
    print("\nPreguntas disponibles para evaluar:")
    for id_p, info in BANCO_PREGUNTAS.items():
        print(f" [{id_p}] - {info['pregunta']}")
        
    
    try:
        id_elegido = int(input("\nSeleccione el número de la pregunta a responder: "))
        if id_elegido not in BANCO_PREGUNTAS:
            print("❌ Pregunta no válida.")
            return
    except ValueError:
        print("❌ Entrada inválida. Debe ingresar un número.")
        return

    info_pregunta = BANCO_PREGUNTAS[id_elegido]
    
   
    nn = NeuralNetwork()
    nombre_cerebro = f"cerebro_pregunta_{id_elegido}.json"
    
    try:
        nn.load(nombre_cerebro)
    except FileNotFoundError:
        print(f"\n❌ Error: No se encontró el cerebro entrenado para esta pregunta ('{nombre_cerebro}').")
        print("Asegúrate de ejecutar primero 'evaluador_entrenar.py' para calibrarlo.")
        return

   
    print("\n" + "="*50)
    print(f"PREGUNTA: {info_pregunta['pregunta']}")
    print("="*50)
    respuesta_alumno = input("Escriba su respuesta: ")
    
    
    vector_entradas = transformar_respuesta_a_vector(respuesta_alumno, info_pregunta["criterios"])
    print(f"\n[INFO] Vector binario extraído de su texto: {vector_entradas}")
    
   
    prediccion = nn.forward(vector_entradas)
    
    
    nota_final = float(prediccion[0]) * 10
    
    
    print("\n" + "="*50)
    print("               RESULTADO DE LA EVALUACIÓN          ")
    print("="*50)
    print(f"▶️ NOTA ASIGNADA POR LA IA: {nota_final:.2f} / 10.00")
    
    
    if nota_final >= 9.0:
        print("▶️ RETROALIMENTACIÓN: ¡Excelente! Incluiste todos los conceptos requeridos de forma precisa.")
    elif nota_final >= 6.0:
        print("▶️ RETROALIMENTACIÓN: Respuesta regular. Mencionas ideas clave, pero te faltó profundizar o incluir más criterios técnicos.")
    else:
        print("▶️ RETROALIMENTACIÓN: Respuesta insuficiente. No se detectaron las palabras clave o criterios mínimos solicitados.")
    print("="*50 + "\n")

if __name__ == "__main__":
    evaluar_examen_alumno()