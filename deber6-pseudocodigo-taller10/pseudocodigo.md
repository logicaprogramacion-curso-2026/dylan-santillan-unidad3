# Pseudocódigo del Sistema de Evaluación Docente

## 1. Estructura General del Sistema
SISTEMA EvaluacionDocente

CONFIGURACIONES:
PUERTO = ":8080"
GROQ_API_KEY = variable_entorno("GROQ_API_KEY")
MODELO_GROQ = "llama3-70b-8192"
TEMPERATURA = 0.3

RECURSOS:

Archivos estáticos (HTML, CSS, JS)

Endpoint API (/api/evaluar)

INICIO:
SI GROQ_API_KEY está vacío ENTONCES
MOSTRAR "ERROR: API Key no configurada"
TERMINAR
FIN SI

CONFIGURAR enrutador HTTP
DEFINIR ruta "/" -> servir index.html
DEFINIR ruta "/static/" -> servir archivos estáticos
DEFINIR ruta "/api/evaluar" (POST) -> manejadorEvaluacion

INICIAR servidor en PUERTO
MOSTRAR "Servidor corriendo en http://localhost" + PUERTO
FIN

text

## 2. Pseudocódigo del Manejador de Evaluación
FUNCIÓN manejadorEvaluacion(peticion, respuesta, apiKey)

// Validación de método HTTP
SI peticion.metodo != "POST" ENTONCES
respuesta.estado = 405
respuesta.mensaje = "Método no permitido. Usa POST"
RETORNAR
FIN SI

// Lectura y validación de datos
cuerpo = LEER peticion.cuerpo
SI error al leer ENTONCES
respuesta.estado = 400
respuesta.mensaje = "Error al leer la petición"
RETORNAR
FIN SI

datos = DECODIFICAR JSON(cuerpo) como EvaluacionRequest
SI error al decodificar ENTONCES
respuesta.estado = 400
respuesta.mensaje = "Formato JSON inválido"
RETORNAR
FIN SI

// Limpiar y validar el tema
tema = LIMPIAR_ESPACIOS(datos.tema)
SI tema está vacío ENTONCES
respuesta.estado = 400
respuesta.mensaje = "El tema es obligatorio"
RETORNAR
FIN SI

// Procesar evaluación
prompt = construirPrompt(tema)
textoGenerado = llamarGroqAPI(apiKey, prompt)

SI error al llamar Groq ENTONCES
respuesta.estado = 500
respuesta.mensaje = "Error al comunicarse con el servicio de IA"
RETORNAR
FIN SI

// Procesar respuesta de Groq
resultado = procesarRespuestaGroq(textoGenerado, tema)

// Enviar respuesta exitosa
respuesta.estado = 200
respuesta.tipo = "application/json"
respuesta.cuerpo = JSON(resultado)

FIN FUNCIÓN

text

## 3. Pseudocódigo para Construir el Prompt
FUNCIÓN construirPrompt(tema)

prompt = """
Evalúa el tema '%s' para ser impartido por un docente en el ámbito educativo.

Analiza los siguientes 3 pilares fundamentales y asígnales una puntuación del 1 al 3
(1=Muy bajo/Básico, 2=Aceptable/Intermedio, 3=Excelente/Completo):

RECURSOS EDUCATIVOS: Evalúa la calidad, variedad y adecuación de los materiales
didácticos, herramientas tecnológicas, soporte visual y recursos complementarios.

EVALUACIÓN: Evalúa la claridad de los criterios de evaluación, la variedad de
métodos (formativa y sumativa), y la efectividad de los mecanismos de retroalimentación.

EMPODERAMIENTO DEL ESTUDIANTE: Evalúa el potencial para fomentar la autonomía,
el pensamiento crítico, la creatividad y la aplicabilidad en contextos reales.

Finalmente, calcula un NIVEL GENERAL del 1 al 3 basado en el promedio de los 3 pilares.

Devuelve tu respuesta ESTRICTAMENTE en el siguiente formato:

NIVEL_GENERAL: [número 1-3]
RECURSOS: [texto descriptivo]
EVALUACION: [texto descriptivo]
EMPODERAMIENTO: [texto descriptivo]
RESUMEN: [explicación concisa]
"""

RETORNAR prompt CON tema reemplazado
FIN FUNCIÓN

text

## 4. Pseudocódigo para Llamar a Groq API
FUNCIÓN llamarGroqAPI(apiKey, prompt)

// Preparar la petición
peticionGroq = {
modelo: "llama3-70b-8192",
mensajes: [
{
rol: "user",
contenido: prompt
}
],
temperatura: 0.3
}

// Configurar cabeceras
cabeceras = {
"Authorization": "Bearer " + apiKey,
"Content-Type": "application/json"
}

// Realizar petición HTTP
respuesta = POST "https://api.groq.com/openai/v1/chat/completions"
CON cabeceras Y cuerpo = JSON(peticionGroq)

// Validar respuesta
SI respuesta.estado != 200 ENTONCES
error = LEER respuesta.cuerpo
LANZAR error("Groq API error: " + error)
FIN SI

// Extraer contenido
datos = DECODIFICAR JSON(respuesta.cuerpo)
textoGenerado = datos.choices[0].message.content

RETORNAR textoGenerado
FIN FUNCIÓN

text

## 5. Pseudocódigo para Procesar la Respuesta de Groq
FUNCIÓN procesarRespuestaGroq(texto, tema)

// Inicializar resultado
resultado = {
tema: tema,
nivel: 1,
detalles: {
recursos: "",
evaluacion: "",
empoderamiento: ""
},
resumen: ""
}

// Expresiones regulares para extraer campos
patronNivel = "NIVEL_GENERAL:\s(\d+)"
patronRecursos = "RECURSOS:\s(.+?)(?:\n|)" patronEvaluacion = "EVALUACION:\s*(.+?)(?:\n|)"
patronEmpoderamiento = "EMPODERAMIENTO:\s*(.+?)(?:\n|)" patronResumen = "RESUMEN:\s*(.+?)(?:\n|)"

// Extraer nivel
nivelEncontrado = BUSCAR(patronNivel, texto)
SI nivelEncontrado existe ENTONCES
nivel = CONVERTIR_A_ENTERO(nivelEncontrado)
// Asegurar que esté entre 1 y 3
SI nivel < 1 ENTONCES nivel = 1
SI nivel > 3 ENTONCES nivel = 3
resultado.nivel = nivel
FIN SI

// Extraer cada detalle
resultado.detalles.recursos = EXTRAER(patronRecursos, texto)
SI está vacío ENTONCES
resultado.detalles.recursos = "No se pudo extraer información de recursos"
FIN SI

resultado.detalles.evaluacion = EXTRAER(patronEvaluacion, texto)
SI está vacío ENTONCES
resultado.detalles.evaluacion = "No se pudo extraer información de evaluación"
FIN SI

resultado.detalles.empoderamiento = EXTRAER(patronEmpoderamiento, texto)
SI está vacío ENTONCES
resultado.detalles.empoderamiento = "No se pudo extraer información de empoderamiento"
FIN SI

resultado.resumen = EXTRAER(patronResumen, texto)
SI está vacío ENTONCES
resultado.resumen = "Evaluación completada exitosamente."
FIN SI

RETORNAR resultado
FIN FUNCIÓN

text

## 6. Pseudocódigo del Frontend (JavaScript)
FUNCIÓN iniciarAplicacion()

// Obtener elementos del DOM
inputTema = OBTENER_ELEMENTO("temaInput")
btnEvaluar = OBTENER_ELEMENTO("btnEvaluar")
contenedorResultado = OBTENER_ELEMENTO("resultado")

// Función para mostrar carga
FUNCIÓN mostrarCargando()
contenedorResultado.innerHTML = """

<div class="loading"> <div class="spinner"></div> <p>🔄 Evaluando tema con IA...</p> </div> """ FIN FUNCIÓN
// Función para mostrar error
FUNCIÓN mostrarError(mensaje)
contenedorResultado.innerHTML = """

<div class="resultado-card" style="border-left: 6px solid #fc8181;"> <h3>❌ Error</h3> <p>{mensaje}</p> </div> """ FIN FUNCIÓN
// Función para renderizar resultados
FUNCIÓN renderizarResultados(datos)
// Determinar nivel y estilos
SEGUN datos.nivel
CASO 1: nivelInfo = "Nivel 1 - Básico", emoji = "🔴"
CASO 2: nivelInfo = "Nivel 2 - Intermedio", emoji = "🟡"
CASO 3: nivelInfo = "Nivel 3 - Excelente", emoji = "🟢"
FIN SEGUN

// Construir HTML
html = """

<div class="resultado-card"> <h2>📋 Resultados para: "{datos.tema}"</h2> <span class="nivel-badge">{emoji} {nivelInfo}</span> <div class="detalle"> <h4>📚 Recursos Educativos</h4> <p>{datos.detalles.recursos}</p> </div> <div class="detalle"> <h4>📝 Evaluación</h4> <p>{datos.detalles.evaluacion}</p> </div> <div class="detalle"> <h4>💪 Empoderamiento</h4> <p>{datos.detalles.empoderamiento}</p> </div> <div class="resumen"> <h4>📌 Resumen</h4> <p>{datos.resumen}</p> </div> </div> """
contenedorResultado.innerHTML = html
FIN FUNCIÓN

// Función principal de evaluación
FUNCIÓN ASÍNCRONA evaluarTema()
tema = LIMPIAR_ESPACIOS(inputTema.value)

SI tema está vacío ENTONCES
mostrarError("Por favor, ingresa un tema")
RETORNAR
FIN SI

mostrarCargando()

TRY
respuesta = ESPERAR FETCH("/api/evaluar", {
metodo: "POST",
cabeceras: {"Content-Type": "application/json"},
cuerpo: JSON.stringify({tema: tema})
})

SI !respuesta.ok ENTONCES
errorData = ESPERAR respuesta.json()
LANZAR error(errorData.error o "Error del servidor")
FIN SI

datos = ESPERAR respuesta.json()
renderizarResultados(datos)

CATCH error
mostrarError(error.mensaje)
FIN TRY
FIN FUNCIÓN

// Configurar eventos
btnEvaluar.addEventListener("click", evaluarTema)
inputTema.addEventListener("keypress", FUNCION(e)
SI e.key == "Enter" ENTONCES
evaluarTema()
FIN SI
FIN FUNCIÓN)

// Enfocar input
inputTema.focus()
FIN FUNCIÓN

text

## 7. Diagrama de Flujo del Sistema
INICIO (Usuario ingresa tema)
|
v
[Validar entrada del usuario]
|
v
[Mostrar indicador de carga]
|
v
[Enviar POST a /api/evaluar]
|
v
[Servidor Go recibe petición]
|
v
[Validar método y formato JSON]
|
v
[Construir prompt para Groq]
|
v
[Llamar a API de Groq]
|
v
[¿Respuesta exitosa?] ---NO---> [Mostrar error 500]
| SI
v
[Extraer datos de la respuesta]
|
v
[Procesar y parsear nivel y detalles]
|
v
[Construir respuesta JSON]
|
v
[Enviar respuesta al frontend]
|
v
[JavaScript recibe y renderiza]
|
v
[Mostrar nivel y detalles al usuario]
|
v
FIN

text

## 8. Estructuras de Datos

### Backend (Go)
// Petición del cliente
Estructura EvaluacionRequest {
Tema string
}

// Respuesta al cliente
Estructura EvaluacionResponse {
Tema string
Nivel int
Detalles struct {
Recursos string
Evaluacion string
Empoderamiento string
}
Resumen string
}

// Petición a Groq
Estructura GroqRequest {
Model string
Messages []struct {
Role string
Content string
}
Temperature float64
}

// Respuesta de Groq
Estructura GroqResponse {
Choices []struct {
Message struct {
Content string
}
}
}

text

### Frontend (JavaScript)
// Objeto de datos recibido del servidor
Objeto DatosEvaluacion {
tema: string,
nivel: number (1-3),
detalles: {
recursos: string,
evaluacion: string,
empoderamiento: string
},
resumen: string
}

text

## 9. Validaciones y Manejo de Errores
LISTA DE VALIDACIONES:

Backend (Go):
✓ API Key de Groq configurada
✓ Método HTTP es POST
✓ Cuerpo de petición es JSON válido
✓ Campo "tema" no está vacío
✓ Respuesta de Groq es exitosa (200)
✓ Extracción correcta de datos de Groq

Frontend (JavaScript):
✓ Input no está vacío
✓ Conexión al servidor establecida
✓ Respuesta del servidor es OK (200)
✓ Datos recibidos tienen el formato esperado

Groq API:
✓ Autenticación correcta (API Key válida)
✓ Modelo disponible y funcionando
✓ Respuesta en el formato esperado
✓ Tiempo de respuesta adecuado

text

## 10. Métricas de Evaluación
CÁLCULO DEL NIVEL GENERAL:

nivelGeneral = REDONDEAR( (recursos + evaluacion + empoderamiento) / 3 )

Donde cada pilar se evalúa como:

1: Muy Bajo o Básico

2: Aceptable o Intermedio

3: Excelente o Completo

Interpretación:

Nivel 1 (1.0 - 1.5): Básico - Necesita mejorar significativamente

Nivel 2 (1.6 - 2.5): Intermedio - Bueno pero con áreas de mejora

Nivel 3 (2.6 - 3.0): Excelente - Muy completo y bien estructurado