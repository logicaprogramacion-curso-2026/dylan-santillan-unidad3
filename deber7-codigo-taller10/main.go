package main

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"
	"os"
	"regexp"
	"strconv"
	"strings"
)

// Estructura para recibir la petición del cliente
type EvaluacionRequest struct {
	Tema string `json:"tema"`
}

// Estructura para enviar la respuesta al cliente
type EvaluacionResponse struct {
	Tema     string            `json:"tema"`
	Nivel    int               `json:"nivel"`
	Detalles DetallesEvaluacion `json:"detalles"`
	Resumen  string            `json:"resumen"`
}

type DetallesEvaluacion struct {
	Recursos       string `json:"recursos"`
	Evaluacion     string `json:"evaluacion"`
	Empoderamiento string `json:"empoderamiento"`
}

// Estructura para la petición a Groq API
type GroqRequest struct {
	Model       string          `json:"model"`
	Messages    []GroqMessage   `json:"messages"`
	Temperature float64         `json:"temperature"`
}

type GroqMessage struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

// Estructura para la respuesta de Groq API
type GroqResponse struct {
	Choices []struct {
		Message struct {
			Content string `json:"content"`
		} `json:"message"`
	} `json:"choices"`
}

func main() {
	// Configurar el puerto
	port := ":8080"

	// Verificar que la API Key de Groq esté configurada
	apiKey := os.Getenv("GROQ_API_KEY")
	if apiKey == "" {
		log.Fatal("ERROR: La variable de entorno GROQ_API_KEY no está configurada")
	}

	// Crear el enrutador
	mux := http.NewServeMux()

	// Servir archivos estáticos (HTML, CSS, JS)
	mux.HandleFunc("/", serveIndex)
	mux.Handle("/static/", http.StripPrefix("/static/", http.FileServer(http.Dir("./static"))))

	// Endpoint de la API para evaluar temas
	mux.HandleFunc("/api/evaluar", func(w http.ResponseWriter, r *http.Request) {
		handleEvaluacion(w, r, apiKey)
	})

	// Iniciar el servidor
	fmt.Printf("🚀 Servidor corriendo en http://localhost%s\n", port)
	fmt.Println("📚 Sistema de Evaluación Docente con Groq")
	fmt.Println("💡 Presiona Ctrl+C para detener el servidor")
	
	log.Fatal(http.ListenAndServe(port, mux))
}

// Servir el archivo index.html
func serveIndex(w http.ResponseWriter, r *http.Request) {
	if r.URL.Path != "/" {
		http.NotFound(w, r)
		return
	}
	http.ServeFile(w, r, "index.html")
}

// Manejador principal de la evaluación
func handleEvaluacion(w http.ResponseWriter, r *http.Request, apiKey string) {
	// Verificar que sea método POST
	if r.Method != http.MethodPost {
		http.Error(w, "Método no permitido. Usa POST", http.StatusMethodNotAllowed)
		return
	}

	// Leer el cuerpo de la petición
	body, err := io.ReadAll(r.Body)
	if err != nil {
		http.Error(w, "Error al leer la petición", http.StatusBadRequest)
		return
	}
	defer r.Body.Close()

	// Parsear JSON
	var req EvaluacionRequest
	if err := json.Unmarshal(body, &req); err != nil {
		http.Error(w, "Formato JSON inválido", http.StatusBadRequest)
		return
	}

	// Validar que el tema no esté vacío
	req.Tema = strings.TrimSpace(req.Tema)
	if req.Tema == "" {
		http.Error(w, "El tema es obligatorio", http.StatusBadRequest)
		return
	}

	// Construir el prompt para Groq
	prompt := construirPrompt(req.Tema)

	// Llamar a la API de Groq
	respuestaGroq, err := llamarGroqAPI(apiKey, prompt)
	if err != nil {
		log.Printf("Error al llamar a Groq API: %v", err)
		http.Error(w, "Error al comunicarse con el servicio de IA", http.StatusInternalServerError)
		return
	}

	// Procesar la respuesta de Groq
	resultado, err := procesarRespuestaGroq(respuestaGroq, req.Tema)
	if err != nil {
		log.Printf("Error al procesar respuesta de Groq: %v", err)
		http.Error(w, "Error al procesar la evaluación", http.StatusInternalServerError)
		return
	}

	// Enviar respuesta al cliente
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(http.StatusOK)
	json.NewEncoder(w).Encode(resultado)
}

// Construir el prompt detallado para Groq
func construirPrompt(tema string) string {
	return fmt.Sprintf(`Evalúa el tema '%s' para ser impartido por un docente en el ámbito educativo.

Analiza los siguientes 3 pilares fundamentales y asígnales una puntuación del 1 al 3 (1=Muy bajo/Básico, 2=Aceptable/Intermedio, 3=Excelente/Completo):

1. RECURSOS EDUCATIVOS: Evalúa la calidad, variedad y adecuación de los materiales didácticos, herramientas tecnológicas, soporte visual y recursos complementarios para el aprendizaje.

2. EVALUACIÓN: Evalúa la claridad de los criterios de evaluación, la variedad de métodos de evaluación (formativa y sumativa), y la efectividad de los mecanismos de retroalimentación.

3. EMPODERAMIENTO DEL ESTUDIANTE: Evalúa el potencial del tema para fomentar la autonomía, el pensamiento crítico, la creatividad, la resolución de problemas y la aplicabilidad en contextos reales.

Finalmente, calcula un NIVEL GENERAL del 1 al 3 basado en el promedio de los 3 pilares.

Devuelve tu respuesta ESTRICTAMENTE en el siguiente formato de texto:

NIVEL_GENERAL: [número 1-3]
RECURSOS: [breve texto descriptivo con la puntuación implícita]
EVALUACION: [breve texto descriptivo con la puntuación implícita]
EMPODERAMIENTO: [breve texto descriptivo con la puntuación implícita]
RESUMEN: [explicación concisa del porqué del nivel general, máximo 2 líneas]

Asegúrate de que cada línea comience exactamente con la etiqueta correspondiente.`, tema)
}

// Llamar a la API de Groq
func llamarGroqAPI(apiKey string, prompt string) (string, error) {
	// Preparar la petición a Groq
	groqReq := GroqRequest{
		Model: "llama3-70b-8192",
		Messages: []GroqMessage{
			{
				Role:    "user",
				Content: prompt,
			},
		},
		Temperature: 0.3, // Baja temperatura para respuestas más consistentes
	}

	jsonData, err := json.Marshal(groqReq)
	if err != nil {
		return "", err
	}

	// Crear la petición HTTP
	req, err := http.NewRequest("POST", "https://api.groq.com/openai/v1/chat/completions", bytes.NewBuffer(jsonData))
	if err != nil {
		return "", err
	}

	req.Header.Set("Authorization", "Bearer "+apiKey)
	req.Header.Set("Content-Type", "application/json")

	// Ejecutar la petición
	client := &http.Client{}
	resp, err := client.Do(req)
	if err != nil {
		return "", err
	}
	defer resp.Body.Close()

	// Verificar estado de la respuesta
	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		return "", fmt.Errorf("Groq API error: %s - %s", resp.Status, string(body))
	}

	// Leer la respuesta
	var groqResp GroqResponse
	if err := json.NewDecoder(resp.Body).Decode(&groqResp); err != nil {
		return "", err
	}

	if len(groqResp.Choices) == 0 {
		return "", fmt.Errorf("no se recibieron respuestas de Groq")
	}

	return groqResp.Choices[0].Message.Content, nil
}

// Procesar la respuesta de Groq y extraer la información
func procesarRespuestaGroq(texto string, tema string) (EvaluacionResponse, error) {
	var resultado EvaluacionResponse
	resultado.Tema = tema

	// Expresiones regulares para extraer cada campo
	reNivel := regexp.MustCompile(`(?i)NIVEL_GENERAL:\s*(\d+)`)
	reRecursos := regexp.MustCompile(`(?i)RECURSOS:\s*(.+?)(?:\n|$)`)
	reEvaluacion := regexp.MustCompile(`(?i)EVALUACION:\s*(.+?)(?:\n|$)`)
	reEmpoderamiento := regexp.MustCompile(`(?i)EMPODERAMIENTO:\s*(.+?)(?:\n|$)`)
	reResumen := regexp.MustCompile(`(?i)RESUMEN:\s*(.+?)(?:\n|$)`)

	// Extraer nivel
	matchNivel := reNivel.FindStringSubmatch(texto)
	if len(matchNivel) > 1 {
		nivel, err := strconv.Atoi(matchNivel[1])
		if err == nil {
			// Asegurar que el nivel esté entre 1 y 3
			if nivel < 1 {
				nivel = 1
			} else if nivel > 3 {
				nivel = 3
			}
			resultado.Nivel = nivel
		} else {
			resultado.Nivel = 1 // Nivel por defecto si hay error
		}
	} else {
		resultado.Nivel = 1 // Nivel por defecto si no se encuentra
	}

	// Extraer detalles
	matchRecursos := reRecursos.FindStringSubmatch(texto)
	if len(matchRecursos) > 1 {
		resultado.Detalles.Recursos = strings.TrimSpace(matchRecursos[1])
	} else {
		resultado.Detalles.Recursos = "No se pudo extraer información de recursos"
	}

	matchEvaluacion := reEvaluacion.FindStringSubmatch(texto)
	if len(matchEvaluacion) > 1 {
		resultado.Detalles.Evaluacion = strings.TrimSpace(matchEvaluacion[1])
	} else {
		resultado.Detalles.Evaluacion = "No se pudo extraer información de evaluación"
	}

	matchEmpoderamiento := reEmpoderamiento.FindStringSubmatch(texto)
	if len(matchEmpoderamiento) > 1 {
		resultado.Detalles.Empoderamiento = strings.TrimSpace(matchEmpoderamiento[1])
	} else {
		resultado.Detalles.Empoderamiento = "No se pudo extraer información de empoderamiento"
	}

	matchResumen := reResumen.FindStringSubmatch(texto)
	if len(matchResumen) > 1 {
		resultado.Resumen = strings.TrimSpace(matchResumen[1])
	} else {
		resultado.Resumen = "Evaluación completada exitosamente."
	}

	return resultado, nil
}