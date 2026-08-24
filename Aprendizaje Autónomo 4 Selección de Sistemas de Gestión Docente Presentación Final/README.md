# 🚀 Evaluaciones de Respuesta Abierta (con LLM)

> Sistema de evaluación académica con red neuronal propia + LLM en cascada.  
> Proyecto de curso — Inteligencia Artificial · Estación Espacial Evaluadora v3.0

---

## 📋 Descripción

Sistema que evalúa respuestas abiertas de estudiantes (preguntas de desarrollo, ensayos cortos) usando una arquitectura híbrida:

- **Red neuronal propia** (PyTorch) entrenada por pregunta, predice la nota (0–10)
- **Embeddings semánticos en cascada**: Gemini → Groq → Cohere → NVIDIA → vocabulario local offline → vector de ceros
- **LLM para feedback**: Gemini (gemini-2.5-flash) → Groq (llama-3.3-70b) como respaldo
- **Detección de plagio** con tres métricas combinadas: SequenceMatcher + Jaccard + bigramas
- **Interfaz web** (Flask) y **CLI** de 9 opciones

---

## 🗂️ Estructura del proyecto

```
├── app.py                          # Servidor Flask — rutas y lógica web
├── evaluador_datos.py              # BD SQLite, embeddings, plagio, usuarios
├── main_api.py                     # CLI de 9 opciones + feedback LLM
├── opo.py                          # Red neuronal moderna (PyTorch)
├── evaluador_entrenar.py           # Entrenamiento del evaluador por pregunta
├── generar_dataset.py              # Genera dataset sintético con Groq
├── entrenar_vocabulario_pytorch.py # Vocabulario Word2Vec + MiniGPT Transformer
├── Layer.py                        # Capa de neuronas (versión anterior, numpy)
├── neuron.py                       # Neurona individual (versión anterior, numpy)
│
├── templates/                      # Plantillas HTML (Flask + Jinja2)
│   ├── index.html                  # Página principal con efecto hiperespacio
│   ├── evaluar.html                # Formulario de evaluación (público)
│   ├── resultado.html              # Reporte de misión con nota y feedback
│   ├── login.html                  # Panel de acceso
│   ├── docente.html                # Panel del docente
│   ├── dashboard.html              # Estadísticas y gráficos (Chart.js)
│   └── historial.html              # Registro completo de evaluaciones
│
├── data/
│   ├── banco_preguntas.json        # Preguntas y criterios de evaluación
│   ├── vocabulario_pytorch.json    # Vocabulario Word2Vec (27,525 palabras)
│   ├── corpus/                     # 42 libros de texto para el vocabulario
│   └── dataset_pregunta_{id}.csv  # Datasets sintéticos por pregunta
│
├── cerebro_pregunta_{id}.json      # Modelos entrenados por pregunta
├── sistema_evaluacion.db           # Base de datos SQLite
├── .env                            # API keys (no subir a GitHub)
└── requirements.txt
```

---

## ⚙️ Requisitos

- Python 3.11 o superior (recomendado 3.11 por compatibilidad con gensim)
- [wkhtmltopdf](https://wkhtmltopdf.org/downloads.html) — para exportar PDF

---

## 🔧 Instalación

### 1. Clonar el repositorio

```bash
git clone https://github.com/tu-usuario/evaluaciones-respuesta-abierta.git
cd evaluaciones-respuesta-abierta
```

### 2. Crear entorno virtual e instalar dependencias

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux / Mac
source venv/bin/activate

pip install -r requirements.txt
```

### 3. Configurar las API keys

Crear un archivo `.env` en la raíz del proyecto con el siguiente contenido:

```env
# APIs de embeddings (cascada: Gemini → Groq → Cohere → NVIDIA → local)
GEMINI_API_KEY=tu_clave_de_gemini
GROQ_API_KEY=tu_clave_de_groq
API_KEY_COHERE=tu_clave_de_cohere
API_KEY_NVIDIA=tu_clave_de_nvidia

# Flask
FLASK_SECRET_KEY=una_clave_secreta_larga_y_aleatoria

# wkhtmltopdf (ajustar según tu sistema)
WKHTMLTOPDF_PATH=C:/Program Files/wkhtmltopdf/bin/wkhtmltopdf.exe
```

> **Nota:** El sistema funciona sin API keys gracias al vocabulario local offline. Si Gemini, Groq, Cohere y NVIDIA fallan, usa el vocabulario de 27,525 palabras entrenado localmente como último recurso.

### 4. (Opcional) Agregar columna de proveedor a una BD existente

Si ya tenés datos en `sistema_evaluacion.db` de una versión anterior, ejecutá esto una sola vez:

```bash
python -c "
import sqlite3
conn = sqlite3.connect('sistema_evaluacion.db')
try:
    conn.execute(\"ALTER TABLE historial_evaluaciones ADD COLUMN proveedor_embedding TEXT DEFAULT 'Desconocido'\")
    conn.commit()
    print('✅ Columna agregada')
except Exception as e:
    print('Ya existe:', e)
conn.close()
"
```

---

## 🚀 Cómo correr el proyecto

### Interfaz web (Flask)

```bash
python app.py
```

Abrir en el navegador: [http://localhost:5000](http://localhost:5000)

### CLI (menú de 9 opciones)

```bash
python main_api.py
```

---

## 👥 Usuarios por defecto

| Usuario | Contraseña | Rol      |
|---------|-----------|----------|
| admin   | admin123  | Docente  |
| profe   | profe123  | Docente  |
| alumno  | alumno123 | Estudiante |
| lucas   | lucas123  | Estudiante |

---

## 📚 Flujo completo

### El docente crea una misión

1. Ir a **Panel del Docente** → ingresar código, descripción y 5 grupos de criterios
2. El sistema genera automáticamente un dataset sintético (via Groq) y entrena la red neuronal
3. El modelo queda guardado como `cerebro_pregunta_{id}.json`

### El estudiante evalúa su respuesta

1. Ir a **Evaluar Respuesta** (sin necesidad de login)
2. Ingresar nombre, seleccionar misión y escribir la respuesta
3. El sistema:
   - Verifica plagio (SequenceMatcher + Jaccard + bigramas, umbral 0.45)
   - Genera el embedding semántico en cascada (Gemini → ... → local)
   - Predice la nota con la red neuronal entrenada
   - Genera feedback cualitativo con el LLM
4. Se muestra el reporte con nota, criterios, integridad y feedback

### El docente consulta resultados

- **Dashboard**: estadísticas agregadas, gráficos, distribución de notas
- **Historial**: tabla completa con proveedor de embedding por evaluación
- **Descargar PDF**: reporte completo para imprimir o archivar

---

## 🧠 Arquitectura de la red neuronal

```
Entrada (768 dims — embedding semántico)
    ↓
Capa 1: 768 → 256  (ReLU, Dropout 0.2)
Capa 2: 256 → 128  (ReLU, Dropout 0.2)
Capa 3: 128 → 64   (ReLU, Dropout 0.1)
Capa 4:  64 → 1    (Sigmoid)
    ↓
Salida × 10 = Nota final (0–10)
```

Protecciones implementadas:
- **BatchNorm + batch size 1**: si el último mini-batch queda con un solo ejemplo, se descarta en vez de tronar con `RuntimeError`
- **Early stopping**: detiene el entrenamiento si la pérdida de validación no mejora en N épocas
- **Gradient clipping**: evita explosión de gradientes

---

## 📡 Cascada de embeddings

| Plan | Proveedor | Modelo | Dims |
|------|-----------|--------|------|
| A | Gemini | gemini-embedding-2 | 768 |
| B | Groq | llama-3.2-3b-preview | 768 |
| C | Cohere | embed-multilingual-v3.0 | 768 |
| D | NVIDIA | nv-embedqa-e5-v5 | 768 |
| E | Vocabulario local | Word2Vec propio (27,525 palabras) | 768 |
| Emergencia | — | Vector de ceros | 768 |

Si un proveedor falla (error de API, rate limit, sin internet), el sistema pasa automáticamente al siguiente sin interrumpir la evaluación.

---

## 🛡️ Detección de plagio

Combina tres métricas sobre texto normalizado (sin acentos, sin puntuación, sin stopwords):

| Métrica | Peso | Qué detecta |
|---------|------|-------------|
| SequenceMatcher | 25% | Copia literal carácter a carácter |
| Jaccard | 45% | Mismo vocabulario aunque reordenado |
| Bigramas | 30% | Frases copiadas aunque mezcladas |

**Umbral combinado: 0.45** (más sensible que el SequenceMatcher solo a 0.70).  
Se compara contra **todas** las respuestas anteriores de la misma pregunta y retorna la más similar.

---

## 📖 Vocabulario local (Word2Vec)

- **42 libros** de texto procesados (guerra/historia, Python/ML/IA, narrativa, académico, filosofía, ingeniería aeroespacial)
- **27,525 palabras** en vocabulario
- **150 dimensiones** por palabra
- **64 épocas** de entrenamiento acumuladas (reanudable)
- Loss final: 0.3594

---

## 🗃️ Base de datos

**Tabla `historial_evaluaciones`**

| Columna | Descripción |
|---------|-------------|
| id | ID autoincremental |
| estudiante | Nombre del estudiante |
| id_pregunta | ID de la pregunta evaluada |
| pregunta | Texto de la pregunta |
| respuesta_estudiante | Respuesta completa |
| vector_neurona | Primeros 5 valores del embedding (truncado) |
| puntuacion_ia | Nota en formato `X.XX / 10.00` |
| feedback_ia | Retroalimentación generada por el LLM |
| alerta_plagio | Resultado del escaneo de integridad |
| proveedor_embedding | Proveedor que generó el embedding |
| fecha | Timestamp de la evaluación |

**Tabla `usuarios`**

| Columna | Descripción |
|---------|-------------|
| id | ID autoincremental |
| usuario | Nombre de usuario (único) |
| password_hash | SHA-256 de la contraseña |
| rol | `docente` o `estudiante` |
| nombre_completo | Nombre para mostrar |
| fecha_creacion | Timestamp de creación |

---

## 📦 Dependencias principales

```
flask
python-dotenv
torch
numpy
pandas
openpyxl
pdfkit
google-generativeai
groq
cohere
requests
difflib (stdlib)
```

---

## 📝 Lecciones aprendidas

### Lo que funcionó bien
- La **cascada de embeddings** como estrategia de resiliencia fue clave — el sistema nunca queda bloqueado por un proveedor caído
- Separar la **predicción de nota** (red neuronal) del **feedback** (LLM) da más consistencia entre evaluaciones similares que depender del LLM para ambas cosas
- El vocabulario local de 27,525 palabras como fallback offline permite evaluar sin internet

### Desafíos encontrados
- **Compatibilidad de Python**: gensim 4.3.3 es incompatible con Python 3.14 (sin wheels precompilados) y con SciPy 1.14+. Se resolvió implementando Word2Vec directamente en PyTorch
- **BatchNorm con batch size 1**: PyTorch lanza `RuntimeError` si el último mini-batch de una época tiene un solo ejemplo. Se resolvió descartando ese batch cuando `len(train) > 1`
- **JSON embebido con comillas**: interpolar texto del LLM directamente en strings JSON rompía `JSON.parse()` si el feedback contenía comillas dobles o saltos de línea. Se resolvió usando el filtro `| tojson` de Jinja2
- **Conteo inconsistente en estadísticas**: `COUNT(*)` incluía registros con nota en formato inválido (`N/A`), pero el promedio los excluía, generando números que no cuadraban. Se resolvió centralizando el cálculo en un helper `_calcular_estadisticas()`

---

## 👨‍💻 Autor

Proyecto desarrollado para el curso de Inteligencia Artificial.  
Estación Espacial Evaluadora v3.0 — 42 libros | 27,525 palabras | 64 épocas
