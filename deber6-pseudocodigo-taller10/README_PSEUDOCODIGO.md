# 📚 Documentación del Pseudocódigo - Sistema de Evaluación Docente

## 📖 Descripción General

Este documento contiene el pseudocódigo completo del **Sistema de Evaluación de Temas Docentes**, una aplicación que utiliza Go como servidor backend, JavaScript para el frontend, y la API de Groq para evaluar temas educativos.

## 🎯 Objetivo del Pseudocódigo

El pseudocódigo presentado tiene como objetivo:

1. **Describir la lógica** del sistema de forma clara y lenguaje natural
2. **Servir como guía** para desarrolladores que quieran implementar el sistema
3. **Documentar el flujo** completo de la aplicación (frontend + backend + API)
4. **Facilitar la comprensión** de la arquitectura del sistema

## 📂 Estructura del Documento

### 1. Estructura General del Sistema
Descripción de alto nivel de la configuración y componentes principales.

### 2. Manejador de Evaluación
Lógica del endpoint principal que procesa las peticiones de evaluación.

### 3. Construcción del Prompt
Cómo se construye la instrucción detallada para el modelo de IA.

### 4. Llamada a Groq API
Comunicación con el servicio externo de inteligencia artificial.

### 5. Procesamiento de Respuesta
Extracción y validación de la información devuelta por Groq.

### 6. Frontend (JavaScript)
Lógica del lado del cliente para interactuar con el usuario.

### 7. Diagrama de Flujo
Representación visual del flujo de datos a través del sistema.

### 8. Estructuras de Datos
Definición de los objetos y formatos utilizados en el sistema.

### 9. Validaciones y Manejo de Errores
Puntos de control y recuperación en caso de fallos.

### 10. Métricas de Evaluación
Cálculo e interpretación del nivel final.

## 🛠️ Tecnologías Representadas

| Tecnología | Rol en el Sistema | Pseudocódigo Referenciado |
|------------|-------------------|---------------------------|
| **Go** | Servidor backend, API REST | Secciones 1, 2, 3, 4, 5 |
| **JavaScript** | Interacción frontend | Sección 6 |
| **Groq API** | Inteligencia artificial | Secciones 3, 4 |
| **HTML/CSS** | Interfaz de usuario | Sección 6 (renderizado) |

## 🔄 Flujo de Datos
