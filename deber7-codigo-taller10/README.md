# 📚 Sistema de Evaluación de Temas Docentes

Sistema completo que utiliza Go para el servidor backend y consume la API de Groq para evaluar temas educativos en tres niveles (1: Básico, 2: Intermedio, 3: Excelente).

## 🚀 Características

- **Backend**: Servidor HTTP en Go con endpoints RESTful
- **Frontend**: HTML, CSS y JavaScript puro
- **IA**: Integración con Groq API (modelo Llama 70B)
- **Evaluación**: 3 pilares fundamentales (Recursos, Evaluación, Empoderamiento)
- **Diseño**: Responsive con gradientes modernos

## 📋 Requisitos Previos

- Go 1.21 o superior
- Cuenta y API Key de Groq (gratuita)
- Navegador web moderno

## 🛠️ Instalación

1. **Clonar o descargar el proyecto**

2. **Configurar la API Key de Groq**:
   ```bash
   # En Linux/Mac
   export GROQ_API_KEY="tu-api-key-aqui"
   
   # En Windows (CMD)
   set GROQ_API_KEY=tu-api-key-aqui
   
   # En Windows (PowerShell)
   $env:GROQ_API_KEY="tu-api-key-aqui"