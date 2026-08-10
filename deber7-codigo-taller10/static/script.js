// Esperar a que el DOM esté completamente cargado
document.addEventListener('DOMContentLoaded', function() {
    const inputTema = document.getElementById('temaInput');
    const btnEvaluar = document.getElementById('btnEvaluar');
    const contenedorResultado = document.getElementById('resultado');

    // Función para mostrar estado de carga
    function mostrarCargando() {
        contenedorResultado.innerHTML = `
            <div class="loading">
                <div class="spinner"></div>
                <p style="margin-top: 15px; color: #4a5568;">
                    🔄 Evaluando tema con IA...
                </p>
                <p style="font-size: 0.9rem; color: #a0aec0;">
                    Analizando recursos educativos, evaluación y empoderamiento
                </p>
            </div>
        `;
    }

    // Función para mostrar error
    function mostrarError(mensaje) {
        contenedorResultado.innerHTML = `
            <div class="resultado-card" style="border-left: 6px solid #fc8181; background: #fff5f5; padding: 20px;">
                <h3 style="color: #c53030;">❌ Error</h3>
                <p style="color: #742a2a;">${mensaje}</p>
                <p style="font-size: 0.9rem; color: #718096; margin-top: 10px;">
                    💡 Verifica tu conexión a internet y que el servidor esté corriendo
                </p>
            </div>
        `;
    }

    // Función para renderizar resultados
    function renderizarResultados(datos) {
        // Determinar nivel y estilos
        let nivelInfo;
        let colorClase;
        let emoji;

        switch(datos.nivel) {
            case 1:
                nivelInfo = 'Nivel 1 - Básico';
                colorClase = 'nivel-1';
                emoji = '🔴';
                break;
            case 2:
                nivelInfo = 'Nivel 2 - Intermedio';
                colorClase = 'nivel-2';
                emoji = '🟡';
                break;
            case 3:
                nivelInfo = 'Nivel 3 - Excelente/Completo';
                colorClase = 'nivel-3';
                emoji = '🟢';
                break;
            default:
                nivelInfo = 'Nivel no determinado';
                colorClase = '';
                emoji = '⚪';
        }

        // Construir la tarjeta de resultados
        const html = `
            <div class="resultado-card ${colorClase}">
                <h2>📋 Resultados para: <span style="color: #2d3748;">"${datos.tema}"</span></h2>
                
                <div style="display: flex; align-items: center; gap: 12px; margin: 15px 0;">
                    <span class="nivel-badge n${datos.nivel}">${emoji} ${nivelInfo}</span>
                    <span style="font-size: 0.9rem; color: #718096;">
                        Evaluación basada en 3 pilares
                    </span>
                </div>

                <div class="detalle">
                    <h4>📚 Recursos Educativos</h4>
                    <p>${datos.detalles.recursos}</p>
                </div>

                <div class="detalle">
                    <h4>📝 Evaluación</h4>
                    <p>${datos.detalles.evaluacion}</p>
                </div>

                <div class="detalle">
                    <h4>💪 Empoderamiento del Estudiante</h4>
                    <p>${datos.detalles.empoderamiento}</p>
                </div>

                <div class="resumen">
                    <h4>📌 Resumen</h4>
                    <p style="font-size: 1.05rem; font-weight: 500;">${datos.resumen}</p>
                </div>
            </div>
        `;

        contenedorResultado.innerHTML = html;
    }

    // Función principal para evaluar el tema
    async function evaluarTema() {
        const tema = inputTema.value.trim();

        // Validar que el campo no esté vacío
        if (!tema) {
            contenedorResultado.innerHTML = `
                <div class="resultado-card" style="border-left: 6px solid #fc8181; background: #fff5f5; padding: 20px;">
                    <h3 style="color: #c53030;">⚠️ Campo requerido</h3>
                    <p style="color: #742a2a;">Por favor, ingresa un tema para evaluar.</p>
                </div>
            `;
            return;
        }

        // Mostrar estado de carga
        mostrarCargando();

        try {
            // Realizar la petición al servidor Go
            const respuesta = await fetch('/api/evaluar', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ tema: tema })
            });

            // Verificar si la respuesta fue exitosa
            if (!respuesta.ok) {
                let mensajeError = `Error en el servidor (${respuesta.status})`;
                
                try {
                    const errorData = await respuesta.json();
                    if (errorData.error) {
                        mensajeError = errorData.error;
                    }
                } catch (e) {
                    // Si no se puede parsear JSON, usar mensaje genérico
                }
                
                throw new Error(mensajeError);
            }

            // Parsear la respuesta JSON
            const datos = await respuesta.json();

            // Renderizar los resultados
            renderizarResultados(datos);

        } catch (error) {
            console.error('Error:', error);
            mostrarError(error.message || 'Error al conectar con el servidor. Asegúrate de que el servidor Go esté corriendo en el puerto 8080.');
        }
    }

    // Event listeners
    btnEvaluar.addEventListener('click', evaluarTema);

    // Permitir evaluar presionando Enter en el input
    inputTema.addEventListener('keypress', function(e) {
        if (e.key === 'Enter') {
            evaluarTema();
        }
    });

    // Enfocar el input al cargar la página
    inputTema.focus();

    // Mostrar mensaje de bienvenida
    console.log('📚 Sistema de Evaluación Docente cargado');
    console.log('💡 Ingresa un tema y presiona Evaluar');
});