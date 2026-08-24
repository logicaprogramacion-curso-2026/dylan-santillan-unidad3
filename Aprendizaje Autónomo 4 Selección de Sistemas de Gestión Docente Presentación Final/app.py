import os
import json
import pdfkit
import datetime
import secrets
import hmac
import html
import time
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor
from functools import wraps
from pathlib import Path
from threading import Lock

import numpy as np
from flask import (
    Flask, abort, flash, jsonify, redirect, render_template, request,
    session, url_for
)
from dotenv import load_dotenv
from opo import NeuralNetwork
from evaluador_datos import (
    transformar_respuesta_a_vector,
    guardar_registro_estudiante,
    verificar_plagio,
    agregar_pregunta_dinamica,
    cargar_banco_preguntas,
    conectar_db,
    crear_tabla_usuarios,
    verificar_usuario,
    registrar_usuario,
    normalizar_criterios,
    pregunta_existe,
    validar_id_pregunta,
    eliminar_pregunta_dinamica
)
from main_api import generar_retroalimentacion_ia
from generar_dataset import generar_dataset_pregunta, ruta_dataset_pregunta
from evaluador_entrenar import (
    entrenar_evaluador,
    cargar_metadata_modelo,
    localizar_modelo_pregunta,
    ruta_modelo_pregunta,
    ruta_metadata_pregunta,
    GRAFICAS_DIR,
)

load_dotenv()
app = Flask(__name__)
_secret = os.environ.get("FLASK_SECRET_KEY")
if not _secret:
    _secret = secrets.token_hex(32)
    print("⚠️ FLASK_SECRET_KEY no definida; se usará una clave temporal.")
app.secret_key = _secret
app.config.update(
    SESSION_COOKIE_HTTPONLY=True,
    SESSION_COOKIE_SAMESITE="Lax",
    SESSION_COOKIE_SECURE=os.environ.get("SESSION_COOKIE_SECURE", "false").lower() == "true",
    MAX_CONTENT_LENGTH=1 * 1024 * 1024,
)

BASE_DIR = Path(__file__).resolve().parent
LOGIN_RATE_MAX = int(os.environ.get("LOGIN_RATE_MAX", "5"))
LOGIN_RATE_WINDOW = int(os.environ.get("LOGIN_RATE_WINDOW", "300"))
EVALUATE_RATE_MAX = int(os.environ.get("EVALUATE_RATE_MAX", "10"))
EVALUATE_RATE_WINDOW = int(os.environ.get("EVALUATE_RATE_WINDOW", "60"))
TEACHER_RATE_MAX = int(os.environ.get("TEACHER_RATE_MAX", "3"))
TEACHER_RATE_WINDOW = int(os.environ.get("TEACHER_RATE_WINDOW", "60"))

# ============================================================
# SEGURIDAD: CSRF Y LÍMITES DE SOLICITUDES
# ============================================================

def _csrf_token():
    token = session.get("_csrf_token")
    if not token:
        token = secrets.token_urlsafe(32)
        session["_csrf_token"] = token
    return token


@app.context_processor
def _inyectar_csrf():
    return {"csrf_token": _csrf_token}


@app.before_request
def _proteger_csrf():
    if request.method in {"POST", "PUT", "PATCH", "DELETE"}:
        recibido = request.form.get("csrf_token") or request.headers.get("X-CSRF-Token")
        esperado = session.get("_csrf_token", "")
        if not recibido or not esperado or not hmac.compare_digest(recibido, esperado):
            abort(400, description="Token CSRF inválido o ausente")


_rate_lock = Lock()
_rate_buckets = defaultdict(deque)

def limite_solicitudes(max_solicitudes, ventana_segundos):
    """Límite en memoria por IP y endpoint; suficiente para ejecución local."""
    def decorador(func):
        @wraps(func)
        def envuelta(*args, **kwargs):
            if request.method == "POST":
                ahora = time.monotonic()
                ip = request.remote_addr or "desconocida"
                clave = (ip, request.endpoint)
                with _rate_lock:
                    eventos = _rate_buckets[clave]
                    while eventos and ahora - eventos[0] >= ventana_segundos:
                        eventos.popleft()
                    if len(eventos) >= max_solicitudes:
                        abort(429, description="Demasiadas solicitudes. Intenta nuevamente más tarde.")
                    eventos.append(ahora)
            return func(*args, **kwargs)
        return envuelta
    return decorador


# ============================================================
# ENTRENAMIENTO EN SEGUNDO PLANO Y LIMPIEZA TRANSACCIONAL
# ============================================================

_entrenador = ThreadPoolExecutor(max_workers=1, thread_name_prefix="entrenamiento")
_trabajos_lock = Lock()
_trabajos_entrenamiento = {}

def _actualizar_trabajo(codigo, **datos):
    with _trabajos_lock:
        trabajo = _trabajos_entrenamiento.setdefault(codigo, {})
        trabajo.update(datos)
        trabajo["actualizado"] = datetime.datetime.now().isoformat(timespec="seconds")

def _snapshot_trabajos():
    with _trabajos_lock:
        return [dict(codigo=codigo, **datos) for codigo, datos in reversed(list(_trabajos_entrenamiento.items()))][:20]

def _artefactos_pregunta(codigo):
    return [
        Path(ruta_dataset_pregunta(codigo)),
        Path(ruta_modelo_pregunta(codigo)),
        Path(ruta_metadata_pregunta(codigo)),
        Path(GRAFICAS_DIR) / f"loss_pregunta_{codigo}.png",
    ]

def _limpiar_artefactos_pregunta(codigo):
    for ruta in _artefactos_pregunta(codigo):
        for candidata in (ruta, ruta.with_suffix(ruta.suffix + ".tmp"), ruta.with_suffix(".tmp")):
            try:
                candidata.unlink(missing_ok=True)
            except OSError as exc:
                print(f"⚠️ No se pudo eliminar {candidata}: {exc}")
    eliminar_pregunta_dinamica(codigo)

def _crear_pregunta_en_segundo_plano(codigo, pregunta, criterios):
    _actualizar_trabajo(codigo, estado="procesando", progreso="Generando dataset", error=None)
    try:
        generar_dataset_pregunta(codigo, pregunta, criterios, n_respuestas=50)
        _actualizar_trabajo(codigo, progreso="Generando embeddings y entrenando")
        _, metadata = entrenar_evaluador(codigo)
        _actualizar_trabajo(codigo, progreso="Publicando pregunta")
        agregar_pregunta_dinamica(codigo, pregunta, criterios)
        _actualizar_trabajo(
            codigo, estado="completado", progreso="Listo",
            proveedor=metadata.get("proveedor_embedding"), error=None
        )
    except Exception as exc:
        print(f"❌ Entrenamiento {codigo} falló: {exc}")
        _limpiar_artefactos_pregunta(codigo)
        _actualizar_trabajo(codigo, estado="error", progreso="Detenido", error=str(exc)[:500])


@app.after_request
def _agregar_cabeceras_seguridad(response):
    response.headers.setdefault("X-Content-Type-Options", "nosniff")
    response.headers.setdefault("X-Frame-Options", "DENY")
    response.headers.setdefault("Referrer-Policy", "same-origin")
    response.headers.setdefault(
        "Permissions-Policy", "camera=(), microphone=(), geolocation=()"
    )
    response.headers.setdefault("Cross-Origin-Opener-Policy", "same-origin")
    if request.endpoint in {"login", "docente", "dashboard", "historial"}:
        response.headers.setdefault("Cache-Control", "no-store")
    return response


@app.errorhandler(400)
def error_solicitud_invalida(error):
    return f"Solicitud inválida: {html.escape(str(error.description))}", 400


@app.errorhandler(429)
def error_demasiadas_solicitudes(error):
    return f"Demasiadas solicitudes: {html.escape(str(error.description))}", 429


# Ruta de wkhtmltopdf: se puede sobreescribir con la variable de entorno
# WKHTMLTOPDF_PATH (en .env) para que el proyecto corra en cualquier máquina
# sin tocar el código. Si no se define, se usa tu ruta de Windows como antes.
WKHTMLTOPDF_PATH = os.environ.get(
    "WKHTMLTOPDF_PATH",
    "C:/Program Files/wkhtmltopdf/bin/wkhtmltopdf.exe"
)

# Crear tablas al iniciar
crear_tabla_usuarios()

# Los usuarios de demostración solo se crean cuando se habilitan expresamente.
def crear_usuarios_prueba():
    registrar_usuario("profe", "profe1234", "docente", "María García")
    registrar_usuario("alumno", "alumno1234", "estudiante", "Juan Pérez")


if os.environ.get("CREATE_DEMO_USERS", "false").lower() == "true":
    crear_usuarios_prueba()


# ============================================================
# DECORADORES
# ============================================================

def login_requerido(f):
    """Cualquier usuario logueado (docente o estudiante)"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'usuario' not in session:
            flash("🔐 Debes iniciar sesión primero.", "error")
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function


def docente_requerido(f):
    """Solo docentes"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'usuario' not in session:
            flash("🔐 Debes iniciar sesión primero.", "error")
            return redirect(url_for('login'))
        if session.get('rol') != 'docente':
            flash("❌ Solo los docentes pueden acceder aquí.", "error")
            return redirect(url_for('index'))
        return f(*args, **kwargs)
    return decorated_function


# ============================================================
# RUTA: INICIO
# ============================================================
@app.route("/")
def index():
    user_info = None
    if 'usuario' in session:
        user_info = {
            'usuario': session.get('usuario'),
            'rol': session.get('rol'),
            'nombre': session.get('nombre')
        }
    return render_template("index.html", user=user_info)


# ============================================================
# RUTA: LOGIN
# ============================================================
@app.route("/login", methods=["GET", "POST"])
@limite_solicitudes(LOGIN_RATE_MAX, LOGIN_RATE_WINDOW)
def login():
    # Si ya está logueado, redirigir al index
    if 'usuario' in session:
        return redirect(url_for('index'))
    
    if request.method == "POST":
        usuario = request.form.get("usuario", "").strip()
        password = request.form.get("password", "").strip()
        
        if not usuario or not password:
            flash("❌ Usuario y contraseña son requeridos.", "error")
            return render_template("login.html")
        
        user = verificar_usuario(usuario, password)
        
        if user:
            # Evita fijación de sesión: se crea una sesión nueva al autenticar.
            session.clear()
            session['usuario'] = user['usuario']
            session['rol'] = user['rol']
            session['nombre'] = user['nombre']
            session['user_id'] = user['id']
            flash(f"✅ Bienvenido, {user['nombre'] or user['usuario']}!", "success")
            return redirect(url_for('index'))
        else:
            flash("❌ Usuario o contraseña incorrectos.", "error")
    
    return render_template("login.html")


# ============================================================
# RUTA: LOGOUT
# ============================================================
@app.route("/logout", methods=["POST"])
@login_requerido
def logout():
    session.clear()
    flash("👋 Sesión cerrada correctamente.", "success")
    return redirect(url_for('login'))



def _cargar_preguntas_entrenadas():
    """Devuelve únicamente preguntas con modelo y metadata compatibles."""
    disponibles = {}
    for codigo, info in cargar_banco_preguntas().items():
        try:
            cargar_metadata_modelo(codigo)
            if localizar_modelo_pregunta(codigo).exists():
                disponibles[codigo] = info
        except Exception:
            continue
    return disponibles

# ============================================================
# RUTA: EVALUAR (Estudiantes y Docentes)
# ============================================================
@app.route("/evaluar", methods=["GET", "POST"])
@limite_solicitudes(EVALUATE_RATE_MAX, EVALUATE_RATE_WINDOW)
def evaluar():
    preguntas = _cargar_preguntas_entrenadas()

    if request.method == "POST":
        nombre_alumno = request.form.get("nombre", "Estudiante Anónimo").strip()
        id_elegido = request.form.get("pregunta", "").strip()
        respuesta_alumno = request.form.get("respuesta", "").strip()

        if not id_elegido or id_elegido not in preguntas:
            flash("❌ Debes seleccionar una misión válida.", "error")
            return render_template("evaluar.html", preguntas=preguntas)
        if not respuesta_alumno:
            flash("❌ La respuesta no puede estar vacía.", "error")
            return render_template("evaluar.html", preguntas=preguntas)
        if len(respuesta_alumno) > 5000:
            flash("❌ La respuesta no puede superar 5000 caracteres.", "error")
            return render_template("evaluar.html", preguntas=preguntas)
        if len(nombre_alumno) > 120:
            flash("❌ El nombre no puede superar 120 caracteres.", "error")
            return render_template("evaluar.html", preguntas=preguntas)

        info_pregunta = preguntas[id_elegido]
        try:
            metadata = cargar_metadata_modelo(id_elegido)
            proveedor_modelo = metadata["proveedor_embedding"]
            ruta_modelo = localizar_modelo_pregunta(id_elegido)
            nn = NeuralNetwork()
            nn.load(str(ruta_modelo))
        except Exception as exc:
            print(f"❌ Modelo no disponible: {exc}")
            flash(
                "❌ Esta pregunta usa un modelo antiguo o incompleto. "
                "El docente debe volver a entrenarla.",
                "error",
            )
            return render_template("evaluar.html", preguntas=preguntas)

        es_plagio, porcentaje, sospechoso_de, detalle_plagio = verificar_plagio(
            respuesta_alumno, id_elegido
        )
        plagio_string = "Ninguno"
        if es_plagio:
            plagio_string = (
                f"Sospecha alta ({porcentaje:.1f}% similitud con {sospechoso_de}) — "
                f"vocabulario: {detalle_plagio.get('score_vocabulario', 0)}%, "
                f"frases: {detalle_plagio.get('score_frases', 0)}%"
            )

        try:
            vector_entradas, proveedor_embedding = transformar_respuesta_a_vector(
                respuesta_alumno,
                proveedor_preferido=proveedor_modelo,
                permitir_fallback=False,
            )
        except Exception as exc:
            print(f"❌ No se pudo generar el embedding: {exc}")
            flash(
                f"❌ El proveedor {proveedor_modelo} no está disponible. "
                "No se asignó ninguna nota.",
                "error",
            )
            return render_template("evaluar.html", preguntas=preguntas)

        try:
            prediccion_np = nn.predict(vector_entradas)
            nota_final = float(np.asarray(prediccion_np).reshape(-1)[0]) * 10
            nota_final = max(0.0, min(10.0, nota_final))
        except Exception as exc:
            print(f"❌ Error al predecir: {exc}")
            flash("❌ El modelo no pudo calcular la nota.", "error")
            return render_template("evaluar.html", preguntas=preguntas)

        feedback = generar_retroalimentacion_ia(
            info_pregunta["pregunta"],
            respuesta_alumno,
            nota_final,
            info_pregunta.get("criterios", []),
        )

        criterios_resultado = feedback.get("criterios") or [
            {
                "grupo": f"Grupo {indice + 1}: {', '.join(grupo)}",
                "cumplido": None,
                "evidencia": "Análisis no disponible",
            }
            for indice, grupo in enumerate(info_pregunta.get("criterios", []))
        ]

        try:
            guardar_registro_estudiante(
                nombre_estudiante=nombre_alumno or "Estudiante Anónimo",
                id_pregunta=id_elegido,
                pregunta_texto=info_pregunta["pregunta"],
                respuesta_alumno=respuesta_alumno,
                vector_generado=vector_entradas,
                puntuacion_ia=f"{nota_final:.2f} / 10.00",
                feedback_ia=feedback.get("retroalimentacion_alumno", ""),
                plagio_info=plagio_string,
                proveedor_embedding=proveedor_embedding,
            )
        except Exception as exc:
            print(f"⚠️ La nota se calculó, pero no se guardó: {exc}")
            flash("⚠️ La evaluación se calculó, pero no pudo guardarse.", "warning")

        return render_template(
            "resultado.html",
            nombre=nombre_alumno or "Estudiante Anónimo",
            nota=f"{nota_final:.2f}",
            criterios=criterios_resultado,
            plagio=plagio_string,
            analisis=feedback.get("analisis_error", "Análisis no disponible"),
            retroalimentacion=feedback.get(
                "retroalimentacion_alumno", "Sin retroalimentación"
            ),
        )

    return render_template("evaluar.html", preguntas=preguntas)


# ============================================================
# RUTA: DOCENTE (Solo docentes)
# ============================================================
@app.route("/docente", methods=["GET", "POST"])
@docente_requerido
@limite_solicitudes(TEACHER_RATE_MAX, TEACHER_RATE_WINDOW)
def docente():
    if request.method == "POST":
        try:
            id_p = validar_id_pregunta(request.form.get("id_pregunta", "").strip())
            preg_texto = " ".join(request.form.get("pregunta_texto", "").strip().split())
            if len(preg_texto) < 10:
                raise ValueError("La pregunta debe tener al menos 10 caracteres.")
            if pregunta_existe(id_p):
                raise ValueError(f"Ya existe una pregunta con el código '{id_p}'.")

            with _trabajos_lock:
                estado_actual = _trabajos_entrenamiento.get(id_p, {}).get("estado")
                if estado_actual in {"pendiente", "procesando"}:
                    raise ValueError("Ese código ya se está entrenando.")

            artefactos_previos = [str(r) for r in _artefactos_pregunta(id_p) if r.exists()]
            if artefactos_previos:
                raise ValueError("Existen archivos antiguos con ese código. Elimínalos o usa otro código.")

            criterios_crudos = [request.form.get(f"grupo{i}", "").split(",") for i in range(1, 6)]
            criterios = normalizar_criterios(criterios_crudos, exigir_cinco=True)

            _actualizar_trabajo(
                id_p, estado="pendiente", progreso="En cola", pregunta=preg_texto,
                iniciado_por=session.get("usuario"), error=None
            )
            _entrenador.submit(_crear_pregunta_en_segundo_plano, id_p, preg_texto, criterios)
            flash(
                f"✅ La pregunta {id_p} quedó en cola. Puedes seguir usando el sistema; "
                "el estado se actualizará en esta pantalla.",
                "success",
            )
        except Exception as exc:
            print(f"❌ Error al programar la pregunta: {exc}")
            flash(f"❌ No se pudo iniciar el entrenamiento: {exc}", "error")
        return redirect(url_for("docente"))

    return render_template("docente.html", trabajos=_snapshot_trabajos())


@app.route("/api/entrenamientos")
@docente_requerido
def estado_entrenamientos():
    return jsonify({"trabajos": _snapshot_trabajos()})


# ============================================================
# HELPERS DE ESTADÍSTICAS (compartidos por dashboard y PDF)
# ============================================================

def _parsear_nota(nota_str):
    """
    Intenta convertir 'X.XX / 10.00' a float.
    Devuelve None si el formato es inválido (en vez de descartar
    el registro en silencio sin dejar rastro del motivo).
    """
    try:
        return float(nota_str.split("/")[0].strip())
    except (AttributeError, ValueError):
        return None


def _calcular_estadisticas():
    """
    Centraliza el cálculo de estadísticas para que dashboard y el PDF
    usen siempre los mismos números. 'total' ahora cuenta solo las
    notas que sí se pudieron interpretar, para que nunca quede
    desalineado con 'promedio', 'aprobados' o 'distribución'.
    """
    conn, cursor = conectar_db()

    cursor.execute("SELECT puntuacion_ia FROM historial_evaluaciones")
    notas_raw = cursor.fetchall()

    notas = []
    notas_invalidas = 0
    for (nota_str,) in notas_raw:
        valor = _parsear_nota(nota_str)
        if valor is not None:
            notas.append(valor)
        else:
            notas_invalidas += 1

    if notas_invalidas:
        print(f"⚠️ {notas_invalidas} registro(s) con nota en formato inválido, excluidos de las estadísticas.")

    total = len(notas)
    promedio = round(sum(notas) / len(notas), 2) if notas else 0
    aprobados = sum(1 for n in notas if n >= 6)
    reprobados = total - aprobados
    porcentaje_aprobados = round((aprobados / total) * 100, 1) if notas else 0

    cursor.execute("SELECT DISTINCT id_pregunta FROM historial_evaluaciones")
    preguntas_ids = [row[0] for row in cursor.fetchall()]

    stats_por_pregunta = []
    for pid in preguntas_ids:
        cursor.execute("SELECT puntuacion_ia FROM historial_evaluaciones WHERE id_pregunta = ?", (pid,))
        notas_p = [v for (ns,) in cursor.fetchall() if (v := _parsear_nota(ns)) is not None]
        if notas_p:
            stats_por_pregunta.append({
                "id": pid, "total": len(notas_p),
                "promedio": round(sum(notas_p) / len(notas_p), 2),
                "maxima": round(max(notas_p), 2),
                "minima": round(min(notas_p), 2)
            })

    excelente = sum(1 for n in notas if n >= 9)
    bueno = sum(1 for n in notas if 7 <= n < 9)
    regular = sum(1 for n in notas if 5 <= n < 7)
    malo = sum(1 for n in notas if n < 5)
    distribucion = {"excelente": excelente, "bueno": bueno, "regular": regular, "malo": malo}

    cursor.execute("SELECT estudiante, id_pregunta, puntuacion_ia, fecha FROM historial_evaluaciones ORDER BY id DESC LIMIT 20")
    ultimas = cursor.fetchall()

    cursor.execute("SELECT puntuacion_ia, fecha FROM historial_evaluaciones ORDER BY id ASC")
    todas = cursor.fetchall()

    historial_notas = []
    indice = 0
    for nota_str, fecha in todas:
        valor = _parsear_nota(nota_str)
        if valor is not None:
            indice += 1
            historial_notas.append({"indice": indice, "nota": valor, "fecha": fecha})

    conn.close()

    return {
        "total": total, "promedio": promedio, "aprobados": aprobados,
        "reprobados": reprobados, "porcentaje_aprobados": porcentaje_aprobados,
        "stats_por_pregunta": stats_por_pregunta, "distribucion": distribucion,
        "ultimas": ultimas, "historial_notas": historial_notas
    }


# ============================================================
# RUTA: DASHBOARD (Solo docentes)
# ============================================================
@app.route("/dashboard")
@docente_requerido
def dashboard():
    try:
        stats = _calcular_estadisticas()
        # dashboard.html solo usa las últimas 10, igual que en el original
        stats["ultimas"] = stats["ultimas"][:10]
        return render_template("dashboard.html", **stats)
    except Exception as e:
        print(f"❌ Error en dashboard: {e}")
        return render_template(
            "dashboard.html", total=0, promedio=0, aprobados=0, reprobados=0,
            porcentaje_aprobados=0, stats_por_pregunta=[],
            distribucion={"excelente": 0, "bueno": 0, "regular": 0, "malo": 0},
            ultimas=[], historial_notas=[]
        )


# ============================================================
# RUTA: HISTORIAL (Solo docentes)
# ============================================================
@app.route("/historial")
@docente_requerido
def historial():
    try:
        conn, cursor = conectar_db()
        cursor.execute("SELECT id, estudiante, id_pregunta, puntuacion_ia, alerta_plagio, fecha, proveedor_embedding FROM historial_evaluaciones ORDER BY id DESC")
        registros = cursor.fetchall()
        conn.close()
    except Exception as e:
        registros = []
    return render_template("historial.html", registros=registros)


# ============================================================
# RUTA: DESCARGAR PDF (Solo docentes)
# ============================================================
@app.route("/descargar-pdf")
@docente_requerido
def descargar_pdf():
    try:
        stats = _calcular_estadisticas()

        fecha_actual = datetime.datetime.now().strftime('%d/%m/%Y %H:%M')
        año_actual = datetime.datetime.now().year

        filas_mision = ""
        for s in stats["stats_por_pregunta"]:
            filas_mision += (
                "<tr><td>Pregunta " + html.escape(str(s["id"])) + "</td>"
                f"<td>{int(s['total'])}</td><td>{float(s['promedio']):.2f}</td>"
                f"<td style='color:#4ade80;'>{float(s['maxima']):.2f}</td>"
                f"<td style='color:#f87171;'>{float(s['minima']):.2f}</td></tr>"
            )

        filas_ultimas = ""
        for u in stats["ultimas"]:
            nota_str = u[2]
            valor = _parsear_nota(nota_str)
            if valor is not None:
                clase = "color:#4ade80;" if valor >= 7 else ("color:#fbbf24;" if valor >= 5 else "color:#f87171;")
            else:
                clase = ""
            filas_ultimas += (
                "<tr><td>" + html.escape(str(u[0])) + "</td>"
                "<td>Pregunta " + html.escape(str(u[1])) + "</td>"
                f"<td style='{clase}'>" + html.escape(str(nota_str)) + "</td>"
                "<td>" + html.escape(str(u[3])) + "</td></tr>"
            )

        d = stats["distribucion"]
        html_reporte = f"""<!DOCTYPE html><html lang="es"><head><meta charset="UTF-8"><style>
            body{{font-family:Arial,sans-serif;background:#0a0a1a;color:#e0e0f0;padding:30px;}}
            .header{{text-align:center;border-bottom:2px solid #7c8aff;padding-bottom:15px;margin-bottom:25px;}}
            .header h1{{color:#7c8aff;font-size:26px;margin:0;}}
            .header p{{color:#9090b0;font-size:13px;}}
            .stats-grid{{display:flex;gap:12px;margin-bottom:25px;}}
            .stat-card{{flex:1;background:#1a1a2e;border:1px solid #2a2a40;border-radius:8px;padding:18px;text-align:center;}}
            .stat-value{{font-size:28px;font-weight:bold;color:#d0d8ff;}}
            .stat-label{{color:#9090b0;font-size:11px;margin-top:4px;}}
            .section{{margin-bottom:22px;}}
            .section h2{{color:#7c8aff;font-size:16px;border-bottom:1px solid #2a2a40;padding-bottom:6px;}}
            table{{width:100%;border-collapse:collapse;margin-top:10px;}}
            th{{background:#1a1a2e;padding:8px 10px;text-align:left;color:#7c8aff;font-size:11px;border-bottom:2px solid #2a2a40;}}
            td{{padding:7px 10px;border-bottom:1px solid #1a1a2e;font-size:12px;}}
            .footer{{text-align:center;margin-top:35px;padding-top:15px;border-top:1px solid #2a2a40;color:#9090b0;font-size:10px;}}
        </style></head><body>
            <div class="header"><h1>🚀 REPORTE DE MISIÓN</h1><p>Estación Espacial Evaluadora — {fecha_actual}</p></div>
            <div class="stats-grid">
                <div class="stat-card"><div class="stat-value">{stats['total']}</div><div class="stat-label">Misiones Totales</div></div>
                <div class="stat-card"><div class="stat-value">{stats['promedio']}</div><div class="stat-label">Promedio General</div></div>
                <div class="stat-card"><div class="stat-value">{stats['porcentaje_aprobados']}%</div><div class="stat-label">Tasa de Aprobación</div></div>
                <div class="stat-card"><div class="stat-value">{stats['aprobados']}/{stats['reprobados']}</div><div class="stat-label">Aprobados/Reprobados</div></div>
            </div>
            <div class="section"><h2>📊 Distribución de Notas</h2><table>
                <tr><th>Categoría</th><th>Cantidad</th></tr>
                <tr><td>🌟 Excelente (9-10)</td><td>{d['excelente']}</td></tr>
                <tr><td>🛰️ Bueno (7-8)</td><td>{d['bueno']}</td></tr>
                <tr><td>🔧 Regular (5-6)</td><td>{d['regular']}</td></tr>
                <tr><td>⚠️ Malo (&lt;5)</td><td>{d['malo']}</td></tr>
            </table></div>
            <div class="section"><h2>📋 Rendimiento por Misión</h2><table>
                <tr><th>Misión</th><th>Total</th><th>Promedio</th><th>Máxima</th><th>Mínima</th></tr>
                {filas_mision}</table></div>
            <div class="section"><h2>🕐 Últimas Misiones</h2><table>
                <tr><th>Tripulante</th><th>Misión</th><th>Puntuación</th><th>Fecha</th></tr>
                {filas_ultimas}</table></div>
            <div class="footer"><p>📡 Estación Espacial Evaluadora v3.0 | Reporte generado automáticamente</p><p>© {año_actual}</p></div>
        </body></html>"""

        config = pdfkit.configuration(wkhtmltopdf=WKHTMLTOPDF_PATH)
        pdf = pdfkit.from_string(html_reporte, False, configuration=config)

        from flask import make_response
        response = make_response(pdf)
        response.headers['Content-Type'] = 'application/pdf'
        response.headers['Content-Disposition'] = 'attachment; filename=reporte_mision.pdf'
        return response

    except Exception as e:
        print(f"❌ Error al generar PDF: {e}")
        flash("❌ Error al generar el PDF.")
        return redirect(url_for("dashboard"))


# ============================================================
# INICIO
# ============================================================
if __name__ == "__main__":
    modo_debug = os.environ.get("FLASK_DEBUG", "false").lower() == "true"
    app.run(debug=modo_debug)