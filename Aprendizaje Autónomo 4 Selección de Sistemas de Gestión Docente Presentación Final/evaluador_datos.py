import os
import re
import json
import sqlite3
import unicodedata
import hashlib
import numpy as np
import requests
import torch
import time
import shutil
import secrets
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()
API_KEY_GEMINI = os.environ.get("GEMINI_API_KEY")
API_KEY_GROQ = os.environ.get("GROQ_API_KEY")
API_KEY_COHERE = os.environ.get("API_KEY_COHERE")
API_KEY_NVIDIA = os.environ.get("API_KEY_NVIDIA")

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)
PREGUNTAS_FILE = DATA_DIR / "banco_preguntas.json"
LEGACY_PREGUNTAS_FILE = BASE_DIR / "banco_preguntas.json"
DB_FILE = str(BASE_DIR / "sistema_evaluacion.db")

_client_gemini = None
_client_groq = None


def _obtener_cliente_gemini():
    global _client_gemini
    if not API_KEY_GEMINI:
        raise RuntimeError("Falta GEMINI_API_KEY en el archivo .env")
    if _client_gemini is None:
        try:
            from google import genai
        except ImportError as exc:
            raise RuntimeError(
                "Falta instalar google-genai: pip install google-genai"
            ) from exc
        _client_gemini = genai.Client(
            api_key=API_KEY_GEMINI,
            http_options={"api_version": "v1"}
        )
    return _client_gemini


def _obtener_cliente_groq():
    global _client_groq
    if not API_KEY_GROQ:
        raise RuntimeError("Falta GROQ_API_KEY en el archivo .env")
    if _client_groq is None:
        try:
            from groq import Groq
        except ImportError as exc:
            raise RuntimeError("Falta instalar groq: pip install groq") from exc
        _client_groq = Groq(api_key=API_KEY_GROQ)
    return _client_groq

# ============================================================
# 1. CONTROL Y PERSISTENCIA CON BASE DE DATOS (SQLite)
# ============================================================

def conectar_db():
    """Conecta a SQLite y mantiene actualizada la estructura mínima de la BD."""
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS historial_evaluaciones (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            estudiante TEXT NOT NULL,
            id_pregunta TEXT NOT NULL,
            pregunta TEXT,
            respuesta_estudiante TEXT,
            vector_neurona TEXT,
            puntuacion_ia TEXT,
            feedback_ia TEXT,
            alerta_plagio TEXT,
            proveedor_embedding TEXT DEFAULT 'Desconocido',
            fecha TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # Migración automática para bases creadas con versiones anteriores.
    cursor.execute("PRAGMA table_info(historial_evaluaciones)")
    columnas = {fila[1] for fila in cursor.fetchall()}
    if "proveedor_embedding" not in columnas:
        cursor.execute(
            "ALTER TABLE historial_evaluaciones "
            "ADD COLUMN proveedor_embedding TEXT DEFAULT 'Desconocido'"
        )

    conn.commit()
    return conn, cursor


def cargar_banco_preguntas():
    """Carga el banco de preguntas desde ``data/banco_preguntas.json``.

    Si existe el archivo usado por versiones anteriores en la raíz del proyecto,
    lo migra automáticamente a la carpeta ``data``.
    """
    banco_base = {
        "1": {
            "pregunta": "¿En qué año se dio la Revolución Francesa y qué sistema cayó?",
            "criterios": []
        },
        "2": {
            "pregunta": "¿Cómo se repite un bloque de código en Python de forma controlada?",
            "criterios": []
        }
    }

    if not PREGUNTAS_FILE.exists() and LEGACY_PREGUNTAS_FILE.exists():
        shutil.copy2(LEGACY_PREGUNTAS_FILE, PREGUNTAS_FILE)
        print(f"📦 Banco de preguntas migrado a {PREGUNTAS_FILE}")

    if not PREGUNTAS_FILE.exists():
        with PREGUNTAS_FILE.open("w", encoding="utf-8") as archivo:
            json.dump(banco_base, archivo, ensure_ascii=False, indent=4)
        return banco_base

    try:
        with PREGUNTAS_FILE.open("r", encoding="utf-8") as archivo:
            banco = json.load(archivo)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"El banco de preguntas contiene JSON inválido: {exc}"
        ) from exc

    if not isinstance(banco, dict):
        raise ValueError("El banco de preguntas debe ser un objeto JSON")
    return banco



def validar_id_pregunta(id_pregunta):
    """Valida que el código sea seguro para usarlo en nombres de archivos."""
    codigo = str(id_pregunta).strip()
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,30}", codigo):
        raise ValueError(
            "El código de la pregunta solo puede contener letras, números, "
            "guion y guion bajo; máximo 30 caracteres."
        )
    return codigo


def normalizar_criterios(lista_criterios, exigir_cinco=False):
    """Limpia grupos de criterios y elimina valores vacíos o repetidos."""
    grupos_limpios = []
    for grupo in lista_criterios or []:
        if isinstance(grupo, str):
            elementos = grupo.split(",")
        else:
            elementos = list(grupo or [])

        vistos = set()
        limpios = []
        for criterio in elementos:
            criterio_limpio = " ".join(str(criterio).strip().split())
            clave = criterio_limpio.casefold()
            if criterio_limpio and clave not in vistos:
                vistos.add(clave)
                limpios.append(criterio_limpio)

        if limpios:
            grupos_limpios.append(limpios)

    if exigir_cinco and len(grupos_limpios) != 5:
        raise ValueError(
            "Debes registrar exactamente 5 grupos de criterios no vacíos."
        )
    return grupos_limpios


def pregunta_existe(id_pregunta):
    codigo = validar_id_pregunta(id_pregunta)
    return codigo in cargar_banco_preguntas()

def agregar_pregunta_dinamica(
    id_pregunta, pregunta_texto, lista_criterios=None, sobrescribir=False
):
    """Guarda una pregunta junto con sus criterios reales."""
    codigo = validar_id_pregunta(id_pregunta)
    pregunta = " ".join(str(pregunta_texto).strip().split())
    if len(pregunta) < 10:
        raise ValueError("La pregunta debe tener al menos 10 caracteres")

    criterios = normalizar_criterios(lista_criterios, exigir_cinco=True)
    banco = cargar_banco_preguntas()
    if codigo in banco and not sobrescribir:
        raise ValueError(
            f"Ya existe una pregunta con el código '{codigo}'. Usa otro código."
        )

    banco[codigo] = {
        "pregunta": pregunta,
        "criterios": criterios
    }

    temporal = PREGUNTAS_FILE.with_suffix(".json.tmp")
    with temporal.open("w", encoding="utf-8") as archivo:
        json.dump(banco, archivo, ensure_ascii=False, indent=4)
    temporal.replace(PREGUNTAS_FILE)

    print(f"\n[Docente] Pregunta {codigo} guardada con 5 grupos de criterios.")
    return banco[codigo]


def eliminar_pregunta_dinamica(id_pregunta):
    """Elimina una pregunta del banco de forma atómica si existe."""
    codigo = validar_id_pregunta(id_pregunta)
    banco = cargar_banco_preguntas()
    if codigo not in banco:
        return False

    banco.pop(codigo, None)
    temporal = PREGUNTAS_FILE.with_suffix(".json.tmp")
    with temporal.open("w", encoding="utf-8") as archivo:
        json.dump(banco, archivo, ensure_ascii=False, indent=4)
    temporal.replace(PREGUNTAS_FILE)
    return True


def guardar_registro_estudiante(
    nombre_estudiante, id_pregunta, pregunta_texto, respuesta_alumno,
    vector_generado, puntuacion_ia="N/A", feedback_ia="N/A",
    plagio_info="Ninguno", proveedor_embedding="Desconocido"
):
    """Guarda una evaluación y el proveedor que generó su embedding."""
    conn = None
    try:
        conn, cursor = conectar_db()

        if isinstance(vector_generado, torch.Tensor):
            vector_np = vector_generado.detach().squeeze(0).cpu().numpy()
        elif isinstance(vector_generado, np.ndarray):
            vector_np = vector_generado.ravel()
        else:
            vector_np = np.asarray(vector_generado, dtype=np.float32).ravel()

        if vector_np.size == 0:
            raise ValueError("El vector generado está vacío")

        vector_str = ",".join(f"{x:.4f}" for x in vector_np[:5])
        vector_str += f"... [Truncado Semántico {vector_np.size}]"

        cursor.execute("""
            INSERT INTO historial_evaluaciones
            (
                estudiante, id_pregunta, pregunta, respuesta_estudiante,
                vector_neurona, puntuacion_ia, feedback_ia, alerta_plagio,
                proveedor_embedding
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            nombre_estudiante,
            str(id_pregunta),
            pregunta_texto,
            respuesta_alumno,
            vector_str,
            puntuacion_ia,
            feedback_ia,
            plagio_info,
            proveedor_embedding or "Desconocido"
        ))

        conn.commit()
        print(
            f"[Sistema - SQLite] Registro de '{nombre_estudiante}' guardado "
            f"con embedding de {proveedor_embedding}."
        )
        return True
    except Exception as e:
        if conn is not None:
            conn.rollback()
        print(f"❌ Error al interactuar con SQLite: {e}")
        raise
    finally:
        if conn is not None:
            conn.close()


# ============================================================
# 2. ANÁLISIS DE PLAGIO OPTIMIZADO DESDE SQLITE
# ============================================================

def normalizar_texto(texto):
    """Limpia el texto: minúsculas, sin acentos, sin puntuación."""
    if not texto:
        return ""
    texto = texto.lower()
    texto = ''.join(
        c for c in unicodedata.normalize('NFD', texto)
        if unicodedata.category(c) != 'Mn'
    )
    texto = re.sub(re.compile(r'[.,;:?¿!¡"()\'\-]'), ' ', texto)
    return ' '.join(texto.split())


# Palabras vacías en español que no aportan significado
_STOPWORDS = {
    'el','la','los','las','un','una','unos','unas','de','del','al','a','en',
    'y','o','que','se','su','sus','por','para','con','sin','es','son','fue',
    'ser','como','pero','si','lo','le','les','me','te','nos','cuando','donde',
    'este','esta','estos','estas','ese','esa','esos','esas','hay','han','era',
    'ante','bajo','sobre','tras','entre','durante','mediante','segun','hacia',
    'mas','muy','bien','mal','asi','tambien','aun','ya','todo','toda','cada'
}


def _palabras_significativas(texto):
    """Devuelve solo las palabras con contenido semántico real."""
    return [p for p in normalizar_texto(texto).split()
            if p not in _STOPWORDS and len(p) > 2]


def _similitud_jaccard(t1, t2):
    """
    Compara conjuntos de palabras significativas.
    Detecta mismo vocabulario aunque el orden cambie.
    """
    s1 = set(_palabras_significativas(t1))
    s2 = set(_palabras_significativas(t2))
    if not s1 or not s2:
        return 0.0
    return len(s1 & s2) / len(s1 | s2)


def _similitud_ngramas(t1, t2, n=2):
    """
    Compara bigramas de palabras significativas.
    Detecta frases copiadas aunque estén mezcladas con otro texto.
    """
    from collections import Counter

    def ngramas(texto):
        words = _palabras_significativas(texto)
        return Counter(tuple(words[i:i + n]) for i in range(len(words) - n + 1))

    c1, c2 = ngramas(t1), ngramas(t2)
    if not c1 or not c2:
        return 0.0
    interseccion = sum((c1 & c2).values())
    return interseccion / max(sum(c1.values()), sum(c2.values()))


def _similitud_combinada(t1, t2):
    """
    Combina tres métricas complementarias:
      - SequenceMatcher (25%): detecta copia literal carácter a carácter
      - Jaccard         (45%): detecta mismo vocabulario reordenado
      - Bigramas        (30%): detecta frases copiadas aunque estén mezcladas

    Retorna (score_combinado, score_seq, score_jac, score_ngr)
    """
    from difflib import SequenceMatcher as SM
    seq = SM(None, normalizar_texto(t1), normalizar_texto(t2)).ratio()
    jac = _similitud_jaccard(t1, t2)
    ngr = _similitud_ngramas(t1, t2, n=2)
    combinado = round(0.25 * seq + 0.45 * jac + 0.30 * ngr, 4)
    return combinado, seq, jac, ngr


def verificar_plagio(respuesta_nueva, id_pregunta, umbral=0.45):
    """
    Detecta posible plagio comparando la respuesta nueva contra todas las
    respuestas anteriores de la misma pregunta.

    Usa tres métricas combinadas (SequenceMatcher + Jaccard + bigramas)
    con umbral calibrado en 0.45 — más sensible que el SequenceMatcher
    solo (0.70), capturando paráfrasis leves sin generar falsos positivos
    en respuestas distintas sobre el mismo tema.

    Retorna:
        (es_plagio: bool, porcentaje: float, estudiante_sospechoso: str,
         detalle: dict con los 3 scores individuales)
    """
    try:
        conn, cursor = conectar_db()
        cursor.execute("""
            SELECT estudiante, respuesta_estudiante
            FROM historial_evaluaciones
            WHERE id_pregunta = ?
        """, (str(id_pregunta),))
        registros = cursor.fetchall()
        conn.close()
    except Exception:
        return False, 0.0, "", {}

    if not registros:
        return False, 0.0, "", {}

    # Calcular similitud contra TODOS los registros y quedarse con el más alto
    mejor_similitud = 0.0
    mejor_estudiante = ""
    mejor_detalle = {}

    for estudiante_anterior, respuesta_anterior in registros:
        if not respuesta_anterior:
            continue
        comb, seq, jac, ngr = _similitud_combinada(respuesta_nueva, respuesta_anterior)
        if comb > mejor_similitud:
            mejor_similitud = comb
            mejor_estudiante = estudiante_anterior
            mejor_detalle = {
                "score_combinado": round(comb * 100, 1),
                "score_secuencia": round(seq * 100, 1),
                "score_vocabulario": round(jac * 100, 1),
                "score_frases": round(ngr * 100, 1)
            }

    if mejor_similitud >= umbral:
        return True, mejor_similitud * 100, mejor_estudiante, mejor_detalle

    return False, mejor_similitud * 100, "", mejor_detalle


# ============================================================
# 3. FUNCIONES DE EMBEDDING
#    Proveedores admitidos: Gemini, Cohere, NVIDIA y vocabulario local.
#    Groq se reserva para generación de texto; no se califican vectores nulos.
# ============================================================

def reducir_dimensiones(embedding, dim_objetivo=768):
    dim_original = len(embedding)
    if dim_original == dim_objetivo:
        return embedding

    chunk_size = dim_original / dim_objetivo
    embedding_reducido = np.array([
        np.mean(embedding[int(i*chunk_size):int((i+1)*chunk_size)])
        for i in range(dim_objetivo)
    ], dtype=np.float32)

    return embedding_reducido


def obtener_embedding_gemini(texto):
    cliente = _obtener_cliente_gemini()
    try:
        from google.genai import types
        resultado = cliente.models.embed_content(
            model="gemini-embedding-2",
            contents=texto,
            config=types.EmbedContentConfig(output_dimensionality=768),
        )
        embedding = np.array(resultado.embeddings[0].values, dtype=np.float32)
        return embedding
    except Exception as exc:
        print(f"      ⚠️ Error Gemini: {str(exc)[:150]}")
        raise


def obtener_embedding_groq(texto):
    raise RuntimeError(
        "Groq se usa para generar texto, pero no ofrece un modelo de "
        "embeddings compatible con este proyecto."
    )


def obtener_embedding_cohere(texto):
    if not API_KEY_COHERE:
        raise Exception("Falta API_KEY_COHERE en .env")

    url = "https://api.cohere.com/v1/embed"
    headers = {
        "Authorization": f"Bearer {API_KEY_COHERE}",
        "Content-Type": "application/json"
    }
    payload = {
        "texts": [texto],
        "model": "embed-multilingual-v3.0",
        "input_type": "search_document"
    }

    try:
        respuesta = requests.post(url, json=payload, headers=headers, timeout=10)
        if respuesta.status_code == 200:
            embedding = np.array(respuesta.json()["embeddings"][0], dtype=np.float32)
            dims = len(embedding)
            print(f"      📏 Cohere: {dims} dimensiones → reduciendo a 768")
            embedding = reducir_dimensiones(embedding, 768)
            return embedding
        else:
            raise Exception(f"Código {respuesta.status_code}: {respuesta.text[:200]}")
    except requests.exceptions.Timeout:
        raise Exception("Timeout en Cohere API")
    except requests.exceptions.RequestException as e:
        raise Exception(f"Error de red Cohere: {str(e)[:100]}")


def obtener_embedding_nvidia(texto):
    """Plan D: NVIDIA NIM embeddings"""
    if not API_KEY_NVIDIA:
        raise Exception("Falta API_KEY_NVIDIA en .env")

    try:
        respuesta = requests.post(
            "https://integrate.api.nvidia.com/v1/embeddings",
            headers={
                "Authorization": f"Bearer {API_KEY_NVIDIA}",
                "Content-Type": "application/json"
            },
            json={
                "model": "nvidia/nv-embedqa-e5-v5",
                "input": texto,
                "encoding_format": "float"
            },
            timeout=10
        )
        if respuesta.status_code == 200:
            embedding = np.array(respuesta.json()["data"][0]["embedding"], dtype=np.float32)
            dims = len(embedding)
            print(f"      📏 NVIDIA: {dims} dimensiones → reduciendo a 768")
            embedding = reducir_dimensiones(embedding, 768)
            return embedding
        else:
            raise Exception(f"Código {respuesta.status_code}: {respuesta.text[:200]}")
    except requests.exceptions.Timeout:
        raise Exception("Timeout en NVIDIA API")
    except requests.exceptions.RequestException as e:
        raise Exception(f"Error de red NVIDIA: {str(e)[:100]}")


def _preparar_embedding(embedding, proveedor):
    """Valida, normaliza la forma y devuelve (tensor, proveedor)."""
    vector = np.asarray(embedding, dtype=np.float32).reshape(-1)

    if vector.size == 0:
        raise ValueError(f"{proveedor} devolvió un embedding vacío")
    if not np.all(np.isfinite(vector)):
        raise ValueError(f"{proveedor} devolvió valores no válidos")

    if vector.size < 768:
        vector = np.pad(vector, (0, 768 - vector.size), mode="constant")
    elif vector.size > 768:
        vector = reducir_dimensiones(vector, 768)

    norma = float(np.linalg.norm(vector))
    if norma <= 1e-12:
        raise ValueError(f"{proveedor} devolvió un vector sin información")
    vector = vector / norma

    tensor = torch.tensor(vector, dtype=torch.float32).unsqueeze(0)
    return tensor, proveedor



def generar_embedding_con_proveedor(texto, proveedor):
    """Genera un vector usando exclusivamente el proveedor indicado."""
    nombre = str(proveedor).strip()
    funciones = {
        "Gemini": obtener_embedding_gemini,
        "Cohere": obtener_embedding_cohere,
        "NVIDIA": obtener_embedding_nvidia,
        "Vocabulario local": texto_a_vector_local,
    }
    if nombre not in funciones:
        raise ValueError(f"Proveedor de embedding no reconocido: {nombre}")

    embedding = funciones[nombre](texto)
    tensor, proveedor_real = _preparar_embedding(embedding, nombre)
    if nombre == "Vocabulario local" and torch.count_nonzero(tensor).item() == 0:
        raise RuntimeError(
            "El vocabulario local no reconoce palabras suficientes de la respuesta"
        )
    return tensor, proveedor_real

def transformar_respuesta_a_vector(
    respuesta_alumno,
    lista_criterios_dummy=None,
    umbral_dummy=None,
    proveedor_preferido=None,
    permitir_fallback=True,
):
    """Genera un embedding válido y devuelve ``(tensor, proveedor)``.

    Cuando ``proveedor_preferido`` se especifica y ``permitir_fallback`` es
    ``False``, se usa exclusivamente ese proveedor. Esto garantiza que una red
    sea evaluada en el mismo espacio vectorial con el que fue entrenada.
    """
    if not isinstance(respuesta_alumno, str) or not respuesta_alumno.strip():
        raise ValueError("La respuesta del estudiante no puede estar vacía")

    if proveedor_preferido:
        try:
            return generar_embedding_con_proveedor(
                respuesta_alumno, proveedor_preferido
            )
        except Exception:
            if not permitir_fallback:
                raise

    orden = ["Gemini", "Cohere", "NVIDIA", "Vocabulario local"]
    if proveedor_preferido in orden:
        orden.remove(proveedor_preferido)

    errores = []
    for proveedor in orden:
        try:
            print(f"      🧠 [{proveedor}] Generando embedding...")
            tensor, nombre = generar_embedding_con_proveedor(
                respuesta_alumno, proveedor
            )
            print(f"      ✅ Embedding listo: {tuple(tensor.shape)}")
            return tensor, nombre
        except Exception as exc:
            mensaje = str(exc)[:160]
            errores.append(f"{proveedor}: {mensaje}")
            print(f"      ⚠️ {proveedor} falló: {mensaje}")
            if proveedor == "Gemini" and (
                "429" in mensaje or "RESOURCE_EXHAUSTED" in mensaje
            ):
                time.sleep(3)

    raise RuntimeError(
        "No fue posible generar un embedding válido. La evaluación no se "
        "calificó. Detalle: " + " | ".join(errores)
    )


# ============================================================
# 4. FUNCIONES DE USUARIOS (LOGIN)
# ============================================================

def crear_tabla_usuarios():
    """Crea la tabla y genera un administrador inicial seguro si está vacía."""
    from werkzeug.security import generate_password_hash

    conn, cursor = conectar_db()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS usuarios (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            usuario TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            rol TEXT NOT NULL DEFAULT 'docente',
            nombre_completo TEXT,
            fecha_creacion TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    cursor.execute("SELECT COUNT(*) FROM usuarios")
    if cursor.fetchone()[0] == 0:
        password_inicial = os.environ.get("ADMIN_INITIAL_PASSWORD")
        generada = False
        if not password_inicial:
            password_inicial = secrets.token_urlsafe(12)
            generada = True
        cursor.execute(
            "INSERT INTO usuarios "
            "(usuario, password_hash, rol, nombre_completo) VALUES (?, ?, ?, ?)",
            (
                "admin",
                generate_password_hash(password_inicial),
                "docente",
                "Administrador Principal",
            ),
        )
        print("✅ Usuario administrador inicial creado: admin")
        if generada:
            print(f"🔐 Contraseña temporal generada: {password_inicial}")
            print("   Guárdala ahora y cámbiala antes de publicar el sistema.")
    conn.commit()
    conn.close()


def verificar_usuario(usuario, password):
    """Verifica hashes seguros y migra automáticamente hashes SHA-256 antiguos."""
    from werkzeug.security import check_password_hash, generate_password_hash

    conn, cursor = conectar_db()
    cursor.execute(
        "SELECT id, usuario, rol, nombre_completo, password_hash "
        "FROM usuarios WHERE usuario = ?",
        (usuario,),
    )
    fila = cursor.fetchone()
    if not fila:
        conn.close()
        return None

    hash_guardado = fila[4]
    es_sha256_antiguo = bool(re.fullmatch(r"[0-9a-f]{64}", hash_guardado or ""))
    valido = False
    if es_sha256_antiguo:
        valido = hashlib.sha256(password.encode()).hexdigest() == hash_guardado
        if valido:
            cursor.execute(
                "UPDATE usuarios SET password_hash = ? WHERE id = ?",
                (generate_password_hash(password), fila[0]),
            )
            conn.commit()
    else:
        try:
            valido = check_password_hash(hash_guardado, password)
        except ValueError:
            valido = False

    conn.close()
    if valido:
        return {
            "id": fila[0],
            "usuario": fila[1],
            "rol": fila[2],
            "nombre": fila[3],
        }
    return None


def registrar_usuario(usuario, password, rol="docente", nombre_completo=""):
    from werkzeug.security import generate_password_hash

    usuario = str(usuario).strip()
    rol = str(rol).strip().lower()
    if not re.fullmatch(r"[A-Za-z0-9_.-]{3,40}", usuario):
        raise ValueError("El usuario debe tener entre 3 y 40 caracteres válidos")
    if len(str(password)) < 8:
        raise ValueError("La contraseña debe tener al menos 8 caracteres")
    if rol not in {"docente", "estudiante"}:
        raise ValueError("El rol debe ser docente o estudiante")

    conn, cursor = conectar_db()
    try:
        cursor.execute(
            "INSERT INTO usuarios "
            "(usuario, password_hash, rol, nombre_completo) VALUES (?, ?, ?, ?)",
            (
                usuario,
                generate_password_hash(password),
                rol,
                str(nombre_completo).strip(),
            ),
        )
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        print(f"❌ El usuario '{usuario}' ya existe.")
        return False
    finally:
        conn.close()


# ============================================================
# 5. VOCABULARIO LITERARIO (Embeddings de palabras - OFFLINE)
# ============================================================

_modelo_vocabulario = None

def cargar_modelo_vocabulario():
    """Carga un Word2Vec local compatible con el evaluador.

    Acepta el formato legado ``data/vocabulario.json`` o el par
    ``data/vocabulario_pytorch.json`` + ``data/vocabulario_pytorch.pt``.
    El Transformer ``transformer_model.pt`` no se usa como embedding de frases.
    """
    global _modelo_vocabulario

    ruta_legacy = DATA_DIR / "vocabulario.json"
    if ruta_legacy.exists():
        try:
            with ruta_legacy.open("r", encoding="utf-8") as archivo:
                data = json.load(archivo)
            requeridos = {"palabra_a_idx", "embeddings", "vector_size"}
            if not requeridos.issubset(data):
                raise ValueError("faltan campos en vocabulario.json")
            _modelo_vocabulario = data
            return True
        except Exception as exc:
            print(f"⚠️ No se pudo cargar el vocabulario legado: {exc}")

    ruta_json = DATA_DIR / "vocabulario_pytorch.json"
    ruta_pt = DATA_DIR / "vocabulario_pytorch.pt"
    if ruta_json.exists() and ruta_pt.exists():
        try:
            with ruta_json.open("r", encoding="utf-8") as archivo:
                vocab_data = json.load(archivo)
            checkpoint = torch.load(ruta_pt, map_location="cpu")
            pesos = checkpoint["model_state_dict"]["in_embed.weight"]
            _modelo_vocabulario = {
                "palabra_a_idx": vocab_data["palabra_a_idx"],
                "embeddings": pesos.detach().cpu().numpy(),
                "vector_size": int(pesos.shape[1]),
            }
            return True
        except Exception as exc:
            print(f"⚠️ No se pudo cargar el Word2Vec PyTorch: {exc}")

    print(
        "⚠️ No existe un vocabulario local compatible. Se requiere "
        "data/vocabulario.json o vocabulario_pytorch.json + .pt."
    )
    return False


def palabra_a_vector(palabra):
    global _modelo_vocabulario
    if _modelo_vocabulario is None:
        if not cargar_modelo_vocabulario():
            return None
    palabra = palabra.lower().strip()
    if palabra in _modelo_vocabulario['palabra_a_idx']:
        idx = _modelo_vocabulario['palabra_a_idx'][palabra]
        return np.array(_modelo_vocabulario['embeddings'][idx], dtype=np.float32)
    return np.zeros(_modelo_vocabulario['vector_size'], dtype=np.float32)


def texto_a_vector_local(texto):
    global _modelo_vocabulario
    if _modelo_vocabulario is None:
        if not cargar_modelo_vocabulario():
            return None
    texto = texto.lower()
    texto = re.sub(r'[^\w\sáéíóúüñ]', ' ', texto)
    texto = re.sub(r'\d+', '', texto)
    palabras = texto.split()
    vectores = []
    for palabra in palabras:
        if palabra in _modelo_vocabulario['palabra_a_idx']:
            idx = _modelo_vocabulario['palabra_a_idx'][palabra]
            vectores.append(np.array(_modelo_vocabulario['embeddings'][idx], dtype=np.float32))
    if vectores:
        return np.mean(vectores, axis=0)
    else:
        return np.zeros(_modelo_vocabulario['vector_size'], dtype=np.float32)


def palabras_similares(palabra, topn=5):
    global _modelo_vocabulario
    if _modelo_vocabulario is None:
        if not cargar_modelo_vocabulario():
            return []
    palabra = palabra.lower().strip()
    if palabra not in _modelo_vocabulario['palabra_a_idx']:
        print(f"❌ '{palabra}' no está en el vocabulario")
        return []
    idx = _modelo_vocabulario['palabra_a_idx'][palabra]
    vec = np.array(_modelo_vocabulario['embeddings'][idx])
    similitudes = []
    palabra_a_idx = _modelo_vocabulario["palabra_a_idx"]
    for p, i in palabra_a_idx.items():
        if p != palabra:
            vec_p = np.asarray(_modelo_vocabulario["embeddings"][i], dtype=np.float32)
            sim = np.dot(vec, vec_p) / (np.linalg.norm(vec) * np.linalg.norm(vec_p) + 1e-8)
            similitudes.append((p, float(sim)))
    similitudes.sort(key=lambda x: x[1], reverse=True)
    return similitudes[:topn]


def comparar_palabras(palabra1, palabra2):
    vec1 = palabra_a_vector(palabra1)
    vec2 = palabra_a_vector(palabra2)
    if vec1 is None or vec2 is None:
        return None
    sim = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2) + 1e-8)
    return float(sim)


# Cargar banco de preguntas al iniciar
BANCO_PREGUNTAS = cargar_banco_preguntas()