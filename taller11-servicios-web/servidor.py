import json
import threading
from dataclasses import dataclass, asdict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import List


# ---------- Preguntas ----------

@dataclass
class Pregunta:
    """Representa un ejercicio con su enunciado, sus 3 opciones y la solución correcta (A, B o C)."""
    pregunta: str
    opcion_a: str
    opcion_b: str
    opcion_c: str
    solucion: str


# Preguntas de sintaxis de Python (usa &blank como marcador del hueco).
PREGUNTAS: List[Pregunta] = [
    Pregunta(
        pregunta="for i &blank range(10):",
        opcion_a="in",
        opcion_b="on",
        opcion_c="of",
        solucion="A",
    ),
    Pregunta(
        pregunta="if x &blank 5:",
        opcion_a="==",
        opcion_b="=",
        opcion_c=":=",
        solucion="A",
    ),
    Pregunta(
        pregunta="def suma(a, b)&blank\n    return a + b",
        opcion_a=":",
        opcion_b=";",
        opcion_c="->",
        solucion="A",
    ),
]


# ---------- Resultados (ranking en memoria) ----------

@dataclass
class Resultado:
    nombre: str
    tiempo_ms: int
    tiempo_texto: str


def formatear_tiempo(ms: int) -> str:
    total_seg = ms // 1000
    minutos = total_seg // 60
    segundos = total_seg % 60
    return f"{minutos:02d}:{segundos:02d}"


class RankingStore:
    """Guarda los resultados en memoria y los mantiene ordenados por tiempo ascendente."""

    def __init__(self):
        self._lock = threading.Lock()
        self._resultados: List[Resultado] = []

    def agregar(self, nombre: str, tiempo_ms: int) -> None:
        nuevo = Resultado(
            nombre=nombre,
            tiempo_ms=tiempo_ms,
            tiempo_texto=formatear_tiempo(tiempo_ms),
        )
        with self._lock:
            self._resultados.append(nuevo)
            self._resultados.sort(key=lambda r: r.tiempo_ms)

    def obtener_ordenados(self) -> List[Resultado]:
        with self._lock:
            return list(self._resultados)


ranking_store = RankingStore()


# ---------- Página ----------

class Pagina:
    """Encapsula la plantilla HTML e inyecta el arreglo de preguntas como JSON."""

    _PLACEHOLDER = "__PREGUNTAS_JSON__"

    _TEMPLATE = """<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="UTF-8">
<title>Completa el for</title>
<style>
  body {
    font-family: -apple-system, Arial, sans-serif;
    background: #f2f2f2;
    display: flex;
    justify-content: center;
    align-items: center;
    height: 100vh;
    margin: 0;
  }
  .card {
    background: #fff;
    padding: 32px 40px;
    border-radius: 12px;
    box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    width: 480px;
    box-sizing: border-box;
  }

  /* ---------- Pantalla inicio ---------- */
  #pantalla-inicio {
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 18px;
  }
  #pantalla-inicio h2 {
    margin: 0;
    color: #333;
  }
  #inputNombre {
    width: 100%;
    box-sizing: border-box;
    font-size: 20px;
    padding: 12px;
    border-radius: 8px;
    border: 1px solid #ccc;
    text-align: center;
  }
  #btnIniciar {
    width: 100%;
    font-size: 22px;
    padding: 14px;
    border: none;
    border-radius: 8px;
    background: #007acc;
    color: #fff;
    cursor: pointer;
  }
  #btnIniciar:hover {
    background: #005f99;
  }

  /* ---------- Pantalla pregunta ---------- */
  #pantalla-pregunta {
    display: none;
    flex-direction: column;
    align-items: center;
  }
  .code {
    width: 100%;
    box-sizing: border-box;
    min-height: 160px;
    font-family: 'Courier New', monospace;
    font-size: 22px;
    background: #1e1e1e;
    color: #d4d4d4;
    border-radius: 8px;
    margin-bottom: 20px;
    padding: 16px;
    display: flex;
    justify-content: center;
    align-items: center;
    text-align: center;
    white-space: pre;
    box-sizing: border-box;
  }
  .blank {
    font-weight: bold;
    color: #ffcc00;
  }
  .blank.correcto {
    color: #4caf50;
  }
  .blank.incorrecto {
    color: #e74c3c;
  }
  .variable {
    width: 100%;
    font-size: 26px;
    color: #555;
    margin-bottom: 20px;
    min-height: 40px;
    display: flex;
    justify-content: center;
    align-items: center;
    text-align: center;
  }
  .buttons {
    display: flex;
    justify-content: center;
    gap: 14px;
  }
  .buttons button {
    width: 143px;
    height: 143px;
    display: flex;
    justify-content: center;
    align-items: center;
    font-family: 'Courier New', monospace;
    font-size: 32px;
    padding: 0;
    border: none;
    border-radius: 8px;
    background: #007acc;
    color: #fff;
    cursor: pointer;
  }
  .buttons button:hover {
    background: #005f99;
  }

  /* ---------- Pantalla final ---------- */
  #pantalla-final {
    display: none;
    flex-direction: column;
    align-items: center;
    text-align: center;
    gap: 10px;
  }
  #pantalla-final h2 {
    margin: 0;
    color: #333;
  }
  #tiempoFinal {
    font-size: 20px;
    color: #007acc;
    font-weight: bold;
    margin-bottom: 6px;
  }
  #tituloRanking {
    margin: 10px 0 0 0;
    color: #333;
  }
  #ranking {
    width: 100%;
    box-sizing: border-box;
    text-align: left;
    background: #f7f7f7;
    border-radius: 8px;
    padding: 12px 20px;
    margin: 0;
    list-style: none;
  }
  #ranking li {
    display: flex;
    justify-content: space-between;
    padding: 6px 0;
    border-bottom: 1px solid #e0e0e0;
    font-size: 16px;
    color: #333;
  }
  #ranking li:last-child {
    border-bottom: none;
  }
  #ranking li.mejor {
    font-weight: bold;
    color: #4caf50;
  }
</style>
</head>
<body>
  <div class="card">

    <!-- Pantalla 1: pedir nombre -->
    <div id="pantalla-inicio">
      <h2>¿Cuál es tu nombre?</h2>
      <input id="inputNombre" type="text" placeholder="Escribe tu nombre">
      <button id="btnIniciar" onclick="iniciar()">Iniciar</button>
    </div>

    <!-- Pantalla 2: pregunta -->
    <div id="pantalla-pregunta">
      <div class="code" id="code"></div>
      <div class="variable" id="variable">Elige un operador</div>
      <div class="buttons">
        <button id="btnA"></button>
        <button id="btnB"></button>
        <button id="btnC"></button>
      </div>
    </div>

    <!-- Pantalla 3: final -->
    <div id="pantalla-final">
      <h2 id="tituloFinal"></h2>
      <div id="tiempoFinal"></div>
      <h3 id="tituloRanking">Ranking (menor tiempo primero)</h3>
      <ol id="ranking"></ol>
    </div>

  </div>

  <script>
    const preguntas = __PREGUNTAS_JSON__;
    let nombre = '';
    let indice = 0;
    let inicioTimestamp = 0;
    const respondidas = [];

    function iniciar() {
      const valor = document.getElementById('inputNombre').value.trim();
      if (!valor) {
        document.getElementById('inputNombre').focus();
        return;
      }
      nombre = valor;
      inicioTimestamp = Date.now();
      document.getElementById('pantalla-inicio').style.display = 'none';
      document.getElementById('pantalla-pregunta').style.display = 'flex';
      cargarPregunta(indice);
    }

    function cargarPregunta(i) {
      const p = preguntas[i];

      const html = p.pregunta
        .replace('&blank', '<span id="blank" class="blank">___</span>')
        .replace(/\\n/g, '<br>');
      document.getElementById('code').innerHTML = html;
      document.getElementById('variable').textContent = 'Elige una opción';

      const btnA = document.getElementById('btnA');
      const btnB = document.getElementById('btnB');
      const btnC = document.getElementById('btnC');

      btnA.textContent = p.opcion_a;
      btnB.textContent = p.opcion_b;
      btnC.textContent = p.opcion_c;

      btnA.onclick = () => elegir(p.opcion_a, 'A');
      btnB.onclick = () => elegir(p.opcion_b, 'B');
      btnC.onclick = () => elegir(p.opcion_c, 'C');
    }

    function elegir(valor, letra) {
      const p = preguntas[indice];
      const blank = document.getElementById('blank');
      blank.textContent = valor;

      if (letra === p.solucion) {
        blank.classList.remove('incorrecto');
        blank.classList.add('correcto');
        document.getElementById('variable').textContent = '¡Correcto!';

        const textoCompleto = p.pregunta.replace('&blank', valor);
        respondidas.push(textoCompleto);

        setTimeout(() => {
          indice++;
          if (indice < preguntas.length) {
            cargarPregunta(indice);
          } else {
            finalizar();
          }
        }, 700);
      } else {
        blank.classList.remove('correcto');
        blank.classList.add('incorrecto');
        document.getElementById('variable').textContent = 'Incorrecto, intenta de nuevo';
      }
    }

    function formatearTiempo(ms) {
      const totalSeg = Math.floor(ms / 1000);
      const min = Math.floor(totalSeg / 60);
      const seg = totalSeg % 60;
      const pad = n => String(n).padStart(2, '0');
      return pad(min) + ':' + pad(seg);
    }

    async function finalizar() {
      document.getElementById('pantalla-pregunta').style.display = 'none';
      document.getElementById('pantalla-final').style.display = 'flex';

      const tiempoMs = Date.now() - inicioTimestamp;

      document.getElementById('tituloFinal').textContent =
        '¡Felicidades, ' + nombre + '!';
      document.getElementById('tiempoFinal').textContent =
        'Tu tiempo: ' + formatearTiempo(tiempoMs);

      try {
        await fetch('/api/resultado', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ nombre: nombre, tiempo_ms: tiempoMs })
        });
      } catch (e) {
        console.error('No se pudo guardar el resultado', e);
      }

      try {
        const resp = await fetch('/api/resultados');
        const datos = await resp.json();
        const ranking = document.getElementById('ranking');
        ranking.innerHTML = '';
        datos.forEach((r, idx) => {
          const li = document.createElement('li');
          if (idx === 0) li.classList.add('mejor');
          li.innerHTML = '<span>' + (idx + 1) + '. ' + r.nombre + '</span><span>' + r.tiempo_texto + '</span>';
          ranking.appendChild(li);
        });
      } catch (e) {
        console.error('No se pudo obtener el ranking', e);
      }
    }
  </script>
</body>
</html>"""

    def __init__(self, preguntas: List[Pregunta]):
        self.preguntas = preguntas

    def render(self) -> bytes:
        preguntas_json = json.dumps(
            [asdict(p) for p in self.preguntas], ensure_ascii=False
        )
        html = self._TEMPLATE.replace(self._PLACEHOLDER, preguntas_json)
        return html.encode("utf-8")


# ---------- Handler / Servidor ----------

class ForHandler(BaseHTTPRequestHandler):
    """Maneja las peticiones HTTP: página principal y API del ranking."""

    pagina = Pagina(PREGUNTAS)

    def do_GET(self):
        if self.path == "/":
            self._responder_html(self.pagina.render())
        elif self.path == "/api/resultados":
            self._responder_json(
                [asdict(r) for r in ranking_store.obtener_ordenados()]
            )
        else:
            self.send_error(404, "No encontrado")

    def do_POST(self):
        if self.path != "/api/resultado":
            self.send_error(404, "No encontrado")
            return

        largo = int(self.headers.get("Content-Length", 0))
        cuerpo = self.rfile.read(largo)

        try:
            datos = json.loads(cuerpo)
            nombre = str(datos["nombre"])
            tiempo_ms = int(datos["tiempo_ms"])
        except (json.JSONDecodeError, KeyError, ValueError):
            self.send_error(400, "JSON inválido")
            return

        ranking_store.agregar(nombre, tiempo_ms)
        self._responder_json({"ok": True})

    def _responder_html(self, contenido: bytes):
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(contenido)))
        self.end_headers()
        self.wfile.write(contenido)

    def _responder_json(self, data):
        contenido = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(contenido)))
        self.end_headers()
        self.wfile.write(contenido)

    def log_message(self, format, *args):
        print("%s - %s" % (self.client_address[0], format % args))


class Servidor:
    """Encapsula la creación y ejecución del servidor HTTP."""

    def __init__(self, host: str = "localhost", puerto: int = 8080):
        self.host = host
        self.puerto = puerto
        self._httpd = ThreadingHTTPServer((self.host, self.puerto), ForHandler)

    def iniciar(self):
        print(f"Servidor corriendo en http://{self.host}:{self.puerto}")
        try:
            self._httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nServidor detenido.")
        finally:
            self._httpd.server_close()


if __name__ == "__main__":
    Servidor(host="localhost", puerto=8080).iniciar()
