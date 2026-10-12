r"""
Muestra un video o una cámara en vivo, con las placas detectadas y leídas encima.

  ver_en_vivo.cmd video.mp4
  ver_en_vivo.cmd rtsp://usuario:clave@IP:554/ruta
  ver_en_vivo.cmd 192.168.0.100 --credenciales credenciales.txt   (la clave queda en el archivo)
  ver_en_vivo.cmd video.mp4 --ignorar 0.88,0,1,0.15   (zona a ignorar: logos, fecha y hora de
                                                        la cámara; en fracciones x0,y0,x1,y1)

Abre http://127.0.0.1:8801 : el video con un rectángulo en cada placa y, al lado, cada placa
leída con su recorte, para comparar a ojo si la leyó bien. Va al ritmo real del video (si el
análisis se atrasa, se saltan cuadros). No guarda nada en disco: las placas de la calle son
datos personales.
"""
import base64
import io
import json
import os
import sys
import threading
import time
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

CARPETA = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, CARPETA)
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import av
from PIL import Image, ImageDraw

from camara_ip import ES_IP, conectar
from camara import (CAMBIO_DE_CARRO, SILENCIO_PARA_CERRAR, VOTOS_MINIMOS, Pista, confiable, encima, fuente,
                    marcar_cortadas, repetida)
from lector import bonito
from placas import LectorDePlacas

PUERTO = 8801
ANALIZAR_POR_SEGUNDO = 10  # en la calle los carros pasan rápido: se mira más seguido que en un portón
ANCHO_PANTALLA = 1280
MOSTRAR_POR_SEGUNDO = 12  # la página no necesita los 25 cuadros: cada JPEG cuesta procesador


class Estado:
    """Lo que ve la página: el último cuadro dibujado y las placas leídas (solo en memoria)."""

    aviso = threading.Condition()
    jpg = None
    mirando = 0  # páginas abiertas viendo el video; si no hay ninguna, no se dibuja nada
    lecturas = []
    recientes = deque(maxlen=10)
    datos = {"segundo": 0, "duracion": 0, "cuadros_por_segundo": 0, "analisis_por_segundo": 0, "detectadas": 0}


PAGINA = """<!doctype html><html lang="es"><head><meta charset="utf-8"><title>Placas en vivo</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
:root { --fondo:#0f1419; --tarjeta:#171d24; --borde:#26303b; --texto:#e6ebf1; --suave:#8d99a8; --ok:#3fbf7f; --aviso:#e0a82e; }
* { box-sizing:border-box } body { margin:0; background:var(--fondo); color:var(--texto); font-family:"Segoe UI",system-ui,sans-serif; font-size:14px }
header { display:flex; justify-content:space-between; align-items:center; padding:12px 20px; border-bottom:1px solid var(--borde) }
h1 { margin:0; font-size:17px; font-weight:600 } .suave { color:var(--suave) }
main { display:grid; grid-template-columns:minmax(0,1fr) 340px; gap:16px; padding:16px 20px }
@media (max-width:900px) { main { grid-template-columns:1fr } }
#video { width:100%; border-radius:8px; background:#000; display:block }
.lista { display:flex; flex-direction:column; gap:10px; max-height:calc(100vh - 110px); overflow:auto }
.item { background:var(--tarjeta); border:1px solid var(--borde); border-radius:8px; padding:10px; display:flex; gap:10px; align-items:center }
.item.revisar { border-color:var(--aviso) }
.item img { width:150px; border-radius:4px; background:#000 }
.placa { font-family:Bahnschrift,"Arial Narrow",sans-serif; font-weight:700; font-size:19px; color:#111; padding:5px 8px 3px; border-radius:4px;
  background:linear-gradient(to bottom,#2368d8 0 3px,#fff 3px calc(100% - 4px),#2368d8 calc(100% - 4px)); white-space:nowrap }
.detalle { color:var(--suave); font-size:12px; margin-top:4px; line-height:1.4 } .detalle b { color:var(--aviso) }
.recientes { display:grid; grid-template-columns:repeat(auto-fill,minmax(150px,1fr)); gap:8px }
.reciente { background:var(--tarjeta); border:1px solid var(--borde); border-radius:6px; padding:6px; font-size:12px }
.reciente img { width:100%; border-radius:3px; display:block; margin-bottom:4px; background:#000 }
.reciente.mala { color:var(--suave) }
</style></head><body>
<header><h1>Placas en vivo <span class="suave" id="titulo"></span></h1><div class="suave" id="datos"></div></header>
<main><div><img id="video" src="/flujo" alt="Video con las placas detectadas">
<div class="suave" style="margin:14px 0 8px">Lo último que vio (aunque todavía no lo anote)</div>
<div class="recientes" id="recientes"></div></div>
<div><div class="suave" style="margin-bottom:8px" id="cuenta">Placas leídas</div><div class="lista" id="lista"></div></div></main>
<script>
const esc = t => String(t).replace(/[&<>"']/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));
let vistas = 0;
async function actualizar() {
  try {
    const e = await (await fetch("/api/estado", {cache:"no-store"})).json();
    const d = e.datos, m = s => Math.floor(s/60) + ":" + String(Math.floor(s%60)).padStart(2,"0");
    document.getElementById("datos").textContent = (d.duracion ? `${m(d.segundo)} de ${m(d.duracion)}` : "en vivo")
      + ` · ${d.cuadros_por_segundo} cuadros/s · ${d.analisis_por_segundo} análisis/s · ${d.detectadas} placas detectadas`;
    document.getElementById("recientes").innerHTML = e.recientes.map(r => `
      <div class="reciente ${r.valida ? "" : "mala"}"><img src="data:image/jpeg;base64,${r.recorte}" alt="">
      ${esc(r.texto)}<br>${esc(r.seguridad)} · ${r.ancho} px</div>`).join("");
    document.getElementById("cuenta").textContent = `${e.lecturas.length} placas leídas`;
    if (e.lecturas.length !== vistas) {
      vistas = e.lecturas.length;
      document.getElementById("lista").innerHTML = e.lecturas.slice().reverse().map(l => `
        <div class="item ${l.estado === "revisar" ? "revisar" : ""}"><img src="data:image/jpeg;base64,${l.recorte}" alt="">
        <div><span class="placa">${esc(l.placa)}</span><div class="detalle">${esc(l.hora)} · ${l.votos} votos · ${esc(l.seguridad)}
        ${l.estado === "revisar" ? "<br><b>revisar</b>" : ""}</div></div></div>`).join("");
    }
  } catch {}
}
actualizar(); setInterval(actualizar, 1000);
</script></body></html>"""


class Atender(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def do_GET(self):
        if self.path == "/":
            cuerpo = PAGINA.encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(cuerpo)
        elif self.path == "/api/estado":
            cuerpo = json.dumps({"lecturas": Estado.lecturas, "recientes": list(Estado.recientes)[::-1],
                                 "datos": Estado.datos}, ensure_ascii=False).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(cuerpo)
        elif self.path == "/flujo":
            # MJPEG: el navegador muestra cada JPEG nuevo que llega, como un video.
            self.send_response(200)
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=cuadro")
            self.end_headers()
            with Estado.aviso:
                Estado.mirando += 1
            try:
                while True:
                    with Estado.aviso:
                        Estado.aviso.wait(timeout=5)
                        jpg = Estado.jpg
                    if jpg:
                        self.wfile.write(b"--cuadro\r\nContent-Type: image/jpeg\r\nContent-Length: "
                                         + str(len(jpg)).encode() + b"\r\n\r\n" + jpg + b"\r\n")
            except (ConnectionError, OSError):
                pass
            finally:
                with Estado.aviso:
                    Estado.mirando -= 1
        else:
            self.send_response(404)
            self.end_headers()


class Lecturas:
    """Igual que en camara.py: cada vehículo es una pista con sus votos; se anota cuando la
    lectura es estable. Aquí se anota en memoria, con un recorte para mostrarlo en la página."""

    def __init__(self):
        self.pistas = []
        self.ultima_vez = {}

    def anotar(self, pasada, momento):
        texto = pasada.ganadora()
        if texto is None or pasada.votos[texto] < VOTOS_MINIMOS:
            return
        if repetida(texto, momento, self.ultima_vez):
            return
        self.ultima_vez[texto] = momento
        seguridad, foto, lectura = pasada.mejor_foto[texto]
        es_confiable = confiable(pasada, texto, seguridad, lectura)
        x0, y0, x1, y1 = lectura["caja"]
        w, h = x1 - x0, y1 - y0
        recorte = foto.crop((x0 - w * 0.3, y0 - h * 0.5, x1 + w * 0.3, y1 + h * 0.5))
        recorte.thumbnail((300, 150))
        jpg = io.BytesIO()
        recorte.save(jpg, "JPEG", quality=85)
        Estado.lecturas.append({
            "id": len(Estado.lecturas), "texto": texto,
            "placa": lectura["placa"], "hora": f"{int(momento // 60)}:{momento % 60:04.1f}",
            "votos": pasada.votos[texto], "seguridad": f"{seguridad:.0%}",
            "estado": "ok" if es_confiable else "revisar", "recorte": base64.b64encode(jpg.getvalue()).decode(),
        })
        print(f"  {int(momento // 60)}:{momento % 60:04.1f}   {lectura['placa']:<10} {pasada.votos[texto]:2} votos  {seguridad:.0%}"
              + ("" if es_confiable else "   revisar"))

    def actualizar(self, lecturas, foto, momento):
        libres = list(self.pistas)
        for lectura in sorted((p for p in lecturas if p["valida"]), key=lambda p: -p["seguridad"]):
            pista = max(libres, key=lambda p: encima(p.caja, lectura["caja"]), default=None)
            if pista and (encima(pista.caja, lectura["caja"]) >= 0.2 or pista.pasada.ganadora() == lectura["texto"]):
                libres.remove(pista)
            else:
                pista = Pista(lectura["caja"], momento)
                self.pistas.append(pista)
            pista.caja, pista.visto = lectura["caja"], momento
            pista.pasada.votar(lectura, foto)
            if pista.pasada.seguidas >= CAMBIO_DE_CARRO:
                nueva = pista.pasada.separar()
                if not pista.anotada:
                    self.anotar(pista.pasada, momento)
                pista.pasada, pista.anotada = nueva, False
            if not pista.anotada and pista.estable():
                self.anotar(pista.pasada, momento)
                pista.anotada = True
        for pista in [p for p in self.pistas if momento - p.visto > SILENCIO_PARA_CERRAR]:
            if not pista.anotada:
                self.anotar(pista.pasada, momento)
            self.pistas.remove(pista)


def dibujar(vista, lecturas, registro, letra, escala):
    """Rectángulo en cada placa: verde si ya se anotó, amarillo si todavía junta votos,
    gris si lo que leyó no tiene formato de placa salvadoreña. Las cajas vienen en el tamaño
    original del video; `escala` las lleva al tamaño de la vista."""
    d = ImageDraw.Draw(vista)
    grosor = max(2, vista.width // 400)
    alto_letra = letra.size + 6
    for p in lecturas:
        caja = [v * escala for v in p["caja"]]
        pista = next((t for t in registro.pistas if t.caja is p["caja"]), None)
        if not p["valida"] or pista is None:
            color, etiqueta = (160, 160, 160), "cortada" if p.get("cortada") else f"{p['texto'] or ''}?"
        else:
            texto = pista.pasada.ganadora()
            color = (0, 230, 90) if pista.anotada else (250, 200, 0)
            etiqueta = f"{bonito(texto)} ({pista.pasada.votos[texto]})"
        d.rectangle(caja, outline=color, width=grosor)
        # El texto va debajo de la placa; si no cabe (placa abajo en la imagen), arriba.
        y = caja[3] + 4 if caja[3] + alto_letra < vista.height else max(0, caja[1] - alto_letra)
        d.text((caja[0], y), etiqueta, font=letra, fill=color, stroke_width=3, stroke_fill=(0, 0, 0))


class Analisis(threading.Thread):
    """Analiza siempre el cuadro más reciente, aparte de la reproducción: así el video se ve
    fluido y, si el análisis tarda (tarjeta ocupada, cámara en vivo), no se acumula atraso."""

    def __init__(self, sistema, registro, ignorar):
        super().__init__(daemon=True)
        self.sistema, self.registro, self.ignorar = sistema, registro, ignorar
        self.hay_nuevo = threading.Event()
        self.libre = threading.Event()
        self.libre.set()
        self.cuadro = None
        self.cerrojo = threading.Lock()  # protege las pistas y las lecturas mientras se dibujan
        self.lecturas = []
        self.ultimo_reciente = 0
        self.hechos, self.reloj = 0, time.time()

    def entregar(self, imagen, momento):
        """Recibe una copia propia de la imagen (un arreglo de numpy), no el cuadro de video: si
        dos hilos usan el mismo cuadro de PyAV al mismo tiempo, FFmpeg se traba."""
        self.cuadro = (imagen, momento)
        self.hay_nuevo.set()

    def ocupado(self):
        return self.hay_nuevo.is_set() or not self.libre.is_set()

    def guardar_recientes(self, foto, lecturas):
        """Lo último que vio, aunque no lo haya anotado: para entender qué está pasando."""
        if not lecturas or time.time() - self.ultimo_reciente < 0.5:
            return
        self.ultimo_reciente = time.time()
        for p in lecturas:
            x0, y0, x1, y1 = p["caja"]
            w, h = x1 - x0, y1 - y0
            recorte = foto.crop((x0 - w * 0.2, y0 - h * 0.3, x1 + w * 0.2, y1 + h * 0.3))
            recorte.thumbnail((200, 100))
            jpg = io.BytesIO()
            recorte.save(jpg, "JPEG", quality=80)
            Estado.recientes.append({
                "texto": p["placa"] if p["valida"] else ("cortada en el borde" if p.get("cortada")
                                                          else f"{p['texto'] or 'nada'} (no es placa SV)"),
                "seguridad": f"{p['seguridad']:.0%}", "valida": p["valida"], "ancho": round(w),
                "recorte": base64.b64encode(jpg.getvalue()).decode(),
            })

    def run(self):
        while True:
            self.hay_nuevo.wait()
            self.hay_nuevo.clear()
            self.libre.clear()
            inicio = time.time()
            imagen, momento = self.cuadro
            self.procesar(Image.fromarray(imagen), momento)
            self.libre.set()
            espera = 1 / ANALIZAR_POR_SEGUNDO - (time.time() - inicio)
            if espera > 0:
                time.sleep(espera)

    def procesar(self, foto, momento):
        lecturas = marcar_cortadas(self.sistema.leer(foto), foto.width, foto.height)
        if self.ignorar:  # lo que cae en la zona ignorada (un logo, la hora impresa) no cuenta
            x0, y0, x1, y1 = (self.ignorar[0] * foto.width, self.ignorar[1] * foto.height,
                              self.ignorar[2] * foto.width, self.ignorar[3] * foto.height)
            lecturas = [p for p in lecturas
                        if not (x0 <= (p["caja"][0] + p["caja"][2]) / 2 <= x1 and y0 <= (p["caja"][1] + p["caja"][3]) / 2 <= y1)]
        with self.cerrojo:
            self.registro.actualizar(lecturas, foto, momento)
            self.lecturas = lecturas
        Estado.datos["detectadas"] += len(lecturas)
        self.guardar_recientes(foto, lecturas)
        self.hechos += 1
        if time.time() - self.reloj >= 2:
            Estado.datos["analisis_por_segundo"] = round(self.hechos / (time.time() - self.reloj), 1)
            self.hechos, self.reloj = 0, time.time()


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return
    fuente = sys.argv[1]
    ignorar = None
    if "--ignorar" in sys.argv:
        ignorar = [float(v) for v in sys.argv[sys.argv.index("--ignorar") + 1].split(",")]
    sistema = LectorDePlacas(insistir=False)  # video: una sola mirada por cuadro
    registro = Lecturas()
    servidor = ThreadingHTTPServer(("127.0.0.1", PUERTO), Atender)
    threading.Thread(target=servidor.serve_forever, daemon=True).start()
    print(f"Abre http://127.0.0.1:{PUERTO}  (Ctrl+C para salir)\n")

    en_vivo = bool(ES_IP.match(fuente)) or fuente.lower().startswith(("rtsp://", "http://", "https://"))

    def abrir():
        if ES_IP.match(fuente):  # una IP: usuario y clave salen del archivo, sin mostrarse nunca
            return conectar(fuente, sys.argv[sys.argv.index("--credenciales") + 1])[0]
        opciones = {"rtsp_transport": "tcp", "timeout": "5000000"} if fuente.lower().startswith("rtsp://") else {}
        return av.open(fuente, options=opciones)

    # --rapido: recorre el video lo más rápido que se pueda (para medir), analizando 5 cuadros
    # por cada segundo de video; no sirve para cámaras en vivo.
    rapido = "--rapido" in sys.argv and not en_vivo
    analisis = Analisis(sistema, registro, ignorar)
    if not rapido:
        analisis.start()
    while True:
        try:
            contenedor = abrir()
        except (av.error.FFmpegError, ConnectionError, OSError):
            if not en_vivo:
                raise
            print("No se pudo conectar. Reintento en 5 s…")
            time.sleep(5)
            continue
        try:
            momento = reproducir(contenedor, analisis, registro, rapido, en_vivo)
        except av.error.FFmpegError:
            print("Se cortó la señal de la cámara. Reconectando…")  # sin detalle: podría incluir la clave
            continue
        finally:
            contenedor.close()
        if en_vivo:
            print("La cámara dejó de mandar imagen. Reconectando…")
            continue
        break
    analisis.libre.wait(timeout=10)
    with analisis.cerrojo:
        for pista in registro.pistas:
            if not pista.anotada:
                registro.anotar(pista.pasada, momento)
    print(f"\nTerminó el video: {len(Estado.lecturas)} placas leídas. La página sigue abierta (Ctrl+C para salir).")
    while True:
        time.sleep(60)


def reproducir(contenedor, analisis, registro, rapido, en_vivo):
    """Muestra cada cuadro con las placas dibujadas. Devuelve el momento del último cuadro."""
    flujo = contenedor.streams.video[0]
    flujo.thread_type = "AUTO"
    if flujo.duration and flujo.time_base:
        Estado.datos["duracion"] = float(flujo.duration * flujo.time_base)
    ultimo_rapido = -1e9
    letra = None
    inicio, t0, momento = None, None, 0.0
    mostrados, reloj_fps, ultimo_mostrado = 0, time.time(), 0.0
    for cuadro in contenedor.decode(flujo):
        momento = cuadro.time or 0.0
        if rapido:
            if momento - ultimo_rapido >= 0.2:
                ultimo_rapido = momento
                analisis.procesar(cuadro.to_image(), momento)
                Estado.datos["segundo"] = momento
            continue
        if not en_vivo:
            # Un video se muestra a su ritmo real: si vamos adelantados se espera; si atrasados,
            # se salta el cuadro. (En vivo no: los cuadros llegan cuando llegan, y si se tomara
            # una pausa como atraso, se descartarían todos los siguientes y la imagen se congelaría.)
            if inicio is None:
                inicio, t0 = time.time(), momento
            adelanto = (momento - t0) - (time.time() - inicio)
            if adelanto > 0:
                time.sleep(adelanto)
            elif adelanto < -0.1:
                continue
        if not analisis.ocupado():  # el análisis toma el cuadro más reciente cuando se desocupa
            analisis.entregar(cuadro.to_ndarray(format="rgb24"), momento)
        Estado.datos["segundo"] = momento
        # Dibujar y comprimir la vista es lo que más procesador gasta: solo si alguien la mira,
        # y no más de MOSTRAR_POR_SEGUNDO veces por segundo. El análisis no cambia.
        if not Estado.mirando or time.time() - ultimo_mostrado < 1 / MOSTRAR_POR_SEGUNDO:
            continue
        ultimo_mostrado = time.time()
        escala = min(1.0, ANCHO_PANTALLA / cuadro.width)
        ancho, alto = round(cuadro.width * escala), round(cuadro.height * escala)
        if letra is None:
            letra = fuente(max(16, ancho // 45))
        vista = cuadro.to_image(width=ancho, height=alto)
        with analisis.cerrojo:
            dibujar(vista, analisis.lecturas, registro, letra, escala)
        jpg = io.BytesIO()
        vista.save(jpg, "JPEG", quality=80)
        with Estado.aviso:
            Estado.jpg = jpg.getvalue()
            Estado.aviso.notify_all()
        mostrados += 1
        Estado.datos["segundo"] = momento
        if time.time() - reloj_fps >= 2:
            Estado.datos["cuadros_por_segundo"] = round(mostrados / (time.time() - reloj_fps))
            mostrados, reloj_fps = 0, time.time()
    return momento


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nDetenido.")
