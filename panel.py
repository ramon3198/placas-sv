r"""
Panel web con los carros que han pasado: placa, hora, captura. Se actualiza solo.

  panel.cmd                 → abre http://127.0.0.1:8800 con lo que hay en resultados\registro.csv
  camara.cmd ... --panel    → lo mismo, mientras la cámara está leyendo

Solo responde en esta PC (127.0.0.1): las placas son datos personales y no se publican en la red.
"""
import csv
import json
import os
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

CARPETA = os.path.dirname(os.path.abspath(__file__))
RESULTADOS = os.path.join(CARPETA, "resultados")
REGISTRO = os.path.join(RESULTADOS, "registro.csv")
CAPTURAS = os.path.join(RESULTADOS, "capturas")
PUERTO = 8800


def leer_registro(cuantos=300):
    if not os.path.exists(REGISTRO):
        return []
    with open(REGISTRO, encoding="utf-8-sig", newline="") as f:
        filas = list(csv.DictReader(f))
    return filas[-cuantos:][::-1]  # lo más reciente primero


class Atender(BaseHTTPRequestHandler):
    def responder(self, codigo, tipo, cuerpo):
        self.send_response(codigo)
        self.send_header("Content-Type", tipo)
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(cuerpo)

    def do_GET(self):
        ruta = self.path.split("?")[0]
        if ruta == "/":
            with open(os.path.join(CARPETA, "panel.html"), "rb") as f:
                return self.responder(200, "text/html; charset=utf-8", f.read())
        if ruta == "/api/registro":
            return self.responder(200, "application/json", json.dumps(leer_registro(), ensure_ascii=False).encode())
        if ruta == "/registro.csv" and os.path.exists(REGISTRO):
            with open(REGISTRO, "rb") as f:
                return self.responder(200, "text/csv; charset=utf-8", f.read())
        if ruta.startswith("/capturas/"):
            nombre = os.path.basename(ruta)  # nada de ../ : solo archivos de la carpeta de capturas
            archivo = os.path.join(CAPTURAS, nombre)
            if nombre.endswith(".jpg") and os.path.isfile(archivo):
                with open(archivo, "rb") as f:
                    return self.responder(200, "image/jpeg", f.read())
        self.responder(404, "text/plain; charset=utf-8", "No existe".encode())

    def log_message(self, *args):
        pass  # sin una línea en la consola por cada pedido


def iniciar_panel(puerto=PUERTO):
    servidor = ThreadingHTTPServer(("127.0.0.1", puerto), Atender)
    threading.Thread(target=servidor.serve_forever, daemon=True).start()
    print(f"Panel: http://127.0.0.1:{puerto}")
    return servidor


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    servidor = ThreadingHTTPServer(("127.0.0.1", PUERTO), Atender)
    print(f"Panel: http://127.0.0.1:{PUERTO}   (Ctrl+C para cerrar)")
    try:
        servidor.serve_forever()
    except KeyboardInterrupt:
        pass
