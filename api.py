"""
Las placas leídas, para otros programas (una pluma, un sistema de parqueo, una hoja de cálculo):

  GET /api/placas              todas las placas leídas (las últimas 1.000), en JSON
  GET /api/placas?desde=12     solo las que llegaron después de la placa 12
  GET /api/placas/ultima       la última placa leída
  GET /api/placas/12/foto      la foto de la placa 12 (JPEG)
  GET /api/eventos             cada placa nueva apenas se lee (Server-Sent Events)

Con --webhook URL, además, cada placa nueva se manda por POST a esa dirección, con la foto.
Todo queda en memoria: al cerrar el programa no queda nada guardado.
"""
import base64
import json
import threading
import urllib.request
from datetime import datetime


class Placas:
    MAXIMO = 1000

    def __init__(self, webhook=None):
        self.lista = []  # (placa en JSON, foto JPEG)
        self.ultimo_id = 0
        self.cambio = threading.Condition()
        self.webhook = webhook

    def agregar(self, placa, texto, seguridad, votos, estado, jpg):
        with self.cambio:
            self.ultimo_id += 1
            p = {"id": self.ultimo_id, "placa": placa, "texto": texto,
                 "fecha": datetime.now().isoformat(timespec="seconds"), "seguridad": round(seguridad, 3),
                 "votos": votos, "estado": estado, "foto": f"/api/placas/{self.ultimo_id}/foto"}
            self.lista.append((p, jpg))
            del self.lista[:-self.MAXIMO]
            self.cambio.notify_all()
        if self.webhook:  # en otro hilo: un servidor lento no debe frenar la lectura de placas
            threading.Thread(target=self.avisar, args=(p, jpg), daemon=True).start()
        return p

    def desde(self, id_):
        with self.cambio:
            return [p for p, _ in self.lista if p["id"] > id_]

    def ultima(self):
        with self.cambio:
            return self.lista[-1][0] if self.lista else None

    def foto(self, id_):
        with self.cambio:
            return next((jpg for p, jpg in self.lista if p["id"] == id_), None)

    def esperar(self, id_, segundos):
        """Las placas después de id_; si no hay, espera a que llegue alguna (o a que pasen los segundos)."""
        with self.cambio:
            self.cambio.wait_for(lambda: self.ultimo_id > id_, timeout=segundos)
        return self.desde(id_)

    def avisar(self, p, jpg):
        cuerpo = json.dumps({**p, "foto_jpg": base64.b64encode(jpg).decode()}, ensure_ascii=False).encode()
        pedido = urllib.request.Request(self.webhook, data=cuerpo, method="POST",
                                        headers={"Content-Type": "application/json; charset=utf-8"})
        try:
            urllib.request.urlopen(pedido, timeout=5).close()
        except OSError as e:
            print(f"  webhook: no se pudo avisar la placa {p['placa']} ({type(e).__name__})")
