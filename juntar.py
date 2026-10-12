r"""
Junta recortes de placas reales (de la cámara o de un video) para revisarlos a mano y después
entrenar el lector con ellos.

Se activa con --juntar en camara.cmd o ver_en_vivo.cmd. Por cada vehículo anotado guarda hasta
6 recortes de su placa, de distintos cuadros, en datos\reales\pendientes\, y una línea por
recorte en datos\reales\pendientes.jsonl con lo que leyó el sistema. Después se revisan con
revisar.cmd.

Son placas reales: datos personales. No salen de esta PC.
"""
import json
import os
import re
from datetime import datetime

CARPETA = os.path.dirname(os.path.abspath(__file__))
REALES = os.path.join(CARPETA, "datos", "reales")
PENDIENTES = os.path.join(REALES, "pendientes")
INDICE = os.path.join(REALES, "pendientes.jsonl")
REVISADAS = os.path.join(REALES, "revisadas.jsonl")


class Recolector:
    def __init__(self, origen):
        self.origen = re.sub(r"[^\w.-]+", "_", origen)[:40]
        os.makedirs(PENDIENTES, exist_ok=True)

    def guardar(self, pasada, texto, momento, estado):
        if not pasada.recortes:
            return
        vehiculo = f"{self.origen}_{datetime.now():%Y%m%d-%H%M%S-%f}_{texto}"
        with open(INDICE, "a", encoding="utf-8") as f:
            for k, (seguridad, recorte, leido, caja) in enumerate(pasada.recortes):
                archivo = f"{vehiculo}_{k}.jpg"
                recorte.save(os.path.join(PENDIENTES, archivo), quality=92)
                f.write(json.dumps({
                    "vehiculo": vehiculo, "archivo": archivo, "propuesta": texto, "leido": leido,
                    "seguridad": round(seguridad, 3), "caja": caja, "origen": self.origen,
                    "momento": round(momento, 1), "estado": estado,
                }, ensure_ascii=False) + "\n")
