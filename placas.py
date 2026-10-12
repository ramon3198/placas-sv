"""
El sistema completo: foto → detector → recorte → lector → texto de la placa.

    from placas import LectorDePlacas
    sistema = LectorDePlacas()
    for p in sistema.leer(Image.open("carro.jpg")):
        print(p["placa"], p["valida"], p["seguridad"])
"""
import os
import sys
import urllib.request
from collections import Counter

CARPETA = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, CARPETA)

import torch
import torchvision.transforms.functional as TF

import detector
import lector

URL_MODELOS = "https://github.com/ramon3198/placas-sv/releases/download/v1.0/"


def bajar_si_falta(ruta):
    """La primera vez, baja el modelo de las Releases de GitHub (unos 95 MB entre los dos)."""
    if os.path.exists(ruta):
        return
    nombre = os.path.basename(ruta)
    print(f"Bajando {nombre} (solo la primera vez)…")
    os.makedirs(os.path.dirname(ruta), exist_ok=True)
    parcial = ruta + ".parcial"
    try:
        urllib.request.urlretrieve(URL_MODELOS + nombre, parcial)
    except OSError as e:
        raise SystemExit(f"No se pudo bajar {nombre} ({e}). Bájalo a mano de "
                         f"https://github.com/ramon3198/placas-sv/releases y ponlo en {os.path.dirname(ruta)}")
    os.replace(parcial, ruta)


# El lector se entrenó con recortes un poco más grandes que la placa, como los que salen del
# detector. Se lee el mismo recorte con tres márgenes distintos y se vota: si dos o tres
# coinciden, la lectura es mucho más confiable que una sola.
MARGENES = ((0.05, 0.08), (0.0, 0.0), (0.10, 0.16))


class LectorDePlacas:
    def __init__(self, dispositivo=None, umbral_deteccion=0.35, detector_ruta=None, lector_ruta=None, insistir=True):
        """insistir: si no encuentra placa, mirar otra vez con más y con menos detalle. Sirve para
        fotos sueltas; en video no, porque 0,2 s después llega otro cuadro y en un parqueo vacío
        triplicaría el trabajo de la tarjeta."""
        self.insistir = insistir
        self.dispositivo = dispositivo or ("cuda" if torch.cuda.is_available() else "cpu")
        if self.dispositivo == "cuda":
            # PyTorch abre un hilo por núcleo y los deja girando a la espera de trabajo; con la
            # tarjeta haciendo el cálculo, eso solo calienta el procesador.
            torch.set_num_threads(2)
        bajar_si_falta(detector_ruta or detector.MODELO)
        bajar_si_falta(lector_ruta or lector.MODELO)
        self.detector = detector.cargar_entrenado(self.dispositivo, detector_ruta)
        self.lector = lector.cargar_entrenado(self.dispositivo, lector_ruta)
        self.umbral = umbral_deteccion

    @torch.no_grad()
    def detectar(self, foto, tamano):
        """Busca placas mirando la foto escalada a `tamano` píxeles por el lado corto."""
        self.detector.transform.min_size = (tamano,)
        self.detector.transform.max_size = tamano * 5 // 3
        # Los bytes van tal cual a la tarjeta y allá se pasan a 0..1 (4 veces menos que mover decimales).
        s = self.detector([TF.pil_to_tensor(foto).to(self.dispositivo).float() / 255])[0]
        return [(caja, seg) for caja, seg in zip(s["boxes"].tolist(), s["scores"].tolist()) if seg >= self.umbral]

    def buscar(self, foto):
        """Las cajas de placa en la foto, con su seguridad, de la más segura a la menos.
        Si no encuentra nada, vuelve a mirar con más detalle (placa lejos en una foto grande)
        y con menos (placa que llena casi toda la foto, como al tomarla de cerca con el teléfono)."""
        foto = foto.convert("RGB")
        cajas = self.detectar(foto, detector.TAMANO)
        if not self.insistir:
            return cajas
        if not cajas and min(foto.size) > 900:
            cajas = self.detectar(foto, detector.TAMANO * 2)
        if not cajas:
            cajas = self.detectar(foto, detector.TAMANO * 5 // 8)
        self.detector.transform.min_size = (detector.TAMANO,)
        self.detector.transform.max_size = detector.TAMANO * 5 // 3
        return cajas

    @torch.no_grad()
    def leer_recorte(self, foto, caja):
        """Lee una placa ya ubicada. Devuelve (texto, seguridad de la lectura, votos)."""
        x0, y0, x1, y1 = caja
        w, h = x1 - x0, y1 - y0
        recortes = [
            foto.crop((x0 - mx * w, y0 - my * h, x1 + mx * w, y1 + my * h)) for mx, my in MARGENES
        ]
        lote = torch.stack([lector.a_tensor(r) for r in recortes]).to(self.dispositivo)
        lecturas = lector.decodificar(self.lector(lote))
        validas = [(t, s) for t, s in lecturas if lector.interpretar(t)]
        candidatas = validas or lecturas
        votos = Counter(t for t, _ in candidatas)
        texto = max(votos, key=lambda t: (votos[t], max(s for tt, s in candidatas if tt == t)))
        seguridad = max(s for t, s in candidatas if t == texto)
        return texto, seguridad, votos[texto]

    def leer(self, foto):
        foto = foto.convert("RGB")
        resultados = []
        for caja, seg_det in self.buscar(foto):
            texto, seg_lec, votos = self.leer_recorte(foto, caja)
            resultados.append({
                "caja": caja,
                "texto": texto,
                "placa": lector.bonito(texto),
                "valida": lector.interpretar(texto) is not None,
                "seguridad": seg_det * seg_lec,
                "votos": votos,
            })
        return resultados
