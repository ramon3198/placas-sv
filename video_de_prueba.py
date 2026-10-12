r"""
Arma un video de prueba de un portón para probar camara.py sin cámara.

Junta fotos del examen: cada carro se va acercando unos segundos y luego el portón queda vacío.
Deja resultados\video_prueba.mp4 y, en video_prueba_respuestas.txt, las placas que pasan.

  video_de_prueba.cmd        → 8 carros
  video_de_prueba.cmd 20     → otra cantidad
"""
import json
import os
import random
import sys

CARPETA = os.path.dirname(os.path.abspath(__file__))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import av
from PIL import Image, ImageEnhance

EXAMEN = os.path.join(CARPETA, "datos", "examen_fotos")
SALIDA = os.path.join(CARPETA, "resultados", "video_prueba.mp4")
ANCHO, ALTO, CUADROS = 640, 412, 15  # cuadros por segundo


def main():
    carros = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    r = random.Random(3)
    with open(os.path.join(EXAMEN, "respuestas.json"), encoding="utf-8") as f:
        elegidas = r.sample(json.load(f), carros)

    video = av.open(SALIDA, "w")
    flujo = video.add_stream("mpeg4", rate=CUADROS)
    flujo.width, flujo.height, flujo.pix_fmt = ANCHO, ALTO, "yuv420p"
    flujo.bit_rate = 2_000_000

    def escribir(imagen):
        for paquete in flujo.encode(av.VideoFrame.from_image(imagen)):
            video.mux(paquete)

    vacio = Image.new("RGB", (ANCHO, ALTO), (70, 70, 70))
    for _ in range(CUADROS):
        escribir(vacio)
    for e in elegidas:
        foto = Image.open(os.path.join(EXAMEN, e["foto"])).convert("RGB")
        n = int(CUADROS * r.uniform(2.0, 3.5))  # el carro se queda 2 a 3,5 segundos
        luz = r.uniform(0.8, 1.15)
        for k in range(n):
            zoom = 0.82 + 0.18 * k / n  # se va acercando
            w, h = foto.width * zoom, foto.height * zoom
            x0, y0 = (foto.width - w) / 2, (foto.height - h) * 0.7
            cuadro = foto.crop((x0, y0, x0 + w, y0 + h)).resize((ANCHO, ALTO), Image.Resampling.BILINEAR)
            escribir(ImageEnhance.Brightness(cuadro).enhance(luz))
        for _ in range(int(CUADROS * r.uniform(1.5, 3.0))):  # portón vacío
            escribir(vacio)
    for paquete in flujo.encode():
        video.mux(paquete)
    video.close()

    with open(os.path.join(CARPETA, "resultados", "video_prueba_respuestas.txt"), "w", encoding="utf-8") as f:
        f.writelines(e["texto"] + "\n" for e in elegidas)
    print(f"Video: {SALIDA}")
    print("Placas que pasan, en orden: " + ", ".join(e["texto"] for e in elegidas))


if __name__ == "__main__":
    main()
