r"""
Prueba el detector y dibuja los rectángulos encontrados.

  probar_detector.cmd                    → 12 fotos al azar del examen final
  probar_detector.cmd foto.jpg carpeta\  → tus propias fotos (o carpetas de fotos)

Deja el resultado en resultados\detector.jpg y los recortes de cada placa en
resultados\recortes\, que es lo que después recibe el lector.
"""
import glob
import os
import random
import sys

CARPETA = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, CARPETA)
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import torch
import torchvision.transforms.functional as TF
from PIL import Image, ImageDraw, ImageFont

from detector import cargar_entrenado

UMBRAL = 0.5  # solo se dibuja lo que el modelo cree con más de 50 % de seguridad
RESULTADOS = os.path.join(CARPETA, "resultados")


def fotos_pedidas():
    if len(sys.argv) == 1:
        todas = glob.glob(os.path.join(CARPETA, "descargas", "dataset", "test", "*.jpg"))
        return random.sample(todas, 12)
    rutas = []
    for a in sys.argv[1:]:
        if os.path.isdir(a):
            rutas += [r for r in glob.glob(os.path.join(a, "*")) if r.lower().endswith((".jpg", ".jpeg", ".png"))]
        else:
            rutas.append(a)
    return rutas


def main():
    dispositivo = "cuda" if torch.cuda.is_available() else "cpu"
    modelo = cargar_entrenado(dispositivo)
    letra = ImageFont.truetype("bahnschrift.ttf", 22)
    os.makedirs(os.path.join(RESULTADOS, "recortes"), exist_ok=True)

    miniaturas = []
    for n, ruta in enumerate(fotos_pedidas()):
        foto = Image.open(ruta).convert("RGB")
        with torch.no_grad():
            s = modelo([TF.to_tensor(foto).to(dispositivo)])[0]
        dibujo = ImageDraw.Draw(foto)
        encontradas = 0
        for caja, seguridad in zip(s["boxes"].tolist(), s["scores"].tolist()):
            if seguridad < UMBRAL:
                continue
            encontradas += 1
            foto.crop(caja).save(os.path.join(RESULTADOS, "recortes", f"{n:02}_{encontradas}.png"))
            grosor = max(2, foto.width // 200)
            dibujo.rectangle(caja, outline=(0, 230, 90), width=grosor)
            dibujo.text((caja[0], caja[1] - 26), f"placa {seguridad:.0%}", fill=(0, 230, 90), font=letra,
                        stroke_width=2, stroke_fill=(0, 0, 0))
        print(f"{os.path.basename(ruta)[:40]:40}  →  {encontradas} placa(s)")
        foto.thumbnail((640, 420))
        miniaturas.append(foto)

    # Junta todas las fotos en una sola imagen, en cuadrícula de 3 columnas.
    columnas = min(3, len(miniaturas))
    filas = (len(miniaturas) + columnas - 1) // columnas
    hoja = Image.new("RGB", (columnas * 640, filas * 420), (20, 20, 20))
    for i, m in enumerate(miniaturas):
        hoja.paste(m, ((i % columnas) * 640 + (640 - m.width) // 2, (i // columnas) * 420 + (420 - m.height) // 2))
    salida = os.path.join(RESULTADOS, "detector.jpg")
    hoja.save(salida, quality=90)
    print(f"\nResultado: {salida}")


if __name__ == "__main__":
    main()
