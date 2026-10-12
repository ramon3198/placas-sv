r"""
Lee las placas de tus fotos y las dibuja.

  probar_lector.cmd                     → 9 fotos al azar del examen
  probar_lector.cmd foto.jpg carpeta\   → tus propias fotos (o carpetas de fotos)

Deja el resultado en resultados\lecturas.jpg.

Para medir con fotos reales: nombra cada foto con su placa (P123456.jpg, o P123456_2.jpg si hay
varias del mismo carro) y al final te dice el porcentaje de aciertos.
"""
import glob
import json
import os
import random
import sys

CARPETA = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, CARPETA)
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from PIL import Image, ImageDraw, ImageFont, ImageOps

from lector import bonito, interpretar
from placas import LectorDePlacas

EXAMEN = os.path.join(CARPETA, "datos", "examen_fotos")


def fotos_pedidas():
    if len(sys.argv) == 1:
        with open(os.path.join(EXAMEN, "respuestas.json"), encoding="utf-8") as f:
            return [os.path.join(EXAMEN, r["foto"]) for r in random.sample(json.load(f), 9)]
    rutas = []
    for a in sys.argv[1:]:
        if os.path.isdir(a):
            rutas += sorted(r for r in glob.glob(os.path.join(a, "*")) if r.lower().endswith((".jpg", ".jpeg", ".png")))
        else:
            rutas.append(a)
    return rutas


def main():
    sistema = LectorDePlacas()
    miniaturas, comparadas, bien = [], 0, 0
    for ruta in fotos_pedidas():
        # Las fotos del teléfono guardan aparte hacia dónde estaba girado: se endereza aquí.
        foto = ImageOps.exif_transpose(Image.open(ruta)).convert("RGB")
        lecturas = sistema.leer(foto)
        d = ImageDraw.Draw(foto)
        letra = ImageFont.truetype("bahnschrift.ttf", max(16, foto.width // 22))
        for p in lecturas:
            color = (0, 230, 90) if p["valida"] else (250, 180, 0)
            d.rectangle(p["caja"], outline=color, width=max(2, foto.width // 250))
            etiqueta = p["placa"] if p["valida"] else f"{p['texto']}?"
            d.text((p["caja"][0], p["caja"][3] + 4), etiqueta, font=letra, fill=color, stroke_width=3, stroke_fill=(0, 0, 0))
        resumen = ", ".join(f"{p['placa']} ({p['seguridad']:.0%})" if p["valida"] else f"{p['texto']} (dudosa)" for p in lecturas)
        # Si el nombre del archivo es la placa, se compara con lo que leyó.
        real = os.path.splitext(os.path.basename(ruta))[0].split("_")[0].upper().replace("-", "")
        marca = ""
        if interpretar(real):
            comparadas += 1
            acerto = any(p["texto"] == real for p in lecturas)
            bien += acerto
            marca = "  ✓" if acerto else f"  ✗ (era {bonito(real)})"
        print(f"{os.path.basename(ruta):30}  →  {resumen or 'no encontró placa'}{marca}")
        foto.thumbnail((640, 420))
        miniaturas.append(foto)

    columnas = min(3, len(miniaturas))
    filas = (len(miniaturas) + columnas - 1) // columnas
    hoja = Image.new("RGB", (columnas * 640, filas * 420), (20, 20, 20))
    for i, m in enumerate(miniaturas):
        hoja.paste(m, ((i % columnas) * 640 + (640 - m.width) // 2, (i // columnas) * 420 + (420 - m.height) // 2))
    salida = os.path.join(CARPETA, "resultados", "lecturas.jpg")
    os.makedirs(os.path.dirname(salida), exist_ok=True)
    hoja.save(salida, quality=90)
    print(f"\nResultado: {salida}")
    if comparadas:
        print(f"Acertó {bien} de {comparadas} fotos con la placa en el nombre ({bien / comparadas:.1%}).")


if __name__ == "__main__":
    main()
