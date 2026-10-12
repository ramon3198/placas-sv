r"""
Examen del sistema completo (ver crear_examen_fotos.py).

  evaluar.cmd                                   → fotos de portones (datos\examen_fotos)
  evaluar.cmd examen_fotos_oi                   → escenas variadas: calle, parqueos, carreteras
  evaluar.cmd examen_fotos_oi detector_resnet50_general.safetensors   → con otro detector

Mide lo que importa en un portón:
  - ¿Encontró la placa?
  - ¿La leyó exactamente bien?
  - Cuando no está seguro, ¿se abstiene en vez de inventar? Una lectura que no tiene
    formato de placa salvadoreña se marca como "dudosa" y no se cuenta como lectura.

Deja en resultados\errores_examen.jpg las fotos donde se equivocó, para revisarlas.
"""
import json
import os
import sys
import time

CARPETA = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, CARPETA)
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import torch
from PIL import Image, ImageDraw, ImageFont
from torchvision.ops import box_iou

from placas import LectorDePlacas

EXAMEN = os.path.join(CARPETA, "datos", sys.argv[1] if len(sys.argv) > 1 else "examen_fotos")
DETECTOR = os.path.join(CARPETA, "modelos", sys.argv[2]) if len(sys.argv) > 2 else None
LECTOR = os.path.join(CARPETA, "modelos", sys.argv[3]) if len(sys.argv) > 3 else None


def main():
    with open(os.path.join(EXAMEN, "respuestas.json"), encoding="utf-8") as f:
        respuestas = json.load(f)
    sistema = LectorDePlacas(detector_ruta=DETECTOR, lector_ruta=LECTOR)
    print(f"Examen: {EXAMEN}  ·  detector: {DETECTOR or 'el que está en uso'}  ·  lector: {LECTOR or 'el que está en uso'}")
    t0 = time.time()
    encontradas = bien = mal = dudosas = 0
    errores, casos = [], []  # casos: (ancho de la placa, seguridad, ¿acertó?) de cada lectura
    for r in respuestas:
        foto = Image.open(os.path.join(EXAMEN, r["foto"])).convert("RGB")
        lecturas = sistema.leer(foto)
        real = torch.tensor([r["caja"]], dtype=torch.float32)
        # La lectura que corresponde a la placa real es la del rectángulo que más se le encima.
        mejor = None
        if lecturas:
            iou = box_iou(torch.tensor([p["caja"] for p in lecturas]), real)[:, 0]
            if iou.max() >= 0.3:
                mejor = lecturas[int(iou.argmax())]
        ancho = r["caja"][2] - r["caja"][0]
        if mejor is None:
            errores.append((foto, r, None))
            casos.append((ancho, None, False, r.get("vehiculo", "carro")))
            continue
        encontradas += 1
        casos.append((ancho, mejor["seguridad"] if mejor["valida"] else None, mejor["texto"] == r["texto"],
                      r.get("vehiculo", "carro")))
        if not mejor["valida"]:
            dudosas += 1
            errores.append((foto, r, mejor))
        elif mejor["texto"] == r["texto"]:
            bien += 1
        else:
            mal += 1
            errores.append((foto, r, mejor))

    n = len(respuestas)
    print(f"\nExamen de {n} fotos en {time.time() - t0:.0f} s ({n / (time.time() - t0):.1f} fotos por segundo)\n")
    print(f"  Encontró la placa:              {encontradas / n:6.1%}")
    print(f"  La leyó exactamente bien:       {bien / n:6.1%}   ← acierto total")
    print(f"  Leyó algo equivocado:           {mal / n:6.1%}")
    print(f"  Dijo «dudosa» y no inventó:     {dudosas / n:6.1%}")
    print(f"  Cuando da una lectura, acierta: {bien / max(1, bien + mal):6.1%}")

    for vehiculo in ("carro", "moto"):
        grupo = [c for c in casos if c[3] == vehiculo]
        if grupo:
            print(f"  Leídas bien en {vehiculo + 's':6}:        {sum(c[2] for c in grupo) / len(grupo):6.1%}   ({len(grupo)} fotos)")
    print("\n  Según el tamaño de la placa en la foto:")
    for desde, hasta in ((0, 60), (60, 100), (100, 10_000)):
        grupo = [c for c in casos if desde <= c[0] < hasta]
        if grupo:
            nombre = f"más de {desde} px" if hasta == 10_000 else f"{desde} a {hasta} px"
            print(f"    {nombre:>14} de ancho: {sum(c[2] for c in grupo) / len(grupo):6.1%} bien  ({len(grupo)} fotos)")

    print("\n  Si solo se aceptan lecturas con cierta seguridad (las demás quedan como «dudosa»):")
    for umbral in (0.0, 0.5, 0.7, 0.85, 0.95):
        aceptadas = [c for c in casos if c[1] is not None and c[1] >= umbral]
        if aceptadas:
            print(f"    seguridad ≥ {umbral:4.0%}:  acepta {len(aceptadas) / n:6.1%} de los carros  ·  "
                  f"de esas acierta {sum(c[2] for c in aceptadas) / len(aceptadas):6.1%}")
    hoja(errores[:24])
    return bien / n


def hoja(errores):
    """Las fotos donde falló, con lo real y lo leído."""
    if not errores:
        return
    letra = ImageFont.truetype("bahnschrift.ttf", 20)
    hoja = Image.new("RGB", (4 * 360, ((len(errores) + 3) // 4) * 260), (25, 25, 25))
    d = ImageDraw.Draw(hoja)
    for i, (foto, r, lectura) in enumerate(errores):
        x0, y0, x1, y1 = r["caja"]
        w, h = x1 - x0, y1 - y0
        recorte = foto.crop((x0 - w, y0 - h, x1 + w, y1 + h))
        escala = min(340 / recorte.width, 200 / recorte.height)
        recorte = recorte.resize((round(recorte.width * escala), round(recorte.height * escala)), Image.Resampling.NEAREST)
        x, y = (i % 4) * 360 + 10, (i // 4) * 260 + 10
        hoja.paste(recorte, (x, y))
        leido = "no la encontró" if lectura is None else lectura["texto"] + ("  (dudosa)" if not lectura["valida"] else "")
        d.text((x, y + 205), f"real:  {r['texto']}", font=letra, fill=(120, 230, 140))
        d.text((x, y + 228), f"leyó: {leido}", font=letra, fill=(250, 140, 120))
    ruta = os.path.join(CARPETA, "resultados", "errores_examen.jpg")
    os.makedirs(os.path.dirname(ruta), exist_ok=True)
    hoja.save(ruta, quality=88)
    print(f"\n  Fotos con errores: {ruta}")


if __name__ == "__main__":
    main()
