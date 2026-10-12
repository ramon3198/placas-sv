r"""
Compara detectores en todos los exámenes: cuántas placas encuentra y cuántas cosas marca que
no son placas. Separa las placas legibles (30 px o más de ancho) de las diminutas, que en
una foto de calle pueden medir 10 px y nadie podría leer.

  comparar_detectores.cmd                                    → el que está en uso
  comparar_detectores.cmd detector.safetensors detector_mobilenet_general.safetensors
"""
import os
import sys
import time

CARPETA = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, CARPETA)
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import torch
import torchvision.transforms.functional as TF
from PIL import Image
from torchvision.ops import box_iou

import detector
from entrenar_detector import Placas

UMBRAL = 0.35  # el mismo con que trabaja el sistema (se cambia con --umbral 0.6)
D = os.path.join(CARPETA, "descargas")
EXAMENES = {
    "escenas variadas (Open Images)": os.path.join(D, "openimages", "validation", "Vehicle registration plate"),
    "carros tipo foto de teléfono": os.path.join(D, "autoria", "test"),
    "de noche": os.path.join(D, "nocturnas", "NightLPBD v1.0", "valid"),
    "portones": os.path.join(D, "dataset", "test"),
    "placas SV en portones": os.path.join(CARPETA, "datos", "examen_fotos"),
    "placas SV en escenas variadas": os.path.join(CARPETA, "datos", "examen_fotos_oi"),
    "placas SV en fotos de carros": os.path.join(CARPETA, "datos", "examen_fotos_autos"),
}


@torch.no_grad()
def medir(modelo, datos, cuantas=600):
    legibles = legibles_bien = todas = todas_bien = marcadas = falsas = 0
    for ruta, cajas in datos.fotos[:cuantas]:
        foto = Image.open(ruta).convert("RGB")
        s = modelo([TF.to_tensor(foto).cuda()])[0]
        pred = s["boxes"][s["scores"] >= UMBRAL].float().cpu()
        reales = torch.tensor(cajas, dtype=torch.float32).reshape(-1, 4)
        marcadas += len(pred)
        if len(reales) and len(pred):
            iou = box_iou(reales, pred)
            encontrada = iou.max(1).values >= 0.5
            falsas += int((iou.max(0).values < 0.3).sum())
        else:
            encontrada = torch.zeros(len(reales), dtype=torch.bool)
            falsas += len(pred)
        anchos = reales[:, 2] - reales[:, 0]
        todas += len(reales)
        todas_bien += int(encontrada.sum())
        legibles += int((anchos >= 30).sum())
        legibles_bien += int((encontrada & (anchos >= 30)).sum())
    return legibles_bien / max(1, legibles), todas_bien / max(1, todas), falsas / max(1, marcadas)


def main():
    global UMBRAL
    argumentos = sys.argv[1:]
    if "--umbral" in argumentos:
        i = argumentos.index("--umbral")
        UMBRAL = float(argumentos[i + 1])
        del argumentos[i:i + 2]
    archivos = argumentos or ["detector.safetensors"]
    datos = {n: Placas(r) for n, r in EXAMENES.items() if os.path.exists(r)}
    for archivo in archivos:
        ruta = os.path.join(CARPETA, "modelos", archivo)
        modelo = detector.cargar_entrenado("cuda", ruta)
        t0 = time.time()
        print(f"\n{archivo}  ({detector.arquitectura_de(ruta)})")
        print(f"   {'examen':32} {'legibles':>9} {'todas':>7} {'marcas falsas':>14}")
        for nombre, d in datos.items():
            legibles, todas, falsas = medir(modelo, d)
            print(f"   {nombre:32} {legibles:9.1%} {todas:7.1%} {falsas:14.1%}")
        print(f"   ({time.time() - t0:.0f} s)")
        del modelo
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
