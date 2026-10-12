"""
Entrena el detector de placas con fotos de cámaras de parqueo ya marcadas a mano.

Es el mismo ciclo que el mini-gpt (predecir, medir el error, ajustar, repetir), pero aquí
la entrada son píxeles y la salida son rectángulos.

  entrenar_detector.cmd          → desde los pesos de COCO, 2 épocas
  entrenar_detector.cmd afinar   → parte del detector ya entrenado y lo afina con fotos que
                                   tienen placas salvadoreñas pegadas (crear_examen_fotos --entreno)
"""
import json
import math
import os
import random
import shutil
import sys
import time
from collections import defaultdict

CARPETA = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, CARPETA)
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import torch
import torchvision.transforms.functional as TF
from PIL import Image
from safetensors.torch import load_file, save_file
from torchvision.ops import box_iou

from detector import MODELO, crear_modelo, guardar_ficha

DATOS = os.path.join(CARPETA, "descargas", "dataset")
SV_ENTRENO = os.path.join(CARPETA, "datos", "fotos_sv_entreno")
SV_EXAMEN = os.path.join(CARPETA, "datos", "examen_fotos")
AFINAR = "afinar" in sys.argv
EPOCAS = 1 if AFINAR else 2  # cuántas veces ve todas las fotos
LOTE = 8
APRENDIZAJE = 0.003 if AFINAR else 0.01

dispositivo = "cuda" if torch.cuda.is_available() else "cpu"
random.seed(0)
torch.manual_seed(0)


class Placas(torch.utils.data.Dataset):
    """Lee las fotos y sus rectángulos (formato COCO: x, y, ancho, alto en píxeles)."""

    def __init__(self, carpeta, aumentar=False):
        if os.path.exists(os.path.join(carpeta, "_annotations.coco.json")):
            with open(os.path.join(carpeta, "_annotations.coco.json"), encoding="utf-8") as f:
                coco = json.load(f)
            cajas = defaultdict(list)
            for a in coco["annotations"]:
                x, y, w, h = a["bbox"]
                if w >= 2 and h >= 2:
                    cajas[a["image_id"]].append([x, y, x + w, y + h])
            self.fotos = [(os.path.join(carpeta, i["file_name"]), cajas[i["id"]]) for i in coco["images"]]
        elif os.path.isdir(os.path.join(carpeta, "images")) and os.path.isdir(os.path.join(carpeta, "labels")):
            # Formato YOLO: images/ y labels/, con "clase x_centro y_centro ancho alto" de 0 a 1.
            self.fotos = []
            for nombre in sorted(os.listdir(os.path.join(carpeta, "images"))):
                etiqueta = os.path.join(carpeta, "labels", os.path.splitext(nombre)[0] + ".txt")
                if not nombre.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")) or not os.path.exists(etiqueta):
                    continue
                ruta = os.path.join(carpeta, "images", nombre)
                with Image.open(ruta) as img:  # solo lee el encabezado, para saber el tamaño
                    W, H = img.size
                cajas = []
                with open(etiqueta, encoding="utf-8") as f:
                    for linea in f:
                        partes = linea.split()
                        if len(partes) >= 5 and partes[0] == "0":  # clase 0 = placa (otros datasets marcan también letras)
                            xc, yc, w, h = (float(v) for v in partes[1:5])
                            cajas.append([(xc - w / 2) * W, (yc - h / 2) * H, (xc + w / 2) * W, (yc + h / 2) * H])
                self.fotos.append((ruta, [c for c in cajas if c[2] - c[0] >= 2 and c[3] - c[1] >= 2]))
        elif os.path.isdir(os.path.join(carpeta, "Label")):
            # Open Images: un .txt por foto, con líneas "Vehicle registration plate x0 y0 x1 y1".
            self.fotos = []
            for nombre in sorted(os.listdir(carpeta)):
                etiqueta = os.path.join(carpeta, "Label", os.path.splitext(nombre)[0] + ".txt")
                if nombre.lower().endswith(".jpg") and os.path.exists(etiqueta):
                    with open(etiqueta, encoding="utf-8") as f:
                        cajas = [[float(v) for v in linea.split()[-4:]] for linea in f if linea.strip()]
                    cajas = [c for c in cajas if c[2] - c[0] >= 2 and c[3] - c[1] >= 2]
                    self.fotos.append((os.path.join(carpeta, nombre), cajas))
        else:  # el examen de fotos compuestas guarda sus respuestas en otro formato
            with open(os.path.join(carpeta, "respuestas.json"), encoding="utf-8") as f:
                self.fotos = [(os.path.join(carpeta, r["foto"]), [r["caja"]]) for r in json.load(f)]
        self.aumentar = aumentar

    def __len__(self):
        return len(self.fotos)

    def __getitem__(self, k):
        ruta, cajas = self.fotos[k]
        x = TF.to_tensor(Image.open(ruta).convert("RGB"))
        b = torch.tensor(cajas, dtype=torch.float32).reshape(-1, 4)
        if self.aumentar:
            # Variaciones al azar para que no memorice las fotos: espejo, luz y contraste.
            if random.random() < 0.5:
                x = x.flip(-1)
                b[:, [0, 2]] = x.shape[-1] - b[:, [2, 0]]
            x = TF.adjust_brightness(x, random.uniform(0.6, 1.4))
            x = TF.adjust_contrast(x, random.uniform(0.7, 1.3))
        return x, {"boxes": b, "labels": torch.ones(len(b), dtype=torch.int64)}


def juntar(lote):
    return tuple(zip(*lote))


@torch.no_grad()
def evaluar(modelo, datos, cuantas=None, umbral=0.5):
    """De las placas reales, ¿cuántas encontró? Y de lo que marcó, ¿cuánto era placa de verdad?"""
    modelo.eval()
    indices = list(range(len(datos)))[: cuantas or len(datos)]
    acertadas = falsas = perdidas = 0
    ajustes = []
    for i in range(0, len(indices), LOTE):
        fotos, metas = zip(*(datos[k] for k in indices[i : i + LOTE]))
        with torch.autocast("cuda", enabled=dispositivo == "cuda"):
            salidas = modelo([f.to(dispositivo) for f in fotos])
        for s, m in zip(salidas, metas):
            marcadas = s["boxes"][s["scores"] >= umbral].float().cpu()
            reales = m["boxes"]
            usadas = set()
            for caja in marcadas:  # vienen ordenadas de la más segura a la menos
                iou = box_iou(caja[None], reales)[0] if len(reales) else torch.zeros(0)
                if len(iou):
                    iou[list(usadas)] = 0
                if len(iou) and iou.max() >= 0.5:
                    usadas.add(int(iou.argmax()))
                    ajustes.append(float(iou.max()))
                    acertadas += 1
                else:
                    falsas += 1
            perdidas += len(reales) - len(usadas)
    modelo.train()
    encontradas = acertadas / max(1, acertadas + perdidas)
    precision = acertadas / max(1, acertadas + falsas)
    ajuste = sum(ajustes) / max(1, len(ajustes))
    return encontradas, precision, ajuste


def main():
    entreno = Placas(os.path.join(DATOS, "train"), aumentar=True)
    prueba, examen = Placas(os.path.join(DATOS, "valid")), Placas(os.path.join(DATOS, "test"))
    sv = Placas(SV_EXAMEN) if os.path.exists(SV_EXAMEN) else None
    if AFINAR:
        # Las fotos originales más las de placa salvadoreña (dos veces), para que aprenda las
        # nuevas sin olvidar las que ya sabía.
        con_sv = Placas(SV_ENTRENO, aumentar=True)
        entreno = torch.utils.data.ConcatDataset([entreno, con_sv, con_sv])
    print(f"\nFotos para entrenar: {len(entreno):,}  ·  para medir: {len(prueba):,}  ·  examen final: {len(examen):,}")
    cargador = torch.utils.data.DataLoader(entreno, batch_size=LOTE, shuffle=True, collate_fn=juntar)

    modelo = crear_modelo().to(dispositivo)
    if AFINAR:
        modelo.load_state_dict(load_file(MODELO))
        shutil.copy(MODELO, MODELO.replace(".safetensors", "_antes_de_afinar.safetensors"))  # por si sale peor
    pesos = [p for p in modelo.parameters() if p.requires_grad]
    print(f"El detector tiene {sum(p.numel() for p in modelo.parameters()):,} pesos "
          f"({sum(p.numel() for p in pesos):,} se van a ajustar; el resto queda como venía de COCO).")
    print(f"Entrenando en: {torch.cuda.get_device_name(0) if dispositivo == 'cuda' else 'CPU'}\n")

    e, p, a = evaluar(modelo, prueba, cuantas=200)
    print(f"Antes de entrenar  →  encuentra {e:5.1%} de las placas  ·  acierta {p:5.1%} de lo que marca")
    if sv:
        e, p, a = evaluar(modelo, sv)
        print(f"   con placas salvadoreñas  →  encuentra {e:5.1%}  ·  acierta {p:5.1%}  ·  rectángulos {a:5.1%}")
    print()

    optimizador = torch.optim.SGD(pesos, lr=APRENDIZAJE, momentum=0.9, weight_decay=1e-4)
    escalador = torch.amp.GradScaler("cuda", enabled=dispositivo == "cuda")
    total = EPOCAS * len(cargador)
    paso, t0 = 0, time.time()
    for epoca in range(1, EPOCAS + 1):
        suma = defaultdict(float)
        for fotos, metas in cargador:
            # Primero ajustes pequeños, luego grandes, y al final cada vez más finos.
            lr = APRENDIZAJE * min(1, (paso + 1) / 300) * 0.5 * (1 + math.cos(math.pi * paso / total))
            for g in optimizador.param_groups:
                g["lr"] = lr
            fotos = [f.to(dispositivo) for f in fotos]
            metas = [{k: v.to(dispositivo) for k, v in m.items()} for m in metas]
            with torch.autocast("cuda", enabled=dispositivo == "cuda"):
                errores = modelo(fotos, metas)  # 1. predecir y medir el error
            # Faster R-CNN mide 4 errores; los juntamos en dos ideas:
            es_placa = errores["loss_objectness"] + errores["loss_classifier"]  # ¿hay placa ahí?
            donde = errores["loss_rpn_box_reg"] + errores["loss_box_reg"]  # ¿qué tan bien ajustó el rectángulo?
            optimizador.zero_grad()
            escalador.scale(es_placa + donde).backward()  # 2. hacia dónde mover cada peso
            escalador.step(optimizador)  # 3. moverlos un poquito
            escalador.update()
            paso += 1
            suma["placa"] += es_placa.item()
            suma["donde"] += donde.item()
            suma["n"] += 1
            if paso % 100 == 0:
                print(f"época {epoca}  paso {paso:5}/{total}  |  error ¿es placa? {suma['placa'] / suma['n']:.3f}"
                      f"  |  error ¿dónde? {suma['donde'] / suma['n']:.3f}  |  {time.time() - t0:4.0f} s")
                suma.clear()
        e, p, a = evaluar(modelo, prueba, cuantas=400)
        print(f"\n── Fin de la época {epoca}  →  encuentra {e:5.1%} de las placas  ·  acierta {p:5.1%} de lo que marca"
              f"  ·  el rectángulo coincide {a:5.1%} con el marcado a mano\n")

    os.makedirs(os.path.dirname(MODELO), exist_ok=True)
    save_file({k: v.contiguous() for k, v in modelo.state_dict().items()}, MODELO)
    guardar_ficha(MODELO, "mobilenet")

    print("Examen final con fotos que nunca vio…")
    e, p, a = evaluar(modelo, examen)
    print(f"  Encontró {e:.1%} de las placas  ·  acertó {p:.1%} de lo que marcó  ·  rectángulos {a:.1%} de coincidencia")
    if sv:
        e, p, a = evaluar(modelo, sv)
        print(f"  Con placas salvadoreñas: encontró {e:.1%}  ·  acertó {p:.1%}  ·  rectángulos {a:.1%}")
    print(f"\nListo en {(time.time() - t0) / 60:.1f} minutos. Detector guardado en {MODELO}")


if __name__ == "__main__":
    main()
