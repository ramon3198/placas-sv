r"""
Entrena un detector de placas "general": que las encuentre en cualquier escena (parqueos con
varios carros, calle, placas lejos o en ángulo, de noche), no solo frente a un portón.

  entrenar_detector_general.cmd mobilenet      → el liviano y rápido
  entrenar_detector_general.cmd resnet50       → el preciso, más lento
  entrenar_detector_general.cmd resnet50 3     → con otra cantidad de épocas

Junta varias fuentes de fotos (las que estén descargadas):
  - Open Images: miles de fotos de calle, parqueos y carreteras de muchos países.
  - Las de portones de parqueo (el dataset del principio), solo la mitad para que no dominen.
  - AUTO.RIA: fotos de carros de anuncios, de frente, atrás, de lado y de cerca (como las de
    un teléfono).
  - NightLPBD: calles de noche, con faros, reflejos y poca luz.
  - Las tres primeras también con placas salvadoreñas pegadas encima (crear_examen_fotos.py).
Y a cada foto le aplica variaciones fuertes: mosaicos de 4 fotos (varios carros y placas
chicas), alejar y acercar, espejo, luz, blanco y negro (cámaras de noche) y desenfoque. Al final
cada foto se mete, achicada al azar, en un lienzo cuadrado fijo.

Guarda en modelos\detector_<arquitectura>_general.safetensors: no reemplaza al que está en uso.
"""
import math
import os
import random
import sys
import time
from collections import defaultdict

CARPETA = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, CARPETA)
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
# cuDNN guarda en la RAM un "plan" por cada forma de cálculo distinta, hasta 10.000. Con el
# lienzo fijo (ver LADO) entrenar usa unos pocos cientos; el límite es para los exámenes, donde
# cada foto tiene otro tamaño. (Con 64 los planes no alcanzaban y se rehacían en cada paso.)
os.environ.setdefault("TORCH_CUDNN_V8_API_LRU_CACHE_LIMIT", "2000")

import numpy
import torch
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
from safetensors.torch import save_file

from detector import TAMANO, crear_modelo, guardar_ficha
from entrenar_detector import Placas, evaluar

ARQUITECTURA = sys.argv[1] if len(sys.argv) > 1 else "mobilenet"
EPOCAS = int(sys.argv[2]) if len(sys.argv) > 2 else (3 if ARQUITECTURA == "mobilenet" else 2)
LOTE = int(sys.argv[3]) if len(sys.argv) > 3 else 4
APRENDIZAJE = 0.01 * LOTE / 8  # el ritmo de aprendizaje va en proporción al tamaño del lote
# Cada foto se mete en un lienzo cuadrado de este lado, como en los entrenamientos tipo YOLO.
# La variedad de tamaños se logra escalando la foto adentro del lienzo, pero la tarjeta siempre
# ve la misma forma. Con formas distintas en cada paso, cuDNN armaba un plan nuevo cada vez: el
# entrenamiento iba lentísimo y la memoria crecía 1 GB por minuto hasta agotarse.
LADO = 768 if ARQUITECTURA == "mobilenet" else 640
SALIDA = os.path.join(CARPETA, "modelos", f"detector_{ARQUITECTURA}_general.safetensors")

DATOS = os.path.join(CARPETA, "descargas", "dataset")
OI = os.path.join(CARPETA, "descargas", "openimages")
AUTOS = os.path.join(CARPETA, "descargas", "autoria")
NOCHE = os.path.join(CARPETA, "descargas", "nocturnas", "NightLPBD v1.0")
dispositivo = "cuda" if torch.cuda.is_available() else "cpu"
random.seed(0)
torch.manual_seed(0)


class Variado(torch.utils.data.Dataset):
    """Junta varias fuentes y le aplica a cada foto variaciones al azar."""

    def __init__(self, fuentes):
        self.fotos = [f for fuente in fuentes for f in fuente]

    def __len__(self):
        return len(self.fotos)

    def abrir(self, k):
        ruta, cajas = self.fotos[k]
        img = Image.open(ruta).convert("RGB")
        cajas = [list(c) for c in cajas]
        if max(img.size) > 1280:  # las fotos muy grandes se achican: el modelo igual las ve a <=768
            f = 1280 / max(img.size)
            img = img.resize((round(img.width * f), round(img.height * f)), Image.Resampling.BILINEAR)
            cajas = [[v * f for v in c] for c in cajas]
        return img, cajas

    def __getitem__(self, k):
        r = random
        img, cajas = self.abrir(k)
        if r.random() < 0.35:  # mosaico: 4 fotos en una, con placas más chicas y varios carros
            img, cajas = mosaico([(img, cajas)] + [self.abrir(r.randrange(len(self))) for _ in range(3)], r)
        u = r.random()
        if u < 0.2:
            img, cajas = alejar(img, cajas, r)
        elif u < 0.55:
            img, cajas = acercar(img, cajas, r)
        if r.random() < 0.5:
            img = ImageOps.mirror(img)
            cajas = [[img.width - c[2], c[1], img.width - c[0], c[3]] for c in cajas]
        img = ImageEnhance.Brightness(img).enhance(r.uniform(0.5, 1.5))
        img = ImageEnhance.Contrast(img).enhance(r.uniform(0.6, 1.4))
        img = ImageEnhance.Color(img).enhance(r.uniform(0.5, 1.5))
        if r.random() < 0.15:  # cámara de noche: blanco y negro
            img = img.convert("L").convert("RGB")
        if r.random() < 0.15:
            img = img.filter(ImageFilter.GaussianBlur(r.uniform(0.3, 2.0)))
        img, cajas = lienzo_fijo(img, cajas, r)
        # Se manda como arreglo de numpy en enteros de 0 a 255, no como tensor: en Windows, los
        # tensores que los procesos auxiliares le pasan al principal por memoria compartida no se
        # liberan, y la memoria virtual se agotaba en una hora.
        x = numpy.asarray(img, dtype=numpy.uint8)
        cajas = [c for c in cajas if c[2] - c[0] >= 3 and c[3] - c[1] >= 3]
        return x, numpy.asarray(cajas, dtype=numpy.float32).reshape(-1, 4)


def lienzo_fijo(img, cajas, r):
    """Mete la foto, achicada al azar, en un cuadrado gris de LADO×LADO."""
    escala = min(LADO / img.width, LADO / img.height) * r.uniform(0.55, 1.0)
    w, h = max(1, round(img.width * escala)), max(1, round(img.height * escala))
    ox, oy = r.randint(0, LADO - w), r.randint(0, LADO - h)
    lienzo = Image.new("RGB", (LADO, LADO), (114, 114, 114))
    lienzo.paste(img.resize((w, h), Image.Resampling.BILINEAR), (ox, oy))
    return lienzo, [[c[0] * escala + ox, c[1] * escala + oy, c[2] * escala + ox, c[3] * escala + oy] for c in cajas]


def mosaico(piezas, r, lado=896):
    lienzo = Image.new("RGB", (lado, lado), (114, 114, 114))
    cx, cy = int(lado * r.uniform(0.35, 0.65)), int(lado * r.uniform(0.35, 0.65))
    zonas = [(0, 0, cx, cy), (cx, 0, lado, cy), (0, cy, cx, lado), (cx, cy, lado, lado)]
    todas = []
    for (img, cajas), (x0, y0, x1, y1) in zip(piezas, zonas):
        escala = min((x1 - x0) / img.width, (y1 - y0) / img.height)
        w, h = max(1, int(img.width * escala)), max(1, int(img.height * escala))
        ox, oy = x0 + (x1 - x0 - w) // 2, y0 + (y1 - y0 - h) // 2
        lienzo.paste(img.resize((w, h), Image.Resampling.BILINEAR), (ox, oy))
        todas += [[c[0] * escala + ox, c[1] * escala + oy, c[2] * escala + ox, c[3] * escala + oy] for c in cajas]
    return lienzo, todas


def alejar(img, cajas, r):
    """La foto se ve desde más lejos: queda chica dentro de un lienzo más grande."""
    f = r.uniform(1.3, 2.8)
    W, H = int(img.width * f), int(img.height * f)
    lienzo = Image.new("RGB", (W, H), tuple(int(v) for v in img.resize((1, 1)).getpixel((0, 0))))
    ox, oy = r.randint(0, W - img.width), r.randint(0, H - img.height)
    lienzo.paste(img, (ox, oy))
    # Se vuelve al tamaño original: todo se ve más chico y lejano, sin agrandar la imagen.
    lienzo = lienzo.resize(img.size, Image.Resampling.BILINEAR)
    return lienzo, [[(c[0] + ox) / f, (c[1] + oy) / f, (c[2] + ox) / f, (c[3] + oy) / f] for c in cajas]


def acercar(img, cajas, r):
    """Se recorta un pedazo de la foto: las placas se ven más grandes. Si una placa queda
    cortada a la mitad, se descarta el recorte."""
    for _ in range(10):
        s = r.uniform(0.45, 0.85)
        w, h = int(img.width * s), int(img.height * s)
        x0, y0 = r.randint(0, img.width - w), r.randint(0, img.height - h)
        nuevas, cortada = [], False
        for c in cajas:
            dentro = [max(c[0], x0), max(c[1], y0), min(c[2], x0 + w), min(c[3], y0 + h)]
            if dentro[2] <= dentro[0] or dentro[3] <= dentro[1]:
                continue  # quedó afuera
            area = (dentro[2] - dentro[0]) * (dentro[3] - dentro[1])
            if area < 0.7 * (c[2] - c[0]) * (c[3] - c[1]):
                cortada = True
                break
            nuevas.append([dentro[0] - x0, dentro[1] - y0, dentro[2] - x0, dentro[3] - y0])
        if not cortada:
            return img.crop((x0, y0, x0 + w, y0 + h)), nuevas
    return img, cajas


def juntar(lote):
    return tuple(zip(*lote))


def sembrar(_):
    """Cada proceso que prepara fotos necesita su propio azar, si no todos variarían igual."""
    random.seed(torch.initial_seed() % 2**32)


def id_de(ruta):
    """El dataset de portones trae algunas fotos de Open Images: se reconocen por su código."""
    return os.path.basename(ruta).split("_jpg")[0].split(".")[0]


def fuente(ruta, mitad=False, excluir=frozenset()):
    if not os.path.exists(ruta):
        print(f"  (no está {ruta}, se omite)")
        return []
    fotos = [f for f in Placas(ruta).fotos if id_de(f[0]) not in excluir]
    if mitad:
        fotos = random.Random(1).sample(fotos, len(fotos) // 2)
    return fotos


def main():
    # En Windows, cuando la tarjeta se llena, el controlador sigue usando la RAM del sistema sin
    # avisar: todo se vuelve lentísimo y la memoria virtual se agota. Con este techo, PyTorch se
    # queda dentro de la tarjeta (va aquí y no arriba para que los procesos auxiliares no la usen).
    torch.cuda.set_per_process_memory_fraction(0.85)
    oi_entreno = os.path.join(OI, "train", "Vehicle registration plate")
    oi_prueba = os.path.join(OI, "validation", "Vehicle registration plate")
    # Ninguna foto de examen puede colarse al entrenamiento, aunque venga por otro dataset.
    de_examen = {id_de(f[0]) for d in ("valid", "test") for f in Placas(os.path.join(DATOS, d)).fotos}
    de_examen |= {id_de(f[0]) for f in Placas(oi_prueba).fotos}
    entreno = Variado([
        fuente(oi_entreno, excluir=de_examen),
        fuente(os.path.join(DATOS, "train"), mitad=True, excluir=de_examen),
        fuente(os.path.join(CARPETA, "datos", "fotos_sv_entreno")),
        fuente(os.path.join(CARPETA, "datos", "fotos_sv_oi_entreno")),
        fuente(os.path.join(AUTOS, "val")),  # fotos de carros de anuncios: de frente, atrás, de lado
        fuente(os.path.join(CARPETA, "datos", "fotos_sv_autos_entreno")),
        fuente(os.path.join(NOCHE, "train")),  # calles de noche: faros, reflejos, poca luz
    ])
    examenes = {
        "escenas variadas (Open Images)": Placas(oi_prueba),
        "portones": Placas(os.path.join(DATOS, "test")),
        "placas SV en portones": Placas(os.path.join(CARPETA, "datos", "examen_fotos")),
    }
    opcionales = {
        "placas SV en escenas variadas": os.path.join(CARPETA, "datos", "examen_fotos_oi"),
        "carros tipo foto de teléfono": os.path.join(AUTOS, "test"),
        "placas SV en fotos de carros": os.path.join(CARPETA, "datos", "examen_fotos_autos"),
        "de noche": os.path.join(NOCHE, "valid"),
    }
    for nombre, ruta in opcionales.items():
        if os.path.exists(ruta):
            examenes[nombre] = Placas(ruta)
    print(f"\nDetector {ARQUITECTURA}: {len(entreno):,} fotos para entrenar, {EPOCAS} épocas, lotes de {LOTE}")
    cargador = torch.utils.data.DataLoader(entreno, batch_size=LOTE, shuffle=True, collate_fn=juntar,
                                           num_workers=6, worker_init_fn=sembrar, prefetch_factor=2)

    modelo = crear_modelo(ARQUITECTURA).to(dispositivo)
    pesos = [p for p in modelo.parameters() if p.requires_grad]
    print(f"Pesos: {sum(p.numel() for p in modelo.parameters()):,}  ·  en {torch.cuda.get_device_name(0)}\n")

    def examinar(cuantas=300):
        modelo.transform.min_size, modelo.transform.max_size = (TAMANO,), TAMANO * 5 // 3
        for nombre, datos in examenes.items():
            e, p, a = evaluar(modelo, datos, cuantas=cuantas)
            print(f"   {nombre:32} encuentra {e:6.1%}  ·  acierta {p:6.1%}  ·  rectángulos {a:6.1%}")
        modelo.transform.min_size, modelo.transform.max_size = (LADO,), LADO

    optimizador = torch.optim.SGD(pesos, lr=APRENDIZAJE, momentum=0.9, weight_decay=1e-4)
    escalador = torch.amp.GradScaler("cuda")
    total = EPOCAS * len(cargador)
    modelo.transform.min_size, modelo.transform.max_size = (LADO,), LADO
    paso, t0 = 0, time.time()
    for epoca in range(1, EPOCAS + 1):
        suma = defaultdict(float)
        for fotos, metas in cargador:
            lr = APRENDIZAJE * min(1, (paso + 1) / 500) * 0.5 * (1 + math.cos(math.pi * paso / total))
            for g in optimizador.param_groups:
                g["lr"] = lr
            try:
                fotos = [torch.from_numpy(f).to(dispositivo).permute(2, 0, 1).float().div_(255) for f in fotos]
                metas = [{"boxes": torch.from_numpy(m).to(dispositivo),
                          "labels": torch.ones(len(m), dtype=torch.int64, device=dispositivo)} for m in metas]
                with torch.autocast("cuda"):
                    errores = modelo(fotos, metas)
                es_placa = errores["loss_objectness"] + errores["loss_classifier"]
                donde = errores["loss_rpn_box_reg"] + errores["loss_box_reg"]
                optimizador.zero_grad()
                escalador.scale(es_placa + donde).backward()
                escalador.unscale_(optimizador)
                torch.nn.utils.clip_grad_norm_(pesos, 10)
                escalador.step(optimizador)
                escalador.update()
            except torch.OutOfMemoryError:
                # Un lote que no cupo en la tarjeta se salta, en vez de desbordar a la RAM.
                del fotos, metas
                optimizador.zero_grad()
                torch.cuda.empty_cache()
                suma["saltados"] += 1
                paso += 1
                continue
            paso += 1
            if paso % 50 == 0:
                torch.cuda.empty_cache()  # que la memoria de la tarjeta no se fragmente con los tamaños variables
            suma["placa"] += es_placa.item()
            suma["donde"] += donde.item()
            suma["n"] += 1
            if paso % 200 == 0:
                faltan = (time.time() - t0) / paso * (total - paso)
                print(f"época {epoca}  paso {paso:5}/{total}  |  error ¿es placa? {suma['placa'] / max(1, suma['n']):.3f}"
                      f"  |  ¿dónde? {suma['donde'] / max(1, suma['n']):.3f}  |  {(time.time() - t0) / 60:4.0f} min, faltan ~{faltan / 60:.0f}"
                      + (f"  |  {int(suma['saltados'])} lotes no cupieron" if suma["saltados"] else ""))
                suma.clear()
        print(f"\n── Fin de la época {epoca}:")
        examinar()
        os.makedirs(os.path.dirname(SALIDA), exist_ok=True)
        save_file({k: v.contiguous() for k, v in modelo.state_dict().items()}, SALIDA)
        guardar_ficha(SALIDA, ARQUITECTURA)
        print()

    print("Examen final, con todas las fotos de cada examen:")
    examinar(cuantas=None)
    print(f"\nListo en {(time.time() - t0) / 60:.0f} minutos. Guardado en {SALIDA}")


if __name__ == "__main__":
    main()
