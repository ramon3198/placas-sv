r"""
Arma exámenes de todo el sistema con fotos reales: de portones de parqueo y de escenas variadas.

Toma fotos que el detector nunca usó para entrenar y, encima de cada placa original, pega una
placa salvadoreña falsa con la fuente que el lector no conoce. Le ajusta la luz y la nitidez
para que combine con la foto. No reemplaza a las fotos reales de aquí, pero es lo más parecido
que se puede medir sin ellas.

Deja las fotos en datos\examen_fotos\ con sus respuestas en respuestas.json.
Con --entreno arma otras parecidas, a partir de las fotos de entrenamiento, para afinar el detector.
Con --oi y --oi-entreno hace lo mismo con las fotos de Open Images (calle, parqueos, carreteras):
el examen queda en datos\examen_fotos_oi\ y las de entrenamiento en datos\fotos_sv_oi_entreno\.
"""
import json
import math
import os
import random
import sys

CARPETA = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, CARPETA)
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageStat

from generar_placas import FUENTES_ENTRENO, FUENTES_EXAMEN, dibujar_placa, texto_al_azar

DATOS = os.path.join(CARPETA, "descargas", "dataset")
OI = os.path.join(CARPETA, "descargas", "openimages")
AUTOS = os.path.join(CARPETA, "descargas", "autoria")
SALIDA = os.path.join(CARPETA, "datos", "examen_fotos")


def tapar(foto, x0, y0, x1, y1, r):
    """Borra un rectángulo de la foto rellenándolo con un degradado entre la fila de arriba y la
    de abajo, con un poco de ruido para que no quede liso."""
    x0, y0 = max(0, round(x0)), max(1, round(y0))
    x1, y1 = min(foto.width, round(x1)), min(foto.height - 1, round(y1))
    if x1 - x0 < 2 or y1 - y0 < 2:
        return
    arriba = foto.crop((x0, y0 - 1, x1, y0)).filter(ImageFilter.BoxBlur(2))
    abajo = foto.crop((x0, y1, x1, y1 + 1)).filter(ImageFilter.BoxBlur(2))
    alto = y1 - y0
    relleno = Image.new("RGB", (x1 - x0, alto))
    for k in range(alto):
        relleno.paste(Image.blend(arriba, abajo, k / max(1, alto - 1)), (0, k))
    ruido = Image.effect_noise(relleno.size, r.uniform(4, 12)).convert("RGB")
    relleno = Image.blend(relleno, ruido, 0.08).filter(ImageFilter.GaussianBlur(1))
    foto.paste(relleno, (x0, y0))


def fotos_de_portones(partes):
    """(ruta, [cajas x0 y0 x1 y1]) de las fotos del dataset de portones."""
    fotos = []
    for parte in partes:
        with open(os.path.join(DATOS, parte, "_annotations.coco.json"), encoding="utf-8") as f:
            coco = json.load(f)
        cajas = {}
        for a in coco["annotations"]:
            x, y, w, h = a["bbox"]
            cajas.setdefault(a["image_id"], []).append([x, y, x + w, y + h])
        fotos += [(os.path.join(DATOS, parte, i["file_name"]), cajas.get(i["id"], [])) for i in coco["images"]]
    return fotos


def fotos_de_open_images(carpeta, excluir=frozenset()):
    """(ruta, [cajas]) de una carpeta de Open Images (un .txt por foto en Label/)."""
    fotos = []
    for nombre in sorted(os.listdir(carpeta)):
        codigo = os.path.splitext(nombre)[0]
        etiqueta = os.path.join(carpeta, "Label", codigo + ".txt")
        if not nombre.lower().endswith(".jpg") or not os.path.exists(etiqueta) or codigo in excluir:
            continue
        with open(etiqueta, encoding="utf-8") as f:
            cajas = [[float(v) for v in linea.split()[-4:]] for linea in f if linea.strip()]
        fotos.append((os.path.join(carpeta, nombre), cajas))
    return fotos


def fotos_de_yolo(carpeta):
    """(ruta, [cajas]) de una carpeta en formato YOLO (images/ y labels/, valores de 0 a 1)."""
    fotos = []
    for nombre in sorted(os.listdir(os.path.join(carpeta, "images"))):
        etiqueta = os.path.join(carpeta, "labels", os.path.splitext(nombre)[0] + ".txt")
        if not nombre.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")) or not os.path.exists(etiqueta):
            continue
        ruta = os.path.join(carpeta, "images", nombre)
        with Image.open(ruta) as img:
            W, H = img.size
        cajas = []
        with open(etiqueta, encoding="utf-8") as f:
            for linea in f:
                partes = linea.split()
                if len(partes) >= 5 and partes[0] == "0":  # clase 0 = placa (otros datasets marcan también letras)
                    xc, yc, w, h = (float(v) for v in partes[1:5])
                    cajas.append([(xc - w / 2) * W, (yc - h / 2) * H, (xc + w / 2) * W, (yc + h / 2) * H])
        fotos.append((ruta, cajas))
    return fotos


def pegar(foto, caja, r, fuentes):
    """Tapa la placa de esa caja y pega encima una salvadoreña falsa del tamaño que tendría.
    Devuelve (caja nueva, texto, ¿moto?)."""
    x0, y0, x1, y1 = caja
    w, h = x1 - x0, y1 - y0
    proporcion = w / max(1, h)
    tipo, serie, moto = texto_al_azar(r, proporcion < 1.8)
    placa = dibujar_placa(tipo, serie, r, fuentes, moto)
    # La salvadoreña mide 30 cm de ancho: como las de EE. UU., más angosta que las europeas o
    # la vietnamita de carro (52 cm). Las de moto miden parecido en todos lados.
    if moto or proporcion < 2.6:
        ancho = round(w * r.uniform(0.85, 1.05))
    else:
        ancho = round(w * r.uniform(0.6, 0.8))
    alto = round(ancho * placa.height / placa.width)
    placa = placa.resize((max(1, ancho), max(1, alto)), Image.Resampling.LANCZOS)
    placa = ImageEnhance.Color(placa).enhance(r.uniform(0.5, 1.0))

    # Que combine con la foto: misma luz que la placa original y un poco de desenfoque.
    original = foto.crop((x0, y0, x1, y1))
    # La original se tapa pasando suavemente de lo que hay arriba a lo que hay abajo,
    # columna por columna. (Un rectángulo de un solo color le enseñaba al detector un
    # atajo: "placa = algo dentro de un rectángulo liso", y las reales no lo tienen.)
    tapar(foto, x0 - 3, y0 - 3, x1 + 3, y1 + 3, r)
    luz = ImageStat.Stat(original.convert("L")).mean[0] / max(1, ImageStat.Stat(placa.convert("L")).mean[0])
    placa = Image.eval(placa, lambda v, k=min(1.3, max(0.35, luz)): min(255, int(v * k)))
    placa = placa.filter(ImageFilter.GaussianBlur(r.uniform(0.3, 1.0)))

    # La mayoría se pega en perspectiva, como se vería en un carro en ángulo.
    placa = deformar(placa, r) if r.random() < 0.6 else placa.convert("RGBA")
    px, py = round(x0 + w / 2 - placa.width / 2), round(y0 + h / 2 - placa.height / 2)
    foto.paste(placa, (px, py), placa)
    a, b, c, d = placa.getchannel("A").getbbox()
    return [px + a, py + b, px + c, py + d], tipo + serie, moto


def deformar(placa, r, fuerza=0.35):
    """Pone la placa en perspectiva: un lado más lejos (más bajo), algo más angosta y girada.
    Devuelve una imagen con transparencia alrededor."""
    W, H = placa.size
    lejos = r.uniform(0, fuerza) * H
    izquierdo = r.random() < 0.5  # ¿cuál lado queda más lejos?
    angosta = 1 - r.uniform(0, fuerza * 0.6)
    esquinas = [(0, lejos if izquierdo else 0), (W * angosta, 0 if izquierdo else lejos),
                (W * angosta, H - (0 if izquierdo else lejos)), (0, H - (lejos if izquierdo else 0))]
    angulo = math.radians(r.uniform(-8, 8))
    cx, cy = W * angosta / 2, H / 2
    esquinas = [(cx + (x - cx) * math.cos(angulo) - (y - cy) * math.sin(angulo) + r.uniform(-0.03, 0.03) * W,
                 cy + (x - cx) * math.sin(angulo) + (y - cy) * math.cos(angulo) + r.uniform(-0.04, 0.04) * H)
                for x, y in esquinas]
    x_min, y_min = min(x for x, _ in esquinas), min(y for _, y in esquinas)
    esquinas = [(x - x_min, y - y_min) for x, y in esquinas]
    tamano = (max(1, math.ceil(max(x for x, _ in esquinas))), max(1, math.ceil(max(y for _, y in esquinas))))
    # PERSPECTIVE necesita la cuenta al revés: para cada punto del resultado, de dónde sale en la placa.
    filas, valores = [], []
    for (x, y), (u, v) in zip(esquinas, [(0, 0), (W, 0), (W, H), (0, H)]):
        filas += [[x, y, 1, 0, 0, 0, -u * x, -u * y], [0, 0, 0, x, y, 1, -v * x, -v * y]]
        valores += [u, v]
    coeficientes = numpy.linalg.solve(numpy.array(filas, float), numpy.array(valores, float))
    return placa.convert("RGBA").transform(tamano, Image.Transform.PERSPECTIVE, coeficientes.tolist(),
                                           resample=Image.Resampling.BICUBIC)


def elegible(caja, minimo_carro=50, minimo_moto=40):
    w, h = caja[2] - caja[0], caja[3] - caja[1]
    return w >= (minimo_moto if w / max(1, h) < 1.8 else minimo_carro)


def componer(fotos, salida, fuentes, semilla, una_placa=True):
    """Pega placas salvadoreñas falsas sobre las placas de esas fotos.
    Con una_placa=True (exámenes) solo usa fotos con una sola placa, y la reemplaza. Si no,
    reemplaza todas las placas grandes de cada foto y deja las chicas como estaban."""
    os.makedirs(salida, exist_ok=True)
    r = random.Random(semilla)
    respuestas = []
    for ruta, cajas in fotos:
        if una_placa and (len(cajas) != 1 or not elegible(cajas[0])):
            continue
        if not una_placa and not any(elegible(c, 30, 30) for c in cajas):
            continue
        foto = Image.open(ruta).convert("RGB")
        placas = []
        for caja in cajas:
            if una_placa or elegible(caja, 30, 30):
                nueva, texto, moto = pegar(foto, caja, r, fuentes)
                placas.append({"texto": texto, "caja": nueva, "vehiculo": "moto" if moto else "carro"})
            else:
                placas.append({"texto": None, "caja": [round(v) for v in caja], "vehiculo": None})
        nombre = f"{len(respuestas):05}.jpg"
        foto.save(os.path.join(salida, nombre), quality=r.randint(60, 90))
        fila = {"foto": nombre, "ancho": foto.width, "alto": foto.height, "placas": placas}
        fila.update(placas[0])  # los exámenes tienen una sola placa: texto, caja y vehiculo arriba
        respuestas.append(fila)

    with open(os.path.join(salida, "respuestas.json"), "w", encoding="utf-8") as f:
        json.dump(respuestas, f, indent=1)
    return respuestas


def guardar_coco(respuestas, salida):
    """Las mismas respuestas en formato COCO, como el dataset original, para entrenar el detector."""
    coco = {"images": [], "annotations": []}
    for i, r in enumerate(respuestas):
        coco["images"].append({"id": i, "file_name": r["foto"], "width": r["ancho"], "height": r["alto"]})
        for p in r["placas"]:
            x0, y0, x1, y1 = p["caja"]
            coco["annotations"].append({"id": len(coco["annotations"]), "image_id": i, "category_id": 0,
                                        "bbox": [x0, y0, x1 - x0, y1 - y0]})
    with open(os.path.join(salida, "_annotations.coco.json"), "w", encoding="utf-8") as f:
        json.dump(coco, f)


def ids_de_examen():
    """Códigos de las fotos de examen: el dataset de portones trae algunas de Open Images."""
    ids = {os.path.basename(ruta).split("_jpg")[0] for ruta, _ in fotos_de_portones(("valid", "test"))}
    ids |= {os.path.splitext(n)[0] for n in os.listdir(os.path.join(OI, "validation", "Vehicle registration plate"))}
    return ids


def main():
    if "--entreno" in sys.argv:
        # Fotos de entrenamiento con las fuentes de entreno: para que el detector conozca las
        # placas salvadoreñas.
        salida = os.path.join(CARPETA, "datos", "fotos_sv_entreno")
        respuestas = componer(fotos_de_portones(("train",)), salida, FUENTES_ENTRENO, 30_000_000)
        guardar_coco(respuestas, salida)
        print(f"{len(respuestas)} fotos de portón para entrenar el detector en {salida}")
    elif "--oi" in sys.argv:
        salida = os.path.join(CARPETA, "datos", "examen_fotos_oi")
        fotos = fotos_de_open_images(os.path.join(OI, "validation", "Vehicle registration plate"))
        respuestas = componer(fotos, salida, FUENTES_EXAMEN, 40_000_000)
        print(f"{len(respuestas)} fotos de examen de escenas variadas en {salida}")
    elif "--oi-entreno" in sys.argv:
        salida = os.path.join(CARPETA, "datos", "fotos_sv_oi_entreno")
        fotos = fotos_de_open_images(os.path.join(OI, "train", "Vehicle registration plate"), ids_de_examen())
        respuestas = componer(fotos, salida, FUENTES_ENTRENO, 50_000_000, una_placa=False)
        guardar_coco(respuestas, salida)
        print(f"{len(respuestas)} fotos de escenas variadas para entrenar el detector en {salida}")
    elif "--autos" in sys.argv:
        # Fotos de carros como las de un anuncio o un teléfono: de frente, atrás, de lado, de cerca.
        salida = os.path.join(CARPETA, "datos", "examen_fotos_autos")
        respuestas = componer(fotos_de_yolo(os.path.join(AUTOS, "test")), salida, FUENTES_EXAMEN, 60_000_000)
        print(f"{len(respuestas)} fotos de examen de carros (tipo teléfono) en {salida}")
    elif "--autos-entreno" in sys.argv:
        salida = os.path.join(CARPETA, "datos", "fotos_sv_autos_entreno")
        respuestas = componer(fotos_de_yolo(os.path.join(AUTOS, "val")), salida, FUENTES_ENTRENO, 70_000_000, una_placa=False)
        guardar_coco(respuestas, salida)
        print(f"{len(respuestas)} fotos de carros (tipo teléfono) para entrenar el detector en {salida}")
    else:
        fotos = fotos_de_portones(("test", "valid"))  # el detector nunca las usó
        respuestas = componer(fotos, SALIDA, FUENTES_EXAMEN, 20_000_000)
        print(f"{len(respuestas)} fotos de examen en {SALIDA}")


if __name__ == "__main__":
    main()
