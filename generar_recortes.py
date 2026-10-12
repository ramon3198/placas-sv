r"""
Recortes de placas salvadoreñas dentro de fotos reales, para entrenar el lector.

Sale de las fotos compuestas de entrenamiento (crear_examen_fotos.py --entreno, --oi-entreno y
--autos-entreno): placas en perspectiva, con la luz, el fondo y la compresión de fotos de verdad.
Cada placa se recorta dos veces con márgenes al azar, como los recortes que salen del detector.
Usan las fuentes de entreno; los exámenes usan otra fuente, así que no se mezclan.

Deja datos\recortes_sv\ (etiquetas.tsv + cache.pt, el mismo formato de datos\placas\).
"""
import io
import json
import os
import random
import sys
from multiprocessing import Pool

CARPETA = os.path.dirname(os.path.abspath(__file__))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from PIL import Image

ALTO, ANCHO = 64, 128
FUENTES = ("fotos_sv_entreno", "fotos_sv_oi_entreno", "fotos_sv_autos_entreno")
SALIDA = os.path.join(CARPETA, "datos", "recortes_sv")


def recortar(trabajo):
    carpeta, fila, semilla = trabajo
    r = random.Random(semilla)
    foto = Image.open(os.path.join(CARPETA, "datos", carpeta, fila["foto"])).convert("RGB")
    return recortar_foto(foto, [p for p in fila["placas"] if p["texto"]], r)


def recortar_nuevas(trabajo):
    """Pega placas nuevas en una foto original (en memoria, sin guardarla) y las recorta."""
    sys.path.insert(0, CARPETA)
    from crear_examen_fotos import elegible, pegar
    from generar_placas import FUENTES_ENTRENO

    ruta, cajas, semilla = trabajo
    r = random.Random(semilla)
    foto = Image.open(ruta).convert("RGB")
    placas = []
    for caja in cajas:
        if elegible(caja, 30, 30):
            nueva, texto, _ = pegar(foto, caja, r, FUENTES_ENTRENO)
            placas.append({"texto": texto, "caja": nueva})
    return recortar_foto(foto, placas, r)


def recortar_foto(foto, placas, r):
    salida = []
    for placa in placas:
        x0, y0, x1, y1 = placa["caja"]
        w, h = x1 - x0, y1 - y0
        if w < 12:
            continue
        for _ in range(2):
            mx, my = r.uniform(0.0, 0.12) * w, r.uniform(0.0, 0.18) * h  # nunca más apretado que la placa
            caja = (x0 - mx - r.uniform(0, 0.04) * w, y0 - my - r.uniform(0, 0.06) * h,
                    x1 + mx + r.uniform(0, 0.04) * w, y1 + my + r.uniform(0, 0.06) * h)
            gris = foto.crop(caja).convert("L").resize((ANCHO, ALTO), Image.Resampling.BILINEAR)
            jpg = io.BytesIO()
            gris.save(jpg, "JPEG", quality=r.randint(50, 95))
            salida.append((placa["texto"], Image.open(jpg).convert("L").tobytes()))
    return salida


def preparar_reales(por_imagen=6):
    """Placas reales revisadas a mano (revisar.cmd) → datos de entrenamiento y examen del lector.
    El 15 % de los vehículos se aparta como examen: ninguna de sus fotos se usa para entrenar."""
    import hashlib

    import torch

    sys.path.insert(0, CARPETA)
    from juntar import PENDIENTES, REALES, REVISADAS

    if not os.path.exists(REVISADAS):
        print("Todavía no hay placas reales revisadas (revisar.cmd).")
        return
    with open(REVISADAS, encoding="utf-8") as f:
        ultimas = {}
        for linea in f:
            if linea.strip():
                fila = json.loads(linea)
                ultimas[fila["vehiculo"]] = fila  # si se decidió dos veces, vale la última
    partes = {"entreno": [], "examen": []}
    vehiculos = {"entreno": 0, "examen": 0}
    for vehiculo, fila in ultimas.items():
        if fila["decision"] == "descartada":
            continue
        parte = "examen" if int(hashlib.md5(vehiculo.encode()).hexdigest(), 16) % 100 < 15 else "entreno"
        vehiculos[parte] += 1
        r = random.Random(vehiculo)
        for recorte in fila["recortes"]:
            ruta = os.path.join(PENDIENTES, recorte["archivo"])
            if not os.path.exists(ruta):
                continue
            foto = Image.open(ruta).convert("RGB")
            x0, y0, x1, y1 = recorte["caja"]
            w, h = x1 - x0, y1 - y0
            # Para el examen, un solo recorte con el margen que usa el sistema; para entrenar, varios al azar.
            margenes = [(0.05, 0.08)] if parte == "examen" else [(r.uniform(0, 0.12), r.uniform(0, 0.18)) for _ in range(por_imagen)]
            for mx, my in margenes:
                gris = foto.crop((x0 - mx * w, y0 - my * h, x1 + mx * w, y1 + my * h)).convert("L")
                gris = gris.resize((ANCHO, ALTO), Image.Resampling.BILINEAR)
                partes[parte].append((fila["texto"], gris.tobytes()))
    for parte, filas in partes.items():
        if filas:
            x = torch.frombuffer(bytearray(b"".join(p for _, p in filas)), dtype=torch.uint8).view(-1, 1, ALTO, ANCHO)
            torch.save({"x": x, "textos": [t for t, _ in filas]}, os.path.join(REALES, f"{parte}.pt"))
        print(f"  {parte}: {vehiculos[parte]} vehículos, {len(filas)} recortes")


def main():
    if "--reales" in sys.argv:
        preparar_reales()
        return
    trabajos = []
    for carpeta in FUENTES:
        ruta = os.path.join(CARPETA, "datos", carpeta, "respuestas.json")
        if not os.path.exists(ruta):
            print(f"  (no está {ruta}, se omite)")
            continue
        with open(ruta, encoding="utf-8") as f:
            trabajos += [(carpeta, fila, 80_000_000 + len(trabajos) + k) for k, fila in enumerate(json.load(f))]
    with Pool(max(1, os.cpu_count() - 2)) as pool:
        filas = [x for lote in pool.map(recortar, trabajos, chunksize=50) for x in lote]
        # Con --rondas N, además se pegan placas nuevas N veces sobre las fotos originales de
        # entrenamiento: más textos distintos en contextos reales, sin guardar fotos nuevas.
        rondas = int(sys.argv[sys.argv.index("--rondas") + 1]) if "--rondas" in sys.argv else 0
        if rondas:
            sys.path.insert(0, CARPETA)
            from crear_examen_fotos import AUTOS, OI, fotos_de_open_images, fotos_de_portones, fotos_de_yolo, ids_de_examen

            originales = fotos_de_portones(("train",))
            originales += fotos_de_open_images(os.path.join(OI, "train", "Vehicle registration plate"), ids_de_examen())
            originales += fotos_de_yolo(os.path.join(AUTOS, "val"))
            for ronda in range(rondas):
                nuevos = [(ruta, cajas, 90_000_000 + ronda * 100_000 + k) for k, (ruta, cajas) in enumerate(originales)]
                filas += [x for lote in pool.map(recortar_nuevas, nuevos, chunksize=50) for x in lote]
                print(f"  ronda {ronda + 1}: {len(filas):,} recortes en total")

    import torch

    os.makedirs(SALIDA, exist_ok=True)
    with open(os.path.join(SALIDA, "etiquetas.tsv"), "w", encoding="utf-8") as f:
        f.writelines(f"{k:08}.jpg\t{texto}\n" for k, (texto, _) in enumerate(filas))
    x = torch.frombuffer(bytearray(b"".join(p for _, p in filas)), dtype=torch.uint8).view(-1, 1, ALTO, ANCHO)
    torch.save({"x": x, "textos": [t for t, _ in filas]}, os.path.join(SALIDA, "cache.pt"))
    # Una hoja con algunos recortes, para ver cómo son.
    hoja = Image.new("L", (ANCHO * 8, (ALTO + 2) * 6), 30)
    for k in range(48):
        i = k * len(filas) // 48
        hoja.paste(Image.frombytes("L", (ANCHO, ALTO), filas[i][1]), ((k % 8) * ANCHO, (k // 8) * (ALTO + 2)))
    hoja.save(os.path.join(CARPETA, "datos", "muestra_recortes.jpg"))
    print(f"{len(filas):,} recortes de {len(trabajos):,} fotos en {SALIDA}")


if __name__ == "__main__":
    main()
