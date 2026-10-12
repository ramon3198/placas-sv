r"""
Fabrica placas salvadoreñas falsas para entrenar el lector: de carro y de moto, del diseño
2011 (el de la bandera), del 2000 (blanco con el mapa) y provisionales de cartón.

Cada placa lleva un texto al azar con el formato real: la letra del tipo de vehículo
(P, A, C, M, AB…) y hasta 6 caracteres, que en los registros nuevos usan solo 0-9 y A-F.
Después se "ensucia" a propósito (inclinación, desenfoque, poca luz, ruido, baja
resolución) para que se parezca a un recorte de cámara de verdad.

  generar_placas.cmd           → 80.000 placas en datos\placas\
  generar_placas.cmd 5000      → otra cantidad

También deja datos\muestra_placas.jpg para ver cómo quedaron.
"""
import io
import math
import os
import random
import sys
from multiprocessing import Pool

from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont

CARPETA = os.path.dirname(os.path.abspath(__file__))
SALIDA = os.path.join(CARPETA, "datos", "placas")
EXAMEN = os.path.join(CARPETA, "datos", "examen_lector")
ALTO, ANCHO = 64, 128  # tamaño con el que el lector ve cada placa

# P (particular) es por mucho la más común en la calle, pero las otras letras (M, V, I…) solo
# aparecen como tipo: si salen poco, el lector no las aprende bien y confunde M con V.
# La M (moto) sale con las placas de moto, que son un tercio del total.
TIPOS = ["P"] * 8 + ["A", "C", "AB", "CC", "CD", "MI", "V", "MI", "V", "N", "N"]
HEX = "0123456789ABCDEF"
NUMEROS = "0123456789"
# La fuente real de las placas no está en Windows: se entrena con varias parecidas para que
# el lector aprenda la forma de cada carácter y no los detalles de una sola fuente.
FUENTES_ENTRENO = [("bahnschrift.ttf", "Bold Condensed")] * 4 + [
    ("bahnschrift.ttf", "SemiBold Condensed"),
    ("bahnschrift.ttf", "Bold SemiCondensed"),
    ("impact.ttf", None),
    ("tahomabd.ttf", None),
    ("segoeuib.ttf", None),
    ("verdanab.ttf", None),
    ("trebucbd.ttf", None),
    ("ebrimabd.ttf", None),
]
# El examen usa una fuente que el lector nunca vio, para medir cómo le iría con la real.
FUENTES_EXAMEN = [("ARIALNB.TTF", None)]
_fuentes = {}


def fuente(tamano, estilo="Bold Condensed", archivo="bahnschrift.ttf"):
    if (archivo, tamano, estilo) not in _fuentes:
        f = ImageFont.truetype(archivo, tamano)
        if estilo:
            f.set_variation_by_name(estilo)
        _fuentes[archivo, tamano, estilo] = f
    return _fuentes[archivo, tamano, estilo]


def pintar(placa, texto, x, y, f, compresion, tinta, relieve=None):
    """Pinta el texto con su centro vertical en y, angostándolo si la fuente es muy ancha.
    Con relieve=(dx, dy, color) dibuja antes un borde corrido, como las letras repujadas."""
    x0, y0, x1, y1 = f.getbbox(texto)
    mascara = Image.new("L", (x1 - x0 + 4, y1 - y0 + 4), 0)
    ImageDraw.Draw(mascara).text((2 - x0, 2 - y0), texto, font=f, fill=255)
    mascara = mascara.resize((max(1, round(mascara.width * compresion)), mascara.height))
    esquina = (round(x), round(y - mascara.height / 2))
    if relieve:
        dx, dy, color = relieve
        placa.paste(color, (esquina[0] + dx, esquina[1] + dy), mascara)
    placa.paste(tinta, esquina, mascara)
    return mascara.width


def texto_al_azar(r, moto=None):
    """Devuelve (tipo, serie, moto). Un tercio son motos, como en la calle (704 mil de 2 millones)."""
    if moto is None:
        moto = r.random() < 0.35
    tipo = "M" if moto else r.choice(TIPOS)
    simbolos = HEX if r.random() < 0.5 else NUMEROS  # placas viejas: solo números
    # En la calle se ven sobre todo 6 caracteres ("P813-472") y 5 ("P 35-792", "P 61-AE2"),
    # y algunas series más cortas: "P 1 CAB", "N 12-345".
    largo = r.choices([6, 5, 4, 3, 2, 1], weights=[48, 34, 11, 4, 2, 1])[0]
    serie = "".join(r.choice(simbolos) for _ in range(largo))
    if largo == 4 and r.random() < 0.6:
        serie = "1" + "".join(r.choice(HEX) for _ in range(3))
    return tipo, serie, moto


def elegir_fuente(r, fuentes, alto_px):
    """Una de las fuentes, del tamaño justo para que las letras midan alto_px."""
    archivo, estilo = r.choice(fuentes)
    f = fuente(100, estilo, archivo)
    caja = f.getbbox("P08")
    return fuente(max(8, round(100 * alto_px / (caja[3] - caja[1]))), estilo, archivo)


def relieve_al_azar(r):
    if r.random() < 0.6:  # las placas reales son repujadas: la luz deja un borde claro u oscuro
        gris = r.choice([r.randint(90, 150), r.randint(200, 255)])
        return r.randint(-4, 4), r.randint(-4, 4), (gris, gris, gris)
    return None


def caracteres(placa, tipo, serie, r, fuentes, centro_y, alto, margen, adorno=None):
    """Pinta tipo + serie en una línea. Con adorno=(ancho, dibujar), deja ese hueco antes de los
    últimos 3 caracteres y dibuja ahí el escudo o el monumento: «P123 ⊕ 456», «P  1 ⊕ CAB»."""
    W = placa.width
    f = elegir_fuente(r, fuentes, alto)
    tinta = tuple([r.randint(0, 35)] * 3)
    relieve = relieve_al_azar(r)
    if adorno and len(serie) > 3:
        izq, der = tipo + ("  " if len(serie) < 6 else "") + serie[:-3], serie[-3:]
    else:
        izq, der = tipo + serie, ""
    hueco = adorno[0] if adorno and der else 0
    natural = f.getlength(izq) + (f.getlength(der) if der else 0)
    compresion = min(1.0, (W - 2 * margen - hueco) / natural) * r.uniform(0.9, 1.0)
    total = natural * compresion + hueco
    x0, y = (W - total) / 2 + r.uniform(-6, 6), centro_y + r.uniform(-5, 5)
    ancho_izq = pintar(placa, izq, x0, y, f, compresion, tinta, relieve)
    if der:
        adorno[1](ImageDraw.Draw(placa), x0 + ancho_izq + hueco / 2, y)
        pintar(placa, der, x0 + ancho_izq + hueco, y, f, compresion, tinta, relieve)


def escudo(marino, escala=1.0):
    """El escudo de la bandera, simplificado."""
    def dibujar(d, cx, y):
        a, b = 22 * escala, 28 * escala
        d.ellipse((cx - a, y - b, cx + a, y + b), fill=(240, 240, 235), outline=marino, width=max(1, round(2 * escala)))
        d.polygon([(cx - 15 * escala, y + 10 * escala), (cx + 15 * escala, y + 10 * escala), (cx, y - 16 * escala)], fill=(40, 140, 70))
        d.line((cx - 14 * escala, y + 17 * escala, cx + 14 * escala, y + 17 * escala), fill=(200, 170, 40), width=max(1, round(3 * escala)))
    return dibujar


def monumento(azul, escala=1.0):
    """El Salvador del Mundo de las placas 2000: una figura sobre un globo."""
    def dibujar(d, cx, y):
        e = escala
        d.rectangle((cx - 6 * e, y - 55 * e, cx + 6 * e, y + 50 * e), fill=azul)
        d.ellipse((cx - 17 * e, y + 5 * e, cx + 17 * e, y + 39 * e), fill=azul)
        d.ellipse((cx - 8 * e, y - 72 * e, cx + 8 * e, y - 56 * e), fill=azul)
        d.polygon([(cx - 16 * e, y - 40 * e), (cx + 16 * e, y - 40 * e), (cx, y - 28 * e)], fill=azul)
    return dibujar


def bandera(d, W, H, r, arriba, abajo):
    """Franjas azules onduladas arriba y abajo, blanco en medio."""
    azul = (r.randint(20, 60), r.randint(90, 130), r.randint(190, 230))
    fase, onda = r.uniform(0, 6.3), r.uniform(0.03, 0.06) * H
    linea_arriba = [(x, arriba + onda * math.sin(x / W * 7 + fase)) for x in range(0, W + 1, 20)]
    linea_abajo = [(x, abajo + onda * math.sin(x / W * 7 + fase + 1)) for x in range(0, W + 1, 20)]
    d.polygon(linea_arriba + [(W, 0), (0, 0)], fill=azul)
    d.polygon(linea_abajo + [(W, H), (0, H)], fill=azul)


def placa_2011_carro(tipo, serie, r, fuentes):
    """La de carro actual, 600×300 (12×6 pulgadas)."""
    W, H = 600, 300
    fondo = tuple(r.randint(232, 250) for _ in range(3))
    if tipo == "N" and r.random() < 0.7:  # la N suele tener el cuerpo azulado
        fondo = (r.randint(140, 190), r.randint(170, 210), r.randint(220, 250))
    placa = Image.new("RGB", (W, H), fondo)
    d = ImageDraw.Draw(placa)
    bandera(d, W, H, r, 92, 215)
    marino = (r.randint(5, 25), r.randint(25, 45), r.randint(80, 110))
    d.text((W / 2 - 20, 38), "EL SALVADOR", font=fuente(52, "Bold"), fill=marino, anchor="mm")
    d.text((W - 60, 32), "2011", font=fuente(36, "SemiBold"), fill=marino, anchor="mm")
    d.text((W / 2, 268), "CENTRO AMERICA", font=fuente(40, "Bold"), fill=(235, 240, 250), anchor="mm")
    for x in (175, 425):  # los agujeros de los tornillos
        d.ellipse((x - 9, 52, x + 9, 70), fill=(90, 90, 90))
    caracteres(placa, tipo, serie, r, fuentes, 160, r.uniform(135, 150), 15, (64, escudo(marino)))
    return placa


def placa_2011_moto(tipo, serie, r, fuentes):
    """La de moto actual: más chica y angosta, con los caracteres casi del alto de la placa."""
    W, H = 420, 240
    placa = Image.new("RGB", (W, H), tuple(r.randint(232, 250) for _ in range(3)))
    d = ImageDraw.Draw(placa)
    bandera(d, W, H, r, 58, 196)
    marino = (r.randint(5, 25), r.randint(25, 45), r.randint(80, 110))
    d.text((W / 2, 26), "EL SALVADOR", font=fuente(40, "Bold"), fill=marino, anchor="mm")
    for x in (70, 350):
        d.ellipse((x - 8, 44, x + 8, 60), fill=(90, 90, 90))
    caracteres(placa, tipo, serie, r, fuentes, 130, r.uniform(118, 138), 10)
    escudo(marino, 0.6)(d, 34, 215)
    d.text((W / 2 + 10, 222), "CENTRO AMERICA", font=fuente(24, "Bold"), fill=(235, 240, 250), anchor="mm")
    return placa


def placa_2000(tipo, serie, r, fuentes, moto):
    """Las del diseño anterior: blancas, con el mapa del país en celeste y el monumento."""
    W, H = (420, 240) if moto else (600, 300)
    placa = Image.new("RGB", (W, H), tuple(r.randint(232, 250) for _ in range(3)))
    d = ImageDraw.Draw(placa)
    celeste = (r.randint(120, 170), r.randint(170, 205), r.randint(225, 250))
    puntos = []
    for k in range(16):  # una mancha alargada, como la silueta del mapa
        a = k / 16 * 2 * math.pi
        radio = r.uniform(0.75, 1.05)
        puntos.append((W / 2 + math.cos(a) * 0.42 * W * radio, H * 0.55 + math.sin(a) * 0.3 * H * radio))
    d.polygon(puntos, fill=celeste)
    azul = (r.randint(20, 60), r.randint(70, 110), r.randint(170, 220))
    if moto:
        d.text((W / 2, 26), "EL SALVADOR", font=fuente(42, "Bold"), fill=(15, 15, 15), anchor="mm")
        caracteres(placa, tipo, serie, r, fuentes, 112, r.uniform(95, 112), 12, (40, monumento(azul, 0.65)))
        d.text((W / 2 + 20, 212), "2000", font=fuente(44, "Bold"), fill=(15, 15, 15), anchor="mm")
        escudo((20, 40, 100), 0.55)(d, W / 2 - 70, 212)
    else:
        d.text((W / 2 - 40, 36), "EL SALVADOR", font=fuente(60, "Bold Condensed"), fill=(15, 15, 15), anchor="mm")
        d.text((W - 70, 30), "2000", font=fuente(36, "SemiBold"), fill=(15, 15, 15), anchor="mm")
        d.text((W / 2, 272), "CENTRO AMERICA", font=fuente(38, "Bold"), fill=(15, 15, 15), anchor="mm")
        caracteres(placa, tipo, serie, r, fuentes, 160, r.uniform(140, 160), 15, (56, monumento(azul)))
    return placa


def placa_provisional(tipo, serie, r, fuentes, moto):
    """Las provisionales de cartón, mientras sale la definitiva."""
    W, H = (420, 240) if moto else (600, 300)
    carton = r.randint(200, 238)
    placa = Image.new("RGB", (W, H), (carton, carton, carton - r.randint(0, 20)))
    d = ImageDraw.Draw(placa)
    d.rectangle((8, 8, W - 9, H - 9), outline=(20, 20, 20), width=5)
    escala = 0.75 if moto else 1.0
    d.text((W / 2, 40 * escala), "EL SALVADOR", font=fuente(round(46 * escala), "Bold"), fill=(15, 15, 15), anchor="mm")
    caracteres(placa, tipo, serie, r, fuentes, H * 0.5, r.uniform(0.38, 0.45) * H, 20)
    d.text((W / 2, H - 52 * escala), "PROVISIONAL", font=fuente(round(40 * escala), "Bold"), fill=(15, 15, 15), anchor="mm")
    d.text((W / 2, H - 22 * escala), "2011", font=fuente(round(26 * escala), "Bold"), fill=(15, 15, 15), anchor="mm")
    return placa


def dibujar_placa(tipo, serie, r, fuentes=FUENTES_ENTRENO, moto=False, diseno=None):
    """La placa limpia. Si no se pide un diseño, sale uno al azar: casi todas son del 2011."""
    if diseno is None:
        u = r.random()
        diseno = "2011" if u < 0.78 else "2000" if u < 0.88 else "provisional"
    if diseno == "2011":
        return placa_2011_moto(tipo, serie, r, fuentes) if moto else placa_2011_carro(tipo, serie, r, fuentes)
    if diseno == "2000":
        return placa_2000(tipo, serie, r, fuentes, moto)
    return placa_provisional(tipo, serie, r, fuentes, moto)


def ensuciar(placa, r):
    """Hace que la placa limpia parezca recortada de una cámara de parqueo."""
    W, H = placa.size
    M = 150
    fondo = tuple(r.randint(10, 230) for _ in range(3))  # el color del carro alrededor
    lienzo = Image.new("RGB", (W + 2 * M, H + 2 * M), fondo)
    lienzo.paste(placa, (M, M))
    for _ in range(r.randint(0, 4)):  # manchas de suciedad
        x, y = r.uniform(M, M + W), r.uniform(M, M + H)
        tam = r.uniform(8, 40)
        ImageDraw.Draw(lienzo).ellipse((x - tam, y - tam / 2, x + tam, y + tam / 2), fill=(r.randint(80, 150),) * 3)
    if r.random() < 0.35:  # marco de plástico de la agencia, que tapa parte de arriba y de abajo
        d = ImageDraw.Draw(lienzo)
        marco = tuple([r.randint(5, 60)] * 3)
        arriba, abajo, lado = r.randint(0, 50), r.randint(10, 70), r.randint(4, 22)
        d.rectangle((M - lado, M - lado, M + W + lado, M + arriba), fill=marco)
        d.rectangle((M - lado, M + H - abajo, M + W + lado, M + H + lado), fill=marco)
        d.rectangle((M - lado, M, M, M + H), fill=marco)
        d.rectangle((M + W, M, M + W + lado, M + H), fill=marco)
        if abajo > 35 and r.random() < 0.7:
            d.text((M + W / 2, M + H - abajo / 2), r.choice(["AUTOMOTRIZ", "MOTORS", "GRUPO", "AUTOS S.A.", "SERVICIO"]),
                   font=fuente(abajo - 12, "Bold"), fill=(220, 220, 220), anchor="mm")
    if r.random() < 0.2:  # reflejo de luz
        brillo = Image.new("L", lienzo.size, 0)
        x, y, tam = r.uniform(M, M + W), r.uniform(M, M + H), r.uniform(30, 120)
        ImageDraw.Draw(brillo).ellipse((x - tam, y - tam / 2, x + tam, y + tam / 2), fill=r.randint(120, 230))
        brillo = brillo.filter(ImageFilter.GaussianBlur(25))
        lienzo = Image.composite(Image.new("RGB", lienzo.size, (255, 255, 255)), lienzo, brillo)

    # El recorte del detector nunca es perfecto: queda flojo, apretado, inclinado o en perspectiva.
    mx, my = r.uniform(-0.02, 0.12) * W, r.uniform(-0.03, 0.18) * H
    lejos_izq = lejos_der = 0
    if r.random() < 0.35:
        # Vista de lado, como una cámara de parqueo en ángulo: el lado más lejano de la placa se
        # ve más bajo, y toda la placa más angosta.
        lejos = r.uniform(0.1, 0.45) * H
        lejos_izq, lejos_der = (lejos, 0) if r.random() < 0.5 else (0, lejos)
        mx += r.uniform(0.05, 0.25) * W
    esquinas = [(-mx, -my - lejos_izq), (-mx, H + my + lejos_izq), (W + mx, H + my + lejos_der), (W + mx, -my - lejos_der)]
    angulo = math.radians(r.uniform(-10, 10))
    quad = []
    for x, y in esquinas:
        x, y = x - W / 2, y - H / 2
        x, y = x * math.cos(angulo) - y * math.sin(angulo), x * math.sin(angulo) + y * math.cos(angulo)
        quad += [x + W / 2 + M + r.uniform(-0.05, 0.05) * W, y + H / 2 + M + r.uniform(-0.06, 0.06) * H]
    img = lienzo.transform((ANCHO * 2, ALTO * 2), Image.Transform.QUAD, quad, resample=Image.Resampling.BILINEAR)

    # Luz: más oscura o quemada, con poco contraste y a veces una sombra de lado a lado.
    img = ImageEnhance.Brightness(img).enhance(r.uniform(0.35, 1.35))
    img = ImageEnhance.Contrast(img).enhance(r.uniform(0.45, 1.3))
    if r.random() < 0.4:
        sombra = Image.linear_gradient("L").rotate(r.uniform(0, 360)).resize(img.size)
        img = Image.composite(img, ImageEnhance.Brightness(img).enhance(0.4), sombra)

    # Cámaras de noche ven en blanco y negro (infrarrojo): el azul puede verse claro u oscuro.
    rr, gg, bb = r.random(), r.random(), r.random()
    s = rr + gg + bb
    img = img.convert("L", (rr / s, gg / s, bb / s, 0))

    img = img.filter(ImageFilter.GaussianBlur(r.choice([0, 0, 0.4, 0.8, 1.2, 1.8])))
    if r.random() < 0.25:  # el carro se movía: desenfoque de lado a lado
        img = img.filter(ImageFilter.Kernel((5, 5), [0] * 10 + [1] * 5 + [0] * 10, scale=5))
    # Baja resolución: la placa lejos de la cámara ocupa pocos píxeles.
    # Casi la mitad muy chiquitas (26-70 px de ancho), que es donde más se equivoca.
    ancho_real = r.randint(26, 70) if r.random() < 0.45 else r.randint(70, 220)
    img = img.resize((ancho_real, max(8, ancho_real // 2)), Image.Resampling.BILINEAR)
    img = img.resize((ANCHO, ALTO), Image.Resampling.BILINEAR)
    ruido = Image.effect_noise((ANCHO, ALTO), r.uniform(5, 40))
    return Image.blend(img, ruido, r.uniform(0, 0.15))


def fabricar(trabajo):
    i, carpeta, fuentes = trabajo
    r = random.Random(i)
    tipo, serie, moto = texto_al_azar(r)
    texto = tipo + serie
    img = ensuciar(dibujar_placa(tipo, serie, r, fuentes, moto), r)
    nombre = f"{i:08}.jpg"
    jpg = io.BytesIO()
    img.save(jpg, "JPEG", quality=r.randint(30, 92))  # la compresión de la cámara también ensucia
    with open(os.path.join(carpeta, nombre), "wb") as f:
        f.write(jpg.getvalue())
    return nombre, texto, Image.open(jpg).convert("L").tobytes()


def fabricar_lote(carpeta, semillas, fuentes):
    os.makedirs(carpeta, exist_ok=True)
    with Pool(max(1, os.cpu_count() - 2)) as pool:
        filas = pool.map(fabricar, [(i, carpeta, fuentes) for i in semillas], chunksize=200)
    with open(os.path.join(carpeta, "etiquetas.tsv"), "w", encoding="utf-8") as f:
        f.writelines(f"{nombre}\t{texto}\n" for nombre, texto, _ in filas)
    # Además deja todas las placas juntas en un solo archivo, que el entrenamiento lee rápido.
    import torch

    x = torch.frombuffer(bytearray(b"".join(pixeles for _, _, pixeles in filas)), dtype=torch.uint8)
    torch.save({"x": x.view(-1, 1, ALTO, ANCHO), "textos": [t for _, t, _ in filas]}, os.path.join(carpeta, "cache.pt"))


def muestra():
    """Una hoja con placas limpias y su versión ensuciada, para ver qué aprende el lector."""
    r = random.Random(7)
    hoja = Image.new("RGB", (4 * 330, 6 * 175), (30, 30, 30))
    d = ImageDraw.Draw(hoja)
    combinaciones = [("2011", False), ("2011", True), ("2000", False), ("2000", True), ("provisional", False), ("provisional", True)]
    for k in range(12):
        diseno, moto = combinaciones[k % 6]
        tipo, serie, moto = texto_al_azar(r, moto)
        texto = tipo + serie
        limpia = dibujar_placa(tipo, serie, r, moto=moto, diseno=diseno)
        sucia = ensuciar(limpia, r).convert("RGB").resize((256, 128), Image.Resampling.NEAREST)
        x, y = (k % 2) * 660, (k // 2) * 175
        hoja.paste(limpia.resize((round(128 * limpia.width / limpia.height), 128)), (x + 20, y + 10))
        hoja.paste(sucia, (x + 350, y + 10))
        d.text((x + 20, y + 142), f"«{texto}» limpia", font=fuente(22, "Regular"), fill=(220, 220, 220))
        d.text((x + 350, y + 142), "como la ve el lector", font=fuente(22, "Regular"), fill=(220, 220, 220))
    ruta = os.path.join(CARPETA, "datos", "muestra_placas.jpg")
    hoja.save(ruta, quality=90)
    return ruta


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    cantidad = int(sys.argv[1]) if len(sys.argv) > 1 else 80000
    print(f"Muestra: {muestra()}")
    print(f"Fabricando {cantidad:,} placas falsas para entrenar…")
    fabricar_lote(SALIDA, range(cantidad), FUENTES_ENTRENO)
    print("Fabricando 2,000 placas de examen, con una fuente que el lector nunca ve al entrenar…")
    fabricar_lote(EXAMEN, range(10_000_000, 10_002_000), FUENTES_EXAMEN)
    print(f"Listo: {SALIDA}  y  {EXAMEN}")


if __name__ == "__main__":
    main()
