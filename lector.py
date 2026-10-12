"""
El lector de placas: recibe el recorte de una placa y devuelve el texto.

Usa la técnica CRNN + CTC, la clásica para leer texto en imágenes:
  1. Una red convolucional (CNN) convierte la placa en 32 "rebanadas" verticales,
     cada una descrita por una lista de números.
  2. Una red recurrente (GRU) lee esas rebanadas de izquierda a derecha y de derecha a
     izquierda, para que cada una sepa qué hay a sus lados.
  3. Para cada rebanada dice qué carácter ve, o "nada" (el espacio entre letras).
  4. CTC junta el resultado: "PP·11··22·3" → "P123". Así no hay que marcar dónde está
     cada letra; para entrenar basta con saber el texto completo de la placa.
"""
import os

import torch
import torch.nn as nn
from PIL import Image

CARPETA = os.path.dirname(os.path.abspath(__file__))
MODELO = os.path.join(CARPETA, "modelos", "lector.safetensors")
ALTO, ANCHO = 64, 128

# Solo lo que puede aparecer en una placa salvadoreña: números, A-F (series nuevas) y las
# letras de los tipos de vehículo. Así el lector nunca puede confundir un 0 con una O.
# La N va al final para que los lectores ya entrenados sigan sirviendo de punto de partida.
SIMBOLOS = "0123456789ABCDEFIMPV" + "N"
TIPOS = ("AB", "CC", "CD", "MI", "P", "A", "C", "M", "V", "N")  # los de dos letras van primero
SERIE = set("0123456789ABCDEF")


class Lector(nn.Module):
    def __init__(self, ancho=1.0):
        """ancho multiplica la cantidad de filtros: 1.0 es el lector normal, 1.5 uno más grande."""
        super().__init__()
        self.ancho = ancho
        c = [round(n * ancho) for n in (32, 64, 128, 256)]

        def capa(entra, sale):
            return [nn.Conv2d(entra, sale, 3, padding=1, bias=False), nn.BatchNorm2d(sale), nn.ReLU(inplace=True)]

        self.cnn = nn.Sequential(
            *capa(1, c[0]), nn.MaxPool2d(2),  # 32 × 32 × 64 (con ancho 1)
            *capa(c[0], c[1]), nn.MaxPool2d(2),  # 64 × 16 × 32
            *capa(c[1], c[2]), *capa(c[2], c[2]), nn.MaxPool2d((2, 1)),  # 128 × 8 × 32
            *capa(c[2], c[3]), *capa(c[3], c[3]), nn.MaxPool2d((2, 1)),  # 256 × 4 × 32
        )
        oculto = round(128 * ancho)
        self.rnn = nn.GRU(c[3] * 4, oculto, num_layers=2, bidirectional=True, batch_first=True, dropout=0.2)
        self.salida = nn.Linear(oculto * 2, len(SIMBOLOS) + 1)  # +1 por "nada", que es el índice 0

    def forward(self, x):  # x: (lote, 1, 64, 128), valores entre -1 y 1
        f = self.cnn(x)  # (lote, 256, 4, 32)
        f = f.permute(0, 3, 1, 2).flatten(2)  # 32 rebanadas de 1024 números
        f, _ = self.rnn(f)
        return self.salida(f)  # (lote, 32 rebanadas, 21 opciones)


def a_tensor(imagen):
    """Una imagen PIL (de cualquier tamaño) → lo que recibe el lector."""
    gris = imagen.convert("L").resize((ANCHO, ALTO), Image.Resampling.BILINEAR)
    x = torch.frombuffer(bytearray(gris.tobytes()), dtype=torch.uint8).view(1, ALTO, ANCHO)
    return x.float() / 127.5 - 1


def decodificar(salida):
    """De las 32 rebanadas al texto: junta repetidos y quita los "nada" (CTC voraz).
    Devuelve el texto y la seguridad de la letra más dudosa."""
    prob = salida.softmax(-1)
    mejor, idx = prob.max(-1)
    textos = []
    for p, ids in zip(mejor.tolist(), idx.tolist()):
        texto, seguridades, anterior = "", [], 0
        for pi, i in zip(p, ids):
            if i != 0 and i != anterior:
                texto += SIMBOLOS[i - 1]
                seguridades.append(pi)
            elif i != 0 and seguridades:
                seguridades[-1] = max(seguridades[-1], pi)
            anterior = i
        textos.append((texto, min(seguridades) if seguridades else 0.0))
    return textos


def interpretar(texto):
    """¿Tiene el formato de una placa salvadoreña? Devuelve (tipo, serie) o None.
    La serie casi siempre tiene 6 caracteres, pero hay más cortas: "P1CAB", "P1E12"."""
    for tipo in TIPOS:
        serie = texto[len(tipo):]
        if texto.startswith(tipo) and 1 <= len(serie) <= 6 and set(serie) <= SERIE:
            return tipo, serie
    return None


def bonito(texto):
    """P123456 → P123-456, como se ve en la placa: el escudo separa los últimos 3."""
    if not interpretar(texto) or len(texto) < 5:
        return texto
    return f"{texto[:-3]}-{texto[-3:]}"


def guardar_ficha(ruta, ancho):
    import json

    with open(ruta.replace(".safetensors", ".json"), "w", encoding="utf-8") as f:
        json.dump({"ancho": ancho}, f)


def cargar_entrenado(dispositivo, ruta=None):
    import json

    from safetensors.torch import load_file

    ruta = ruta or MODELO
    ficha = ruta.replace(".safetensors", ".json")
    ancho = 1.0
    if os.path.exists(ficha):  # los primeros lectores no tenían ficha: eran de ancho 1
        with open(ficha, encoding="utf-8") as f:
            ancho = json.load(f)["ancho"]
    modelo = Lector(ancho)
    pesos = load_file(ruta)
    faltan = len(SIMBOLOS) + 1 - pesos["salida.bias"].shape[0]
    if faltan > 0:
        # Un lector de antes de agregar letras (como la N): las salidas nuevas quedan apagadas,
        # así sigue leyendo igual que siempre, solo que nunca devuelve esas letras.
        pesos["salida.weight"] = torch.cat([pesos["salida.weight"], torch.zeros(faltan, pesos["salida.weight"].shape[1])])
        pesos["salida.bias"] = torch.cat([pesos["salida.bias"], torch.full((faltan,), -1e4)])
    modelo.load_state_dict(pesos)
    return modelo.to(dispositivo).eval()
