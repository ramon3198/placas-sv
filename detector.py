"""
El detector de placas: dada una foto, devuelve los rectángulos donde hay placas.

Es un Faster R-CNN de torchvision (licencia BSD) con MobileNet v3: liviano y rápido, sirve hasta
sin tarjeta de video. Su última capa decide entre dos clases: fondo o placa.
"""
import json
import os

from safetensors.torch import load_file
from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_fpn, fasterrcnn_resnet50_fpn_v2

CARPETA = os.path.dirname(os.path.abspath(__file__))
MODELO = os.path.join(CARPETA, "modelos", "detector.safetensors")
TAMANO = 640  # la foto se escala para que su lado corto mida esto
ARQUITECTURAS = {
    "mobilenet": fasterrcnn_mobilenet_v3_large_fpn,
    "resnet50": fasterrcnn_resnet50_fpn_v2,
}


def crear_modelo(arquitectura="mobilenet"):
    # Sin pesos de partida: todos los pesos salen del detector entrenado.
    return ARQUITECTURAS[arquitectura](weights=None, weights_backbone=None, num_classes=2,
                                       min_size=TAMANO, max_size=TAMANO * 5 // 3)


def arquitectura_de(ruta):
    """Cada detector guardado lleva al lado un .json que dice de qué tipo es."""
    ficha = ruta.replace(".safetensors", ".json")
    if os.path.exists(ficha):
        with open(ficha, encoding="utf-8") as f:
            return json.load(f)["arquitectura"]
    return "mobilenet"


def cargar_entrenado(dispositivo, ruta=None):
    ruta = ruta or MODELO
    modelo = crear_modelo(arquitectura_de(ruta))
    modelo.load_state_dict(load_file(ruta))
    return modelo.to(dispositivo).eval()
