"""
El detector de placas: dado una foto, devuelve los rectángulos donde hay placas.

Usamos Faster R-CNN de torchvision (licencia BSD), en dos tamaños:
  - "mobilenet": liviano y rápido, sirve hasta sin tarjeta de video.
  - "resnet50": más pesado pero más preciso con placas pequeñas o lejanas.
No partimos de cero: arrancamos con un modelo que ya aprendió a reconocer 91 tipos de objetos
(carros, personas, perros…) con el dataset COCO, y le cambiamos la última capa para que
reconozca solo "placa" o "no placa". Ya sabe ver bordes, letras y formas; solo tiene que
especializarse.
"""
import json
import os

CARPETA = os.path.dirname(os.path.abspath(__file__))
# torchvision busca los pesos de COCO en <TORCH_HOME>/hub/checkpoints: así los toma del disco
# en vez de descargarlos.
os.environ["TORCH_HOME"] = os.path.join(CARPETA, "pesos")

import torch
from safetensors.torch import load_file
from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_fpn, fasterrcnn_resnet50_fpn_v2
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

MODELO = os.path.join(CARPETA, "modelos", "detector.safetensors")
TAMANO = 640  # la foto se escala para que su lado corto mida esto
ARQUITECTURAS = {
    "mobilenet": fasterrcnn_mobilenet_v3_large_fpn,
    "resnet50": fasterrcnn_resnet50_fpn_v2,
}


def crear_modelo(arquitectura="mobilenet"):
    modelo = ARQUITECTURAS[arquitectura](weights="COCO_V1", min_size=TAMANO, max_size=TAMANO * 5 // 3)
    # La última capa decidía entre 91 objetos de COCO; la cambiamos por una de 2: fondo o placa.
    entradas = modelo.roi_heads.box_predictor.cls_score.in_features
    modelo.roi_heads.box_predictor = FastRCNNPredictor(entradas, 2)
    return modelo


def arquitectura_de(ruta):
    """Cada detector guardado lleva al lado un .json que dice de qué tipo es."""
    ficha = ruta.replace(".safetensors", ".json")
    if os.path.exists(ficha):
        with open(ficha, encoding="utf-8") as f:
            return json.load(f)["arquitectura"]
    return "mobilenet"  # los primeros detectores no tenían ficha y todos eran mobilenet


def guardar_ficha(ruta, arquitectura):
    with open(ruta.replace(".safetensors", ".json"), "w", encoding="utf-8") as f:
        json.dump({"arquitectura": arquitectura}, f)


def cargar_entrenado(dispositivo, ruta=None):
    ruta = ruta or MODELO
    modelo = crear_modelo(arquitectura_de(ruta))
    modelo.load_state_dict(load_file(ruta))
    return modelo.to(dispositivo).eval()
