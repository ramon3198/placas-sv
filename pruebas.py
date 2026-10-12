"""
Pruebas rápidas de la lógica que no depende del entrenamiento (formato, CTC, votos).

  pruebas.cmd
"""
import os
import sys

CARPETA = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, CARPETA)
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import torch

from camara import Pasada
from lector import SIMBOLOS, bonito, decodificar, interpretar


def salida_ctc(secuencia):
    """Arma una salida falsa del lector donde cada rebanada elige el símbolo indicado ('-' = nada)."""
    x = torch.full((1, len(secuencia), len(SIMBOLOS) + 1), -10.0)
    for t, c in enumerate(secuencia):
        x[0, t, 0 if c == "-" else SIMBOLOS.index(c) + 1] = 10.0
    return x


def test_formato():
    assert interpretar("P123456") == ("P", "123456")
    assert interpretar("AB1F3C2E") == ("AB", "1F3C2E")
    assert interpretar("A1F3C2E") == ("A", "1F3C2E")  # A + 6, no AB + 5
    assert interpretar("MI09B11F") == ("MI", "09B11F")
    assert interpretar("P1CAB") == ("P", "1CAB")  # series cortas, como «P 1 CAB»
    assert interpretar("P1234567") is None  # le sobra uno
    assert interpretar("PP12345") is None  # PP no es un tipo
    assert interpretar("P") is None  # sin serie
    assert interpretar("P12G456") is None  # G no existe en las placas
    assert bonito("P123456") == "P123-456"
    assert bonito("CD0AF123") == "CD0AF-123"
    assert bonito("P1CAB") == "P1-CAB"


def test_ctc():
    texto, _ = decodificar(salida_ctc("PP-11--22-3--44-55-66"))[0]
    assert texto == "P123456", texto
    texto, _ = decodificar(salida_ctc("P-1-1-7"))[0]  # letras repetidas separadas por "nada"
    assert texto == "P117", texto


def test_votos():
    p = Pasada()
    for texto in ["P123456", "P123456", "P123486", "P123456"]:
        p.votar({"texto": texto, "seguridad": 0.9, "caja": [0, 0, 1, 1], "placa": texto}, None)
    assert p.ganadora() == "P123456"
    assert p.votos["P123456"] == 3

    # Llega otro carro pegado: tres cuadros seguidos con otra placa la separan en otra pasada.
    for _ in range(3):
        p.votar({"texto": "C0AF123", "seguridad": 0.9, "caja": [0, 0, 1, 1], "placa": "C0AF-123"}, None)
    assert p.seguidas == 3
    nueva = p.separar()
    assert p.ganadora() == "P123456" and nueva.ganadora() == "C0AF123"


def test_reglas_de_calle():
    from camara import confiable, marcar_cortadas, repetida

    # Una placa que toca el borde de la imagen no cuenta: se ve solo una parte.
    lecturas = marcar_cortadas([{"caja": [0, 500, 120, 560], "valida": True},
                                {"caja": [400, 500, 520, 560], "valida": True}], 1920, 1080)
    assert not lecturas[0]["valida"] and lecturas[0]["cortada"] and lecturas[1]["valida"]
    # Una segunda lectura casi igual (2 caracteres distintos) dentro del minuto es el mismo carro.
    assert repetida("P92507D", 30, {"P25B7D": 10})
    assert not repetida("P813472", 30, {"P25B7D": 10})
    assert not repetida("P92507D", 100, {"P25B7D": 10})  # ya pasó el minuto
    # Una serie muy corta se manda a revisar aunque todos los votos coincidan.
    p = Pasada()
    for _ in range(5):
        p.votar({"texto": "P31", "seguridad": 0.95, "caja": [0, 0, 120, 60], "placa": "P31"}, None)
    assert not confiable(p, "P31", 0.95, {"caja": [0, 0, 120, 60]})
    q = Pasada()
    for _ in range(5):
        q.votar({"texto": "P31A08", "seguridad": 0.95, "caja": [0, 0, 120, 60], "placa": "P31-A08"}, None)
    assert confiable(q, "P31A08", 0.95, {"caja": [0, 0, 120, 60]})


if __name__ == "__main__":
    for nombre, prueba in list(globals().items()):
        if nombre.startswith("test_"):
            prueba()
            print(f"  ✓ {nombre[5:]}")
    print("Todas las pruebas pasaron.")
