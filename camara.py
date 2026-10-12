r"""
Lee placas en vivo desde una cámara IP, un video o una webcam, y anota cada carro que pasa.

  camara.cmd rtsp://usuario:clave@IP:554/ruta-del-canal     (cámara IP)
  camara.cmd resultados\video_prueba.mp4                    (un video)
  camara.cmd --webcam "Nombre de la webcam"                 (webcam de la PC)
  camara.cmd 192.168.0.100 --credenciales credenciales.txt  (la clave queda en el archivo)

Agregando --panel al final se abre además el panel web en http://127.0.0.1:8800
Agregando --juntar se guardan recortes de cada placa para revisarlos (revisar.cmd) y entrenar.

Por cada carro guarda:
  - una línea en resultados\registro.csv (fecha, hora, placa, votos, seguridad, estado, captura)
  - la foto del momento más claro en resultados\capturas\

Un carro frente a la cámara aparece en muchos cuadros seguidos. En vez de confiar en uno solo,
lee la placa en varios cuadros y se queda con la lectura que más se repitió. Si la cámara ve
varios vehículos a la vez (un parqueo), sigue a cada uno por separado según dónde está su placa.
"""
import csv
import os
import sys
import time
from collections import defaultdict
from datetime import datetime

CARPETA = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, CARPETA)
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import av
from PIL import ImageDraw, ImageFont

from camara_ip import ES_IP, conectar
from lector import interpretar
from placas import LectorDePlacas

ANALIZAR_POR_SEGUNDO = 4  # no hace falta mirar los 30 cuadros de cada segundo
SILENCIO_PARA_CERRAR = 2.0  # segundos sin ver placa: el carro ya pasó
VOTOS_MINIMOS = 2  # una placa vista en un solo cuadro no se anota (puede ser un error)
VOTOS_PARA_ANOTAR = 5  # con 5 lecturas que coinciden se anota de una vez (para abrir la barrera rápido)
CAMBIO_DE_CARRO = 3  # cuadros seguidos con otra placa: llegó otro carro sin pausa
NO_REPETIR = 60  # segundos: la misma placa no se anota dos veces seguidas
# Se marca "revisar" si los votos estuvieron divididos o la mejor lectura fue poco segura.
PARTICIPACION_MINIMA = 0.6
SEGURIDAD_MINIMA = 0.7
ANCHO_MINIMO = 50  # px: una placa más chica que esto no se puede leer con confianza
RESULTADOS = os.path.join(CARPETA, "resultados")
REGISTRO = os.path.join(RESULTADOS, "registro.csv")
CAPTURAS = os.path.join(RESULTADOS, "capturas")


def distancia(a, b):
    """Cuántos caracteres hay que cambiar, agregar o quitar para pasar de a a b."""
    fila = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        nueva = [i]
        for j, cb in enumerate(b, 1):
            nueva.append(min(fila[j] + 1, nueva[j - 1] + 1, fila[j - 1] + (ca != cb)))
        fila = nueva
    return fila[-1]


def repetida(texto, momento, ultima_vez):
    """¿Ya se anotó hace poco esta placa o una casi igual (hasta 2 caracteres distintos)? Las
    segundas lecturas borrosas del mismo carro suelen diferir en uno o dos caracteres."""
    return any(momento - cuando < NO_REPETIR and distancia(texto, otro) <= 2 for otro, cuando in ultima_vez.items())


def confiable(pasada, texto, seguridad, lectura):
    """Si no, se marca "revisar": votos divididos, poca seguridad, placa chica en la imagen o
    serie muy corta (en la calle casi siempre es una placa leída a medias)."""
    participacion = pasada.votos[texto] / sum(pasada.votos.values())
    ancho = lectura["caja"][2] - lectura["caja"][0]
    partes = interpretar(texto)
    return (participacion >= PARTICIPACION_MINIMA and seguridad >= SEGURIDAD_MINIMA
            and ancho >= ANCHO_MINIMO and partes is not None and len(partes[1]) >= 4)


def marcar_cortadas(lecturas, ancho, alto, margen=3):
    """Una placa que toca el borde de la imagen está cortada: lo que se lee es solo una parte,
    así que no cuenta como lectura."""
    for p in lecturas:
        x0, y0, x1, y1 = p["caja"]
        if x0 <= margen or y0 <= margen or x1 >= ancho - margen or y1 >= alto - margen:
            p["valida"] = False
            p["cortada"] = True
    return lecturas


class Pasada:
    """Lo que se vio mientras un carro estuvo frente a la cámara."""

    guardar_recortes = False  # con --juntar: se guardan recortes de la placa para revisarlos y entrenar
    RECORTES_POR_VEHICULO = 6

    def __init__(self):
        self.peso = defaultdict(float)
        self.votos = defaultdict(int)
        self.mejor_foto = {}
        self.recortes = []  # (seguridad, recorte, texto leído en ese cuadro, caja dentro del recorte)
        self.otra, self.seguidas = None, 0

    def votar(self, lectura, foto):
        texto = lectura["texto"]
        self.peso[texto] += lectura["seguridad"]
        self.votos[texto] += 1
        if texto not in self.mejor_foto or lectura["seguridad"] > self.mejor_foto[texto][0]:
            self.mejor_foto[texto] = (lectura["seguridad"], foto, lectura)
        if self.guardar_recortes and foto is not None:
            # La placa con un margen amplio, para después recortarla con márgenes al azar al entrenar.
            x0, y0, x1, y1 = lectura["caja"]
            mx, my = (x1 - x0) * 0.25, (y1 - y0) * 0.4
            recorte = foto.crop((round(x0 - mx), round(y0 - my), round(x1 + mx), round(y1 + my)))
            caja = [round(mx), round(my), round(mx + x1 - x0), round(my + y1 - y0)]
            self.recortes.append((lectura["seguridad"], recorte, texto, caja))
            self.recortes.sort(key=lambda r: -r[0])
            del self.recortes[self.RECORTES_POR_VEHICULO:]
        # ¿Es otra placa distinta a la que va ganando, varias veces seguidas? Si difiere en un
        # solo carácter no cuenta: es casi seguro una mala lectura del mismo carro.
        ganadora = self.ganadora()
        if texto != ganadora and distancia(texto, ganadora) >= 2:
            self.seguidas = self.seguidas + 1 if texto == self.otra else 1
            self.otra = texto
        else:
            self.otra, self.seguidas = None, 0

    def ganadora(self):
        return max(self.peso, key=self.peso.get) if self.peso else None

    def separar(self):
        """Si llegó otro carro pegado al anterior, saca sus votos a una pasada nueva."""
        nueva = Pasada()
        texto = self.otra
        nueva.peso[texto], nueva.votos[texto] = self.peso.pop(texto), self.votos.pop(texto)
        nueva.mejor_foto[texto] = self.mejor_foto.pop(texto)
        nueva.recortes = [r for r in self.recortes if r[2] == texto]
        self.recortes = [r for r in self.recortes if r[2] != texto]
        return nueva


class Registro:
    def __init__(self, es_video, recolector=None):
        self.es_video = es_video
        self.recolector = recolector  # con --juntar: guarda recortes para revisarlos y entrenar
        self.ultima_vez = {}
        os.makedirs(CAPTURAS, exist_ok=True)
        if not os.path.exists(REGISTRO):
            with open(REGISTRO, "w", newline="", encoding="utf-8-sig") as f:  # -sig: Excel muestra bien las tildes
                csv.writer(f).writerow(["fecha", "hora", "placa", "votos", "seguridad", "estado", "captura"])

    def anotar(self, pasada, momento):
        texto = pasada.ganadora()
        if texto is None or pasada.votos[texto] < VOTOS_MINIMOS:
            return
        if repetida(texto, momento, self.ultima_vez):
            return
        self.ultima_vez[texto] = momento
        seguridad, foto, lectura = pasada.mejor_foto[texto]
        estado = "ok" if confiable(pasada, texto, seguridad, lectura) else "revisar"
        ahora = datetime.now()
        hora = f"{int(momento // 60):02}:{momento % 60:04.1f} del video" if self.es_video else ahora.strftime("%H:%M:%S")

        foto = foto.copy()
        d = ImageDraw.Draw(foto)
        d.rectangle(lectura["caja"], outline=(0, 230, 90), width=max(2, foto.width // 300))
        letra = ImageFont.truetype("bahnschrift.ttf", max(18, foto.width // 25))
        d.text((lectura["caja"][0], lectura["caja"][3] + 6), lectura["placa"], font=letra, fill=(0, 230, 90),
               stroke_width=3, stroke_fill=(0, 0, 0))
        nombre = f"{ahora:%Y-%m-%d_%H-%M-%S}_{texto}.jpg"
        foto.save(os.path.join(CAPTURAS, nombre), quality=88)

        with open(REGISTRO, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([ahora.strftime("%Y-%m-%d"), hora, lectura["placa"], pasada.votos[texto],
                                    f"{seguridad:.0%}", estado, nombre])
        print(f"  {hora:>16}   {lectura['placa']:<10}  {pasada.votos[texto]:2} votos   seguridad {seguridad:.0%}"
              + ("" if estado == "ok" else "   ← revisar"))
        if self.recolector:
            self.recolector.guardar(pasada, texto, momento, estado)


def abrir(argumentos):
    if ES_IP.match(argumentos[0]):  # una IP: usuario y clave salen del archivo, sin mostrarse nunca
        return conectar(argumentos[0], argumentos[argumentos.index("--credenciales") + 1])[0], False
    if argumentos[0] == "--webcam":
        return av.open(f"video={argumentos[1]}", format="dshow"), False
    if argumentos[0].lower().startswith("rtsp://"):
        return av.open(argumentos[0], options={"rtsp_transport": "tcp", "timeout": "5000000"}), False
    return av.open(argumentos[0]), True


def encima(a, b):
    """Qué tanto se encima la caja a con la b (0 = nada, 1 = son la misma)."""
    ancho = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    alto = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    comun = ancho * alto
    return comun / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - comun + 1e-9)


class Pista:
    """Un vehículo que se está siguiendo: dónde se vio su placa por última vez y sus votos.
    En un parqueo la cámara ve varios a la vez; cada uno lleva su propia pista."""

    def __init__(self, caja, momento):
        self.caja, self.visto = caja, momento
        self.pasada = Pasada()
        self.anotada = False

    def estable(self):
        """Ya hay lectura segura: se anota sin esperar a que el vehículo se vaya (un carro
        estacionado puede quedarse horas frente a la cámara)."""
        texto = self.pasada.ganadora()
        votos = self.pasada.votos
        return votos[texto] >= VOTOS_PARA_ANOTAR and votos[texto] / sum(votos.values()) >= 0.7


def vigilar(sistema, contenedor, registro, es_video):
    flujo = contenedor.streams.video[0]
    flujo.thread_type = "AUTO"
    inicio, ultimo = time.time(), -1e9
    pistas = []
    for cuadro in contenedor.decode(flujo):
        momento = cuadro.time if es_video and cuadro.time is not None else time.time() - inicio
        if momento - ultimo < 1 / ANALIZAR_POR_SEGUNDO:
            continue
        ultimo = momento
        foto = cuadro.to_image()
        libres = list(pistas)
        for lectura in sorted((p for p in marcar_cortadas(sistema.leer(foto), foto.width, foto.height) if p["valida"]), key=lambda p: -p["seguridad"]):
            # Cada placa se suma a la pista que estaba en el mismo lugar (o que ya tenía ese texto).
            pista = max(libres, key=lambda p: encima(p.caja, lectura["caja"]), default=None)
            if pista and (encima(pista.caja, lectura["caja"]) >= 0.2 or pista.pasada.ganadora() == lectura["texto"]):
                libres.remove(pista)
            else:
                pista = Pista(lectura["caja"], momento)
                pistas.append(pista)
            pista.caja, pista.visto = lectura["caja"], momento
            pista.pasada.votar(lectura, foto)
            if pista.pasada.seguidas >= CAMBIO_DE_CARRO:  # en el mismo lugar llegó otro vehículo
                nueva = pista.pasada.separar()
                if not pista.anotada:
                    registro.anotar(pista.pasada, momento)
                pista.pasada, pista.anotada = nueva, False
            if not pista.anotada and pista.estable():
                registro.anotar(pista.pasada, momento)
                pista.anotada = True
        for pista in [p for p in pistas if momento - p.visto > SILENCIO_PARA_CERRAR]:
            if not pista.anotada:  # se fue antes de juntar suficientes votos: se anota con lo que hay
                registro.anotar(pista.pasada, momento)
            pistas.remove(pista)
    for pista in pistas:
        if not pista.anotada:
            registro.anotar(pista.pasada, ultimo)


def main():
    argumentos = [a for a in sys.argv[1:] if a not in ("--panel", "--juntar")]
    recolector = None
    if "--juntar" in sys.argv:
        from juntar import Recolector

        Pasada.guardar_recortes = True
        recolector = Recolector(argumentos[0] if ES_IP.match(argumentos[0]) else os.path.basename(argumentos[0]))
    if not argumentos:
        print(__doc__)
        return
    if "--panel" in sys.argv:
        from panel import iniciar_panel

        iniciar_panel()
    sistema = LectorDePlacas(insistir=False)  # video: una sola mirada por cuadro
    while True:
        try:
            contenedor, es_video = abrir(argumentos)
        except av.error.FFmpegError as e:
            if argumentos[0].lower().startswith("rtsp://") or ES_IP.match(argumentos[0]):
                print("No se pudo conectar a la cámara. Reintento en 5 s…")
                time.sleep(5)
                continue
            raise
        registro = Registro(es_video, recolector)
        print(f"Vigilando {'el video' if es_video else 'la cámara'}. Cada carro que pase se anota en {REGISTRO}\n")
        try:
            vigilar(sistema, contenedor, registro, es_video)
        except av.error.FFmpegError:
            print("Se cortó la señal. Reconectando…")  # sin el detalle: podría incluir la dirección con la clave
        finally:
            contenedor.close()
        if es_video:
            break  # un video se termina; una cámara se reconecta


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nDetenido.")
