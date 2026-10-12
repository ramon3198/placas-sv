"""
Conecta con una cámara IP a partir de su IP y un archivo de texto con el usuario y la clave.

La clave nunca aparece en pantalla, en la línea de comandos ni en los mensajes de error: el
programa la lee del archivo y arma la dirección RTSP por dentro.

El archivo puede tener cualquiera de estas formas:
    usuario: admin            admin            admin / clave123      rtsp://admin:clave@IP/...
    contraseña: clave123      clave123
"""
import re
from urllib.parse import quote

# Rutas del video según la marca. Se prueban en orden hasta que una responde.
RUTAS = [
    ("Dahua", "/cam/realmonitor?channel=1&subtype={sub}"),
    ("Hikvision", "/Streaming/Channels/10{sub1}"),
    ("ONVIF genérica", "/onvif1"),
    ("genérica", "/live"),
    ("genérica", "/stream1"),
    ("genérica", "/h264"),
]
ES_IP = re.compile(r"^\d{1,3}(\.\d{1,3}){3}$")


def leer_credenciales(archivo):
    with open(archivo, encoding="utf-8-sig") as f:
        texto = f.read().strip()
    url = re.search(r"rtsp://([^:@/\s]+):([^@\s]+)@", texto)
    if url:
        return url.group(1), url.group(2)
    usuario = clave = None
    for linea in texto.splitlines():
        m = re.match(r"\s*(usuario|user|username|login|nombre)\s*[:=]\s*(.+?)\s*$", linea, re.I)
        if m:
            usuario = m.group(2)
        m = re.match(r"\s*(contrase(ñ|n)a|clave|password|pass|pwd)\s*[:=]\s*(.+?)\s*$", linea, re.I)
        if m:
            clave = m.group(3)
    if usuario and clave:
        return usuario, clave
    lineas = [l.strip() for l in texto.splitlines() if l.strip()]
    if len(lineas) == 2:
        return lineas[0], lineas[1]
    if len(lineas) == 1:
        for separador in (" / ", "/", " : ", ":", ";", ",", " "):
            if separador in lineas[0]:
                usuario, clave = lineas[0].split(separador, 1)
                return usuario.strip(), clave.strip()
    raise ValueError("No pude entender el archivo de credenciales: usa 'usuario: ...' y 'contraseña: ...' en dos líneas.")


def ocultar(texto, clave):
    """Borra la clave (tal cual y codificada para URL) de cualquier texto que se vaya a mostrar."""
    for forma in {clave, quote(clave, safe="")}:
        if forma:
            texto = texto.replace(forma, "***")
    return texto


def conectar(ip, archivo, secundario=False):
    """Prueba las rutas conocidas y devuelve (contenedor de PyAV, marca) de la primera que responde."""
    import av

    usuario, clave = leer_credenciales(archivo)
    print(f"Credenciales leídas: usuario {usuario[0]}{'*' * (len(usuario) - 1)}, clave de {len(clave)} caracteres.")
    sub = 1 if secundario else 0
    for marca, ruta in RUTAS:
        url = f"rtsp://{quote(usuario, safe='')}:{quote(clave, safe='')}@{ip}:554" + ruta.format(sub=sub, sub1=sub + 1)
        try:
            contenedor = av.open(url, options={"rtsp_transport": "tcp", "timeout": "5000000"})
            print(f"Conectado: cámara {marca}, canal {'secundario' if secundario else 'principal'}.")
            return contenedor, marca
        except av.error.FFmpegError as e:
            print(f"  ruta de {marca}: no respondió ({ocultar(str(e), clave)})")
    raise ConnectionError("Ninguna ruta conocida respondió. Revisa la IP, el usuario y la clave.")
