# placas-sv

Lector de placas de **El Salvador**, de carro y de moto. Conectas una cámara IP (o abres un
video) y ves cada placa detectada y leída en vivo, o anotas cada vehículo que pasa con su hora y
su foto. Si la cámara ve varios vehículos a la vez, sigue a cada uno por separado.

- Lee el diseño 2011, el diseño 2000, las provisionales y las series cortas como «P 1 CAB».
- Tipos: P, A, C, M, AB, CC, CD, MI, V y N.
- Todo corre local, en la PC: sin servicios en la nube.

## Resultados

Exámenes con fotos que los modelos no vieron al entrenar:

| Examen | Resultado |
|---|---|
| Encontrar la placa: calle, parqueos y carreteras (Open Images) | 91,2 % |
| Encontrar la placa: carros fotografiados de cerca (AUTO.RIA) | 98,6 % |
| Encontrar la placa: de noche (NightLPBD) | 96,8 % |
| Leer la placa completa: fotos de carros (2.453) | 97,9 % |
| Leer la placa completa: escenas variadas (275) | 98,2 % |
| Leer la placa completa: portones de parqueo (2.311) | 96,0 % |

## Instalación

1. Python 3.10 o más nuevo. Instala PyTorch según https://pytorch.org/get-started/locally/ (con
   CUDA si tienes tarjeta NVIDIA; sin tarjeta también funciona, más lento).
2. El resto:
   ```powershell
   pip install -r requirements.txt
   ```

Los modelos (unos 95 MB) se bajan solos de **Releases** la primera vez que lo usas.

## Uso

```powershell
.\ver_en_vivo.cmd rtsp://usuario:clave@IP:554/ruta        # en vivo: abre http://127.0.0.1:8801
.\ver_en_vivo.cmd video.mp4                               # lo mismo con un video
.\camara.cmd rtsp://usuario:clave@IP:554/ruta --panel     # anota cada vehículo + panel: http://127.0.0.1:8800
```

En Linux o Mac, igual con `python ver_en_vivo.py ...` y `python camara.py ...`.

- **La clave de la cámara en un archivo** (para no escribirla en la línea de comandos): pon
  `usuario: ...` y `contraseña: ...` en dos líneas y usa
  `.\ver_en_vivo.cmd 192.168.0.100 --credenciales credenciales.txt`. Prueba las rutas RTSP de Dahua,
  Hikvision y genéricas, y nunca muestra la clave.
- **Registro:** `camara.cmd` anota cada vehículo en `resultados\registro.csv` (se abre en Excel) con
  su foto en `resultados\capturas\`. Las lecturas dudosas salen como `revisar`.
- **Zonas a ignorar**, como la fecha y hora que dibuja la cámara: `--ignorar x0,y0,x1,y1`, en
  fracciones de la imagen.
- **Para leer mejor:** obturador de la cámara en 1/250 s o más rápido (para que el carro no salga
  arrastrado) y la placa de frente, de 130 a 250 px de ancho, donde el carro va despacio.

Las placas son datos personales: las páginas solo se abren desde la misma PC (127.0.0.1).

## Cómo funciona

```
imagen → DETECTOR → recorte de cada placa → LECTOR → "P123456" → formato válido → votos → registro
         ¿dónde hay placas?                  ¿qué dice?            (P123-456)
```

1. **Detector** (`detector.py`): Faster R-CNN con MobileNet v3 de torchvision.
2. **Lector** (`lector.py`): CRNN + CTC que solo conoce los caracteres de una placa salvadoreña
   (0-9, A-F y las letras de tipo), así que no puede confundir un 0 con una O.
3. **Formato:** toda lectura debe ser un tipo seguido de 1 a 6 caracteres 0-9/A-F. Si no cumple,
   se marca como dudosa en vez de inventar.
4. **Votos:** cada placa se lee con tres recortes distintos y en varios cuadros del video; se anota
   la lectura que más se repite.

## Licencias

- Código: MIT (ver `LICENSE`). Detector: torchvision (BSD).
- Los modelos se entrenaron con estos datos, todos CC BY 4.0:
  - *Vehicle Registration Plates* (Augmented Startups, Roboflow Universe), vía
    `keremberke/license-plate-object-detection` en Hugging Face.
  - Open Images (Google), anotaciones vía `shravya11/vehicle-registration-plate`.
  - AUTO.RIA Numberplate Dataset (ARS Online OU), vía `AY000554/Car_plate_detecting_dataset`.
  - NightLPBD (Hasan y Ahsan, IEEE COMPAS 2025).
