# placas-sv

Lector de placas de **El Salvador**, de carro y de moto, para cámaras de portón o de parqueo, videos
y fotos tomadas con el teléfono. Encuentra cada placa en la imagen, la lee y anota cada vehículo
que pasa. Si la cámara ve varios vehículos a la vez, sigue a cada uno por separado.

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
2. El resto de las dependencias:
   ```powershell
   pip install -r requirements.txt
   ```
3. Baja los modelos de **Releases** (`detector.safetensors` y `lector.safetensors`) y ponlos en
   `modelos\`, junto a sus fichas `.json`:
   ```powershell
   gh release download v1.0 -R ramon3198/placas-sv -D modelos
   ```
4. La primera vez, torchvision baja los pesos base de COCO (unos 75 MB) a `pesos\`.

Los `.cmd` usan el `python` del sistema. Para usar otro: `set PYTHON=C:\ruta\a\python.exe`.

## Cómo usarlo

```powershell
.\ver_en_vivo.cmd video.mp4                        # video con las placas marcadas: http://127.0.0.1:8801
.\ver_en_vivo.cmd rtsp://usuario:clave@IP:554/ruta # lo mismo, desde una cámara IP
.\camara.cmd rtsp://usuario:clave@IP:554/ruta --panel   # registro de vehículos + panel: http://127.0.0.1:8800
.\probar_lector.cmd C:\fotos\placas                # leer placas de fotos
.\pruebas.cmd                                      # pruebas rápidas
```

- **Clave de la cámara en un archivo:** `ver_en_vivo.cmd 192.168.0.100 --credenciales credenciales.txt`,
  con `usuario: ...` y `contraseña: ...` en dos líneas. Prueba las rutas RTSP conocidas (Dahua,
  Hikvision y genéricas) y nunca muestra la clave.
- **Zonas a ignorar** (logos, fecha y hora de la cámara): `--ignorar x0,y0,x1,y1` en fracciones de
  la imagen.
- **Registro:** `camara.cmd` anota cada vehículo en `resultados\registro.csv` con su foto en
  `resultados\capturas\`. Las lecturas dudosas salen como `revisar`.
- **Para medir con tus fotos:** nombra cada foto con su placa (`P123456.jpg`) y corre
  `probar_lector.cmd` con la carpeta.

**Consejos para la cámara:** obturador rápido (1/250 s o menos) para que el carro no salga
arrastrado, y la placa de frente, de 130 a 250 px de ancho, donde el carro va despacio. Las placas
son datos personales: las páginas solo se abren desde la misma PC (127.0.0.1).

### Afinar el lector con tu cámara

```powershell
.\ver_en_vivo.cmd 192.168.0.100 --credenciales credenciales.txt --juntar   # guarda recortes en datos\reales
.\revisar.cmd                                      # confirmarlos o corregirlos: http://127.0.0.1:8802
.\generar_recortes.cmd --reales                    # arma entrenamiento y examen
.\entrenar_lector.cmd 6 1.5 afinado --desde lector.safetensors
```

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
4. **Votos:** cada placa se lee con tres recortes distintos y, en video, en varios cuadros; se
   anota la lectura que más se repite.

## Entrenar desde cero

Baja los datasets de la sección de licencias a `descargas\` (`dataset\`, `openimages\`,
`autoria\` y `nocturnas\NightLPBD v1.0\`; las rutas exactas están al inicio de cada script).

```powershell
.\generar_placas.cmd                         # placas sintéticas + examen
.\crear_examen_fotos.cmd --entreno           # placas pegadas en fotos de entrenamiento
.\crear_examen_fotos.cmd --oi-entreno
.\crear_examen_fotos.cmd --autos-entreno
.\entrenar_detector_general.cmd mobilenet 2 8   # detector
.\generar_recortes.cmd --rondas 2            # recortes de placas en fotos reales
.\entrenar_lector.cmd 25 1.5 grande          # lector
.\crear_examen_fotos.cmd                     # exámenes
.\crear_examen_fotos.cmd --oi
.\crear_examen_fotos.cmd --autos
.\evaluar.cmd examen_fotos_autos             # medir todo el sistema
```

## Límites

- El diseño de moto 2011 se dibujó a partir de pocas fotos.
- Las placas de 1994 y anteriores no están modeladas.
- Si cambia el diseño de la placa, hay que ajustar `generar_placas.py` y reentrenar el lector.

## Licencias

- Código: MIT (ver `LICENSE`).
- Detector: torchvision (BSD).
- Datos con los que se entrenaron los modelos, todos CC BY 4.0:
  - *Vehicle Registration Plates* (Augmented Startups, Roboflow Universe), vía
    `keremberke/license-plate-object-detection` en Hugging Face.
  - Open Images (Google), anotaciones vía `shravya11/vehicle-registration-plate`.
  - AUTO.RIA Numberplate Dataset (ARS Online OU), vía `AY000554/Car_plate_detecting_dataset`.
  - NightLPBD (Hasan y Ahsan, IEEE COMPAS 2025).
