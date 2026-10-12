r"""
Entrena el lector con las placas falsas de datos\placas y lo examina con datos\examen_lector,
que usa una fuente que el lector nunca vio al entrenar.

  entrenar_lector.cmd                 → 20 épocas, lector normal, en modelos\lector.safetensors
  entrenar_lector.cmd 30 1.5 grande   → 30 épocas, 1,5 veces más ancho, en modelos\lector_grande.safetensors
"""
import os
import sys
import time

CARPETA = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, CARPETA)
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from safetensors.torch import save_file

from lector import ALTO, ANCHO, MODELO, SIMBOLOS, Lector, decodificar, guardar_ficha, interpretar

EPOCAS = int(sys.argv[1]) if len(sys.argv) > 1 else 20
TAMANO_RED = float(sys.argv[2]) if len(sys.argv) > 2 else 1.0
SALIDA = MODELO.replace("lector.safetensors", f"lector_{sys.argv[3]}.safetensors") if len(sys.argv) > 3 and not sys.argv[3].startswith("--") else MODELO
# --desde X: parte de un lector ya entrenado (modelos/X) y lo afina, con ajustes más suaves.
DESDE = sys.argv[sys.argv.index("--desde") + 1] if "--desde" in sys.argv else None
MAXIMO_LR = 6e-4 if DESDE else 2e-3
LOTE = 128
dispositivo = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(0)


def cargar(carpeta):
    """Lee todas las placas a memoria. La primera vez deja una copia en un solo archivo
    (cache.pt), porque abrir 40.000 archivos sueltos en Windows es lento."""
    etiquetas, cache = os.path.join(carpeta, "etiquetas.tsv"), os.path.join(carpeta, "cache.pt")
    if os.path.exists(cache) and os.path.getmtime(cache) > os.path.getmtime(etiquetas):
        guardado = torch.load(cache)
        return guardado["x"], guardado["textos"]
    with open(etiquetas, encoding="utf-8") as f:
        filas = [linea.rstrip("\n").split("\t") for linea in f if linea.strip()]
    x = torch.empty(len(filas), 1, ALTO, ANCHO, dtype=torch.uint8)
    for k, (nombre, _) in enumerate(filas):
        gris = Image.open(os.path.join(carpeta, nombre)).convert("L")
        x[k, 0] = torch.frombuffer(bytearray(gris.tobytes()), dtype=torch.uint8).view(ALTO, ANCHO)
    textos = [texto for _, texto in filas]
    torch.save({"x": x, "textos": textos}, cache)
    return x, textos


def mover_un_poco(x):
    """Variaciones al azar en la GPU: corre, agranda, achica e inclina un poco cada placa."""
    n = x.shape[0]
    azar = lambda escala: (torch.rand(n, device=x.device) - 0.5) * escala
    angulo, tamano = azar(0.08), 1 + azar(0.2)
    cos, sen = torch.cos(angulo) * tamano, torch.sin(angulo) * tamano
    # La imagen es el doble de ancha que alta: por eso el 0.5 y el 2 en la rotación.
    theta = torch.stack([torch.stack([cos, -sen * 0.5, azar(0.12)], 1),
                         torch.stack([sen * 2, cos, azar(0.12)], 1)], 1)
    rejilla = F.affine_grid(theta, list(x.shape), align_corners=False)
    return F.grid_sample(x, rejilla, padding_mode="border", align_corners=False)


def distancia(a, b):
    """Cuántas letras hay que cambiar, agregar o quitar para pasar de a a b."""
    fila = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        nueva = [i]
        for j, cb in enumerate(b, 1):
            nueva.append(min(fila[j] + 1, nueva[j - 1] + 1, fila[j - 1] + (ca != cb)))
        fila = nueva
    return fila[-1]


@torch.no_grad()
def medir(modelo, x, textos, mostrar_errores=0):
    modelo.eval()
    leidos = []
    for i in range(0, len(x), 512):
        lote = x[i : i + 512].to(dispositivo).float() / 127.5 - 1
        leidos += [t for t, _ in decodificar(modelo(lote))]
    modelo.train()
    perfectas = sum(a == b for a, b in zip(leidos, textos)) / len(textos)
    letras = 1 - sum(distancia(a, b) for a, b in zip(leidos, textos)) / sum(len(t) for t in textos)
    formato = sum(interpretar(t) is not None for t in leidos) / len(textos)
    if mostrar_errores:
        errores = [(b, a) for a, b in zip(leidos, textos) if a != b][:mostrar_errores]
        print("   Algunos errores (real → leído):  " + "   ".join(f"{b}→{a or '∅'}" for b, a in errores))
    return perfectas, letras, formato


def main():
    t0 = time.time()
    print("\nCargando placas falsas…")
    x, textos = cargar(os.path.join(CARPETA, "datos", "placas"))
    x_examen, textos_examen = cargar(os.path.join(CARPETA, "datos", "examen_lector"))
    x_ent, t_ent, x_val, t_val = x[:-2000], textos[:-2000], x[-2000:], textos[-2000:]
    recortes = os.path.join(CARPETA, "datos", "recortes_sv")
    if os.path.exists(os.path.join(recortes, "cache.pt")):
        # Recortes de placas dentro de fotos reales (generar_recortes.py): en perspectiva y con
        # la luz y el fondo de fotos de verdad.
        x_rec, t_rec = cargar(recortes)
        x_ent, t_ent = torch.cat([x_ent, x_rec]), t_ent + t_rec
        print(f"Más {len(x_rec):,} recortes de placas dentro de fotos reales.")
    reales = os.path.join(CARPETA, "datos", "reales")
    x_real = t_real = None
    if os.path.exists(os.path.join(reales, "entreno.pt")):
        # Placas salvadoreñas de verdad, revisadas a mano: pocas pero valiosas, se repiten 4 veces.
        guardado = torch.load(os.path.join(reales, "entreno.pt"))
        x_ent = torch.cat([x_ent] + [guardado["x"]] * 4)
        t_ent = t_ent + guardado["textos"] * 4
        print(f"Más {len(guardado['x']):,} recortes de placas salvadoreñas reales (×4).")
    if os.path.exists(os.path.join(reales, "examen.pt")):
        guardado = torch.load(os.path.join(reales, "examen.pt"))
        x_real, t_real = guardado["x"], guardado["textos"]
        print(f"Examen con {len(x_real):,} placas salvadoreñas reales que no se usan para entrenar.")
    print(f"Para entrenar: {len(x_ent):,}  ·  para medir: {len(x_val):,}  ·  examen con fuente nueva: {len(x_examen):,}")

    modelo = Lector(TAMANO_RED).to(dispositivo)
    if DESDE:
        from safetensors.torch import load_file

        viejo, nuevo = load_file(os.path.join(os.path.dirname(MODELO), DESDE)), modelo.state_dict()
        for nombre, valor in viejo.items():
            if nuevo[nombre].shape == valor.shape:
                nuevo[nombre] = valor
            else:  # la capa de salida creció (se agregó una letra): se copia lo que ya sabía
                nuevo[nombre][: valor.shape[0]] = valor
        modelo.load_state_dict(nuevo)
        print(f"Afinando el lector {DESDE}.")
    print(f"El lector tiene {sum(p.numel() for p in modelo.parameters()):,} pesos.\n")
    pasos_por_epoca = len(x_ent) // LOTE
    optimizador = torch.optim.AdamW(modelo.parameters(), lr=MAXIMO_LR, weight_decay=1e-4)
    ritmo = torch.optim.lr_scheduler.OneCycleLR(optimizador, max_lr=MAXIMO_LR, total_steps=EPOCAS * pasos_por_epoca, pct_start=0.15)
    ctc = nn.CTCLoss(blank=0, zero_infinity=True)

    for epoca in range(1, EPOCAS + 1):
        orden, suma = torch.randperm(len(x_ent)), 0.0
        for k in range(pasos_por_epoca):
            idx = orden[k * LOTE : (k + 1) * LOTE]
            lote = mover_un_poco(x_ent[idx].to(dispositivo).float() / 127.5 - 1)
            elegidos = [t_ent[i] for i in idx]
            objetivo = torch.tensor([SIMBOLOS.index(c) + 1 for t in elegidos for c in t], device=dispositivo)
            largos = torch.tensor([len(t) for t in elegidos], device=dispositivo)

            salida = modelo(lote)  # 1. leer
            prob = salida.log_softmax(-1).permute(1, 0, 2)  # CTC las quiere como (rebanadas, lote, opciones)
            rebanadas = torch.full((len(idx),), prob.shape[0], device=dispositivo)
            error = ctc(prob, objetivo, rebanadas, largos)  # 2. ¿qué tan lejos del texto real?
            optimizador.zero_grad()
            error.backward()  # 3. hacia dónde mover cada peso
            nn.utils.clip_grad_norm_(modelo.parameters(), 5)
            optimizador.step()
            ritmo.step()
            suma += error.item()

        perf, letras, _ = medir(modelo, x_val, t_val)
        perf_ex, letras_ex, _ = medir(modelo, x_examen, textos_examen)
        print(f"época {epoca:2}  |  error {suma / pasos_por_epoca:.3f}  |  placas perfectas {perf:6.1%} "
              f"(fuente nueva {perf_ex:6.1%})  |  letras bien {letras:6.1%} (fuente nueva {letras_ex:6.1%})"
              + (f"  |  placas REALES {medir(modelo, x_real, t_real)[0]:6.1%}" if x_real is not None else "")
              + f"  |  {time.time() - t0:4.0f} s")

    os.makedirs(os.path.dirname(SALIDA), exist_ok=True)
    save_file(modelo.state_dict(), SALIDA)
    guardar_ficha(SALIDA, TAMANO_RED)
    print("\nExamen final con la fuente que nunca vio:")
    perf, letras, formato = medir(modelo, x_examen, textos_examen, mostrar_errores=12)
    print(f"   Placas leídas perfectas: {perf:.1%}  ·  letras correctas: {letras:.1%}  ·  con formato válido: {formato:.1%}")
    if x_real is not None:
        print("\nExamen con placas salvadoreñas REALES (revisadas a mano, nunca usadas para entrenar):")
        perf, letras, _ = medir(modelo, x_real, t_real, mostrar_errores=12)
        print(f"   Placas leídas perfectas: {perf:.1%}  ·  letras correctas: {letras:.1%}")
    print(f"\nListo en {(time.time() - t0) / 60:.1f} minutos. Lector guardado en {SALIDA}")


if __name__ == "__main__":
    main()
