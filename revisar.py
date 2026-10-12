r"""
Página para revisar a mano las placas reales que juntó el sistema (con --juntar).

  revisar.cmd      → abre http://127.0.0.1:8802

Por cada vehículo muestra sus recortes y la lectura propuesta:
  Enter   la lectura está bien (o la corregiste escribiéndola)
  Esc     descartar: no se lee, está cortada o no es una placa
  Ctrl+Z  deshacer la última

Lo revisado queda en datos\reales\revisadas.jsonl y sirve para entrenar el lector
(generar_recortes.cmd --reales) y para medirlo con placas reales.
"""
import json
import os
import sys
from collections import OrderedDict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

CARPETA = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, CARPETA)
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from juntar import INDICE, PENDIENTES, REVISADAS
from lector import bonito, interpretar

PUERTO = 8802


def cargar():
    """Vehículos pendientes (en orden de llegada) y los ya revisados."""
    vehiculos = OrderedDict()
    if os.path.exists(INDICE):
        with open(INDICE, encoding="utf-8") as f:
            for linea in f:
                r = json.loads(linea)
                vehiculos.setdefault(r["vehiculo"], []).append(r)
    revisados = []
    if os.path.exists(REVISADAS):
        with open(REVISADAS, encoding="utf-8") as f:
            revisados = [json.loads(linea) for linea in f if linea.strip()]
    return vehiculos, revisados


def siguiente():
    vehiculos, revisados = cargar()
    hechos = {r["vehiculo"] for r in revisados}
    pendientes = [v for v in vehiculos if v not in hechos]
    if not pendientes:
        return {"listo": True, "hechas": len(hechos), "quedan": 0}
    recortes = vehiculos[pendientes[0]]
    return {
        "listo": False, "vehiculo": pendientes[0], "propuesta": bonito(recortes[0]["propuesta"]),
        "origen": recortes[0]["origen"], "estado": recortes[0]["estado"],
        "recortes": [{"archivo": r["archivo"], "leido": bonito(r["leido"]), "seguridad": r["seguridad"]} for r in recortes],
        "hechas": len(hechos), "quedan": len(pendientes),
    }


def decidir(datos):
    vehiculos, _ = cargar()
    recortes = vehiculos.get(datos["vehiculo"])
    if not recortes:
        return {"error": "Ese vehículo no existe."}
    texto = "".join(c for c in datos.get("texto", "").upper() if c.isalnum())
    if datos["decision"] != "descartada" and not interpretar(texto):
        return {"error": f"«{texto}» no tiene formato de placa salvadoreña."}
    fila = {
        "vehiculo": datos["vehiculo"], "decision": datos["decision"],
        "texto": texto if datos["decision"] != "descartada" else None,
        "propuesta": recortes[0]["propuesta"],
        "recortes": [{"archivo": r["archivo"], "caja": r["caja"]} for r in recortes],
    }
    with open(REVISADAS, "a", encoding="utf-8") as f:
        f.write(json.dumps(fila, ensure_ascii=False) + "\n")
    return {"ok": True}


def deshacer():
    if not os.path.exists(REVISADAS):
        return {"ok": True}
    with open(REVISADAS, encoding="utf-8") as f:
        lineas = [l for l in f if l.strip()]
    with open(REVISADAS, "w", encoding="utf-8") as f:
        f.writelines(lineas[:-1])
    return {"ok": True}


PAGINA = """<!doctype html><html lang="es"><head><meta charset="utf-8"><title>Revisar placas</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
:root { --fondo:#0f1419; --tarjeta:#171d24; --borde:#26303b; --texto:#e6ebf1; --suave:#8d99a8; --ok:#3fbf7f; --mal:#e5534b; --acento:#5b9bff }
* { box-sizing:border-box } body { margin:0; background:var(--fondo); color:var(--texto); font-family:"Segoe UI",system-ui,sans-serif }
main { max-width:960px; margin:0 auto; padding:20px 16px 40px }
header { display:flex; justify-content:space-between; align-items:baseline; gap:12px; margin-bottom:14px }
h1 { font-size:18px; margin:0; font-weight:600 } .suave { color:var(--suave); font-size:13px }
.barra { height:6px; background:var(--borde); border-radius:3px; overflow:hidden; margin-bottom:18px } .barra div { height:100%; background:var(--acento) }
.recortes { display:grid; grid-template-columns:repeat(auto-fill,minmax(270px,1fr)); gap:10px; margin-bottom:18px }
.recorte { background:var(--tarjeta); border:1px solid var(--borde); border-radius:8px; padding:6px; font-size:12px; color:var(--suave) }
.recorte img { width:100%; border-radius:4px; display:block; margin-bottom:4px; background:#000 }
input { width:100%; font-family:Bahnschrift,"Arial Narrow",sans-serif; font-size:40px; font-weight:700; letter-spacing:2px; text-align:center;
  text-transform:uppercase; padding:12px; border-radius:10px; border:2px solid var(--borde); background:#fff; color:#111 }
input:focus { outline:none; border-color:var(--acento) } input.malo { border-color:var(--mal) }
.aviso { min-height:20px; color:var(--mal); font-size:13px; margin:8px 0 }
.botones { display:flex; gap:10px; flex-wrap:wrap }
button { flex:1; min-width:140px; padding:14px; font-size:15px; border-radius:8px; border:1px solid var(--borde); background:var(--tarjeta); color:var(--texto); cursor:pointer }
button.si { background:var(--ok); border-color:var(--ok); color:#06210f; font-weight:600 } button.no { border-color:var(--mal); color:var(--mal) }
.fin { text-align:center; padding:60px 0; color:var(--suave) }
</style></head><body><main>
<header><h1>Revisar placas reales</h1><span class="suave" id="cuenta"></span></header>
<div class="barra"><div id="avance" style="width:0"></div></div>
<div id="trabajo">
  <div class="suave" id="origen" style="margin-bottom:8px"></div>
  <div class="recortes" id="recortes"></div>
  <input id="texto" autocomplete="off" spellcheck="false">
  <div class="aviso" id="aviso"></div>
  <div class="botones">
    <button class="si" id="guardar">Guardar (Enter)</button>
    <button class="no" id="descartar">No se lee / no es placa (Esc)</button>
    <button id="deshacer">Deshacer (Ctrl+Z)</button>
  </div>
</div>
<div class="fin" id="fin" hidden>No hay placas pendientes. Junta más con <b>--juntar</b> en la cámara o en un video.</div>
</main><script>
const $ = id => document.getElementById(id);
const esc = t => String(t).replace(/[&<>"']/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));
const FORMATO = /^(AB|CC|CD|MI|P|A|C|M|V|N)[0-9A-F]{1,6}$/;
let actual = null;
const limpio = t => t.toUpperCase().replace(/[^0-9A-Z]/g, "");
async function cargar() {
  actual = await (await fetch("/api/siguiente", {cache:"no-store"})).json();
  const total = actual.hechas + actual.quedan;
  $("cuenta").textContent = `${actual.hechas} revisadas · ${actual.quedan} pendientes`;
  $("avance").style.width = total ? (100 * actual.hechas / total) + "%" : "0";
  $("trabajo").hidden = actual.listo; $("fin").hidden = !actual.listo;
  if (actual.listo) return;
  $("origen").textContent = `${actual.origen} · el sistema la marcó como «${actual.estado}»`;
  $("recortes").innerHTML = actual.recortes.map(r => `<div class="recorte"><img src="/recorte/${encodeURIComponent(r.archivo)}" alt="">
    leyó ${esc(r.leido)} · ${Math.round(r.seguridad * 100)}%</div>`).join("");
  $("texto").value = actual.propuesta; $("aviso").textContent = ""; validar();
  $("texto").focus(); $("texto").select();
}
function validar() { const ok = FORMATO.test(limpio($("texto").value)); $("texto").classList.toggle("malo", !ok); return ok; }
async function enviar(decision) {
  if (!actual || actual.listo) return;
  const texto = limpio($("texto").value);
  if (decision !== "descartada" && !validar()) { $("aviso").textContent = "Eso no tiene formato de placa salvadoreña (tipo + hasta 6 caracteres de 0-9 y A-F)."; return; }
  if (decision !== "descartada") decision = texto === limpio(actual.propuesta) ? "correcta" : "corregida";
  const r = await (await fetch("/api/decidir", {method:"POST", body: JSON.stringify({vehiculo: actual.vehiculo, decision, texto})})).json();
  if (r.error) { $("aviso").textContent = r.error; return; }
  cargar();
}
async function deshacer() { await fetch("/api/deshacer", {method:"POST"}); cargar(); }
$("texto").addEventListener("input", validar);
$("guardar").onclick = () => enviar("guardar"); $("descartar").onclick = () => enviar("descartada"); $("deshacer").onclick = deshacer;
document.addEventListener("keydown", e => {
  if (e.key === "Enter") { e.preventDefault(); enviar("guardar"); }
  else if (e.key === "Escape") { e.preventDefault(); enviar("descartada"); }
  else if (e.key.toLowerCase() === "z" && e.ctrlKey) { e.preventDefault(); deshacer(); }
});
cargar();
</script></body></html>"""


class Atender(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def responder(self, codigo, tipo, cuerpo):
        self.send_response(codigo)
        self.send_header("Content-Type", tipo)
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(cuerpo)

    def json(self, datos):
        self.responder(200, "application/json", json.dumps(datos, ensure_ascii=False).encode())

    def do_GET(self):
        if self.path == "/":
            return self.responder(200, "text/html; charset=utf-8", PAGINA.encode())
        if self.path.startswith("/api/siguiente"):
            return self.json(siguiente())
        if self.path.startswith("/recorte/"):
            from urllib.parse import unquote

            nombre = os.path.basename(unquote(self.path[len("/recorte/"):]))  # solo archivos de la carpeta
            ruta = os.path.join(PENDIENTES, nombre)
            if nombre.endswith(".jpg") and os.path.isfile(ruta):
                with open(ruta, "rb") as f:
                    return self.responder(200, "image/jpeg", f.read())
        self.responder(404, "text/plain", b"No existe")

    def do_POST(self):
        largo = int(self.headers.get("Content-Length", 0))
        datos = json.loads(self.rfile.read(largo) or b"{}")
        if self.path == "/api/decidir":
            return self.json(decidir(datos))
        if self.path == "/api/deshacer":
            return self.json(deshacer())
        self.responder(404, "text/plain", b"No existe")


if __name__ == "__main__":
    vehiculos, revisados = cargar()
    print(f"Revisar placas: http://127.0.0.1:{PUERTO}   ({len(vehiculos)} vehículos juntados, {len(revisados)} revisados)")
    print("Solo responde en esta PC: son placas reales. Ctrl+C para cerrar.")
    try:
        ThreadingHTTPServer(("127.0.0.1", PUERTO), Atender).serve_forever()
    except KeyboardInterrupt:
        pass
