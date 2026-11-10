"""Genera las imagenes PGM usadas por los experimentos."""
import json
from pathlib import Path

from PIL import Image

raiz = Path(__file__).resolve().parent.parent
origen = raiz / "imagenes" / "ImagenGrande.jpg"
datos = raiz / "datos"
Image.MAX_IMAGE_PIXELS = None
datos.mkdir(exist_ok=True)

# Se usa un recorte cuadrado para que todas las resoluciones tengan la misma escena.
original = Image.open(origen)
base = original.convert("L")
lado = min(base.size)
x = (base.width - lado) // 2
y = (base.height - lado) // 2
base = base.crop((x, y, x + lado, y + lado))
registro = {"base_original": original.size, "recorte": (x, y, x + lado, y + lado)}

for n in (2000, 4000, 6000):
    imagen = base.resize((n, n), Image.LANCZOS)
    imagen.save(datos / f"base_{n}.pgm")
    registro[f"base_{n}.pgm"] = imagen.size

# Esta imagen permite probar bloques cuyo ultimo grupo no esta completo.
base.resize((1013, 757), Image.LANCZOS).save(datos / "no_multiplo_1013x757.pgm")
(datos / "preparacion.json").write_text(json.dumps(registro, indent=2))
