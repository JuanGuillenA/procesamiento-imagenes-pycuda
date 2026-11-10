"""Elimina los resultados anteriores y crea sus carpetas vacias."""
import shutil
from pathlib import Path

raiz = Path(__file__).resolve().parent.parent
resultados = raiz / "resultados"

if resultados.exists():
    shutil.rmtree(resultados)

for carpeta in ("mediciones", "pruebas", "imagenes"):
    (resultados / carpeta).mkdir(parents=True, exist_ok=True)
