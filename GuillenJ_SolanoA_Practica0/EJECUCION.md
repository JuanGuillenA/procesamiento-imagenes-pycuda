# Ejecución de la práctica

## Ejecutar

```powershell
python -m pip install -r requirements.txt
./ejecutar_todo.bat
```

Los resultados quedan en `resultados/`.

## Limpiar

```powershell
python scripts/limpiar_resultados.py
```

Después ejecuta nuevamente `./ejecutar_todo.bat`.

## Ejecutar por pasos

```powershell
python scripts/preparar_imagenes.py
./compilar.bat
./convolucion.exe
```
