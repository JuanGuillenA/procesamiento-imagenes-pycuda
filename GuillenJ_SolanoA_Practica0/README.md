# Convolución de imágenes mediante CUDA

Esta práctica implementa los filtros de media, nitidez y laplaciano de forma
secuencial en CPU y paralela en GPU mediante CUDA. El programa compara los
resultados de ambas implementaciones y registra los tiempos de ejecución en
archivos CSV.

## Funcionamiento

Las imágenes se convierten a escala de grises y se procesan mediante máscaras
de convolución. La versión secuencial recorre cada píxel en la CPU, mientras
que la versión CUDA asigna un hilo de la GPU a cada píxel de salida.

Se implementan los siguientes filtros:

- **Media:** suaviza la imagen utilizando máscaras de diferentes tamaños.
- **Nitidez:** resalta detalles y variaciones de intensidad.
- **Laplaciano:** detecta bordes y cambios bruscos en la imagen.

La práctica evalúa imágenes de 2000x2000, 4000x4000 y 6000x6000 píxeles,
máscaras de 3x3, 9x9 y 21x21, y configuraciones de 128, 256 y 512 hilos por
bloque.

## Validación y medición

Antes de realizar los experimentos se comprueban casos calculados manualmente
y una imagen cuyas dimensiones no son múltiplos del número de hilos. También
se calcula el error máximo y medio entre las salidas de CPU y GPU.

Cada configuración se ejecuta diez veces después de un calentamiento previo.
En la GPU se registran por separado los tiempos de transferencia CPU-GPU,
ejecución del kernel, transferencia GPU-CPU y tiempo total.

## Requisitos

- Windows
- Python 3
- CUDA Toolkit
- Visual Studio 2022 con herramientas de C++
- GPU NVIDIA compatible con CUDA

## Ejecución

Instalar la dependencia de Python:

```powershell
python -m pip install -r requirements.txt
```

Preparar las imágenes, compilar y ejecutar todos los experimentos:

```powershell
./ejecutar_todo.bat
```

También se puede ejecutar cada etapa por separado:

```powershell
python scripts/preparar_imagenes.py
./compilar.bat
./convolucion.exe
```

## Resultados

Los archivos generados se guardan en las siguientes rutas:

- `resultados/mediciones/mediciones.csv`: tiempos de CPU y GPU.
- `resultados/pruebas/validacion.csv`: comparación numérica entre CPU y GPU.
- `resultados/pruebas/prueba_manual.csv`: resultados de los casos manuales.
- `resultados/imagenes/`: imágenes procesadas por los diferentes filtros.
