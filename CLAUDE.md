# CLAUDE.md — Práctica 02 · Convolución de imágenes con PyCUDA

Computación Paralela (UPS). Pareja: Guillén J. y Solano A. Cierre en AVAC: lunes 16-nov-2026, 14:00.

**Fuente de verdad:** `docs/guia_practica02.pdf`. Si algo de este archivo la contradice, gana el PDF y me avisas.
**Base de partida:** `GuillenJ_SolanoA_Practica0/` (práctica 01 ya terminada, en CUDA C/C++, en este mismo repositorio). **No se modifica, renombra ni borra ningún archivo de esa carpeta.** La práctica 02 se agrega en una carpeta nueva, `GuillenJ_SolanoA_Practica02/`. El commit que importó P01 sin cambios está marcado con la etiqueta git `practica01-entrega`. El `README.md` de la raíz es la portada del repositorio (no es de P01): en la Fase 8 se puede reescribir como índice de ambas prácticas, con mi aprobación.

## Qué se construye

Migrar a Python el programa anfitrión de la práctica 01 (carga de imagen, máscaras, selección de filtro, ejecución, validación, mediciones, CSV, guardado de resultados) y ejecutar **el mismo kernel CUDA C** desde Python con PyCUDA. Se comparan cuatro versiones:

| `version` (en los CSV) | Origen |
|---|---|
| `cpp_secuencial` | Práctica 01 (no se reescribe) |
| `cuda_c` | Práctica 01 (no se reescribe) |
| `python_secuencial` | Nueva |
| `pycuda` | Nueva |

Se conservan la lógica, los nombres en español y la distribución de carpetas de la práctica 01. No se "mejora" el algoritmo ni se optimiza el kernel.

## Reglas que no se rompen

### Python secuencial
- Convolución con recorridos explícitos de píxeles y de coeficientes (mismo orden que P01: `y`, `x`, luego `j` externo e `i` interno).
- Prohibido: `numba`/cualquier JIT, `scipy.ndimage`, `cv2.filter2D`, `np.convolve`, vectorizar la convolución con NumPy (slicing, desplazamientos, `stride_tricks`), `multiprocessing`/`threading`, `ctypes` o llamadas al ejecutable en C.
- NumPy solo para almacenar datos y calcular errores.
- Debe tener un límite de tiempo configurable por corrida. Si se excede: abortar, registrar `estado=no_completada`, tiempo transcurrido, filas procesadas y motivo. **Nunca** extrapolar ni presentar estimaciones como tiempos medidos.

### PyCUDA
- El kernel vive en `src/kernels/convolucion.cu` y se carga con `SourceModule`. Es el kernel de `convolucionGPU` de P01, sin cambios de lógica.
- Distribución idéntica a P01: bloques e hilos solo en X, un píxel de salida por hilo, `id = blockIdx.x * blockDim.x + threadIdx.x`, `if (id >= n) return;`, `y = id / ancho`, `x = id % ancho`, vecinos leídos de memoria global, arreglos de entrada y salida separados, extensión con ceros. `bloques = ceil(n / hilos)`.
- Prohibido para la convolución: `gpuarray`, CuPy, `numba.cuda`, memoria compartida/constante/textura, bibliotecas de filtrado.
- Memoria y transferencias explícitas: `cuda.mem_alloc`, `cuda.memcpy_htod`, `cuda.memcpy_dtoh`. Argumentos escalares con tipos explícitos (`np.int32`). Bloque y grid pasados explícitamente.
- Arreglos transferidos: `np.float32` y contiguos (`np.ascontiguousarray`). Documentar los tipos de NumPy y del kernel.
- Flags de compilación equivalentes a P01 (`-O3`, `arch=sm_86`, sin `--use_fast_math`). Confirmar con `nvidia-smi`/`compute_capability` que la GPU es sm_86; si no, ajustar y documentarlo.

### Equivalencia entre las cuatro versiones
- Datos y acumulación en `float32`, también en Python secuencial (`np.float32` en la acumulación). Cuidar que ningún arreglo se cree en `float64` por defecto.
- Máscaras idénticas bit a bit a las de P01 (media = `1.0f/(k*k)` calculada en float32; nitidez y laplaciano con los coeficientes de P01). Verificarlo con un test.
- Mismo tratamiento del resultado al guardar (P01: recorte a 0–255; laplaciano con `abs(x)*4`). El error numérico se calcula **antes** del recorte.
- Referencia numérica: C/C++ secuencial. Error absoluto máximo y medio de Python secuencial, CUDA C y PyCUDA contra esa referencia.
- Tolerancia justificada con la precisión de float32 (ε ≈ 1.19e-7 × número de términos × magnitud de la suma parcial). P01 usó 1e-2. Se razona en el README, no solo se declara.

### Datos
- Usar exactamente los `.pgm` de P01 (`base_2000`, `base_4000`, `base_6000`, `no_multiplo_1013x757`), leyéndolos de `../GuillenJ_SolanoA_Practica0/datos/` (sin duplicarlos). **No regenerarlos**: registrar su SHA-256 en `GuillenJ_SolanoA_Practica02/datos/hashes_p01.json` y comprobarlo al inicio de cada ejecución.
- Las imágenes nuevas (p. ej. la de resolución menor común) se generan con el mismo procedimiento de `preparar_imagenes.py` de P01 (recorte cuadrado centrado + LANCZOS desde `imagenes/ImagenGrande.jpg`) y se guardan en `GuillenJ_SolanoA_Practica02/datos/`. Nunca en la carpeta de P01.
- `resultados/` de P01 es evidencia entregada: jamás se sobrescribe. Si se ejecuta el código C++/CUDA C, las salidas van a `GuillenJ_SolanoA_Practica02/resultados/` (P01 escribe en rutas relativas al directorio de trabajo: ejecutar siempre desde una carpeta de P02).
- Pillow se usa solo para lectura, conversión a escala de grises y escritura.

### Honestidad de los resultados
- Todo número del informe sale de un CSV generado por el código entregado. Ningún tiempo se inventa, se redondea a ojo ni se copia a mano.
- No calcular aceleraciones con ejecuciones incompletas.
- Distinguir en los CSV resultados `nuevo` y `reutilizado_p01`. Los tiempos de P01 solo se reutilizan si son del mismo equipo, datos, código y alcance de medición; si hay duda, se repiten.

## Estructura (espejo de P01)

```
procesamiento-imagenes-pycuda/                   # repositorio (la carpeta local puede llamarse distinto)
├── README.md                                    # portada del repo; se completa en la Fase 8
├── CLAUDE.md                                    # nuevo
├── docs/guia_practica02.pdf                     # nuevo
├── GuillenJ_SolanoA_Practica0/                  # P01 INTACTA (tag practica01-entrega, commit 08b155eab3bbc991cc9a20f9e633f25b41ac223f)
└── GuillenJ_SolanoA_Practica02/                 # NUEVO: todo lo de P02 va aquí
    ├── README.md  EJECUCION.md  requirements.txt  .gitignore
    ├── ejecutar_todo.bat          # prepara, valida, mide, grafica
    ├── main.py                    # equivalente a main.cu
    ├── datos/                     # hashes_p01.json + imágenes NUEVAS (las de P01 se leen de la carpeta de P01)
    ├── referencia_cpp/            # extensión C++/CUDA C: exporta .bin float32 y corre la imagen pequeña;
    │                              # incluye las fuentes de P01 por ruta relativa, sin copiarlas ni modificarlas
    ├── resultados/
    │   ├── mediciones/            # mediciones.csv, inicializacion.csv, entorno.json
    │   ├── pruebas/               # prueba_manual.csv, validacion.csv
    │   ├── imagenes/              # PNG (no PGM de 16 MB)
    │   ├── graficas/
    │   └── referencias/           # .bin float32 (en .gitignore)
    ├── scripts/                   # preparar_imagenes, limpiar_resultados,
    │                              # generar_graficas, registrar_entorno
    ├── src/
    │   ├── configuracion.py       # REPETICIONES, TOLERANCIA, enum de filtros, rutas
    │   ├── experimentos.py        # pruebas y experimentos
    │   ├── convolucion.py         # equivale a filtros.cu (NO llamarlo filtros.py: choca con el paquete filtros/)
    │   ├── imagenes.py
    │   ├── filtros/               # mascaras.py, media.py, nitidez.py, laplaciano.py
    │   ├── secuencial/            # filtros_secuenciales.py, media.py, nitidez.py, laplaciano.py
    │   ├── paralelo/              # filtros_paralelos.py, media.py, nitidez.py, laplaciano.py
    │   └── kernels/convolucion.cu
    └── tests/                     # pytest: manual 5x5, no múltiplo, máscaras, tipos
```

## Medición

- CPU Python: `time.perf_counter()`, solo la convolución.
- GPU: eventos CUDA de PyCUDA. Secuencia idéntica a P01: `e0` → copias H2D (imagen y máscara) → `e1` → kernel → `e2` → copia D2H → `e3`; `e3.synchronize()` antes de leer; `t_h2d`, `t_kernel`, `t_d2h`, `t_total` (= e0→e3).
- Reserva de memoria (`mem_alloc`) fuera del tiempo, una vez por configuración, como en P01. Declararlo en el README.
- Excluidos: lectura/escritura de imágenes y visualización.
- Inicialización del contexto CUDA y compilación/carga del kernel se miden aparte (`inicializacion.csv`) y no se repiten dentro de cada medición. Crear el contexto manualmente (no `pycuda.autoinit`) para poder cronometrarlo. Indicar si la compilación usó caché.
- Calentamiento previo; mínimo 5 repeticiones por configuración viable (P01 usó 10; mantener 10 en C++, CUDA C y PyCUDA). Reportar mediana y desviación estándar (mín–máx opcional).
- Modificar un factor a la vez. Hilos fijos en 256 fuera del experimento de hilos.

## Experimentos obligatorios

| Experimento | Configuración | Versiones |
|---|---|---|
| Filtros | media, nitidez, laplaciano 3×3; 4000×4000 | las 4 |
| Resolución | media 9×9; 2000², 4000², 6000² | las 4 |
| Máscara | media 3×3, 9×9, 21×21; 4000×4000 | las 4 |
| Hilos por bloque | media 9×9; 4000×4000; 128, 256, 512 | `cuda_c` y `pycuda` |
| Resolución menor común | resolución pequeña donde Python secuencial sí termine (propuesta: 1000×1000) | las 4 |

Configuraciones repetidas comparten mediciones. Python secuencial: primero piloto pequeño; las configuraciones inviables se intentan una vez hasta el límite de tiempo y se marcan `no_completada`.

## CSV

`mediciones.csv`: `version,filtro,mascara,ancho,alto,hilos_bloque,bloques,repeticion,t_h2d,t_kernel,t_d2h,t_total,unidad,estado,observaciones,origen`
(`estado` ∈ `completada` | `no_completada`; `origen` ∈ `nuevo` | `reutilizado_p01`; `NA` donde no aplique.)

`validacion.csv` y `prueba_manual.csv`: mismas columnas que P01, extendidas con `version`.

`entorno.json` (o `.md`): CPU, núcleos, RAM, GPU, memoria de GPU, SO, versiones de Python, NumPy, PyCUDA, controlador y CUDA Toolkit, hash del commit y si el árbol de trabajo estaba limpio al medir.

## Convenciones de código

- Nombres de funciones y variables en español, como en P01 (`crear_mascara`, `ejecutar_cpu`, `comparar`). Comentarios breves en español.
- `pathlib`, sin rutas absolutas. Todo se ejecuta desde `GuillenJ_SolanoA_Practica02/`.
- Windows + PowerShell. Para compilar con `nvcc` hace falta `cl.exe` en el PATH (P01 lo resuelve con `vcvars64.bat`); resolverlo también en `ejecutar_todo.bat`.
- Dependencias con versión fija en `requirements.txt` (pillow, numpy, pycuda, matplotlib, pytest).

## Flujo de trabajo

1. Una fase a la vez; al terminar cada una: correr los tests, explicarme qué hiciste y qué falta, y hacer `git commit` con mensaje claro. **Nunca `git push`.**
   Antes de cada commit, comprobar que P01 sigue intacta: `git diff --stat practica01-entrega -- GuillenJ_SolanoA_Practica0` debe salir vacío. Si no lo está, detenerte y avisarme.
   Los commits solo agregan archivos nuevos (en `GuillenJ_SolanoA_Practica02/`, `docs/` y `CLAUDE.md`).
2. No lanzar experimentos completos sin mi aprobación (son largos). Antes, siempre piloto pequeño.
3. Si algo es ambiguo o contradice la guía, preguntar en vez de decidir en silencio.
4. Antes de las mediciones finales el árbol de trabajo debe estar limpio y el commit debe quedar registrado en `entorno.json`: el código entregado tiene que corresponder a las mediciones del informe.

## Definición de terminado

- [ ] Python secuencial y PyCUDA ejecutan los 3 filtros con el kernel propio; bordes y límites correctos.
- [ ] Prueba manual 5×5 y prueba de no múltiplo pasan en las 4 versiones.
- [ ] Errores máx/medio vs C++ registrados y tolerancia justificada.
- [ ] Todos los experimentos con calentamiento, repeticiones, variabilidad y tiempos separados; límites de Python secuencial documentados.
- [ ] Aceleraciones: CUDA C vs C++ y PyCUDA vs Python secuencial (kernel y con transferencias), y CUDA C vs PyCUDA.
- [ ] Gráficas, tablas, evidencia visual (original + 3 filtros + recorte de la misma región).
- [ ] README con instalación y ejecución, entorno, dependencias, commit de P01 (tag `practica01-entrega`) y de P02, resultados nuevos vs reutilizados.
- [ ] `git diff --stat practica01-entrega -- GuillenJ_SolanoA_Practica0` vacío (P01 intacta).
