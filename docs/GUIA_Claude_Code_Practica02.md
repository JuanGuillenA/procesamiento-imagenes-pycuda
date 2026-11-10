# Guía: cómo pedirle a Claude Code la Práctica 02 (PyCUDA)

Cierre en AVAC: **lunes 16 de noviembre, 14:00**. Los pasos están ordenados para que lo más lento (Python secuencial) corra mientras tú haces el informe.

---

## 0. Antes de abrir Claude Code

1. **Usa la computadora Windows que tiene la GPU NVIDIA** donde corriste la práctica 01 (el `compilar.bat` usa Visual Studio 2022 y `-arch=sm_86`). PyCUDA no corre sin GPU NVIDIA, y los tiempos de C++/CUDA C solo se pueden reutilizar si son del mismo equipo.
2. **Trabajas en el mismo repositorio de la práctica 01.** La práctica nueva va en una carpeta nueva al lado de la vieja, sin tocar nada de lo existente:

```
convolucion-imagenes-mediante-cuda/        <- tu repo actual (clonado)
├── README.md                              <- de P01, no se toca
├── CLAUDE.md                              <- nuevo (el que te entrego)
├── docs/guia_practica02.pdf               <- nuevo (tu PDF renombrado)
├── GuillenJ_SolanoA_Practica0/            <- P01, intacta
└── GuillenJ_SolanoA_Practica02/           <- la crea Claude Code en la Fase 1
```

3. Marca el commit de entrega de P01 con una etiqueta (`practica01-entrega`) antes de agregar nada. Ese es el commit que citas en el informe, y sirve para comprobar que P01 no cambió.
4. Abre **la raíz del repo** en VS Code, abre Claude Code y pídele que te resuma las reglas de `CLAUDE.md` en 5 líneas.
5. Para la Fase 0 usa **modo plan** (Shift+Tab hasta "plan mode") para que no escriba nada todavía.
6. Entre fases haz `/clear`; el `CLAUDE.md` mantiene el contexto. Revisa cada diff antes de aceptar: en la defensa ambos deben explicar el código.
7. Tu compañero y tú trabajan en el mismo repo: antes de empezar cada sesión haz `git pull` y después `git push`, para no pisar cambios.

---

## Qué te pide el profesor que tu P01 no cubre

Tu kernel, máscaras, imágenes, pruebas y estructura de experimentos se **reutilizan**. Lo nuevo que exige el PDF:

- Python secuencial con bucles explícitos (sin NumPy vectorizado, numba, scipy, multiprocessing).
- Referencias numéricas en float32 **antes del recorte**: P01 solo guardó PGM recortados a 0–255, así que hay que exportar los resultados crudos de C++ sin tocar P01.
- Columna `estado` en el CSV (para «no completada») y `version` con 4 valores.
- Registro del entorno (CPU, RAM, GPU, versiones de Python, NumPy, PyCUDA, driver, CUDA).
- Tiempos de inicialización de contexto y de compilación del kernel, aparte.
- Aceleraciones: CUDA C vs C++, PyCUDA vs Python secuencial (kernel y con transferencias), y CUDA C vs PyCUDA.
- Una resolución menor común (p. ej. 1000×1000) con las 4 versiones. P01 no tiene esa imagen: C++ y CUDA C deben correrse ahí sin alterar su lógica.
- Trazabilidad: commit de P01 y de P02, resultados nuevos vs reutilizados.

---

## Fase 0 — Reconocimiento y plan (modo plan)

```text
Lee CLAUDE.md, docs/guia_practica02.pdf completo y todo el código de GuillenJ_SolanoA_Practica0/ (src, scripts y los CSV de resultados). No escribas código todavía.

Entrégame:
1. Un resumen de 10 líneas de lo que exige la guía.
2. Una tabla con tres columnas: "se reutiliza de P01", "es nuevo" y "lo que P01 no cubre y la guía sí exige".
3. El árbol de carpetas propuesto, espejo de P01 (respeta el de CLAUDE.md y avísame de cualquier choque de nombres en Python).
4. Un plan por fases con criterio de aceptación para cada una.
5. Dudas y riesgos. En particular dime cómo piensas: (a) exportar las salidas float32 de C++ y CUDA C sin modificar P01, (b) correr C++ y CUDA C sobre una imagen más pequeña sin alterar la lógica de P01, (c) medir inicialización de contexto y compilación por separado, (d) acotar el tiempo de Python secuencial.

Espera mi aprobación antes de continuar.
```

**Qué revisar en su respuesta:** que no proponga tocar archivos de P01, que mencione `float32` en la acumulación de Python y que separe `convolucion.py` del paquete `filtros/`.

---

## Fase 1 — Entorno y esqueleto

```text
Fase 1. Crea GuillenJ_SolanoA_Practica02/ con el esqueleto de carpetas de CLAUDE.md (archivos vacíos con su docstring de una línea, como los comentarios de cabecera de P01).

Además:
- requirements.txt con versiones fijas (pillow, numpy, pycuda, matplotlib, pytest). Instálalas en un .venv y dime las versiones que quedaron.
- scripts/registrar_entorno.py: escribe resultados/mediciones/entorno.json con CPU, núcleos, RAM, GPU, memoria de GPU, capacidad de cómputo, SO, versiones de Python, NumPy, PyCUDA, controlador (nvidia-smi) y CUDA Toolkit (nvcc --version), más el hash del commit y si el árbol está limpio.
- Una prueba mínima de humo de PyCUDA (kernel trivial con SourceModule) para confirmar que compila y ejecuta.
- ejecutar_todo.bat que, igual que compilar.bat de P01, deje cl.exe disponible (vcvars64.bat) antes de usar nvcc.

Si pip install pycuda o SourceModule fallan (cl.exe, versión de Python, arquitectura), arréglalo y documenta en EJECUCION.md exactamente qué hizo falta. Al terminar: commit.
```

---

## Fase 2 — Núcleo Python: imágenes, máscaras y convolución secuencial

```text
Fase 2. Implementa en src/ (nombres en español, como P01):
- imagenes.py: leer PGM con Pillow (convertir a "L" y luego a np.float32 contiguo) y guardar resultado a PNG aplicando el mismo tratamiento que guardarPGM de P01 (recorte 0-255; laplaciano con abs(x)*4).
- filtros/: máscaras media (1/(k*k) calculada en float32), nitidez y laplaciano idénticas a P01. Un test debe comprobar bit a bit que coinciden con las de C++.
- convolucion.py: crear_mascara, convolucion_cpu (port literal de convolucionCPU de P01: mismo orden de bucles, extensión con ceros por comprobación de límites, acumulación en np.float32), ejecutar_cpu con time.perf_counter, y comparar() para error máximo y medio.
- secuencial/: filtros_secuenciales.py, media.py, nitidez.py, laplaciano.py como en P01.
- Límite de tiempo configurable en convolucion_cpu (comprobar el reloj una vez por fila). Al excederlo lanza una excepción propia con tiempo transcurrido y filas procesadas.

Prohibido: numba, scipy, cv2.filter2D, np.convolve, vectorizar la convolución, multiprocessing.

tests/: porta los 12 casos de pruebaManual() de P01 (matriz 5x5, máscara de la figura 1, media, nitidez, laplaciano; léelos de GuillenJ_SolanoA_Practica0/src/experimentos.cu, no los copies de memoria) y la prueba de no múltiplo con datos/no_multiplo_1013x757.pgm cuando exista la parte GPU. Corre pytest y muéstrame la salida. Commit.
```

---

## Fase 3 — PyCUDA

```text
Fase 3. Integra PyCUDA.

- src/kernels/convolucion.cu: el kernel convolucionGPU de P01 copiado sin cambios de lógica. Comenta arriba la conversión id -> (y, x) y el control de límites.
- Creación manual del contexto (cuda.init, make_context; sin pycuda.autoinit), cronometrando la inicialización y, por separado, la compilación/carga con SourceModule (mismos flags que P01: -O3, arch sm_86; indica si usa caché). Escribe ambos tiempos en resultados/mediciones/inicializacion.csv. Estos pasos NO se repiten dentro de cada medición.
- src/paralelo/: ejecutar_gpu con la misma secuencia de eventos que ejecutarGPU de P01 (e0, H2D de imagen y máscara, e1, kernel, e2, D2H, e3, e3.synchronize()). mem_alloc una vez por configuración, fuera del tiempo. Argumentos escalares con np.int32; block=(hilos,1,1), grid=(bloques,1); arreglos float32 contiguos.
- media.py, nitidez.py, laplaciano.py y filtros_paralelos.py como en P01.
- Prohibido gpuarray, CuPy, numba.cuda, memoria compartida/constante.

Tests: los mismos de la Fase 2 ahora en GPU (manual 5x5 y no múltiplo con 128, 256 y 512 hilos), más una verificación de que ningún arreglo transferido es float64 ni no contiguo. Corre pytest. Explícame cómo calculas bloques y por qué el último bloque incompleto no escribe fuera de la imagen. Commit.
```

---

## Fase 4 — Referencias C++/CUDA C y validación de las 4 versiones

```text
Fase 4. La validación debe comparar contra C/C++ secuencial ANTES del recorte, y P01 solo guarda PGM recortados.

- Sin modificar ningún archivo de GuillenJ_SolanoA_Practica0/ ni su resultados/, crea una herramienta aparte en GuillenJ_SolanoA_Practica02/referencia_cpp/ que compile incluyendo por ruta relativa las mismas fuentes de P01 (sin copiarlas) (convolucionCPU, kernel, máscaras) y exporte a resultados/referencias/*.bin los resultados float32 crudos de C++ y de CUDA C para las configuraciones de validación y experimentos. Compílala con los mismos flags de P01.
- Esos .bin van en .gitignore y se regeneran con un script.
- En src/experimentos.py implementa prueba_manual y prueba_no_multiplo para las cuatro versiones, y la validación de cada experimento con error absoluto máximo y medio de python_secuencial, cuda_c y pycuda contra la referencia de C++. Escribe resultados/pruebas/prueba_manual.csv y validacion.csv (columnas de P01 + version).
- Dime qué tolerancia propones y su justificación numérica con float32 (epsilon, número de términos, magnitud de las sumas). Compara con los errores observados.

Para Python secuencial usa solo imágenes pequeñas en esta fase. Commit.
```

---

## Fase 5 — Piloto de Python secuencial (aquí te detienes y decides)

```text
Fase 5. Piloto de viabilidad de Python secuencial. No lances aún los experimentos completos.

1. Mide convolucion_cpu en imágenes pequeñas (por ejemplo 250x250 y 500x500) con media 3x3, 9x9 y 21x21, 3 repeticiones cada una.
2. Con esos tiempos calcula, SOLO PARA PLANIFICAR, cuánto tardaría una corrida en 2000, 4000 y 6000 con cada máscara (el costo crece con píxeles x k x k). Presenta esa tabla marcada claramente como "estimación para planificar, no es resultado".
3. Propón: un límite de tiempo por corrida, qué configuraciones grandes se intentarán una sola vez hasta el límite y se marcarán no_completada, y qué resolución menor común (propuesta 1000x1000) permite completar las 4 versiones con 5+ repeticiones y calentamiento.
4. Dime cuántas horas en total tomaría el plan completo.

Espera mi decisión.
```

**Aquí decides tú:** el límite de tiempo por corrida, la resolución menor común y si dejas la corrida larga en la noche. Mi expectativa (a confirmar con el piloto): con acumulación en `np.float32`, 4000×4000 con 3×3 tarda del orden de minutos por corrida, y 9×9 y 21×21 mucho más, así que lo esperable es que varias configuraciones grandes queden «no completadas». La guía lo permite si queda documentado.

---

## Fase 6 — Experimentos completos

```text
Fase 6. Aprobado el plan del piloto con estos parámetros: límite por corrida = <MINUTOS> min, resolución menor común = <N>x<N>, origen de tiempos de C++ y CUDA C = <repetir ahora / reutilizar de P01>.

Antes de empezar: árbol de trabajo limpio, commit hecho, entorno.json generado.

Implementa y ejecuta los experimentos de CLAUDE.md (Filtros, Resolución, Máscara, Hilos y Resolución menor común) escribiendo resultados/mediciones/mediciones.csv con el esquema de CLAUDE.md (version, estado, observaciones, origen). Calentamiento antes de medir; 10 repeticiones en cpp_secuencial, cuda_c y pycuda; 5+ en python_secuencial cuando sea viable. Las configuraciones repetidas comparten mediciones. Las inviables se intentan una vez hasta el límite y se registran como no_completada con tiempo transcurrido y motivo.

El script debe poder reanudarse sin repetir lo ya medido y mostrar progreso. Hazlo ejecutable por partes (primero las versiones GPU, que son rápidas; Python secuencial por separado) para que yo pueda dejar lo largo corriendo.
Pídeme confirmación antes de lanzar la parte de Python secuencial.
```

> Mientras mide: no uses el equipo, conéctalo a la corriente y desactiva suspensión. Si usas el PC, las mediciones salen contaminadas.

---

## Fase 7 — Gráficas, aceleraciones y tablas

```text
Fase 7. Adapta scripts/generar_graficas.py de P01 (mediana y desviación estándar, escala logarítmica donde corresponda) a las 4 versiones:
- Una gráfica por experimento (filtros, resolución, máscara, hilos, resolución menor común), mostrando tiempos de kernel y con transferencias donde aplique.
- Distribución del tiempo en GPU (H2D, kernel, D2H) para CUDA C y PyCUDA.
- Aceleración CUDA C vs C++ y PyCUDA vs Python secuencial, tanto solo kernel como incluyendo transferencias, más una comparación directa CUDA C vs PyCUDA. NO calcular aceleraciones con ejecuciones no_completada: deben aparecer como "no completada", no como un número.
- Tablas resumen en CSV y Markdown (mediana, desviación, mín-máx, aceleraciones) para pegar al informe.
- Evidencia visual: imagen original y resultado de los tres filtros en PNG, más un recorte de la misma región para que el efecto se aprecie.

Todo debe salir de los CSV; nada escrito a mano. Commit.
```

---

## Fase 8 — Documentación, trazabilidad y auditoría contra la rúbrica

```text
Fase 8. Cierra el repositorio.

1. README.md (raíz y GuillenJ_SolanoA_Practica02/), EJECUCION.md y requirements.txt: instalación del entorno (incluida la parte de cl.exe/CUDA/PyCUDA), comandos exactos de ejecución, cómo obtener o verificar los datos (SHA-256), cómo correr tests, qué hace cada carpeta, tipos usados en NumPy y en el kernel, declaración de si la reserva de memoria forma parte del tiempo conjunto, y la tabla de resultados nuevos vs reutilizados de P01.
2. Commit de P01 (tag practica01-entrega, hash <HASH P01>) y espacio para el commit exacto de P02, enlazado desde el README. Comprueba con git diff --stat practica01-entrega -- GuillenJ_SolanoA_Practica0 README.md que P01 y el README raíz siguen intactos.
3. .gitignore (referencias .bin, __pycache__, .venv, ejecutables).
4. Audita el repositorio contra la rúbrica del PDF (Migración, Validación, Experimentos y medición, Análisis comparativo, Informe y reproducibilidad). Para cada criterio dime: cumplido / parcial / falta, con evidencia (archivo y línea o fila de CSV). No marques "cumplido" sin comprobarlo.
5. Corre desde cero (clon limpio) los comandos del README en lo que sea rápido y dime si algo falla.
```

---

## Fase 9 — Preparar la defensa (opcional, muy recomendable)

```text
Fase 9. Genera docs/defensa.md para mí, explicando en lenguaje simple y con referencia a archivos y líneas:
- qué se trasladó a Python y qué se reutilizó del kernel;
- cómo viaja un dato: imagen -> float32 contiguo -> mem_alloc -> memcpy_htod -> kernel -> memcpy_dtoh -> comparación;
- cómo se convierte el índice global en fila y columna y qué pasa en el último bloque;
- por qué la validación se hace antes del recorte y cómo se justifica la tolerancia;
- qué mide cada tiempo y qué queda fuera;
- 15 preguntas que el profesor podría hacer sobre el código, con respuesta corta.
Después hazme 5 preguntas, una por una, y corrígeme.
```

---

## Errores típicos que debes vigilar en lo que escriba

- **`cl.exe` no encontrado** al compilar con `SourceModule`: falta cargar `vcvars64.bat` (P01 ya lo hace en `compilar.bat`).
- **Arreglos en `float64`:** `np.zeros`/`np.empty` sin `dtype` crean doble y el kernel lee basura.
- **Arreglos no contiguos** o argumentos escalares sin `np.int32`.
- **Medir inicialización/compilación dentro de cada repetición** (el PDF lo prohíbe).
- **Olvidar sincronizar** el evento final antes de leer el tiempo.
- **Python secuencial “rápido”** porque usó NumPy vectorizado, numba o `scipy`: eso invalida la implementación.
- **Tiempos estimados presentados como medidos** en el informe.
- **Regenerar los `.pgm`** con otra versión de Pillow: dejaría de ser la misma entrada que P01.
- **Mediciones con el código sin commitear:** el commit del informe debe corresponder a los CSV.

## Qué decides tú (resumen)

| Decisión | Cuándo |
|---|---|
| Hash del commit de P01 (etiqueta `practica01-entrega`) | Antes de la Fase 0 |
| ¿Reutilizar tiempos de P01 o repetirlos? (reutilizar solo si es el mismo equipo, datos, código y alcance) | Fase 5 |
| Límite de tiempo por corrida de Python secuencial | Fase 5 |
| Resolución menor común | Fase 5 |
| Tolerancia numérica propuesta | Fase 4 |

Cuando tengas los CSV y las gráficas, vuelve y te ayudo con el informe PDF (análisis de las 6 preguntas, conclusiones y diapositivas).
