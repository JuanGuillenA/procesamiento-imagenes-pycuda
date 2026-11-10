"""Genera las graficas de rendimiento a partir de mediciones.csv."""

import csv
from pathlib import Path
from statistics import median, stdev

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


RAIZ = Path(__file__).resolve().parent.parent
ENTRADA = RAIZ / "resultados" / "mediciones" / "mediciones.csv"
SALIDA = RAIZ / "resultados" / "graficas"


def cargar_mediciones():
    with ENTRADA.open(encoding="utf-8", newline="") as archivo:
        return list(csv.DictReader(archivo))


def seleccionar(filas, implementacion, filtro, mascara, ancho, hilos=None):
    resultado = []
    for fila in filas:
        if fila["implementacion"] != implementacion:
            continue
        if fila["filtro"] != filtro:
            continue
        if int(fila["mascara"]) != mascara or int(fila["ancho"]) != ancho:
            continue
        if hilos is not None and fila["hilos_bloque"] != str(hilos):
            continue
        resultado.append(fila)
    return resultado


def resumir(filas, columna):
    valores = [float(fila[columna]) for fila in filas]
    if not valores:
        raise ValueError(f"No existen datos para calcular {columna}.")
    variabilidad = stdev(valores) if len(valores) > 1 else 0.0
    return median(valores), variabilidad


def guardar(nombre):
    plt.tight_layout()
    plt.savefig(SALIDA / nombre, dpi=180, bbox_inches="tight")
    plt.close()


def grafica_barras(nombre, titulo, etiquetas, series, escala_log=False,
                   etiqueta_y="Tiempo (ms)"):
    ancho_barra = 0.8 / len(series)
    posiciones = list(range(len(etiquetas)))

    plt.figure(figsize=(9, 5))
    for indice, (nombre_serie, valores, errores) in enumerate(series):
        desplazamiento = (indice - (len(series) - 1) / 2) * ancho_barra
        x = [posicion + desplazamiento for posicion in posiciones]
        plt.bar(x, valores, ancho_barra, yerr=errores, capsize=3,
                label=nombre_serie)

    plt.xticks(posiciones, etiquetas)
    plt.ylabel(etiqueta_y)
    plt.title(titulo)
    if escala_log:
        plt.yscale("log")
    plt.grid(axis="y", alpha=0.25)
    plt.legend()
    guardar(nombre)


def grafica_filtros(filas):
    configuraciones = [
        ("Media", "media"),
        ("Nitidez", "nitidez"),
        ("Laplaciano", "laplaciano"),
    ]
    cpu, gpu_kernel, gpu_total = [], [], []
    for _, filtro in configuraciones:
        datos_cpu = seleccionar(filas, "CPU", filtro, 3, 4000)
        datos_gpu = seleccionar(filas, "GPU", filtro, 3, 4000, 256)
        cpu.append(resumir(datos_cpu, "t_total"))
        gpu_kernel.append(resumir(datos_gpu, "t_kernel"))
        gpu_total.append(resumir(datos_gpu, "t_total"))

    grafica_barras(
        "comparacion_filtros.png",
        "Comparación de filtros en una imagen de 4000x4000",
        [etiqueta for etiqueta, _ in configuraciones],
        [
            ("CPU", [x[0] for x in cpu], [x[1] for x in cpu]),
            ("GPU kernel", [x[0] for x in gpu_kernel],
             [x[1] for x in gpu_kernel]),
            ("GPU total", [x[0] for x in gpu_total],
             [x[1] for x in gpu_total]),
        ],
        escala_log=True,
    )


def grafica_resolucion(filas):
    resoluciones = [2000, 4000, 6000]
    cpu, gpu_kernel, gpu_total = [], [], []
    for resolucion in resoluciones:
        datos_cpu = seleccionar(filas, "CPU", "media", 9, resolucion)
        datos_gpu = seleccionar(filas, "GPU", "media", 9, resolucion, 256)
        cpu.append(resumir(datos_cpu, "t_total"))
        gpu_kernel.append(resumir(datos_gpu, "t_kernel"))
        gpu_total.append(resumir(datos_gpu, "t_total"))

    grafica_barras(
        "comparacion_resolucion.png",
        "Efecto de la resolución con filtro de media 9x9",
        [f"{r}x{r}" for r in resoluciones],
        [
            ("CPU", [x[0] for x in cpu], [x[1] for x in cpu]),
            ("GPU kernel", [x[0] for x in gpu_kernel],
             [x[1] for x in gpu_kernel]),
            ("GPU total", [x[0] for x in gpu_total],
             [x[1] for x in gpu_total]),
        ],
        escala_log=True,
    )


def grafica_mascaras(filas):
    mascaras = [3, 9, 21]
    cpu, gpu_kernel, gpu_total = [], [], []
    for mascara in mascaras:
        datos_cpu = seleccionar(filas, "CPU", "media", mascara, 4000)
        datos_gpu = seleccionar(filas, "GPU", "media", mascara, 4000, 256)
        cpu.append(resumir(datos_cpu, "t_total"))
        gpu_kernel.append(resumir(datos_gpu, "t_kernel"))
        gpu_total.append(resumir(datos_gpu, "t_total"))

    grafica_barras(
        "comparacion_mascaras.png",
        "Efecto del tamaño de la máscara en una imagen de 4000x4000",
        [f"{k}x{k}" for k in mascaras],
        [
            ("CPU", [x[0] for x in cpu], [x[1] for x in cpu]),
            ("GPU kernel", [x[0] for x in gpu_kernel],
             [x[1] for x in gpu_kernel]),
            ("GPU total", [x[0] for x in gpu_total],
             [x[1] for x in gpu_total]),
        ],
        escala_log=True,
    )


def grafica_hilos(filas):
    hilos = [128, 256, 512]
    kernel, total = [], []
    for cantidad in hilos:
        datos = seleccionar(filas, "GPU", "media", 9, 4000, cantidad)
        kernel.append(resumir(datos, "t_kernel"))
        total.append(resumir(datos, "t_total"))

    grafica_barras(
        "comparacion_hilos.png",
        "Efecto de los hilos por bloque con filtro de media 9x9",
        [str(x) for x in hilos],
        [
            ("GPU kernel", [x[0] for x in kernel], [x[1] for x in kernel]),
            ("GPU total", [x[0] for x in total], [x[1] for x in total]),
        ],
    )


def grafica_distribucion_gpu(filas):
    configuraciones = [
        ("Media 3x3", "media", 3),
        ("Nitidez 3x3", "nitidez", 3),
        ("Laplaciano 3x3", "laplaciano", 3),
        ("Media 9x9", "media", 9),
        ("Media 21x21", "media", 21),
    ]
    h2d, kernel, d2h = [], [], []
    for _, filtro, mascara in configuraciones:
        datos = seleccionar(filas, "GPU", filtro, mascara, 4000, 256)
        valores = [
            resumir(datos, "t_h2d")[0],
            resumir(datos, "t_kernel")[0],
            resumir(datos, "t_d2h")[0],
        ]
        total = sum(valores)
        h2d.append(100 * valores[0] / total)
        kernel.append(100 * valores[1] / total)
        d2h.append(100 * valores[2] / total)

    x = list(range(len(configuraciones)))
    plt.figure(figsize=(10, 5))
    plt.bar(x, h2d, label="CPU-GPU")
    plt.bar(x, kernel, bottom=h2d, label="Kernel")
    base = [a + b for a, b in zip(h2d, kernel)]
    plt.bar(x, d2h, bottom=base, label="GPU-CPU")
    plt.xticks(x, [c[0] for c in configuraciones], rotation=15)
    plt.ylabel("Proporción del tiempo total (%)")
    plt.title("Distribución del tiempo de ejecución en GPU")
    plt.ylim(0, 100)
    plt.legend()
    guardar("distribucion_tiempo_gpu.png")


def grafica_aceleracion(filas):
    configuraciones = [
        ("Media 3x3\n4000", "media", 3, 4000),
        ("Nitidez 3x3\n4000", "nitidez", 3, 4000),
        ("Laplaciano 3x3\n4000", "laplaciano", 3, 4000),
        ("Media 9x9\n2000", "media", 9, 2000),
        ("Media 9x9\n4000", "media", 9, 4000),
        ("Media 9x9\n6000", "media", 9, 6000),
        ("Media 21x21\n4000", "media", 21, 4000),
    ]
    solo_kernel, con_transferencias = [], []
    for _, filtro, mascara, resolucion in configuraciones:
        datos_cpu = seleccionar(filas, "CPU", filtro, mascara, resolucion)
        datos_gpu = seleccionar(filas, "GPU", filtro, mascara, resolucion, 256)
        tiempo_cpu = resumir(datos_cpu, "t_total")[0]
        tiempo_kernel = resumir(datos_gpu, "t_kernel")[0]
        tiempo_total = resumir(datos_gpu, "t_total")[0]
        solo_kernel.append(tiempo_cpu / tiempo_kernel)
        con_transferencias.append(tiempo_cpu / tiempo_total)

    grafica_barras(
        "aceleracion.png",
        "Aceleración obtenida con CUDA",
        [c[0] for c in configuraciones],
        [
            ("Solo kernel", solo_kernel, [0] * len(solo_kernel)),
            ("Con transferencias", con_transferencias,
             [0] * len(con_transferencias)),
        ],
        etiqueta_y="Aceleración (veces)",
    )


def main():
    if not ENTRADA.exists():
        raise FileNotFoundError(
            "No se encontró resultados/mediciones/mediciones.csv."
        )

    SALIDA.mkdir(parents=True, exist_ok=True)
    filas = cargar_mediciones()
    grafica_filtros(filas)
    grafica_resolucion(filas)
    grafica_mascaras(filas)
    grafica_hilos(filas)
    grafica_distribucion_gpu(filas)
    grafica_aceleracion(filas)
    print(f"Gráficas guardadas en: {SALIDA}")


if __name__ == "__main__":
    main()
