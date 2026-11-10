// Pruebas, mediciones y experimentos de la practica.
#include "experimentos.h"
#include "configuracion.h"
#include "filtros.cuh"
#include "imagenes.h"
#include "paralelo/filtros_paralelos.cuh"
#include "secuencial/filtros_secuenciales.h"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <direct.h>
#include <cuda_runtime.h>
#include <vector>

namespace {
FILE* tiempos;
FILE* validacion;

// Carga una imagen PGM y libera sus datos cuando termina de usarse.
struct Imagen {
    int ancho, alto;
    float* datos;

    Imagen(const char* ruta) : datos(leerPGM(ruta, &ancho, &alto)) {}
    ~Imagen() { std::free(datos); }
};

// Calcula una vez en CPU y GPU. Se usa solo en las pruebas de correccion.
void calcularAmbas(const float* entrada, int ancho, int alto,
                   const float* mascara, int k, int hilos,
                   float* cpu, float* gpu) {
    int n = ancho * alto;
    float *dEntrada, *dSalida, *dMascara;

    convolucionCPU(entrada, cpu, mascara, ancho, alto, k);
    cudaMalloc(&dEntrada, n * sizeof(float));
    cudaMalloc(&dSalida, n * sizeof(float));
    cudaMalloc(&dMascara, k * k * sizeof(float));
    ejecutarGPU(entrada, gpu, mascara, dEntrada, dSalida, dMascara,
                ancho, alto, k, hilos);
    cudaFree(dEntrada);
    cudaFree(dSalida);
    cudaFree(dMascara);
}

// Ejecuta una configuracion completa y guarda sus tiempos y validacion.
void ejecutar(const Imagen& imagen, int filtro, int k, int hilos,
             bool medirCPU, const char* salidaCPU = nullptr,
             const char* salidaGPU = nullptr) {
    int n = imagen.ancho * imagen.alto;
    int bloques = (n + hilos - 1) / hilos;
    std::vector<float> cpu(n), gpu(n);

    std::printf("\n[%s %dx%d] imagen %dx%d, %d hilos, %d bloques\n",
                NOMBRE_FILTRO[filtro], k, k, imagen.ancho, imagen.alto,
                hilos, bloques);

    if (medirCPU) {
        // Calentamiento CPU: esta ejecucion no se guarda.
        ejecutarFiltroCPU(filtro, imagen.datos, cpu.data(), imagen.ancho,
                          imagen.alto, k);
        for (int r = 1; r <= REPETICIONES; r++) {
            double t = ejecutarFiltroCPU(filtro, imagen.datos, cpu.data(),
                                         imagen.ancho, imagen.alto, k);
            std::fprintf(tiempos,
                         "CPU,%s,%d,%d,%d,NA,NA,%d,NA,NA,NA,%.4f,ms\n",
                         NOMBRE_FILTRO[filtro], k, imagen.ancho, imagen.alto,
                         r, t);
        }
    } else {
        auto mascara = crearMascara(filtro, k);
        convolucionCPU(imagen.datos, cpu.data(), mascara.data(),
                       imagen.ancho, imagen.alto, k);
    }

    float *dEntrada, *dSalida, *dMascara;
    cudaMalloc(&dEntrada, n * sizeof(float));
    cudaMalloc(&dSalida, n * sizeof(float));
    cudaMalloc(&dMascara, k * k * sizeof(float));

    // Calentamiento GPU durante dos segundos, sin guardar sus tiempos.
    auto inicio = std::chrono::steady_clock::now();
    while (std::chrono::duration<double>(
               std::chrono::steady_clock::now() - inicio).count() <
           SEGUNDOS_CALENTAMIENTO_GPU) {
        ejecutarFiltroGPU(filtro, imagen.datos, gpu.data(), dEntrada,
                          dSalida, dMascara, imagen.ancho, imagen.alto,
                          k, hilos);
    }

    for (int r = 1; r <= REPETICIONES; r++) {
        TiemposGPU t = ejecutarFiltroGPU(
            filtro, imagen.datos, gpu.data(), dEntrada, dSalida, dMascara,
            imagen.ancho, imagen.alto, k, hilos);
        std::fprintf(tiempos,
                     "GPU,%s,%d,%d,%d,%d,%d,%d,%.4f,%.4f,%.4f,%.4f,ms\n",
                     NOMBRE_FILTRO[filtro], k, imagen.ancho, imagen.alto,
                     hilos, bloques, r, t.h2d, t.kernel, t.d2h, t.total);
    }

    cudaFree(dEntrada);
    cudaFree(dSalida);
    cudaFree(dMascara);

    double maximo, medio;
    compararCPUyGPU(cpu.data(), gpu.data(), n, &maximo, &medio);
    bool ok = maximo <= TOLERANCIA;
    std::fprintf(validacion,
                 "experimento,%s,%d,%d,%d,%d,%.3e,%.3e,%.0e,%s\n",
                 NOMBRE_FILTRO[filtro], k, imagen.ancho, imagen.alto,
                 hilos, maximo, medio, TOLERANCIA, ok ? "OK" : "FALLA");

    if (salidaCPU)
        guardarPGM(salidaCPU, cpu.data(), imagen.ancho, imagen.alto,
                   filtro == LAPLACIANO);
    if (salidaGPU)
        guardarPGM(salidaGPU, gpu.data(), imagen.ancho, imagen.alto,
                   filtro == LAPLACIANO);
}

// Comprueba la convolucion con una matriz 5x5 y valores calculados a mano.
void pruebaManual() {
    float imagen[] = {35, 40, 41, 45, 50,
                      40, 40, 42, 46, 52,
                      42, 46, 50, 55, 55,
                      48, 52, 56, 58, 60,
                      56, 60, 65, 70, 75};
    float figura[] = {-2, -1, 0, -1, 1, 1, 0, 1, 2};
    float cpu[25], gpu[25];
    std::vector<float> m[] = {crearMascara(MEDIA, 3),
                               crearMascara(NITIDEZ, 3),
                               crearMascara(LAPLACIANO, 3)};
    const float* mascaras[] = {figura, m[0].data(), m[1].data(), m[2].data()};
    const char* nombres[] = {"figura1", "media", "nitidez", "laplaciano"};
    struct Caso { int m, fila, columna; float esperado; };
    Caso casos[] = {{0, 1, 1, 78}, {0, 1, 2, 87},
                    {1, 0, 0, 155.f / 9}, {1, 2, 2, 445.f / 9},
                    {1, 4, 4, 263.f / 9}, {2, 0, 0, 95},
                    {2, 1, 1, 32}, {2, 2, 2, 51},
                    {3, 0, 0, -60}, {3, 1, 1, 8},
                    {3, 2, 2, -1}, {3, 4, 4, -170}};

    FILE* archivo = std::fopen("resultados/pruebas/prueba_manual.csv", "w");
    std::fprintf(archivo,
                 "mascara,fila,columna,esperado_manual,cpu,gpu,resultado\n");
    for (int i = 0; i < 4; i++) {
        calcularAmbas(imagen, 5, 5, mascaras[i], 3, 256, cpu, gpu);
        for (auto caso : casos) {
            if (caso.m != i) continue;
            float a = cpu[caso.fila * 5 + caso.columna];
            float b = gpu[caso.fila * 5 + caso.columna];
            bool ok = std::fabs(a - caso.esperado) <= 1e-3 &&
                      std::fabs(b - caso.esperado) <= 1e-3;
            std::fprintf(archivo, "%s,%d,%d,%.4f,%.4f,%.4f,%s\n",
                         nombres[i], caso.fila, caso.columna, caso.esperado,
                         a, b, ok ? "OK" : "FALLA");
        }
    }
    std::fclose(archivo);
}

// Prueba que el ultimo bloque incompleto tambien calcule sus pixeles.
void pruebaNoMultiplo() {
    Imagen imagen("datos/no_multiplo_1013x757.pgm");
    int n = imagen.ancho * imagen.alto;
    std::vector<float> cpu(n), gpu(n);
    int filtros[] = {MEDIA, NITIDEZ, LAPLACIANO, MEDIA};
    int ks[] = {3, 3, 3, 9};

    for (int hilos : {128, 256, 512}) {
        for (int i = 0; i < 4; i++) {
            auto mascara = crearMascara(filtros[i], ks[i]);
            calcularAmbas(imagen.datos, imagen.ancho, imagen.alto,
                          mascara.data(), ks[i], hilos, cpu.data(), gpu.data());
            double maximo, medio;
            compararCPUyGPU(cpu.data(), gpu.data(), n, &maximo, &medio);
            std::fprintf(validacion,
                         "no_multiplo,%s,%d,%d,%d,%d,%.3e,%.3e,%.0e,%s\n",
                         NOMBRE_FILTRO[filtros[i]], ks[i], imagen.ancho,
                         imagen.alto, hilos, maximo, medio, TOLERANCIA,
                         maximo <= TOLERANCIA ? "OK" : "FALLA");
        }
    }
}
} // namespace

int ejecutarPractica() {
    _mkdir("resultados");
    _mkdir("resultados/mediciones");
    _mkdir("resultados/pruebas");
    _mkdir("resultados/imagenes");

    tiempos = std::fopen("resultados/mediciones/mediciones.csv", "w");
    validacion = std::fopen("resultados/pruebas/validacion.csv", "w");
    std::fprintf(tiempos,
                 "implementacion,filtro,mascara,ancho,alto,hilos_bloque,"
                 "bloques,repeticion,t_h2d,t_kernel,t_d2h,t_total,unidad\n");
    std::fprintf(validacion,
                 "prueba,filtro,mascara,ancho,alto,hilos_bloque,"
                 "error_max,error_medio,tolerancia,resultado\n");

    pruebaManual();
    pruebaNoMultiplo();

    Imagen i2("datos/base_2000.pgm");
    Imagen i4("datos/base_4000.pgm");
    Imagen i6("datos/base_6000.pgm");

    ejecutar(i4, MEDIA, 3, 256, true,
             "resultados/imagenes/secuencial_media3_4000.pgm",
             "resultados/imagenes/paralelo_media3_4000.pgm");
    ejecutar(i4, NITIDEZ, 3, 256, true,
             "resultados/imagenes/secuencial_nitidez3_4000.pgm",
             "resultados/imagenes/paralelo_nitidez3_4000.pgm");
    ejecutar(i4, LAPLACIANO, 3, 256, true,
             "resultados/imagenes/secuencial_laplaciano3_4000.pgm",
             "resultados/imagenes/paralelo_laplaciano3_4000.pgm");

    ejecutar(i2, MEDIA, 9, 256, true);
    ejecutar(i4, MEDIA, 9, 256, true, nullptr,
             "resultados/imagenes/salida_media9_4000.pgm");
    ejecutar(i6, MEDIA, 9, 256, true);
    ejecutar(i4, MEDIA, 21, 256, true, nullptr,
             "resultados/imagenes/salida_media21_4000.pgm");
    ejecutar(i4, MEDIA, 9, 128, false);
    ejecutar(i4, MEDIA, 9, 512, false);

    std::fclose(tiempos);
    std::fclose(validacion);
    std::printf("Listo. Resultados en resultados.\n");
    return 0;
}
