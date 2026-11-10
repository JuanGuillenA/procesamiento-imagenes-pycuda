// Filtro de nitidez usando CUDA.
#include "filtros_paralelos.cuh"
#include "../filtros/mascaras.h"

TiemposGPU nitidezGPU(const float* entrada, float* salida, float* dEntrada,
                      float* dSalida, float* dMascara, int ancho, int alto,
                      int k, int hilos) {
    auto mascara = crearMascaraNitidez();
    return ejecutarGPU(entrada, salida, mascara.data(), dEntrada, dSalida,
                       dMascara, ancho, alto, k, hilos);
}
