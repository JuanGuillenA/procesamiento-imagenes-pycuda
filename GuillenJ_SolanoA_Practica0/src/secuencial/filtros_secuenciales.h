// Entradas de los tres filtros ejecutados de forma secuencial en CPU.
#pragma once
#include "../filtros.cuh"

double mediaCPU(const float*, float*, int, int, int);
double nitidezCPU(const float*, float*, int, int, int);
double laplacianoCPU(const float*, float*, int, int, int);

// Escoge el filtro CPU solicitado por el experimento.
inline double ejecutarFiltroCPU(int filtro, const float* entrada,
                                float* salida, int ancho, int alto, int k) {
    if (filtro == MEDIA) return mediaCPU(entrada, salida, ancho, alto, k);
    if (filtro == NITIDEZ) return nitidezCPU(entrada, salida, ancho, alto, k);
    return laplacianoCPU(entrada, salida, ancho, alto, k);
}
