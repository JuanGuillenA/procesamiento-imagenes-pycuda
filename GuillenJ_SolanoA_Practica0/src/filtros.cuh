// Funciones compartidas por los filtros CPU y CUDA.
#pragma once
#include "configuracion.h"
#include <vector>

struct TiemposGPU {
    float h2d, kernel, d2h, total;
};

std::vector<float> crearMascara(int filtro, int k);
void convolucionCPU(const float*, float*, const float*, int, int, int);
double ejecutarCPU(const float*, float*, const float*, int, int, int);
TiemposGPU ejecutarGPU(const float*, float*, const float*, float*, float*,
                       float*, int, int, int, int);
void compararCPUyGPU(const float*, const float*, int, double*, double*);
