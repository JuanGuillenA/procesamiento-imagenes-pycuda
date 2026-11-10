// Parametros generales de las mediciones y filtros.
#pragma once
constexpr int REPETICIONES = 10;
constexpr double SEGUNDOS_CALENTAMIENTO_GPU = 2.0;
constexpr float TOLERANCIA = 1e-2f;
enum Filtro { MEDIA = 0, NITIDEZ = 1, LAPLACIANO = 2 };
extern const char* NOMBRE_FILTRO[];
extern const char* NOMBRE_FILTRO[];
