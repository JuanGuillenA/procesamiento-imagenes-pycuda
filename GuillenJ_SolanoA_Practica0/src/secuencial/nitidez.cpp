// Filtro de nitidez en CPU.
#include "filtros_secuenciales.h"
#include "../filtros/mascaras.h"

double nitidezCPU(const float* entrada, float* salida,
                  int ancho, int alto, int k) {
    auto mascara = crearMascaraNitidez();
    return ejecutarCPU(entrada, salida, mascara.data(), ancho, alto, k);
}
