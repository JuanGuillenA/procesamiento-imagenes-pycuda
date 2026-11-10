// Filtro de media en CPU.
#include "filtros_secuenciales.h"
#include "../filtros/mascaras.h"

double mediaCPU(const float* entrada, float* salida,
                int ancho, int alto, int k) {
    auto mascara = crearMascaraMedia(k);
    return ejecutarCPU(entrada, salida, mascara.data(), ancho, alto, k);
}
