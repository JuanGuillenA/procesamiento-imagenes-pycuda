// Filtro laplaciano en CPU.
#include "filtros_secuenciales.h"
#include "../filtros/mascaras.h"

double laplacianoCPU(const float* entrada, float* salida,
                     int ancho, int alto, int k) {
    auto mascara = crearMascaraLaplaciano();
    return ejecutarCPU(entrada, salida, mascara.data(), ancho, alto, k);
}
