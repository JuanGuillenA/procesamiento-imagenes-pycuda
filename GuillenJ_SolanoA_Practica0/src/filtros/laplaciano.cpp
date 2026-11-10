// Mascara fija 3x3 para detectar bordes.
#include "mascaras.h"

std::vector<float> crearMascaraLaplaciano() {
    return {0, 1, 0,
            1, -4, 1,
            0, 1, 0};
}
