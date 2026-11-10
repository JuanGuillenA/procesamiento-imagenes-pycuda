// Mascara fija 3x3 que resalta detalles.
#include "mascaras.h"

std::vector<float> crearMascaraNitidez() {
    return {0, -1, 0,
            -1, 5, -1,
            0, -1, 0};
}
