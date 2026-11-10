// Lectura y escritura de imagenes PGM de 8 bits.
#include "imagenes.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>

float* leerPGM(const char* ruta, int* ancho, int* alto) {
    FILE* archivo = std::fopen(ruta, "rb");
    char tipo[3];
    int maximo;
    std::fscanf(archivo, "%2s %d %d %d", tipo, ancho, alto, &maximo);
    std::fgetc(archivo);

    int n = *ancho * *alto;
    auto bytes = (unsigned char*)std::malloc(n);
    auto datos = (float*)std::malloc(n * sizeof(float));
    std::fread(bytes, 1, n, archivo);
    std::fclose(archivo);
    for (int i = 0; i < n; i++) datos[i] = bytes[i];
    std::free(bytes);
    return datos;
}

// Convierte la salida float a una imagen PGM visible.
void guardarPGM(const char* ruta, const float* datos,
                int ancho, int alto, bool laplaciano) {
    int n = ancho * alto;
    auto bytes = (unsigned char*)std::malloc(n);
    for (int i = 0; i < n; i++) {
        float valor = laplaciano ? std::fabs(datos[i]) * 4 : datos[i];
        bytes[i] = (unsigned char)(valor < 0 ? 0 : valor > 255 ? 255 : valor);
    }

    FILE* archivo = std::fopen(ruta, "wb");
    std::fprintf(archivo, "P5\n%d %d\n255\n", ancho, alto);
    std::fwrite(bytes, 1, n, archivo);
    std::fclose(archivo);
    std::free(bytes);
}
