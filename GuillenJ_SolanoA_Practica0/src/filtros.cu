// Convolucion comun para CPU y CUDA, mas la medicion de tiempos.
#include "filtros.cuh"
#include "filtros/mascaras.h"

#include <chrono>
#include <cmath>
#include <cuda_runtime.h>

const char* NOMBRE_FILTRO[] = {"media", "nitidez", "laplaciano"};

// Escoge los coeficientes del filtro solicitado.
std::vector<float> crearMascara(int filtro, int k) {
    if (filtro == MEDIA) return crearMascaraMedia(k);
    return filtro == NITIDEZ ? crearMascaraNitidez()
                             : crearMascaraLaplaciano();
}

// Version secuencial: recorre todos los pixeles y sus vecinos.
void convolucionCPU(const float* entrada, float* salida,
                    const float* mascara, int ancho, int alto, int k) {
    int radio = k / 2;
    for (int y = 0; y < alto; y++) {
        for (int x = 0; x < ancho; x++) {
            float suma = 0;
            for (int j = -radio; j <= radio; j++) {
                for (int i = -radio; i <= radio; i++) {
                    int yy = y + j, xx = x + i;
                    if (yy >= 0 && yy < alto && xx >= 0 && xx < ancho) {
                        suma += entrada[yy * ancho + xx] *
                                mascara[(j + radio) * k + i + radio];
                    }
                }
            }
            salida[y * ancho + x] = suma;
        }
    }
}

// Kernel CUDA: un hilo calcula un pixel de la imagen.
__global__ void convolucionGPU(const float* entrada, float* salida,
                               const float* mascara, int ancho, int alto,
                               int k) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    int n = ancho * alto;
    if (id >= n) return;

    int y = id / ancho, x = id % ancho, radio = k / 2;
    float suma = 0;
    for (int j = -radio; j <= radio; j++) {
        for (int i = -radio; i <= radio; i++) {
            int yy = y + j, xx = x + i;
            if (yy >= 0 && yy < alto && xx >= 0 && xx < ancho) {
                suma += entrada[yy * ancho + xx] *
                        mascara[(j + radio) * k + i + radio];
            }
        }
    }
    salida[id] = suma;
}

// Mide solamente la convolucion secuencial.
double ejecutarCPU(const float* entrada, float* salida,
                   const float* mascara, int ancho, int alto, int k) {
    auto inicio = std::chrono::steady_clock::now();
    convolucionCPU(entrada, salida, mascara, ancho, alto, k);
    return std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - inicio).count();
}

// Mide transferencia CPU-GPU, kernel y transferencia GPU-CPU.
TiemposGPU ejecutarGPU(const float* entrada, float* salida,
                       const float* mascara, float* dEntrada,
                       float* dSalida, float* dMascara, int ancho,
                       int alto, int k, int hilos) {
    int n = ancho * alto;
    int bloques = (n + hilos - 1) / hilos;
    size_t bytesImagen = n * sizeof(float);
    size_t bytesMascara = k * k * sizeof(float);
    cudaEvent_t e[4];

    for (auto& evento : e) cudaEventCreate(&evento);
    cudaEventRecord(e[0]);
    cudaMemcpy(dEntrada, entrada, bytesImagen, cudaMemcpyHostToDevice);
    cudaMemcpy(dMascara, mascara, bytesMascara, cudaMemcpyHostToDevice);
    cudaEventRecord(e[1]);

    convolucionGPU<<<bloques, hilos>>>(dEntrada, dSalida, dMascara,
                                       ancho, alto, k);
    cudaEventRecord(e[2]);
    cudaMemcpy(salida, dSalida, bytesImagen, cudaMemcpyDeviceToHost);
    cudaEventRecord(e[3]);
    cudaEventSynchronize(e[3]);

    TiemposGPU t;
    cudaEventElapsedTime(&t.h2d, e[0], e[1]);
    cudaEventElapsedTime(&t.kernel, e[1], e[2]);
    cudaEventElapsedTime(&t.d2h, e[2], e[3]);
    cudaEventElapsedTime(&t.total, e[0], e[3]);
    for (auto& evento : e) cudaEventDestroy(evento);
    return t;
}

// Obtiene el error maximo y el error medio entre CPU y GPU.
void compararCPUyGPU(const float* cpu, const float* gpu, int n,
                     double* maximo, double* medio) {
    *maximo = *medio = 0;
    for (int i = 0; i < n; i++) {
        double error = std::fabs(cpu[i] - gpu[i]);
        if (error > *maximo) *maximo = error;
        *medio += error;
    }
    *medio /= n;
}
