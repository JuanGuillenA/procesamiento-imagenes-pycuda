// Funciones para convertir entre archivos PGM y arreglos float.
#pragma once
float* leerPGM(const char*, int*, int*);
void guardarPGM(const char*, const float*, int, int, bool);
