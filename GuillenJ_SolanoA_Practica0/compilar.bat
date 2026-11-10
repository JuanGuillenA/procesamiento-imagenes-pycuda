@echo off
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" >nul
cd /d "%~dp0"
nvcc -O3 -arch=sm_86 -Isrc -o convolucion.exe main.cu src\imagenes.cpp src\filtros.cu src\experimentos.cu src\filtros\media.cpp src\filtros\nitidez.cpp src\filtros\laplaciano.cpp src\secuencial\media.cpp src\secuencial\nitidez.cpp src\secuencial\laplaciano.cpp src\paralelo\media.cu src\paralelo\nitidez.cu src\paralelo\laplaciano.cu
del /q convolucion.exp convolucion.lib 2>nul
