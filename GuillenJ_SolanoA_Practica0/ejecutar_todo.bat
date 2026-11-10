@echo off
REM Flujo completo: prepara imagenes, compila CUDA y ejecuta la practica.
REM Uso: ejecutar_todo.bat
cd /d "%~dp0"
python scripts\preparar_imagenes.py
call "%~dp0compilar.bat"

echo Ejecutando pruebas y experimentos (tarda unos minutos)...
convolucion.exe
echo.
echo Practica terminada. Revise la carpeta resultados.
