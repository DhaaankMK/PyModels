@echo off
cd /d "%~dp0"
echo Instalando dependencias do PyModels Studio...
python -m pip install -r requirements.txt
if errorlevel 1 (
  echo.
  echo Falha ao instalar as dependencias. Verifique se o Python e o pip estao instalados.
  pause
  exit /b 1
)
echo.
echo Instalacao concluida. Iniciando PyModels Studio...
python PyModels.py
pause
