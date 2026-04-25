@echo off
echo ===============================
echo Deploy a GitHub / Vercel
echo ===============================

echo.
echo Agregando cambios...
git add .

echo.
echo Commit...
git commit -m "Auto deploy %date% %time%"

echo.
echo Sincronizando con GitHub...
git pull origin main --allow-unrelated-histories

echo.
echo Subiendo a GitHub...
git push origin main

echo.
echo ===============================
echo Deploy enviado a Vercel
echo ===============================
echo Espera 1-2 minutos y recarga con CTRL+F5
pause