@echo off
setlocal

where py >nul 2>nul
if %errorlevel%==0 (
  py -3 "%~dp0launch_colmena_chile.py" %*
) else (
  python "%~dp0launch_colmena_chile.py" %*
)

endlocal
