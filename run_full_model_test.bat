@echo off
rem run_full_model_test.bat - end-to-end test of the full_model process (tiny settings, CPU, a few minutes).
rem Needs the data (bash get_data.sh) and the feature table (python experiments\common\extract_spectral_features.py) first.
rem Usage:  run_full_model_test.bat                 (CPU, seed 42)
rem         run_full_model_test.bat 7               (seed 7; same as --seed 7; results in full_model\results_test_s7)
rem         run_full_model_test.bat --seed 7 --device cuda
rem         set SEED=7 ^& run_full_model_test.bat   (seed from the environment)
rem Writes only to full_model\results_test, full_model\splits_test, full_model\cache and full_model\logs (all git-ignored).
setlocal
cd /d "%~dp0"

set "PY=.venv\Scripts\python.exe"
if not exist "%PY%" set "PY=python"
"%PY%" -c "import sys; print('python', sys.version.split()[0])" || (echo Python not found. Create the venv: python -m venv .venv ^&^& .venv\Scripts\pip install -r requirements.txt & exit /b 1)

"%PY%" -c "import torch, optuna, sklearn, scipy, soundfile, onnx, onnxruntime, joblib, pandas" 2>nul
if errorlevel 1 (
  echo Missing packages: installing requirements.txt ...
  "%PY%" -m pip install -r requirements.txt || (echo pip install failed & exit /b 1)
)

set "ARGS=%*"
echo %1| findstr /r "^[0-9][0-9]*$" >nul && set "ARGS=--seed %*"
if defined SEED if "%~1"=="" set "ARGS=--seed %SEED%"
"%PY%" full_model\selftest.py %ARGS%
set "RC=%ERRORLEVEL%"
if "%RC%"=="0" (echo. & echo RESULT: PASS) else (echo. & echo RESULT: FAIL, exit code %RC% - see full_model\logs\selftest_*.log)
exit /b %RC%
