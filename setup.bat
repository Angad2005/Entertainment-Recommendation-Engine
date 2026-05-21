@echo off
echo Checking for CUDA availability...
where nvidia-smi >nul 2>nul
if %errorlevel% equ 0 (
    echo NVIDIA GPU detected.
    nvidia-smi
) else (
    echo NVIDIA GPU not detected or drivers not installed.
)

echo Installing dependencies...
pip install -r requirements.txt

echo Setup complete.
pause
