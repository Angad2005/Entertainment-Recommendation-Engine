#!/bin/bash

echo "Checking for CUDA availability..."
if command -v nvidia-smi &> /dev/null
then
    echo "NVIDIA GPU detected."
    nvidia-smi
else
    echo "NVIDIA GPU not detected or drivers not installed."
fi

echo "Installing dependencies..."
python3 -m pip install -r requirements.txt

echo "Setup complete."
