#!/bin/bash

# List of directories and files to format
TARGETS=(
    "data_factory"
    "layers"
    "losses"
    "models"
    "optimizers"
    "scalers"
    "schedulers"
    "strategies"
    "utils"
    "main.py"
    "analysis.py"
)

# Iterate over each target and apply isort and black
for TARGET in "${TARGETS[@]}"; do
    echo "Formatting $TARGET..."
    isort --profile=black "$TARGET"
    black "$TARGET"
    echo "=========================================="
done

echo "Formatting completed."