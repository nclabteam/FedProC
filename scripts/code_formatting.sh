#!/bin/bash

# List of directories and files to format
TARGETS=(
    "analysis"
    "data_factory"
    "layers"
    "losses"
    "models"
    "optimizers"
    "scalers"
    "schedulers"
    "scripts"
    "strategies"
    "utils"
    "topologies"
    "main.py"
)

for TARGET in "${TARGETS[@]}"; do
    echo "Formatting $TARGET..."
    autoflake --remove-all-unused-imports --in-place --recursive "$TARGET"
    isort --profile=black "$TARGET"
    black "$TARGET"
    echo "=========================================="
done

echo "Formatting completed."