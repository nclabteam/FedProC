#!/bin/bash

set -e

cd "$(dirname "$0")/.."
FORMAT_PYTHON="${FORMAT_PYTHON:-python}"
PYTHON_FILES=()
while IFS= read -r -d '' PYTHON_FILE; do
    [[ -f "$PYTHON_FILE" ]] && PYTHON_FILES+=("$PYTHON_FILE")
done < <(
    git ls-files --cached --others --exclude-standard --deduplicate -z -- '*.py'
)

((${#PYTHON_FILES[@]})) || exit 0

"$FORMAT_PYTHON" -m autoflake \
    --remove-all-unused-imports \
    --ignore-init-module-imports \
    --in-place \
    "${PYTHON_FILES[@]}"
"$FORMAT_PYTHON" -m isort --profile=black "${PYTHON_FILES[@]}"
"$FORMAT_PYTHON" -m black --target-version=py310 "${PYTHON_FILES[@]}"

echo "Formatting completed."
