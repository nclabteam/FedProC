#!/bin/bash

# ----------- USER CONFIGURATION ------------

# Base args without strategy or other tunable params (can keep static flags)
base_args="--iterations=500 --epochs=1 --times=5 --patience=20 --model=Linear --scaler=StandardScaler"

# Hyperparameter grid including strategy names
declare -A hyperparams
hyperparams[strategy]="Elastic"
hyperparams[tau]="0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9"
hyperparams[mu]="0.7 0.75 0.8 0.85 0.9 0.95 1.0"
hyperparams[sample_ratio]="0.2 0.3 0.4 0.5"

# ----------- SCRIPT START ------------------

# Generate Cartesian product of hyperparameters
generate_combinations() {
  local keys=("${!hyperparams[@]}")
  combinations=("")

  for key in "${keys[@]}"; do
    new_combinations=()
    for combo in "${combinations[@]}"; do
      for val in ${hyperparams[$key]}; do
        new_combinations+=("$combo --$key=$val")
      done
    done
    combinations=("${new_combinations[@]}")
  done
}

# Convert combo string to a clean run name: remove '--', replace spaces with '_', remove '='
make_run_name() {
  local combo_str="$1"
  combo_str=$(echo "$combo_str" | xargs)                    # trim spaces
  name=$(echo "$combo_str" | sed -E 's/--//g' | sed 's/=//g' | tr ' ' '_')
  echo "$name"
}

# ----------- SCRIPT START ------------------

generate_combinations

run_id=0

for combo in "${combinations[@]}"; do
  run_name=$(make_run_name "$combo")

  echo "=============================================="
  echo "Running with params: $combo"
  echo "=============================================="

  full_cmd="python ./main.py $base_args $combo --name=$run_name"
  eval "$full_cmd"

  echo ""
  ((run_id++))
done
