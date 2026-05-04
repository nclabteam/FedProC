# Analysis Tools Documentation

Analysis tools for processing federated learning experiment results.

```
analysis/
├── metrics.py      # Pure functions: list[float] -> value
├── io.py           # Read CSV/JSON, unit conversion
├── single.py       # Per-experiment: runs/expN/ -> results.csv
├── multi.py        # Cross-experiment: pivot/ranking tables
├── landscape.py    # Loss surface visualization (1D/2D/3D)
├── datasets.py     # Dataset statistics visualization
├── inference.py    # Model evaluation (zero-shot = special case)
├── cli.py          # Unified entry point
└── tables/         # Output directory for multi.py
```

## Data Flow

```
Training (main.py):
  server.train() -> server.csv, client_*.csv per run
  main.py -> timing.json

Per-experiment (single.py):
  runs/exp19/{0,1,2}/results/server.csv -> runs/exp19/results.csv

Cross-experiment (multi.py):
  runs/{exp1,exp2,exp3}/results.csv -> analysis/tables/*.csv

Loss landscape (landscape.py):
  runs/exp19/0/models/server_best.pt -> runs/exp19/landscape/run_0_server_{1d,2d,3d}.png
```

---

# Per-Experiment Analysis (single.py)

Aggregates results across all runs of a single experiment.

## Usage

```bash
# Analyze a single experiment
python -m analysis single --experiment runs/exp19

# Or run the module directly
python -m analysis.single --experiment runs/exp19
```

## Output

Produces `runs/exp19/results.csv` with one row per metric:

```
metric,avg_min,std_min,avg_max,std_max
global_avg_test_loss,0.3406,0.0,0.3406,0.0
efficiency,5.3805,0.0,5.3805,0.0
communication,0.2842,0.0,0.2842,0.0
...
```

## Metrics

| Metric | Description |
|--------|-------------|
| global_avg_test_loss | Average test loss across clients |
| global_avg_train_loss | Average train loss across clients |
| efficiency | Time per training round |
| communication | Total bandwidth per round |
| uplink | Data sent from clients to server |
| downlink | Data sent from server to clients |
| last_improvement_round | Last round with loss improvement |
| longest_improvement_streak | Max consecutive improvements |
| most_frequent_improvement_streak | Most common streak length |
| oscillation_count | Number of loss direction changes |
| improvement_ratio | Fraction of rounds with improvement (0-1) |
| improvement_magnitude | Average loss reduction per improvement |
| time_per_experiment | Wall-clock time per run (from timing.json) |

---

# Cross-Experiment Analysis (multi.py)

Compares results across multiple experiments. Reads `results.csv` produced by `single.py`.

## Usage

```bash
# Compare experiments on a metric
python -m analysis multi --runs-dir runs --metric loss

# Pivot by strategy instead of model
python -m analysis multi --runs-dir runs --metric loss --pivot strategy

# Batch all metrics
python -m analysis multi --runs-dir runs --metric all

# Filter specific experiments
python -m analysis multi --runs-dir runs --metric loss --models DLinear --strategies FedAvg

# Excel-driven batch runs
python -m analysis multi --excels scripts/setting1.xlsx --metric all
```

## Command Line Arguments

| Argument           | Short | Type   | Default           | Description                                          |
|--------------------|-------|--------|-------------------|------------------------------------------------------|
| --runs-dir         | -r    | str    | "runs"            | Directory containing experiment folders              |
| --output-dir       | -o    | str    | "analysis/tables" | Output directory for generated tables                |
| --std-multiplier   | -s    | float  | 1000              | Factor to multiply standard deviation for visibility |
| --decimal-places   | -d    | int    | 3                 | Number of decimal places to display                  |
| --agg-mode         | -a    | choice | "min"             | Aggregation mode: min, max, mean, last, median       |
| --time-unit        | -t    | choice | "s"               | Time unit: s (seconds), ms, m (minutes), h (hours)   |
| --size-unit        | -z    | choice | "mb"              | Size unit: b (bytes), kb, mb, gb, tb                 |
| --metric           | -m    | choice | "loss"            | Metric to analyze (or "all" for batch processing)    |
| --pivot            | -p    | choice | "model"           | Pivot mode: model or strategy                        |
| --no-ranking       |       | flag   | False             | Disable ranking table generation                     |
| --higher-is-better |       | flag   | False             | Higher metric values are better (default: lower)     |
| --verbose          | -v    | flag   | False             | Enable debug logging                                 |
| --excels           | -e    | list   | None              | One or more Excel files providing batch queries      |
| --models           |       | list   | None              | Filter to specific models (e.g. Linear DLinear)      |
| --strategies       |       | list   | None              | Filter to specific strategies (e.g. FedAvg FedProx)  |
| --datasets         |       | list   | None              | Filter to specific datasets (e.g. SolarEnergy)       |
| --output-lens      |       | list   | None              | Filter to specific output lengths (e.g. 24 48 96)    |
| --experiments      |       | list   | None              | Process specific experiments (e.g. exp76 exp77)      |

## Pivot Modes

**Model Pivot (default):**
- Groups tables by model
- Columns represent different strategies
- Useful for comparing strategies within each model

**Strategy Pivot:**
- Groups tables by strategy
- Columns represent different models
- Useful for comparing models within each strategy

## Output Files

Tables are saved to `analysis/tables/` organized by pivot value:
- `{pivot_value}_{metric}.csv`: Main analysis table with mean+/-std format
- `{pivot_value}_{metric}_ranking.csv`: Ranking table (if enabled)

Examples:
- `DLinear_loss.csv` - Loss analysis for DLinear model
- `FedAvg_communication.csv` - Communication cost for FedAvg strategy

## Rankings

Ranking tables display performance comparisons:
1. **Primary Sort**: Mean metric value (lower is better by default)
2. **Tiebreaker**: Standard deviation (lower is better when means are equal)
3. **Best Column**: Shows rank 1 strategy/model for each configuration
4. **Average Rank Row**: Mean ranking across all configurations

## Excel Batch Processing

Provide Excel files via `--excels`. Each row must include `--project=` and `--name=` columns. An optional `script` column is echoed when an experiment is missing.

---

# Loss Landscape (landscape.py)

Visualizes the loss surface around trained model weights by perturbing along random directions.

## Usage

```bash
# Best run (lowest loss) - server model
python -m analysis landscape --experiment runs/exp19

# Specific run
python -m analysis landscape --experiment runs/exp19 --run 0

# All runs
python -m analysis landscape --experiment runs/exp19 --all

# Specific client (personalized FL)
python -m analysis landscape --experiment runs/exp19 --client 0

# All clients (personalized FL)
python -m analysis landscape --experiment runs/exp19 --all-clients
```

## Two FL Modes

**Traditional FL** (`save_local_model=False`):
- One global server model
- Generates: `run_{idx}_server_{1d,2d,3d}.png`

**Personalized FL** (`save_local_model=True`):
- Per-client models
- Generates: `run_{idx}_client_{NNN}_{1d,2d,3d}.png`
- Use `--client N` for a specific client, `--all-clients` for all

## Plot Types

| Type | Description |
|------|-------------|
| 1D | Loss along a single random direction (line plot) |
| 2D | Loss along two random directions (contour plot) |
| 3D | Loss surface along two directions (3D surface) |

## Model Selection

- Default: `server_best.pt` from the run with lowest min loss (from `results.csv`)
- `--run N`: use run N's `server_best.pt`
- `--client N`: use `client_N_best.pt` (personalized FL)

## Output

Saves to `runs/expN/landscape/` (experiment-level directory, survives compact):

```
runs/exp19/landscape/
├── run_0_server_1d.png
├── run_0_server_2d.png
├── run_0_server_3d.png
├── run_0_client_000_1d.png   # personalized FL only
├── run_0_client_000_2d.png
├── run_0_client_000_3d.png
└── ...
```

---

# Inference Evaluation (inference.py)

Evaluates trained models on datasets (same or different).

## Usage

```bash
# Regular inference (same dataset as training)
python -m analysis inference --experiments exp1 --target-dataset SolarEnergy

# Zero-shot evaluation (different dataset)
python -m analysis inference --experiments exp1 --target-dataset ETDatasetHour

# Batch evaluate multiple experiments
python -m analysis inference --experiments exp1 exp2 exp3 --target-dataset ETDatasetHour

# Denormalized metrics only
python -m analysis inference --experiments exp1 --target-dataset SolarEnergy --norm-mode denorm
```

## Arguments

| Argument           | Short | Type   | Default              | Description                                    |
|--------------------|-------|--------|----------------------|------------------------------------------------|
| --runs-dir         | -r    | str    | "runs"               | Directory containing experiment folders        |
| --output-dir       | -o    | str    | "analysis/inference" | Output directory for inference results         |
| --experiments      |       | list   | Required             | Experiments to evaluate                        |
| --target-dataset   |       | str    | Required             | Target dataset for evaluation                  |
| --norm-mode        |       | choice | "both"               | Evaluation mode: norm, denorm, or both         |
| --verbose          | -v    | flag   | False                | Enable debug logging                           |

## Zero-Shot vs Regular

- **Regular inference**: target dataset = training dataset (measures final performance)
- **Zero-shot inference**: target dataset != training dataset (measures generalization)

Both use the same pipeline; the distinction is purely whether the datasets match.

## Output

Saved to `analysis/inference/`:
- `{experiment}_inference_{target_dataset}.csv`

---

# Unified CLI

All analysis tools accessible via `python -m analysis`:

```bash
python -m analysis single --experiment runs/exp19
python -m analysis multi --runs-dir runs --metric loss
python -m analysis landscape --experiment runs/exp19
python -m analysis inference --experiments exp1 --target-dataset ETDatasetHour
```

---

# Internal Modules

## metrics.py

Pure functions for computing metrics from raw value lists. No I/O, fully testable.

```python
from analysis.metrics import compute_per_run_agg, last_improvement_round, improvement_streaks

val = compute_per_run_agg([0.5, 0.4, 0.3], mode="min")  # 0.3
round = last_improvement_round([0.5, 0.4, 0.35, 0.36])  # 2
```

## io.py

Shared I/O helpers: file reading, unit conversion, numeric parsing.

```python
from analysis.io import read_csv, convert_time, parse_numeric_list

data = read_csv("runs/exp19/0/results/server.csv")
secs = convert_time(1500, "ms", "s")  # 1.5
vals = parse_numeric_list(["0.5", "N/A", "0.3"])  # [0.5, 0.3]
```
