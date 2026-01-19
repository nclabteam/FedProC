# Analysis Tools Documentation

This document describes the analysis tools for processing federated learning experiment results.

---

# Experiment Results Analysis

## Overview

Generates comprehensive analysis tables from federated learning experiments with support for multiple metrics, pivot views, and ranking tables.

**Features:**
- Analyze 8 different metrics: loss, efficiency, communication, and 5 stability metrics
- Batch processing with "all" metric option to generate tables for all metrics at once
- Excel-driven batch runs: read multiple Excel files with per-row experiment targets
- Flexible pivot modes: group by model or strategy
- Ranking tables with performance-based ordering
- Unit conversion for time (s, ms, m, h) and bandwidth (b, kb, mb, gb, tb)
- Advanced filtering by models, strategies, datasets, output lengths, or experiment names
- Configurable aggregation modes (min, max, mean, last, median)
- Customizable output precision and standard deviation scaling
- Comprehensive metric descriptions displayed in table headers
- Missing experiments surfaced with their suggested run scripts

---

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

---

## Available Metrics

The analysis tool supports 8 metrics plus an "all" option for batch processing:

| Metric | Description | Interpretation |
|--------|-------------|----------------|
| **loss** | Test loss (model performance) | Average test loss across clients or global model. Lower = better model performance. Primary metric for evaluating model quality and convergence. |
| **efficiency** | Time per training round (computational efficiency) | Average time per federated learning round. Lower = faster training, more efficient system. Critical for deployment feasibility assessment. |
| **communication** | Total communication cost (bandwidth usage) | Combined uplink and downlink data transfer per round. Lower = less bandwidth usage, lower cost. Key metric for scalability and deployment cost. |
| **last_improvement_round** | Round with last improvement (convergence indicator) | Last round where model achieved better loss. Lower = early convergence; higher = continued improvement throughout training. |
| **longest_improvement_streak** | Maximum consecutive improvements (stability measure) | Longest sequence of consecutive improvements. Lower = erratic training with plateaus; higher = stable, consistent optimization. |
| **most_frequent_improvement_streak** | Most common improvement streak (training pattern) | Most common improvement streak length. Lower = short bursts with frequent stagnation; higher = sustained improvement patterns. |
| **oscillation_count** | Number of loss direction changes (instability measure) | Counts how many times loss direction changed. Lower = stable, monotonic improvement (ideal); higher = unstable training, optimization issues. |
| **improvement_ratio** | Fraction of rounds with improvement (training efficiency: 0.0-1.0) | Proportion of rounds with loss reduction. Values >0.5 = productive training; <0.3 = inefficient training, optimization problems. |
| **all** | Process all metrics in sequence | Batch process all 8 metrics at once, generating separate tables for each. |

---

## Usage Examples

```bash
# Basic loss analysis with default settings
python analysis/results.py

# Analyze computational efficiency with minute display
python analysis/results.py --metric efficiency --time-unit m

# Communication cost analysis with gigabyte display
python analysis/results.py --metric communication --size-unit gb

# Stability analysis: longest improvement streak
python analysis/results.py --metric longest_improvement_streak

# Batch process all metrics at once
python analysis/results.py --metric all

# Filter by models and pivot by strategy
python analysis/results.py --models Linear DLinear --pivot strategy

# High precision analysis with custom std multiplier
python analysis/results.py --std-multiplier 1000 --decimal-places 4

# Filter multiple criteria
python analysis/results.py --models DLinear --strategies FedAvg FedProx --datasets SolarEnergy

# Filter by output length
python analysis/results.py --output-lens 24 48 96

# Process specific experiments only
python analysis/results.py --experiments exp76 exp77 exp78

# Disable ranking tables
python analysis/results.py --no-ranking

# Higher is better metric (e.g., improvement_ratio)
python analysis/results.py --metric improvement_ratio --higher-is-better

# Different aggregation mode
python analysis/results.py --agg-mode median

# Enable verbose logging
python analysis/results.py --verbose

# Batch by Excel (rows with --project= and --name=; optional script column for rerun hints)
python analysis/results.py --excels .\scripts\FedProC.xlsx .\scripts\default.xlsx
```

---

## Excel Batch Processing

- Provide one or more Excel files via `--excels`. Each row must include columns `--project=` and `--name=` specifying the runs directory and experiment name. An optional `script` column is echoed in logs when an experiment is missing, so you can rerun it quickly.
- All experiments across all Excel files are aggregated into a single combined table per metric.
- Rows missing the required columns are skipped with a warning; unreadable files are logged as errors.
- When an experiment listed in Excel is absent, the log will show `MISSING EXPERIMENTS` with the experiment name and its `script` value if provided.

---

## Pivot Modes

The tool supports two pivot modes for organizing tables:

**Model Pivot (default):**
- Groups tables by model
- Columns represent different strategies
- Useful for comparing strategies within each model

**Strategy Pivot:**
- Groups tables by strategy
- Columns represent different models
- Useful for comparing models within each strategy

---

## Output Files

The analysis tool saves results to the `analysis/tables/` directory with descriptive filenames based on the metric and pivot mode.

**Generated Files:**

Tables are organized by pivot value (model or strategy name):
- `{pivot_value}_{metric}.csv`: Main analysis table with mean±std format
- `{pivot_value}_{metric}_ranking.csv`: Ranking table (if enabled)

**Examples:**
- `Linear_loss.csv` - Loss analysis for Linear model
- `Linear_efficiency.csv` - Efficiency analysis for Linear model  
- `FedAvg_communication.csv` - Communication cost for FedAvg strategy
- `DLinear_longest_improvement_streak_ranking.csv` - Ranking table for stability metric

When using `--metric all`, separate files are generated for each of the 8 metrics.

---

## Understanding Rankings

Ranking tables display performance comparisons across different configurations:

**Ranking Logic:**
1. **Primary Sort**: Mean metric value (lower is better by default, or higher if `--higher-is-better` is set)
2. **Tiebreaker**: Standard deviation (lower is better when means are equal)
3. **Best Column**: Shows which strategy/model achieved rank 1 for each configuration
4. **Average Rank Row**: Shows mean ranking across all configurations
5. **Most Frequent Winner**: Displayed in bottom-right cell showing overall best performer

**Interpreting Rankings:**
- **Rank 1**: Best performing strategy/model for that configuration
- **Lower average ranks**: Better overall consistency across configurations
- **Frequent winner**: Strategy/model that achieves rank 1 most often
- **N/A entries**: Indicates missing data for that configuration

---

## Filtering Options

The analysis tool supports comprehensive filtering to focus on specific experimental conditions:

- **Model Filtering**: `--models Linear DLinear` - Analyze only specified model architectures
- **Strategy Filtering**: `--strategies FedAvg FedProx` - Focus on specific federated learning strategies
- **Dataset Filtering**: `--datasets SolarEnergy ETDatasetHour` - Limit analysis to particular datasets
- **Output Length Filtering**: `--output-lens 24 48 96` - Filter by prediction horizon lengths
- **Experiment Filtering**: `--experiments exp76 exp77` - Process only specific named experiments

Filters can be combined to create precise analysis subsets. For example:
```bash
python analysis/results.py --models DLinear --strategies FedAvg --output-lens 96 --datasets ETDatasetHour
```

This flexibility allows researchers to perform targeted analysis on specific experimental conditions while maintaining consistent output formats and analysis methodologies.

---

# Inference Evaluation

## Overview

Evaluates federated learning models on datasets (same or different) with comprehensive metric computation, supporting both regular inference and zero-shot evaluation.

**Features:**
- Evaluate models on same training dataset (regular inference)
- Evaluate models on different datasets (zero-shot transfer evaluation)
- Both normalized and denormalized inference evaluation modes
- Per-client and aggregated metric computation
- Batch processing of multiple experiments and trials
- Comprehensive logging and progress tracking
- Both model best and last checkpoints
- Flexible dataset selection for comprehensive evaluation

---

## Command Line Arguments

| Argument           | Short | Type   | Default            | Description                                          |
|--------------------|-------|--------|-------------------|------------------------------------------------------|
| --runs-dir         | -r    | str    | "runs"             | Directory containing experiment folders              |
| --output-dir       | -o    | str    | "analysis/inference" | Output directory for inference results              |
| --experiments      |       | list   | Required           | Experiments to evaluate (e.g. exp1 exp2 exp3)       |
| --target-dataset   |       | str    | Required           | Target dataset for evaluation                        |
| --norm-mode        |       | choice | "both"             | Evaluation mode: norm, denorm, or both              |
| --verbose          | -v    | flag   | False              | Enable debug logging                                 |

---

## Usage Examples

```bash
# Zero-shot evaluation: Evaluate trained model on different dataset
python analysis/inference.py --experiments exp1 --target-dataset ETDatasetHour

# Regular inference: Evaluate model on same training dataset
python analysis/inference.py --experiments exp1 --target-dataset SolarEnergy

# Batch evaluate multiple experiments on same target dataset
python analysis/inference.py --experiments exp1 exp2 exp3 --target-dataset ETDatasetHour

# Evaluate with only denormalized metrics
python analysis/inference.py --experiments exp1 --target-dataset SolarEnergy --norm-mode denorm

# Evaluate with only normalized metrics (keep outputs normalized)
python analysis/inference.py --experiments exp1 --target-dataset SolarEnergy --norm-mode norm

# Custom output directory
python analysis/inference.py --experiments exp1 --target-dataset ETDatasetHour --output-dir my_results/inference

# Enable verbose logging for debugging
python analysis/inference.py --experiments exp1 --target-dataset ETDatasetHour --verbose
```

---

## Evaluation Modes

**Regular Inference:**
- Evaluate trained models on the same dataset they were trained on
- Measures final model performance and training effectiveness
- Target dataset matches source dataset in experiment config

**Zero-Shot Inference:**
- Evaluate trained models on different, unseen datasets
- Measures model generalization and transfer capabilities
- Target dataset differs from source dataset in experiment config
- No fine-tuning on target dataset

**Note:** When target_dataset equals source_dataset, this is regular inference. When different, this is zero-shot evaluation.
Both modes use the same inference pipeline with identical metric computation.

---

## Output Files

Inference results are saved to `analysis/inference/` directory with the following naming convention:
- `{experiment}_inference_{target_dataset}.csv`: Inference evaluation metrics

**CSV Columns:**
- `experiment`: Source experiment name
- `trial`: Trial number within the experiment
- `model_type`: Either "best" or "last" checkpoint
- `normalization`: Evaluation mode (norm or denorm)
- `target_dataset`: Dataset used for evaluation
- `source_dataset`: Original training dataset
- `input_len`: Input sequence length
- `output_len`: Prediction horizon
- `[metric_name]`: Computed metric (mean across clients)
- `[metric_name]_std`: Standard deviation of metric across clients

---

## Normalization Modes

**Normalized Mode (`--norm-mode norm`):**
- Input data is normalized using training statistics
- Output predictions remain in normalized space (no inverse transform)
- Useful for analyzing model prediction distributions

**Denormalized Mode (`--norm-mode denorm`):**
- Input data is normalized
- Output predictions are denormalized using training statistics
- Metrics computed on actual value scale
- More interpretable for domain experts

**Both Mode (`--norm-mode both`):**
- Evaluates in both norm and denorm modes
- Provides comprehensive comparison
- Useful for understanding model behavior across different scales

---

