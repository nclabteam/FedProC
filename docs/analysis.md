# Analysis Tools Documentation

This directory contains analysis tools for processing federated learning experiment results.

---

## Shared Utilities

All argument parsing, experiment filtering, and ranking logic are now centralized in `utils/analysis.py`.  
The analysis scripts (`results.py`, `efficiency.py`, and `stability.py`) import and use these shared utilities to avoid code duplication.

**Key shared utilities:**
- `parse_args`: Unified command-line argument parser for all analysis scripts.
- `filter_experiments`: Unified experiment filtering by model, strategy, dataset, output length, experiment name, or Excel file.
- `create_ranking_table_from_pivot`, `create_avg_time_ranking_table_from_pivot`: Shared ranking logic for loss and efficiency analysis.
- Other helpers: experiment loading, config extraction, and more.

---

## Command Line Arguments

The following arguments are available for `results.py`, `efficiency.py`, and `stability.py` (see `parse_args` in `utils/analysis.py`):

| Argument         | Short | Type   | Default           | Description                                          |
|------------------|-------|--------|-------------------|------------------------------------------------------|
| --runs-dir       | -r    | str    | "runs"            | Directory containing experiment folders              |
| --output-dir     | -o    | str    | "analysis/tables" | Output directory for generated tables                |
| --table-type     | -t    | choice | "model-specific"  | Type of tables: model-specific, comparison, both     |
| --std-multiplier | -s    | float  | 10000             | Factor to multiply standard deviation for visibility |
| --decimal-places | -d    | int    | 3                 | Number of decimal places to display                  |
| --time-unit      |       | choice | "seconds"         | Time unit for efficiency analysis: seconds, minutes, hours |
| --no-display     |       | flag   | False             | Don't display tables to console, only save files     |
| --show-metadata  |       | flag   | False             | Display metadata table to console (always saved)     |
| --quiet          | -q    | flag   | False             | Reduce output verbosity                              |
| --models         |       | list   | None              | Filter to specific models (e.g. linear lstm)         |
| --strategies     |       | list   | None              | Filter to specific strategies (e.g. fedavg fedprox)  |
| --datasets       |       | list   | None              | Filter to specific datasets (e.g. stock crypto)      |
| --output-lens    |       | list   | None              | Filter to specific output lengths (e.g. 24 48 96)    |
| --experiments    |       | list   | None              | Process specific experiments (e.g. exp76 exp77)      |
| --excel          |       | str    | None              | Excel file to filter experiments (column '--name=' must contain experiment names) |

---

## results.py - Results Table Generator

Generates statistical analysis tables from federated learning experiment results with mean±std format and strategy rankings.

**Features:**
- Generate statistical tables from federated learning experiment results with mean±std format
- Create ranking tables showing strategy performance rankings (1=best, lower loss is better)
- Advanced tiebreaking: when win counts are equal, use average ranking as tiebreaker
- Filter experiments by models, strategies, datasets, output lengths, specific experiment names, or an Excel file
- Control output precision with customizable decimal places and standard deviation multipliers
- Create two table types: model-specific (combined mean±std) or comparison (separate mean/std tables)
- Save results to CSV with automatic file naming based on parameters
- Batch processing with quiet mode and optional console display control
- Handle multiple runs by calculating statistics across experimental repetitions
- Flexible input/output with customizable source and destination directories
- Automatic ranking with comprehensive tiebreaking logic

**Usage Examples:**

```bash
# Basic usage with defaults
python analysis/results.py

# Show metadata table in console and save to file
python analysis/results.py --show-metadata

# High precision with small std multiplier
python analysis/results.py --std-multiplier=1000 --decimal-places=4

# Filter specific models and generate both table types with metadata display
python analysis/results.py --models Linear DLinear --table-type both --show-metadata

# Quiet mode, save only, no console output
python analysis/results.py --no-display --quiet

# Filter by strategy and dataset with metadata display
python analysis/results.py --strategies FedAvg FedProx --datasets SolarEnergy --show-metadata

# Filter by output length
python analysis/results.py --output-lens 24 48 96

# Process specific experiments only with full output
python analysis/results.py --experiments exp76 exp77 exp78 --table-type both --show-metadata

# Combine multiple filters: dataset, Excel, and quiet mode
python analysis/results.py --datasets SolarEnergy ETDatasetHour --excel=scripts/default.xlsx --quiet
```

**Output Tables:**

For `model-specific` table type, the tool generates:
1. **Analysis Table**: Mean±std format for easy comparison
2. **Ranking Table**: Strategy rankings with performance ranks

**Ranking Table Features:**
- **Ranking Method**: Strategies ranked by mean performance (1=best, lower loss is better)
- **Primary Tiebreaker**: When means are equal, lower standard deviation wins
- **Best Strategy Column**: Shows which strategy achieved rank 1 for each configuration
- **Average Ranks**: Last row shows average rank across all configurations
- **Most Frequent Winner**: Bottom-right cell shows strategy that wins most often
- **Advanced Tiebreaking**: When win counts are tied, the strategy with lower average rank wins

---

## efficiency.py - Efficiency Table Generator

Generates tables and rankings for time efficiency (total and average time per iteration) for each strategy.

**Features:**
- Generate tables for total time and average time per iteration for each strategy
- Create ranking tables for both total time and average time per iteration
- **Time Unit Conversion**: Convert time values from seconds to minutes or hours for better readability
- All ranking logic and argument parsing is shared with `results.py` via `utils/analysis.py`
- Filter experiments by models, strategies, datasets, output lengths, experiment names, or Excel file
- Output precision and display options are fully configurable
- All tables and rankings are sorted by `dataset`, `in`, `out` for consistency

**Time Unit Options:**
- `--time-unit seconds`: Display raw time values in seconds (default)
- `--time-unit minutes`: Convert to minutes (time ÷ 60) for medium-duration experiments
- `--time-unit hours`: Convert to hours (time ÷ 3600) for long-running experiments

**Usage Examples:**

```bash
# Basic usage with default seconds
python analysis/efficiency.py

# Display times in minutes for better readability
python analysis/efficiency.py --time-unit minutes

# Display times in hours for very long experiments
python analysis/efficiency.py --time-unit hours

# Show both total and average time rankings with metadata
python analysis/efficiency.py --show-metadata --time-unit minutes

# Filter by model and dataset with time unit conversion
python analysis/efficiency.py --models=Linear --datasets=SolarEnergy --time-unit hours

# Filter by output length with minute display
python analysis/efficiency.py --output-lens 24 48 --time-unit minutes

# Save only, no console output, with time conversion
python analysis/efficiency.py --no-display --quiet --time-unit hours
```

**Output Tables:**
- **Total Time Table**: Shows total time used per configuration and strategy (in specified unit)
- **Avg Time Table**: Shows average time per iteration per configuration and strategy (in specified unit)
- **Ranking Table (Total Time)**: Ranks strategies by total time (lower is better), ties broken by avg time
- **Ranking Table (Avg Time)**: Ranks strategies by average time per iteration (lower is better)

**Time Unit Display:**
- Table headers include the time unit (e.g., "TOTAL TIME in minutes")
- All time values are converted automatically
- File names include time unit suffix when not using seconds (e.g., `linear_total_time_minutes.csv`)

---

## stability.py - Training Stability Analyzer

Analyzes convergence patterns and training stability metrics from loss sequences during federated learning experiments.

**Features:**
- Analyze convergence and stability patterns from training loss sequences
- Generate stability metrics tables with mean±std format across multiple runs
- Track improvement streaks, oscillations, and convergence points
- Filter experiments by models, strategies, datasets, output lengths, experiment names, or Excel file
- Configurable improvement threshold for determining meaningful progress
- Detailed explanations for each stability metric
- Save comprehensive stability analysis to CSV files

**Stability Metrics:**
- **Last Improvement Round**: Final round with meaningful improvement (convergence point)
- **Longest Improvement Streak**: Maximum consecutive rounds of improvement (stability measure)
- **Most Frequent Improvement Streak**: Most common improvement streak length (training pattern)
- **Oscillation Count**: Number of loss increases after decreases (instability measure)
- **Improvement Ratio**: Fraction of rounds with improvement (training efficiency: 0.0-1.0)

**Usage Examples:**

```bash
# Basic usage with default improvement threshold (1e-6)
python analysis/stability.py

# Show metadata table in console
python analysis/stability.py --show-metadata

# Filter by specific models and strategies
python analysis/stability.py --models DLinear --strategies FedAWA Elastic

# Filter by output length and dataset
python analysis/stability.py --output-lens 96 720 --datasets ETDatasetHour

# High precision analysis
python analysis/stability.py --decimal-places=4

# Quiet mode, save only
python analysis/stability.py --no-display --quiet

# Combine filters with Excel experiment list
python analysis/stability.py --excel=scripts/selected_experiments.xlsx --datasets SolarEnergy
```

**Interpreting Stability Metrics:**

- **Last Improvement Round**: 
  - Lower values = Early convergence (faster but may underfit)
  - Higher values = Late convergence (thorough learning but potentially inefficient)

- **Longest Improvement Streak**:
  - Lower values = Erratic training with frequent plateaus
  - Higher values = Stable, consistent optimization progress

- **Most Frequent Improvement Streak**:
  - Lower values = Training progresses in short bursts with frequent stagnation
  - Higher values = Sustained improvement patterns dominate the training

- **Oscillation Count**:
  - Lower values = Stable, monotonic improvement (ideal)
  - Higher values = Unstable training, possible learning rate issues

- **Improvement Ratio**:
  - Values >0.5 = Productive training, most rounds contributed to learning
  - Values <0.3 = Inefficient training, optimization problems

---

## Output Files

Analysis tools save results to the `analysis/tables/` directory with descriptive filenames that include parameter suffixes when non-default values are used.

**Generated Files:**

**Results Analysis (`results.py`):**
- `{model}_analysis{suffix}.csv`: Main analysis table with mean±std format
- `{model}_ranking{suffix}.csv`: Strategy ranking table with performance ranks
- `experiment_metadata{suffix}.csv`: Experiment details (always saved)

**Efficiency Analysis (`efficiency.py`):**
- `{model}_total_time{suffix}.csv`: Total time table
- `{model}_avg_time{suffix}.csv`: Average time per iteration table
- `{model}_efficiency_ranking{suffix}.csv`: Efficiency ranking table
- `efficiency_metadata{suffix}.csv`: Experiment details (always saved)

**Stability Analysis (`stability.py`):**
- `{model}_stability_raw{suffix}.csv`: Raw stability data for all metrics
- `{model}_{metric}_mean{suffix}.csv`: Mean values for each stability metric
- `{model}_{metric}_std{suffix}.csv`: Standard deviation values for each stability metric
- `stability_metadata{suffix}.csv`: Experiment details (always saved)

**Filename Suffixes:**
- `_stdx{value}`: Added when std_multiplier != 10000 (results.py, efficiency.py)
- `_dec{value}`: Added when decimal_places != 3 (all tools)
- `_{time_unit}`: Added when time_unit != "seconds" (efficiency.py only)
- `_stability`: Added for stability analysis files
- `_stability_dec{value}`: Added when decimal_places != 3 for stability analysis

**Note:** Metadata files are always saved to disk regardless of the `--show-metadata` flag. The flag only controls whether metadata is displayed in the console output.

---

## Understanding Rankings

**Ranking Logic (results.py and efficiency.py):**
1. **Primary Sort**: Mean loss value or total/avg time (ascending - lower is better)
2. **Performance Tiebreaker**: Standard deviation or avg time (ascending - lower is better)
3. **Best Strategy**: Strategy with rank 1 (lowest loss/time) for each configuration
4. **Average Rank**: Mean ranking across all configurations for each strategy
5. **Most Frequent Winner**: Strategy that achieves rank 1 most often across configurations
6. **Winner Tiebreaker**: When multiple strategies have the same win count, the one with the lowest average rank wins

**Interpreting Results:**
- **Rank 1**: Best performing strategy (lowest loss or time)
- **Lower average ranks**: Better overall performance consistency
- **Simple winner display**: `"FedAvg (3x)"` when there's a clear most frequent winner
- **Tied winner display**: `"FedAvg (tie: FedAvg(2x,avg:1.5), FedProx(2x,avg:1.8))"` showing all tied strategies with counts and average ranks
- **Robust strategies**: Low average rank with low standard deviation and frequent wins
- **Consistent performers**: Strategies that appear often in the "best_strategy" column with good average ranks

**Note:** `stability.py` does not generate ranking tables - it focuses on descriptive analysis of training dynamics with detailed explanations for each metric.

---

## Code Structure

- All core logic for argument parsing, experiment filtering, and ranking is in `utils/analysis.py`.
- `results.py` and `efficiency.py` are thin wrappers that call shared utilities and include ranking logic.
- `stability.py` uses shared utilities but focuses on stability analysis without rankings.
- This ensures maintainability and eliminates code duplication across analysis tools.

---

## Filtering Options

All analysis tools support comprehensive filtering to focus on specific experimental conditions:

- **Model Filtering**: `--models Linear DLinear` - Analyze only specified model architectures
- **Strategy Filtering**: `--strategies FedAWA Elastic FedProx` - Focus on specific federated learning strategies  
- **Dataset Filtering**: `--datasets SolarEnergy ETDatasetHour` - Limit analysis to particular datasets
- **Output Length Filtering**: `--output-lens 24 48 96` - Filter by prediction horizon lengths
- **Experiment Filtering**: `--experiments exp76 exp77` - Process only specific named experiments
- **Excel-based Filtering**: `--excel=scripts/experiments.xlsx` - Use Excel file to specify experiment list

Filters can be combined to create precise analysis subsets. For example:
```bash
python analysis/stability.py --models DLinear --strategies FedAWA --output-lens 96 --datasets ETDatasetHour
```

This flexibility allows researchers to perform targeted analysis on specific experimental conditions while maintaining consistent output formats and analysis methodologies across all tools.

---