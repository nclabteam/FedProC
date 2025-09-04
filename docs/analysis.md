# Analysis Tools Documentation

This directory contains analysis tools for processing federated learning experiment results.

---

## Shared Utilities

All argument parsing, experiment filtering, and ranking logic are now centralized in `utils/analysis.py`.  
Both `results.py` and `efficiency.py` import and use these shared utilities to avoid code duplication.

**Key shared utilities:**
- `parse_args`: Unified command-line argument parser for all analysis scripts.
- `filter_experiments`: Unified experiment filtering by model, strategy, dataset, experiment name, or Excel file.
- `create_ranking_table_from_pivot`, `create_avg_time_ranking_table_from_pivot`: Shared ranking logic for both loss and efficiency analysis.
- Other helpers: experiment loading, config extraction, and more.

---

## Command Line Arguments

The following arguments are available for both `results.py` and `efficiency.py` (see `parse_args` in `utils/analysis.py`):

| Argument         | Short | Type   | Default           | Description                                          |
|------------------|-------|--------|-------------------|------------------------------------------------------|
| --runs-dir       | -r    | str    | "runs"            | Directory containing experiment folders              |
| --output-dir     | -o    | str    | "analysis/tables" | Output directory for generated tables                |
| --table-type     | -t    | choice | "model-specific"  | Type of tables: model-specific, comparison, both     |
| --std-multiplier | -s    | float  | 10000             | Factor to multiply standard deviation for visibility |
| --decimal-places | -d    | int    | 3                 | Number of decimal places to display                  |
| --no-display     |       | flag   | False             | Don't display tables to console, only save files     |
| --show-metadata  |       | flag   | False             | Display metadata table to console (always saved)     |
| --quiet          | -q    | flag   | False             | Reduce output verbosity                              |
| --models         |       | list   | None              | Filter to specific models (e.g. linear lstm)         |
| --strategies     |       | list   | None              | Filter to specific strategies (e.g. fedavg fedprox)  |
| --datasets       |       | list   | None              | Filter to specific datasets (e.g. stock crypto)      |
| --experiments    |       | list   | None              | Process specific experiments (e.g. exp76 exp77)      |
| --excel          |       | str    | None              | Excel file to filter experiments (column '--name=' must contain experiment names) |

---

## results.py - Results Table Generator

Generates statistical analysis tables from federated learning experiment results with mean±std format and strategy rankings.

**Features:**
- Generate statistical tables from federated learning experiment results with mean±std format
- Create ranking tables showing strategy performance rankings (1=best, lower loss is better)
- Advanced tiebreaking: when win counts are equal, use average ranking as tiebreaker
- Filter experiments by models, strategies, datasets, specific experiment names, or an Excel file
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
- All ranking logic and argument parsing is shared with `results.py` via `utils/analysis.py`
- Filter experiments by models, strategies, datasets, experiment names, or Excel file
- Output precision and display options are fully configurable
- All tables and rankings are sorted by `dataset`, `in`, `out` for consistency

**Usage Examples:**

```bash
# Basic usage
python analysis/efficiency.py

# Show both total and average time rankings
python analysis/efficiency.py --show-metadata

# Filter by model and dataset
python analysis/efficiency.py --models=Linear --datasets=SolarEnergy

# Save only, no console output
python analysis/efficiency.py --no-display --quiet
```

**Output Tables:**
- **Total Time Table**: Shows total time used per configuration and strategy
- **Avg Time Table**: Shows average time per iteration per configuration and strategy
- **Ranking Table (Total Time)**: Ranks strategies by total time (lower is better), ties broken by avg time
- **Ranking Table (Avg Time)**: Ranks strategies by average time per iteration (lower is better)

---

## Output Files

Analysis tools save results to the `analysis/tables/` directory with descriptive filenames that include parameter suffixes when non-default values are used.

**Generated Files:**
- `{model}_analysis{suffix}.csv`: Main analysis table with mean±std format (results.py)
- `{model}_ranking{suffix}.csv`: Strategy ranking table with performance ranks (results.py)
- `{model}_total_time{suffix}.csv`: Total time table (efficiency.py)
- `{model}_avg_time{suffix}.csv`: Average time per iteration table (efficiency.py)
- `{model}_efficiency_ranking{suffix}.csv`: Efficiency ranking table (efficiency.py)
- `experiment_metadata{suffix}.csv` or `efficiency_metadata{suffix}.csv`: Experiment details (always saved)

**Filename Suffixes:**
- `_stdx{value}`: Added when std_multiplier != 10000
- `_dec{value}`: Added when decimal_places != 3

**Note:** Metadata files are always saved to disk regardless of the `--show-metadata` flag. The flag only controls whether metadata is displayed in the console output.

---

## Understanding Rankings

**Ranking Logic:**
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

---

## Code Structure

- All core logic for argument parsing, experiment filtering, and ranking is in `utils/analysis.py`.
- Both `results.py` and `efficiency.py` are thin wrappers that call shared utilities.
- This ensures maintainability and eliminates code duplication.

---