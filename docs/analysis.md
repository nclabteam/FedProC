# Analysis Tools Documentation

This directory contains analysis tools for processing federated learning experiment results.

## Available Tools

### results.py - Results Table Generator

Generates statistical analysis tables from federated learning experiment results with mean±std format and strategy rankings.

**Features:**
- Generate statistical tables from federated learning experiment results with mean±std format
- Create ranking tables showing strategy performance rankings (1=best, lower loss is better)
- Filter experiments by models, strategies, datasets, or specific experiment names
- Control output precision with customizable decimal places and standard deviation multipliers
- Create two table types: model-specific (combined mean±std) or comparison (separate mean/std tables)
- Save results to CSV with automatic file naming based on parameters
- Batch processing with quiet mode and optional console display control
- Handle multiple runs by calculating statistics across experimental repetitions
- Flexible input/output with customizable source and destination directories
- Automatic ranking with tiebreaking by standard deviation

**Command Line Arguments:**

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

**Usage Examples:**

```bash
# Basic usage with defaults
python analysis/results.py

# Show metadata table in console and save to file
python analysis/results.py --show-metadata

# High precision with small std multiplier
python analysis/results.py --std-multiplier=1000 --decimal-places=4

# Filter specific models and generate both table types with metadata display
python analysis/results.py --models linear lstm --table-type both --show-metadata

# Quiet mode, save only, no console output
python analysis/results.py --no-display --quiet

# Filter by strategy and dataset with metadata display
python analysis/results.py --strategies fedavg fedprox --datasets stock --show-metadata

# Process specific experiments only with full output
python analysis/results.py --experiments exp76 exp77 exp78 --table-type both --show-metadata
```

## Output Files

Analysis tools typically save results to the `analysis/tables/` directory with descriptive filenames that include parameter suffixes when non-default values are used.

**Generated Files:**
- `{model}_analysis{suffix}.csv`: Main analysis table with mean±std format
- `{model}_ranking{suffix}.csv`: Strategy ranking table with performance ranks
- `experiment_metadata{suffix}.csv`: Experiment details (always saved)

**Filename Suffixes:**
- `_stdx{value}`: Added when std_multiplier != 10000
- `_dec{value}`: Added when decimal_places != 3

**Note:** Metadata files are always saved to disk regardless of the `--show-metadata` flag. The flag only controls whether metadata is displayed in the console output.

## Understanding Rankings

**Ranking Logic:**
1. **Primary Sort**: Mean loss value (ascending - lower is better)
2. **Tiebreaker**: Standard deviation (ascending - lower variability is better)
3. **Best Strategy**: Strategy with rank 1 (lowest loss) for each configuration
4. **Average Rank**: Mean ranking across all configurations for each strategy
5. **Most Frequent Winner**: Strategy that achieves rank 1 most often across configurations

**Interpreting Results:**
- **Rank 1**: Best performing strategy (lowest loss)
- **Lower ranks**: Better average performance
- **Consistent winners**: Strategies appearing frequently in "best_strategy" column
- **Robust strategies**: Low average rank with low standard deviation