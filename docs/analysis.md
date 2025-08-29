# Analysis Tools Documentation

This directory contains analysis tools for processing federated learning experiment results.

## Available Tools

### results.py - Results Table Generator

Generates statistical analysis tables from federated learning experiment results with mean±std format.

**Features:**
- Generate statistical tables from federated learning experiment results with mean±std format
- Filter experiments by models, strategies, datasets, or specific experiment names
- Control output precision with customizable decimal places and standard deviation multipliers
- Create two table types: model-specific (combined mean±std) or comparison (separate mean/std tables)
- Save results to CSV with automatic file naming based on parameters
- Batch processing with quiet mode and optional console display control
- Handle multiple runs by calculating statistics across experimental repetitions
- Flexible input/output with customizable source and destination directories

**Command Line Arguments:**

| Argument         | Short | Type   | Default           | Description                                          |
|------------------|-------|--------|-------------------|----------------------------------------------------- |
| --runs-dir       | -r    | str    | "runs"            | Directory containing experiment folders              |
| --output-dir     | -o    | str    | "analysis/tables" | Output directory for generated tables                |
| --table-type     | -t    | choice | "model-specific"  | Type of tables: model-specific, comparison, both     |
| --std-multiplier | -s    | float  | 10000             | Factor to multiply standard deviation for visibility |
| --decimal-places | -d    | int    | 3                 | Number of decimal places to display                  |
| --no-display     |       | flag   | False             | Don't display tables to console, only save files     |
| --no-metadata    |       | flag   | False             | Don't display or save metadata table                 |
| --quiet          | -q    | flag   | False             | Reduce output verbosity                              |
| --models         |       | list   | None              | Filter to specific models (e.g. linear lstm)         |
| --strategies     |       | list   | None              | Filter to specific strategies (e.g. fedavg fedprox)  |
| --datasets       |       | list   | None              | Filter to specific datasets (e.g. stock crypto)      |
| --experiments    |       | list   | None              | Process specific experiments (e.g. exp76 exp77)      |

**Usage Examples:**

```bash
# Basic usage with defaults
python analysis/results.py

# High precision with small std multiplier
python analysis/results.py --std-multiplier=1000 --decimal-places=4

# Filter specific models and generate both table types
python analysis/results.py --models linear lstm --table-type=both

# Quiet mode, save only, no console output
python analysis/results.py --no-display --quiet --no-metadata

# Filter by strategy and dataset
python analysis/results.py --strategies fedavg fedprox --datasets=electricity

# Process specific experiments only
python analysis/results.py --experiments exp76 exp77 exp78 --table-type=both
```

## Output Files

Analysis tools typically save results to the `analysis/tables/` directory with descriptive filenames that include parameter suffixes when non-default values are used.

Common output formats:
- CSV files for tabular data
- Metadata files with experiment details
- Summary statistics files