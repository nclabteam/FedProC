import json
import os
import re
import sys

import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

# --- Configuration ---
dataset_name = "Traffic"  # <--- CHANGE THIS IF NEEDED
base_data_path = os.path.join("datasets")
# Adjust sub_path if your structure is different
sub_path = os.path.join(dataset_name, f"seq_96-offset_0-pred_96")  # Example
full_data_path = os.path.join(base_data_path, sub_path)
info_file_path = os.path.join(full_data_path, "info.json")
# Define the folder where plots will be saved
output_folder = os.path.join("saved_plots", dataset_name)

# --- Create Output Folder ---
try:
    os.makedirs(output_folder, exist_ok=True)
    print(f"Output folder '{output_folder}' created or already exists.")
except OSError as e:
    print(f"Error creating output folder '{output_folder}': {e}")
    exit()

# --- Data Loading and Preparation ---
# Initialize structure to hold extracted stats
stats = {"trend_strength": {}, "seasonal_strength": {}, "transition": {}, "samples": {}}

if not os.path.exists(info_file_path):
    print(f"Error: File not found at {info_file_path}")
    exit()

print(f"Loading data from: {info_file_path}")
num_clients_from_json = 0
all_features = []  # Initialize feature list

try:
    with open(info_file_path, "r") as f:
        info = json.load(f)
        if isinstance(info, list):
            num_clients_from_json = len(info)
            print(f"Found {num_clients_from_json} clients in the JSON list.")
            all_features_found = set()  # Use a set to avoid duplicate features

            for client_idx, client in enumerate(info):
                client_stats = client.get("stats", {}).get("train", {})
                samples_info = client.get("samples", {}).get("train", {})
                client_features = set(
                    client_stats.keys()
                )  # Features present for this client
                all_features_found.update(client_features)

                # --- Process trend, seasonal, transition stats ---
                if client_stats:
                    for feature, substats in client_stats.items():
                        for stat_key in [
                            "trend_strength",
                            "seasonal_strength",
                            "transition",
                        ]:
                            if stat_key in substats:
                                value_to_append = substats[stat_key]
                                # Handle list values (e.g., from MSTL) - take first element
                                if isinstance(value_to_append, list):
                                    value_to_append = (
                                        value_to_append[0] if value_to_append else None
                                    )
                                # Append if valid number
                                if value_to_append is not None:
                                    try:
                                        stats[stat_key].setdefault(feature, []).append(
                                            (client_idx, float(value_to_append))
                                        )
                                    except (ValueError, TypeError):
                                        # Optionally print warning, but can be verbose
                                        # print(f"Warning: Non-numeric stat '{value_to_append}' for {stat_key}/{feature}, client {client_idx}. Skipping.")
                                        pass

                # --- Process 'samples' count ---
                num_samples_train_x = pd.NA  # Default to Pandas Not Available
                if samples_info and "x" in samples_info:
                    shape_string = str(samples_info["x"])
                    # Regex to find first number in parentheses
                    match = re.search(r"\(\s*(\d+).*?\)", shape_string)
                    if match:
                        try:
                            num_samples_train_x = int(match.group(1))
                        except ValueError:
                            # print(f"Warning: Could not convert extracted sample count '{match.group(1)}' to int for client {client_idx}.")
                            num_samples_train_x = (
                                pd.NA
                            )  # Revert to NA on conversion error
                    # else: # Optional warning if regex fails on existing string
                    # if shape_string: print(f"Warning: Regex failed on shape string '{shape_string}' for client {client_idx}.")

                # Assign samples count (or NA) to all features found for this client
                # Prevents errors if samples key exists but stats doesn't
                if not pd.isna(num_samples_train_x):
                    # Use features identified from the stats section for this client
                    for feature in client_features:
                        stats["samples"].setdefault(feature, []).append(
                            (client_idx, num_samples_train_x)
                        )

            # Finalize the list of all unique features found across clients
            all_features = sorted(list(all_features_found))
            if not all_features:
                print("Warning: No features found in client stats across the dataset.")

        else:
            # Handle case where JSON root is not a list
            print(
                f"Warning: Top-level structure in info.json is not a list (type: {type(info)}). Cannot process client data."
            )
            all_features = []  # Ensure it's defined

except json.JSONDecodeError:
    print(f"Error: Could not decode JSON from {info_file_path}")
    exit()
except Exception as e:
    print(f"An unexpected error occurred during data loading: {e}")
    # For detailed debugging:
    # import traceback
    # traceback.print_exc()
    exit()

# --- Reshape data to long format ---
if not any(v for stat_dict in stats.values() for v in stat_dict.values()):
    print("Warning: No statistics were loaded from the info.json file. Exiting.")
    exit()

# Initialize the list to hold records for the DataFrame
records = []
all_stat_types = list(stats.keys())  # Get stat types from the dictionary keys

# Determine number of clients (robustly, either from JSON list length or inferred)
num_clients = 0
max_client_idx_seen = -1
if num_clients_from_json > 0:
    num_clients = num_clients_from_json
else:
    # Infer from data if JSON wasn't a list or empty
    for stat_type, stat_data in stats.items():
        for feature, client_value_pairs in stat_data.items():
            if client_value_pairs:  # Check if list is not empty
                try:
                    # Check if it's a list/tuple of pairs before finding max index
                    if isinstance(client_value_pairs, (list, tuple)) and all(
                        isinstance(p, (list, tuple)) and len(p) == 2
                        for p in client_value_pairs
                    ):
                        max_idx_in_feature = max(idx for idx, val in client_value_pairs)
                        max_client_idx_seen = max(
                            max_client_idx_seen, max_idx_in_feature
                        )
                    # else: print(f"Debug: Unexpected format in inference: {stat_type}/{feature}") # Optional Debug
                except (ValueError, TypeError):
                    continue  # Skip if error finding max index
    if max_client_idx_seen >= 0:
        num_clients = max_client_idx_seen + 1
        print(f"Inferred number of clients from data: {num_clients}")
    else:
        # Check if num_clients_from_json was 0 but we found no client data either
        if num_clients_from_json == 0:
            print(
                "Error: Could not determine the number of clients from JSON or data. Exiting."
            )
            exit()
        else:  # JSON list was empty
            num_clients = 0  # Proceed with 0 clients, will likely exit later

# Create a lookup dictionary for faster access
client_data_tracker = {}  # {(stat_type, feature): {client_idx: value}}
for stat_type, stat_data in stats.items():
    for feature, client_value_pairs in stat_data.items():
        if not isinstance(client_value_pairs, list):
            continue  # Skip if not a list
        key = (stat_type, feature)
        try:
            # Ensure pairs are valid tuples/lists of length 2 before creating dict
            valid_pairs = [
                p
                for p in client_value_pairs
                if isinstance(p, (list, tuple)) and len(p) == 2
            ]
            # Convert index to int and value to float robustly
            client_data_tracker[key] = {
                int(idx): float(val) for idx, val in valid_pairs
            }
        except (TypeError, ValueError, IndexError) as e:
            # print(f"Warning: Error processing data structure for {key}: {e}. Skipping entry.") # Optional Warning
            client_data_tracker[key] = {}  # Assign empty dict to avoid KeyError
            continue

# Generate records list (POPULATE the 'records' list)
if num_clients > 0 and all_features:
    for i in range(num_clients):
        for feature in all_features:
            for stat_type in all_stat_types:
                key = (stat_type, feature)
                feature_data = client_data_tracker.get(
                    key, {}
                )  # Get data for stat/feature, default empty
                value = feature_data.get(i, pd.NA)  # Get value for client i, default NA
                # Append the dictionary for this row to the records list
                records.append(
                    {
                        "stat_type": stat_type,
                        "feature": feature,
                        "client": f"client_{i+1}",  # 1-based client string
                        "client_id": i,  # 0-based client index
                        "value": value,
                    }
                )
else:
    # Warn if records cannot be generated
    print(
        "Warning: Cannot generate records. Number of clients is zero or no features were found."
    )

# Check if records list was populated before creating DataFrame
if not records:
    print("No records were generated for the DataFrame. Cannot create plots. Exiting.")
    exit()

# Create DataFrame from the populated list of records
df = pd.DataFrame.from_records(records)

# --- Data Type Conversion ---
df["client_id"] = pd.to_numeric(df["client_id"])
df["value"] = pd.to_numeric(
    df["value"], errors="coerce"
)  # Convert values to numeric, force errors to NaN
df["stat_type"] = df["stat_type"].astype("category")  # Convert stat_type to categorical
# Convert feature to ordered categorical using the discovered list of features
if all_features:
    df["feature"] = pd.Categorical(df["feature"], categories=all_features, ordered=True)
else:
    # Fallback if no features were somehow found but records were generated
    df["feature"] = df["feature"].astype("category")

# --- Initial Data Summary ---
print(f"\nDataFrame ready for plotting ({len(df)} rows).")
print(
    f"Total NaN values in 'value' column after processing: {df['value'].isna().sum()}"
)
if all_features:
    print(f"Features found: {all_features}")
print(f"Stat types found: {df['stat_type'].cat.categories.tolist()}")


# --- Plotting and Saving ---

# Set Seaborn theme (e.g., "ticks" style removes grid)
sns.set_theme(style="ticks", palette="tab10")

# Define filenames for the plots
plot_filenames = {
    "boxplot": "boxplot_stats_excl_samples.png",
    "violin": "violinplot_stats_excl_samples.png",
    "heatmap": "heatmap_mean_stats_excl_samples.png",
    "line_faceted": "lineplot_faceted_row_stats_incl_samples.png",  # Includes samples
    "distribution": "distribution_kde_stats_excl_samples.png",  # Excludes samples
}


# --- Plot 1: Faceted Line Chart - INCLUDES SAMPLES ---
# (Shows trends for all stats, including the constant 'samples' count)
print("\nGenerating Faceted Line Chart (Rows = Stat Type, includes 'samples')...")
try:
    # Drop rows with NaN values specifically for this plot
    plot_df_line = df.dropna(subset=["value"])
    if not plot_df_line.empty:
        g = sns.relplot(
            data=plot_df_line,
            x="client_id",
            y="value",
            hue="feature",
            row="stat_type",  # Color by feature, separate row per stat
            kind="line",
            # markers=True, marker='o', markersize=5, # Add markers
            linewidth=1.5,  # Line thickness
            height=3,
            aspect=3.5,  # Control facet size and aspect ratio
            facet_kws={
                "sharey": False,
                "sharex": True,
                "margin_titles": True,
            },  # Independent Y axes
            legend="full",
        )
        # Disable grid on each subplot's axes
        for ax in g.axes.flat:
            ax.grid(False)
        # Add title and labels
        main_title = f"{dataset_name}: Statistics Across Clients (Faceted by Stat Type)"
        g.fig.suptitle(main_title, y=1.03, fontsize=16, weight="bold")
        g.set_axis_labels("Client Index", "Statistic Value")
        g.set_titles(
            row_template="{row_name}", col_template="{col_name}"
        )  # Set facet titles
        if g.legend:
            g.legend.set_title("Feature")  # Set legend title
        # Adjust layout to prevent title overlap
        g.figure.tight_layout(rect=[0, 0.03, 1, 0.97])
        # Save the plot
        save_path = os.path.join(output_folder, plot_filenames["line_faceted"])
        g.figure.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close(g.fig)  # Close the figure to free memory
        print(f"Saved: {save_path}")
    else:
        print("Skipping faceted line chart: DataFrame empty after dropna.")
except Exception as e:
    print(f"Error generating faceted line plot: {e}")
    # import traceback; traceback.print_exc() # Uncomment for detailed debug
    plt.close()  # Attempt to close any open figure


# --- Filter out 'samples' stat_type for subsequent plots ---
print("\nFiltering out 'samples' stat_type for remaining plots...")
original_rows = len(df)
# Create the filtered DataFrame using boolean indexing, copy to avoid warnings
df_filtered = df[df["stat_type"] != "samples"].copy()
print(f"Filtered DataFrame from {original_rows} to {len(df_filtered)} rows.")

# Update the categorical definition in the filtered DataFrame to remove 'samples' level
stat_types_to_plot = []  # Initialize list of stat types to plot
if "stat_type" in df_filtered.columns and pd.api.types.is_categorical_dtype(
    df_filtered["stat_type"]
):
    df_filtered["stat_type"] = df_filtered["stat_type"].cat.remove_unused_categories()
    # Get the remaining categories for ordering plots
    stat_types_to_plot = df_filtered["stat_type"].cat.categories.tolist()
    print(f"Remaining stat_type categories for plots: {stat_types_to_plot}")
else:
    # Fallback if column isn't categorical or doesn't exist
    if "stat_type" in df_filtered.columns:
        stat_types_to_plot = sorted(df_filtered["stat_type"].unique())
        print(
            f"Warning: 'stat_type' not categorical, using unique values: {stat_types_to_plot}"
        )
    else:
        print("Warning: 'stat_type' column not found in filtered data.")


# --- Generate remaining plots using the FILTERED df_filtered ---

# --- Plot 2: Distribution Plot (KDE Line) - EXCLUDES SAMPLES ---
print("\nGenerating Distribution Plot (KDE Line, excludes 'samples', no grid)...")
try:
    # Use the filtered df, drop NaNs specifically for this plot
    plot_df_dist = df_filtered.dropna(subset=["value"])
    if not plot_df_dist.empty:
        # Check if there are any stat types left after filtering and NaN drop
        if plot_df_dist["stat_type"].nunique() > 0:
            h = sns.displot(
                data=plot_df_dist,  # Use filtered, NaN-dropped data
                x="value",
                col="stat_type",  # Facet by remaining statistic types
                kind="kde",  # Use Kernel Density Estimate lines
                col_order=stat_types_to_plot,  # Order facets by remaining types
                fill=True,  # Fill area under curve
                alpha=0.4,  # Transparency of fill
                linewidth=0,  # Hide line if only fill is desired
                # line_kws={'linewidth': 2.0}, # Alternative: style line explicitly
                facet_kws={"sharey": False, "sharex": False},  # Independent axes
                height=4,  # Height of each facet
                aspect=1.3,  # Aspect ratio (width/height)
            )
            # Disable grid on each subplot's axes
            for ax in h.axes.flat:
                ax.grid(False)
            # Add title and labels
            h.fig.suptitle(
                f"{dataset_name}: Distribution Density (KDE, Excl. Samples)",
                y=1.04,
                fontsize=16,
                weight="bold",
            )
            h.set_axis_labels("Statistic Value", "Density")  # Y-axis is density for KDE
            h.set_titles("{col_name}", weight="semibold")  # Set facet titles
            # Adjust layout
            h.figure.tight_layout(rect=[0, 0.03, 1, 0.96])
            # Save the plot
            save_path = os.path.join(output_folder, plot_filenames["distribution"])
            h.figure.savefig(save_path, dpi=300, bbox_inches="tight")
            plt.close(h.fig)  # Close the figure
            print(f"Saved: {save_path}")
        else:
            print(
                "Skipping KDE distribution plot: No valid 'stat_type' data remaining after filtering and dropping NaNs."
            )
    else:
        print(
            "Skipping KDE distribution plot: Filtered DataFrame empty after dropping NaN values."
        )
except Exception as e:
    print(f"Error generating KDE distribution plot: {e}")
    # import traceback; traceback.print_exc() # Uncomment for detailed debug
    plt.close()


# --- Plot 3: Boxplot - EXCLUDES SAMPLES ---
print("\nGenerating Boxplot (excluding 'samples')...")
try:
    # Check if there's data to plot
    if df_filtered.empty or not stat_types_to_plot:
        print("Skipping Boxplot: Filtered DataFrame is empty or no stat types remain.")
    else:
        # Get features present in the filtered data
        features_in_filtered = sorted(df_filtered["feature"].unique())
        plt.figure(
            figsize=(max(10, len(features_in_filtered) * 0.7), 7)
        )  # Dynamic figure size
        # Create the boxplot on the axes 'ax'
        ax = sns.boxplot(
            data=df_filtered,
            x="feature",
            y="value",
            hue="stat_type",
            order=features_in_filtered,
            hue_order=stat_types_to_plot,  # Control order
            linewidth=1.2,  # Box outline width
            flierprops=dict(marker=".", markersize=4, alpha=0.6),
        )  # Customize outliers
        # Disable grid on this specific plot's axes
        ax.grid(False)
        # Add titles and labels
        plt.title(
            f"Boxplot: {dataset_name} Stat Distribution (Excl. Samples)",
            fontsize=16,
            weight="bold",
        )
        plt.xlabel("Feature", fontsize=12)
        plt.ylabel("Statistic Value", fontsize=12)
        plt.xticks(rotation=45, ha="right", fontsize=10)
        plt.yticks(fontsize=10)
        plt.legend(title="Statistic Type", loc="best", fontsize=10, title_fontsize=11)
        plt.tight_layout()  # Adjust layout
        # Save the plot
        save_path = os.path.join(output_folder, plot_filenames["boxplot"])
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()  # Close the figure
        print(f"Saved: {save_path}")
except Exception as e:
    print(f"Error generating or saving boxplot: {e}")
    plt.close()

# --- Plot 4: Violinplot - EXCLUDES SAMPLES ---
print("\nGenerating Violinplot (excluding 'samples')...")
try:
    # Check if there's data to plot
    if df_filtered.empty or not stat_types_to_plot:
        print(
            "Skipping Violinplot: Filtered DataFrame is empty or no stat types remain."
        )
    else:
        features_in_filtered = sorted(df_filtered["feature"].unique())
        plt.figure(figsize=(max(10, len(features_in_filtered) * 0.7), 7))
        # Create the violinplot on the axes 'ax'
        ax = sns.violinplot(
            data=df_filtered,
            x="feature",
            y="value",
            hue="stat_type",
            order=features_in_filtered,
            hue_order=stat_types_to_plot,  # Control order
            split=True,
            inner="quartile",  # Show quartiles inside violins
            linewidth=1.0,  # Violin outline width
            scale="width",
        )  # Scale violins to have same max width
        # Disable grid on this specific plot's axes
        ax.grid(False)
        # Add titles and labels
        plt.title(
            f"Violinplot: {dataset_name} Stat Distribution (Excl. Samples)",
            fontsize=16,
            weight="bold",
        )
        plt.xlabel("Feature", fontsize=12)
        plt.ylabel("Statistic Value", fontsize=12)
        plt.xticks(rotation=45, ha="right", fontsize=10)
        plt.yticks(fontsize=10)
        plt.legend(title="Statistic Type", loc="best", fontsize=10, title_fontsize=11)
        plt.tight_layout()  # Adjust layout
        # Save the plot
        save_path = os.path.join(output_folder, plot_filenames["violin"])
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()  # Close the figure
        print(f"Saved: {save_path}")
except Exception as e:
    print(f"Error generating or saving violin plot: {e}")
    plt.close()


# --- Plot 5: Heatmap - EXCLUDES SAMPLES ---
print("\nGenerating Heatmap (excluding 'samples')...")
try:
    # Prepare data for heatmap: use filtered data, drop NaNs, pivot
    heatmap_df_prep = df_filtered.dropna(subset=["value"])
    if not heatmap_df_prep.empty:
        # Ensure necessary columns exist
        if (
            "stat_type" in heatmap_df_prep.columns
            and "feature" in heatmap_df_prep.columns
        ):
            # Get features and stats actually present in the data being pivoted
            current_features = sorted(heatmap_df_prep["feature"].unique())
            # Use the ordered list derived after filtering for rows
            current_stat_types = [
                st
                for st in stat_types_to_plot
                if st in heatmap_df_prep["stat_type"].unique()
            ]

            if not current_stat_types or not current_features:
                print(
                    "Skipping heatmap: No valid stat types or features left after filtering NaNs."
                )
            else:
                # Create pivot table (mean aggregation by default)
                heatmap_data = pd.pivot_table(
                    heatmap_df_prep,
                    values="value",
                    index="stat_type",
                    columns="feature",
                    aggfunc="mean",
                )
                # Reindex to ensure consistent row/column order based on filtered lists
                heatmap_data = heatmap_data.reindex(
                    index=current_stat_types, columns=current_features
                )

                if not heatmap_data.empty:
                    plt.figure(
                        figsize=(
                            max(8, len(current_features) * 0.8),
                            max(4, len(current_stat_types) * 0.7),
                        )
                    )
                    # Create heatmap (no grid needed to disable here)
                    sns.heatmap(
                        heatmap_data,
                        annot=True,
                        cmap="viridis",
                        fmt=".2f",  # Annotate with values
                        linewidths=0.5,
                        linecolor="lightgray",  # Add cell borders
                        annot_kws={"size": 9},
                    )  # Control annotation font size
                    # Add titles and labels
                    plt.title(
                        f"Heatmap: Mean {dataset_name} Stat Values (Excl. Samples)",
                        fontsize=16,
                        weight="bold",
                    )
                    plt.xlabel("Feature", fontsize=12)
                    plt.ylabel("Statistic Type", fontsize=12)
                    plt.xticks(rotation=45, ha="right", fontsize=10)
                    plt.yticks(rotation=0, fontsize=10)
                    plt.tight_layout()  # Adjust layout
                    # Save the plot
                    save_path = os.path.join(output_folder, plot_filenames["heatmap"])
                    plt.savefig(save_path, dpi=300, bbox_inches="tight")
                    plt.close()  # Close the figure
                    print(f"Saved: {save_path}")
                else:
                    print("Skipping heatmap: Pivot table resulted in empty data.")
        else:
            print(
                "Skipping heatmap: 'stat_type' or 'feature' column missing after filtering."
            )
    else:
        print(
            "Skipping heatmap: No non-NaN data available for pivoting after filtering 'samples'."
        )
except KeyError as e:
    # Catch potential errors if columns used in pivot_table are missing
    print(f"Error creating heatmap data (KeyError): {e}. Check DataFrame columns.")
    plt.close()
except Exception as e:
    print(f"An unexpected error occurred during heatmap creation/saving: {e}")
    plt.close()


# --- Final Message ---
print(f"\n--- All plots saved to folder: {os.path.abspath(output_folder)} ---")
print("Script finished.")
