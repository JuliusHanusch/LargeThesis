import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# Define the base path for the CSV files
CSV_PATH = "/home/julius/LargeThesisCode/headsResults/"

# Define scaling parameters and evaluation types
SCALING_PARAMS = ["num_heads_8", "num_heads_6", "num_heads_4", "num_heads_2"]  # Correct order
EVAL_TYPES = ["in-domain", "zero-shot"]
METRICS = ["MASE", "WQL", "RMSE[mean]", "MAE"]

# Define colors corresponding to layers (updated as per request)
COLORS = {
    "num_heads_8": "blue",     # Layer 8 → 🔵
    "num_heads_6": "orange",   # Layer 6 → 🟠
    "num_heads_4": "green",    # Layer 4 → 🟢
    "num_heads_2": "red"       # Layer 2 → 🔴
}

# Function to generate and save a single benchmark plot containing all scaling parameters
def generate_combined_benchmark_plot(metric):
    # Create a single figure with 1 column and 4 rows
    fig, axes = plt.subplots(4, 1, figsize=(18, 24))  # 1 column, 4 rows for both evaluations (in-domain and zero-shot)
    fig.suptitle(f"Comparison Across All Scaling Parameters for {metric}", fontsize=16)

    # Loop over the evaluation types (in-domain and zero-shot)
    for i, eval_type in enumerate(EVAL_TYPES):
        # Determine the indices for the subplot
        start_idx = i * 2  # Two plots for each evaluation type, one for 1-25, one for 26-50
        end_idx = start_idx + 1
        
        # Loop over the config splits (1-25 and 26-50)
        for j, config_range in enumerate([(1, 25), (26, 50)]):
            ax = axes[start_idx + j]
            ax.set_title(f"{eval_type.title()} Evaluation (Configs {config_range[0]}-{config_range[1]})", fontsize=14)
            ax.set_xlabel("Config ID", fontsize=12)
            ax.set_ylabel(metric, fontsize=12)

            width = 0.15  # Adjusted width for individual bars
            x_positions = np.arange(config_range[0], config_range[1] + 1)  # Config IDs for current split
            
            ax.set_xticks(x_positions - 1)  # Shifting X positions for better display
            ax.set_xticklabels(x_positions, rotation=90, fontsize=8)

            # Loop over each scaling parameter
            for k, scaling_param in enumerate(SCALING_PARAMS):
                file_path = os.path.join(CSV_PATH, f"{scaling_param}_{eval_type}.csv")

                if not os.path.exists(file_path):
                    print(f"Skipping {file_path}, file not found.")
                    continue

                df = pd.read_csv(file_path)
                df = df.rename(columns={"Config ID": "Config_ID"})

                # Map Config IDs correctly for each scaling parameter
                if scaling_param == "num_heads_2":
                    df["Config_ID"] -= 0    # Config IDs are already 1-50
                elif scaling_param == "num_heads_4":
                    df["Config_ID"] -= 50   # Shift Config IDs from 51-100 → 1-50
                elif scaling_param == "num_heads_6":
                    df["Config_ID"] -= 100  # Shift Config IDs from 101-150 → 1-50
                elif scaling_param == "num_heads_8":
                    df["Config_ID"] -= 150  # Shift Config IDs from 151-200 → 1-50

                # Select only the relevant configs for the current split (1-25 or 26-50)
                df = df[(df["Config_ID"] >= config_range[0]) & (df["Config_ID"] <= config_range[1])]
                
                # If the dataframe is empty, skip plotting
                if df.empty:
                    print(f"No data for {scaling_param} in {eval_type} for configs {config_range[0]}-{config_range[1]}")
                    continue
                
                df = df.sort_values(by="Config_ID")

                # Align bars correctly for comparison
                ax.bar(x_positions + (k - 2) * width, df[metric], width,
                       label=scaling_param.replace("_", " ").title(),
                       color=COLORS[scaling_param])

            ax.legend(fontsize=10)

    # Adjust layout for the 1x4 grid of plots
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plot_filename = os.path.join(CSV_PATH, f"benchmark_combined_{metric}.png")
    plt.savefig(plot_filename)
    print(f"Saved plot: {plot_filename}")

# Generate the combined benchmark plot for all metrics
for metric in METRICS:
    generate_combined_benchmark_plot(metric)

print("Combined benchmark plots generated successfully!")
