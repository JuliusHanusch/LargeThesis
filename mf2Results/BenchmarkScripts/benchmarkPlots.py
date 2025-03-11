import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# Define the base path for the CSV files
CSV_PATH = "/home/julius/LargeThesisCode/mf2Results/"

# Define scaling parameters and evaluation types
SCALING_PARAMS = ["context_length", "num_heads", "num_layers", "max_steps"]
EVAL_TYPES = ["in-domain", "zero-shot"]
METRICS = ["MASE", "WQL", "RMSE[mean]", "MAE"]

# Function to generate and save benchmark plots
def generate_benchmark_plots(metric):
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    fig.suptitle(f"Comparison of {metric} Across Scaling Parameters", fontsize=16)
    
    for i, eval_type in enumerate(EVAL_TYPES):
        default_file = os.path.join(CSV_PATH, f"default_{eval_type}.csv")
        
        if not os.path.exists(default_file):
            print(f"Skipping {default_file}, file not found.")
            continue
        
        default_df = pd.read_csv(default_file)
        
        for j, scaling_param in enumerate(SCALING_PARAMS):
            other_file = os.path.join(CSV_PATH, f"{scaling_param}_{eval_type}.csv")
            
            if not os.path.exists(other_file):
                print(f"Skipping {other_file}, file not found.")
                continue
            
            other_df = pd.read_csv(other_file)
            
            # Rename "Config ID" for easier merging
            default_df = default_df.rename(columns={"Config ID": "Config_ID"})
            other_df = other_df.rename(columns={"Config ID": "Config_ID"})
            
            # Adjust Config_IDs to match default
            if scaling_param == "context_length":
                other_df["Config_ID"] = other_df["Config_ID"] - 50
            elif scaling_param == "num_heads":
                other_df["Config_ID"] = other_df["Config_ID"] - 100
            elif scaling_param == "num_layers":
                other_df["Config_ID"] = other_df["Config_ID"] - 150
            # max_steps uses the same Config_IDs as default
            
            # Merge DataFrames
            merged_df = pd.merge(default_df, other_df, on="Config_ID", suffixes=("_default", f"_{scaling_param}"))
            merged_df = merged_df.sort_values(by="Config_ID")
            
            # Plot in correct subplot
            ax = axes[i, j]
            width = 0.4
            x = np.arange(len(merged_df["Config_ID"]))
            
            ax.bar(x - width/2, merged_df[f"{metric}_default"], width, label="Default")
            ax.bar(x + width/2, merged_df[f"{metric}_{scaling_param}"], width, label=scaling_param.replace("_", " ").title())
            
            ax.set_title(f"{scaling_param.replace('_', ' ').title()} ({eval_type})", fontsize=12)
            ax.set_xlabel("Config ID", fontsize=10)
            ax.set_ylabel(metric, fontsize=10)
            ax.set_xticks(x)
            ax.set_xticklabels(merged_df["Config_ID"], rotation=90, fontsize=8)
            ax.legend(fontsize=8)
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust layout to fit title
    plot_filename = os.path.join(CSV_PATH, f"benchmark_{metric}.png")
    plt.savefig(plot_filename)
    print(f"Saved plot: {plot_filename}")

# Generate benchmark plots for all metrics
for metric in METRICS:
    generate_benchmark_plots(metric)

print("All benchmark plots generated successfully!")
