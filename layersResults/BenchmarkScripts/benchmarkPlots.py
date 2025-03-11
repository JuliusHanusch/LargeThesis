import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# Define the base path for the CSV files
CSV_PATH = "/home/julius/LargeThesisCode/layersResults/"

# Define scaling parameters and evaluation types
SCALING_PARAMS = ["num_layers_6", "num_layers_4", "num_layers_2"]  # Correct order
EVAL_TYPES = ["in-domain", "zero-shot"]
METRICS = ["MASE", "WQL", "RMSE[mean]", "MAE"]

# Define colors corresponding to layers
COLORS = {
    "num_layers_6": "blue",    # Layer 6 → 🔵
    "num_layers_4": "orange",  # Layer 4 → 🟠
    "num_layers_2": "green"    # Layer 2 → 🟢
}

# Function to generate and save a single benchmark plot containing all scaling parameters
def generate_combined_benchmark_plot(metric):
    fig, axes = plt.subplots(2, 1, figsize=(10, 10))  # 1 column, 2 rows
    fig.suptitle(f"Comparison of {metric} Across All Scaling Parameters", fontsize=16)
    
    for i, eval_type in enumerate(EVAL_TYPES):
        ax = axes[i]
        ax.set_title(f"{eval_type.title()} Evaluation", fontsize=14)
        ax.set_xlabel("Config ID", fontsize=12)
        ax.set_ylabel(metric, fontsize=12)
        
        width = 0.25  # Adjusted width for multiple bars
        x_positions = np.arange(50)  # Ensure Config IDs are always 1-50
        ax.set_xticks(x_positions)
        ax.set_xticklabels(range(1, 51), rotation=90, fontsize=8)
        
        for j, scaling_param in enumerate(SCALING_PARAMS):
            file_path = os.path.join(CSV_PATH, f"{scaling_param}_{eval_type}.csv")
            
            if not os.path.exists(file_path):
                print(f"Skipping {file_path}, file not found.")
                continue
            
            df = pd.read_csv(file_path)
            df = df.rename(columns={"Config ID": "Config_ID"})
            
            # Map Config IDs correctly
            if scaling_param == "num_layers_6":
                df["Config_ID"] -= 100  # Shift Config IDs from 101-150 → 1-50
            elif scaling_param == "num_layers_4":
                df["Config_ID"] -= 50   # Shift Config IDs from 51-100 → 1-50
            # num_layers_2 stays the same (already 1-50)
            
            df = df.sort_values(by="Config_ID")
            
            # Align bars correctly for comparison
            ax.bar(x_positions + (j - 1) * width, df[metric], width, 
                   label=scaling_param.replace("_", " ").title(), 
                   color=COLORS[scaling_param])
        
        ax.legend(fontsize=10)
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plot_filename = os.path.join(CSV_PATH, f"benchmark_all_{metric}.png")
    plt.savefig(plot_filename)
    print(f"Saved plot: {plot_filename}")

# Generate combined benchmark plots for all metrics
for metric in METRICS:
    generate_combined_benchmark_plot(metric)

print("All combined benchmark plots generated successfully!")