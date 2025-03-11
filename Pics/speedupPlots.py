import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Read the CSV file
data = pd.read_csv("/home/julius/LargeThesisCode/Pics/speedupTraining.csv")

# Convert training_time to minutes
def convert_to_minutes(time_str):
    time_parts = time_str.split()
    total_minutes = 0

    for i, part in enumerate(time_parts):
        if "days" in part:
            total_minutes += int(time_parts[i - 1]) * 1440  # Convert days to minutes
        elif ":" in part:
            h, m, s = map(float, part.split(":"))
            total_minutes += h * 60 + m + s / 60
    
    return total_minutes

data["training_time_minutes"] = data["training_time"].apply(convert_to_minutes)
data["training_time_hours"] = data["training_time_minutes"] / 60  # Convert minutes to hours

# Define which scaling parameters to plot
selected_scaling_params = ["max_steps", "context_length", "num_heads", "num_layers"]  # Modify this list as needed

# Filter data for selected scaling parameters
data = data[data["scaling_parameter"].isin(selected_scaling_params)]

# Get unique selected scaling parameters
scaling_params = data["scaling_parameter"].unique()

# Create a grid of subplots based on the number of selected parameters
num_plots = len(scaling_params)
rows = (num_plots + 2) // 3  # Ensure enough rows for 3 columns
fig, axes = plt.subplots(rows, 2, figsize=(15, 5 * rows))
axes = axes.flatten()

# Loop through each scaling parameter and plot on the corresponding axis
for i, param in enumerate(scaling_params):
    subset = data[data["scaling_parameter"] == param]
    ax = axes[i]

    ax.plot(range(len(subset)), subset["training_time_hours"], marker="o", linestyle="-", label=param)
    ax.set_xlabel("Value of " + param)
    ax.set_ylabel("Training Time (hours)")
    ax.set_title(f"Speedup Plot for {param}")
    ax.legend()
    ax.set_yscale("linear")
    
    ax.set_xticks(range(len(subset)))
    ax.set_xticklabels(subset["value_of_scaling_parameter"], rotation=45)

    for j, txt in enumerate(subset["training_time_minutes"]):
        hours = int(txt // 60)
        minutes = int(txt % 60)
        time_str = f"{hours}h {minutes}min"
        ax.annotate(time_str, (j, subset.iloc[j]["training_time_hours"]), textcoords="offset points", xytext=(0, 5), ha='center')

    y_ticks = np.arange(0, 7, 1)
    ax.set_yticks(y_ticks)
    ax.set_yticklabels([f"{i}h" for i in y_ticks])
    ax.grid(True, which="both", linestyle="--", linewidth=0.5)

# Hide unused subplots
for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
plt.savefig("speedup_selected_plots.png")
plt.show()
