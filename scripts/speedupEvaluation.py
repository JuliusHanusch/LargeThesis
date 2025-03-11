import sqlite3
import pandas as pd
import json

# Connect to SQLite database
connection = sqlite3.connect("Speedup.db")
cursor = connection.cursor()

# Load Configs and TrainingRuns tables
configs_df = pd.read_sql_query("SELECT * FROM Configs", connection)
runs_df = pd.read_sql_query("SELECT * FROM TrainingRuns WHERE start_time IS NOT NULL AND end_time IS NOT NULL", connection)

# Compute training times
runs_df["training_time"] = pd.to_datetime(runs_df["end_time"]) - pd.to_datetime(runs_df["start_time"])

# Default values for each scaling parameter
defaults = {
    "d_ff": 2048, "num_heads": 8, "num_layers": 6, "context_length": 512,
    "n_tokens": 4096, "max_steps": 200000, "d_model": 512
}

# Parse config JSON and extract modified parameters
def parse_config(row):
    config = json.loads(row["config_json"])
    for param, default in defaults.items():
        if config.get(param, default) != default:
            return param, config.get(param, default)
    return None, None  # If no parameter was modified

configs_df[["scaling_parameter", "value_of_scaling_parameter"]] = configs_df.apply(parse_config, axis=1, result_type="expand")

# Extract default configurations (where no parameter was changed)
default_configs = configs_df[configs_df["scaling_parameter"].isna()].copy()
default_configs["scaling_parameter"] = default_configs["scaling_parameter"].fillna("default")

# Merge Configs with TrainingRuns
df = runs_df.merge(configs_df, on="config_id")

# Select relevant columns
final_df = df[["run_id", "config_id", "scaling_parameter", "value_of_scaling_parameter", "training_time"]]

# Drop rows where no parameter was scaled
final_df = final_df.dropna()

# Add the default training time for each scaling parameter
default_entries = []
for param in defaults.keys():
    default_config = default_configs.iloc[0]  # Take the first available default config
    default_config_id = default_config["config_id"]
    default_run = runs_df.loc[runs_df["config_id"] == default_config_id]
    
    if not default_run.empty:
        default_time = default_run.iloc[0]["training_time"]
        default_entries.append({
            "run_id": None,  # No specific run_id for default
            "config_id": default_config_id,
            "scaling_parameter": param,
            "value_of_scaling_parameter": defaults[param],  # Use default value for reference
            "training_time": default_time
        })

# Append default entries to the DataFrame
final_df = pd.concat([final_df, pd.DataFrame(default_entries)], ignore_index=True)

# Sort results by scaling parameter and its value
final_df = final_df.sort_values(by=["scaling_parameter", "value_of_scaling_parameter"], ascending=[True, True])

# Save to CSV
final_df.to_csv("grouped_training_times.csv", index=False)

# Close the database connection
connection.close()

print("Grouped training times saved successfully.")

