import pandas as pd
import scipy.stats as stats
import os

def map_config_id(config_id):
    return (config_id - 1) % 50 + 1

def calculate_spearman_correlation(default_file, scaling_file, metric):
    default_df = pd.read_csv(default_file)
    scaling_df = pd.read_csv(scaling_file)
    
    default_df[metric] = pd.to_numeric(default_df[metric], errors='coerce')
    scaling_df[metric] = pd.to_numeric(scaling_df[metric], errors='coerce')
    
    default_df["MappedConfig"] = default_df["Config ID"].apply(map_config_id)
    scaling_df["MappedConfig"] = scaling_df["Config ID"].apply(map_config_id)
    
    default_df = default_df.sort_values(by=metric).reset_index(drop=True)
    default_df["Rank_default"] = default_df[metric].rank(method="average", ascending=True)
    
    scaling_df = scaling_df.sort_values(by=metric).reset_index(drop=True)
    scaling_df["Rank_scaling"] = scaling_df[metric].rank(method="average", ascending=True)
    
    merged_df = pd.merge(default_df, scaling_df, on="MappedConfig", suffixes=("_default", "_scaling"))
    
    correlation, p_value = stats.spearmanr(merged_df["Rank_default"], merged_df["Rank_scaling"])
    return correlation, p_value

def main():
    base_path = "/home/julius/LargeThesisCode/mf2Results"
    scaling_params = ["context_length", "num_heads", "num_layers", "max_steps"]
    eval_types = ["in-domain", "zero-shot"]
    metrics = ["MASE", "WQL", "RMSE[mean]", "MAE"]
    
    results = []
    
    for eval_type in eval_types:
        default_file = os.path.join(base_path, f"default_{eval_type}.csv")
        for scaling in scaling_params:
            scaling_file = os.path.join(base_path, f"{scaling}_{eval_type}.csv")
            if os.path.exists(default_file) and os.path.exists(scaling_file):
                result_row = [eval_type, scaling]
                for metric in metrics:
                    correlation, p_value = calculate_spearman_correlation(default_file, scaling_file, metric)
                    result_row.append(f"{correlation:.4f} ({p_value:.2e})")
                results.append(result_row)
            else:
                print(f"Missing file(s) for {eval_type} - {scaling}")
    
    results_df = pd.DataFrame(results, columns=["Evaluation Type", "Scaling Parameter", "MASE", "WQL", "RMSE[mean]", "MAE"])
    results_df.to_csv(os.path.join(base_path, "spearman_correlation_results.csv"), index=False)
    print("Results saved to spearman_correlation_results.csv")

if __name__ == "__main__":
    main()
