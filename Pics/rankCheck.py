import pandas as pd

def map_config_id(config_id):
    # Maps IDs so that 51->1, 52->2, ..., 100->50, 101->1, etc.
    return (config_id - 1) % 50 + 1

def process_file(file_path):
    df = pd.read_csv(file_path)
    df["Config ID"] = df["Config ID"].apply(map_config_id)
    df["MASE"] = pd.to_numeric(df["MASE"], errors="coerce")
    df = df.sort_values(by="MASE").reset_index(drop=True)
    df["Rank"] = range(1, len(df) + 1)
    return df

def calculate_spearman(merged_df):
    n = len(merged_df)
    merged_df["Rank Difference Squared"] = merged_df["Rank Difference"] ** 2
    sum_d_squared = merged_df["Rank Difference Squared"].sum()
    print(sum_d_squared)
    spearman_rho = 1 - (6 * sum_d_squared) / (n * (n**2 - 1))
    return spearman_rho

def main():
    file1 = "/home/julius/LargeThesisCode/results/results/context_length_in-domain.csv"
    file2 = "/home/julius/LargeThesisCode/results/results/default_in-domain.csv"
    
    df1 = process_file(file1)
    df2 = process_file(file2)
    
    # Merge on Config ID
    merged_df = df1.merge(df2, on="Config ID", suffixes=("_context_length", "_default"))
    
    # Compute rank difference (context_length relative to default)
    merged_df["Rank Difference"] = merged_df["Rank_context_length"] - merged_df["Rank_default"]
    
    # Check where ranks are the same and where they differ
    merged_df["Rank Match"] = merged_df["Rank_context_length"] == merged_df["Rank_default"]
    
    print("Configuration Rank Comparison:")
    print(merged_df[["Config ID", "Rank_context_length", "Rank_default", "Rank Difference", "Rank Match"]])
    
    # Print mismatched ranks
    mismatched = merged_df[~merged_df["Rank Match"]]
    if not mismatched.empty:
        print("\nMismatched Ranks:")
        print(mismatched[["Config ID", "Rank_context_length", "Rank_default", "Rank Difference"]])
    else:
        print("\nAll ranks match.")
    
    # Calculate Spearman correlation using the manual formula
    spearman_rho = calculate_spearman(merged_df)
    print(f"\nSpearman Correlation (Manual Calculation): {spearman_rho:.4f}")
    
    # Save the merged DataFrame
    merged_df.to_csv("/home/julius/LargeThesisCode/results/results/merged_ranks.csv", index=False)

if __name__ == "__main__":
    main()
