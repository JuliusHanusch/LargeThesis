import pandas as pd
from scipy.stats import gmean  # requires: pip install scipy

print("Python Script to calcuate Metrics")

def agg_relative_score(model_df: pd.DataFrame, baseline_df: pd.DataFrame):
    relative_score = model_df.drop("model", axis="columns") / baseline_df.drop(
        "model", axis="columns"
    )
    return relative_score.agg(gmean)


result_df = pd.read_csv("chronos-forecasting/scripts/evaluation/results/chronos-t5-tiny-in-domain.csv").set_index("dataset")
baseline_df = pd.read_csv("chronos-forecasting/scripts/evaluation/results/seasonal-naive-in-domain.csv").set_index("dataset")

agg_score_df = agg_relative_score(result_df, baseline_df)
agg_score_df.to_csv('chronos-forecasting/scripts/evaluation/results/agg_score.csv', index=False)

print("Finished")
