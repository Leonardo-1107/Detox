import os
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Define the results folder path
results_dir = Path("results")
csv_files = list(results_dir.glob("*.csv"))

if not csv_files:
    print("[ERROR] No CSV files found in the 'results' directory.")
else:
    all_dfs = []

    # Load each CSV and tag with model name (from filename)
    for file in csv_files:
        try:
            df = pd.read_csv(file)
            if {'lang', 'STA', 'SIM', 'XCOMET', 'J'}.issubset(df.columns):
                df["model"] = file.stem
                all_dfs.append(df)
            else:
                print(f"[WARNING] Skipped file with missing columns: {file.name}")
        except Exception as e:
            print(f"[ERROR] Failed to read {file.name}: {e}")

    if not all_dfs:
        print("[ERROR] No valid CSV dataframes to plot.")
    else:
        df_all = pd.concat(all_dfs, ignore_index=True)

        # Melt dataframe for easy plotting
        df_melted = df_all.melt(id_vars=["lang", "model"], value_vars=["STA", "SIM", "XCOMET", "J"],
                                var_name="Metric", value_name="Score")

        # Plot each metric separately
        metrics = ["STA", "SIM", "XCOMET", "J"]
        for metric in metrics:
            plt.figure(figsize=(10, 6))
            subset = df_melted[df_melted["Metric"] == metric]

            # Sort models by name
            sorted_models = sorted(subset["model"].unique())

            for lang in subset["lang"].unique():
                lang_data = subset[subset["lang"] == lang]
                lang_data = lang_data.set_index("model").loc[sorted_models].reset_index()
                plt.plot(lang_data["model"], lang_data["Score"], marker="o", markersize=10, label=lang)

            plt.title(f"{metric} Score across Models by Language")
            plt.xlabel("Model")
            plt.ylabel(f"{metric} Score")
            plt.xticks(rotation=45)
            plt.legend(title="Language")
            plt.tight_layout()
            plt.grid(True)
            plt.show()