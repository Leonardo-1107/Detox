import os
import pandas as pd
import matplotlib.pyplot as plt

# Path to results folder
results_dir = "results"

# Read all CSVs and annotate with model name
df_list = []
for file in os.listdir(results_dir):
    if file.endswith(".csv"):
        model_name = os.path.splitext(file)[0]
        df = pd.read_csv(os.path.join(results_dir, file))
        df["model"] = model_name
        df_list.append(df)

# Combine all into a single DataFrame
df_all = pd.concat(df_list)

# Plot one figure per language
metrics = ["STA", "SIM", "XCOMET", "J"]
colors = ["blue", "green", "orange", "red"]
markers = ["o", "s", "^", "D"]

for lang in df_all["lang"].unique():
    plt.figure(figsize=(10, 4))
    df_lang = df_all[df_all["lang"] == lang].copy()
    df_lang = df_lang.sort_values(by="J")  # sort by J ascending
    x_labels = df_lang["model"].tolist()

    for metric, color, marker in zip(metrics, colors, markers):
        plt.plot(x_labels, df_lang[metric], label=metric, marker=marker, markersize=10, color=color, alpha=0.8)

    plt.title(f"Detoxification Scores for {lang.upper()} ")
    plt.xlabel("Model (sorted by J Score)")
    plt.ylabel("Score")
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.grid(True)
    plt.show()