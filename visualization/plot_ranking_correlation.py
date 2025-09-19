import pandas as pd
from scipy.stats import kendalltau
from itertools import combinations
import matplotlib.pyplot as plt
import numpy as np
import json

track = 1
# Example nested dict (shortened for clarity)
with open(f"ranking/metric_track{track}/track{track}_score_dict.json", "r") as f:
    data = json.load(f)

# Step 1: Flatten JSON into rows: (language, system, metric, value)
records = []
for lang, modes in data.items():
    for system, metrics in modes["proper"].items():
        for metric, value in metrics.items():
            records.append({"language": lang, "system": system, "metric": metric, "value": value})

df = pd.DataFrame(records)
print('df:')
print(df)

# Step 2: Average over languages
avg_df = df.groupby(["system", "metric"])["value"].mean().unstack()
print('avg_df:')
print(avg_df)

# avg_df: rows = systems, cols = metrics

# Step 1: rank per metric
#ranked = avg_df.rank(method="average")

# Step 2: build Kendall’s tau matrix
metrics = avg_df.columns
n = len(metrics)

tau_matrix = pd.DataFrame(np.zeros((n, n)), index=metrics, columns=metrics)
pval_matrix = pd.DataFrame(np.zeros((n, n)), index=metrics, columns=metrics)

for i, m1 in enumerate(metrics):
    for j, m2 in enumerate(metrics):
        if i <= j:
            tau, pval = kendalltau(avg_df[m1], avg_df[m2])
            tau_matrix.loc[m1, m2] = tau
            pval_matrix.loc[m1, m2] = pval
            # symmetry
            tau_matrix.loc[m2, m1] = tau
            pval_matrix.loc[m2, m1] = pval

print("Kendall’s tau correlation matrix:")
print(tau_matrix, "\n")
print("P-value matrix:")
print(pval_matrix)


# --- assume tau_matrix and pval_matrix already computed and renamed ---
name_map = {
    'bleu4': 'BLEU', 'chrf2++': 'ChrF++', 'consistency_frequent': 'Consistency (Freq)',
    'consistency_predefined': 'Consistency (Dict)', 'proper_term_success_rate': 'Term Acc', 'random_term_success_rate': 'Term Acc (Random)',
}
tau_matrix = tau_matrix.rename(index=name_map, columns=name_map)
metrics = tau_matrix.columns
n = len(metrics)

# Mask diagonal + upper triangle
masked_tau = tau_matrix.copy()
mask = np.triu(np.ones_like(masked_tau, dtype=bool))
masked_tau = masked_tau.mask(mask)

fig, ax = plt.subplots(figsize=(6, 5))
cax = ax.imshow(masked_tau, cmap="coolwarm", vmin=-1, vmax=1)

# Colorbar
cbar = plt.colorbar(cax, ax=ax, fraction=0.046, pad=0.04)
cbar.set_label("Kendall's tau")

# Ticks
ax.set_xticks(range(n))
ax.set_yticks(range(n))
ax.set_xticklabels(metrics, rotation=45, ha="right")
ax.set_yticklabels(metrics)

# Annotate lower triangle only
for i in range(n):
    for j in range(n):
        if i > j:  # strictly lower triangle
            tau_val = tau_matrix.iloc[i, j]
            pval = pval_matrix.iloc[i, j]
            star = "*" if pval < 0.05 else ""
            ax.text(j, i, f"{tau_val:.2f}{star}",
                    ha="center", va="center", color="black", fontsize=10)

ax.set_title("Kendall’s tau correlations between metrics")
plt.tight_layout()
plt.savefig(f"generated/ranking_correlation_track{track}.pdf")
