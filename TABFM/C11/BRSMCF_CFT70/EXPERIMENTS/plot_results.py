import matplotlib.pyplot as plt
import pandas as pd
import sys
import numpy as np

input_file=sys.argv[1]

# response_variable,randomforest_r2,randomforest_features,tabpfn_r2,tabpfn_features
# post2:WKLD_@_VO2PK,0.5084084191757556,['pre:SBPseated', 'pre:VO2PK', 'pre:WKLD_@_VO2PK'],0.5182547344292523,['pre:WKLD_@_VO2PK', 'pre:VO2PK', 'pre:VEPK']

results=pd.read_csv(input_file, sep=',', header=0)


experiments=list(results['response_variable'])

rf_r2=list(results['randomforest_r2'])
#print('rf_r2: ', rf_r2)
rf_features=list(results['randomforest_features'])

tabpfn_r2=list(results['tabpfn_r2'])
#print('tabpfn_r2: ', tabpfn_r2)
tabpfn_features=list(results['tabpfn_features'])


# Create x-axis positions
x = np.arange(len(experiments))

# Create figure
fig, ax = plt.subplots(figsize=(16, 8))

# Width of each bar
width = 0.4

# Create grouped bars
bars1 = ax.bar(
    x - width / 2,
    rf_r2,
    width,
    label="Random Forest"
)

bars2 = ax.bar(
    x + width / 2,
    tabpfn_r2,
    width,
    label="Tab-PFN"
)

# Axis labels
ax.set_xlabel("Experiment")
ax.set_ylabel("R²")
ax.set_title("R² Performance by Experiment")


# Experiment names
ax.set_xticks(x)
ax.set_xticklabels(
    experiments,
    rotation=45,
    ha="right",
    fontsize=14,
    fontweight='bold'
)

# R² ranges from -10 to 1
min_rfr2=min(rf_r2)
min_tabpfnr2=min(tabpfn_r2)
ylim_min=min(min_rfr2, min_tabpfnr2)

ax.set_ylim(ylim_min, 1)
ax.set_yticks(np.arange(0, 1.1, 0.1))

# Add R² values above bars
#ax.bar_label(bars1, fmt="%.2f", padding=3)
#ax.bar_label(bars2, fmt="%.2f", padding=3)

# Grid
ax.grid(axis="y", linestyle="--", alpha=0.5)

# Legend
ax.legend()

plt.tight_layout()
plt.savefig('r2_results.png', dpi=300)
