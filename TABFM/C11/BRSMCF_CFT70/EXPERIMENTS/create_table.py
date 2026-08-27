import matplotlib.pyplot as plt
import sys
import pandas as pd
import ast

input_file=sys.argv[1]

# response_variable,randomforest_r2,randomforest_features,tabpfn_r2,tablpfn_features
# post2:WKLD_@_VO2PK,0.5084084191757556,['pre:SBPseated', 'pre:VO2PK', 'pre:WKLD_@_VO2PK'],0.5182547344292523,['pre:WKLD_@_VO2PK', 'pre:VO2PK', 'pre:VEPK']

results=pd.read_csv(input_file, sep=',', header=0)


experiments=list(results['response_variable'])

rf_features=list(results['randomforest_features'])

tabpfn_features=list(results['tabpfn_features'])

top3_rf = [ast.literal_eval(results.iloc[i]['randomforest_features']) for i in range(len(results))]
top3_tabpfn = [ast.literal_eval(results.iloc[i]['tabpfn_features']) for i in range(len(results))]
#print('rf features: ', top3_rf)
#print('tabpfn features: ', top3_tabpfn)


# Convert each list of 3 features into a single string
rf_features = [
    ", ".join(features) for features in top3_rf
]

tabpfn_features = [
    ", ".join(features) for features in top3_tabpfn
]

# Create table data
table_data = [
    [experiment, alg1, alg2]
    for experiment, alg1, alg2
    in zip(experiments, rf_features, tabpfn_features)
]

# Create figure
fig, ax = plt.subplots(figsize=(16, 8))

# Hide axes
ax.axis("off")

# Create table
table = ax.table(
    cellText=table_data,
    colLabels=[
        "Experiment",
        "Random Forest — Top 3 Features",
        "Tab-PFN — Top 3 Features"
    ],
    loc="center",
    cellLoc="left"
)



# Formatting
table.auto_set_font_size(False)
table.set_fontsize(18)
table.scale(1, 2)

for col in range(3):
    table[0, col].set_text_props(weight="bold")

# Adjust column widths
for row in range(len(table_data) + 1):
    table[row, 0].set_width(0.25)
    table[row, 1].set_width(0.375)
    table[row, 2].set_width(0.375)

plt.tight_layout()
plt.savefig('feature_importance_results.png', dpi=300)
