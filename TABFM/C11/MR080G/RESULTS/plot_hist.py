import matplotlib.pyplot as plt
import sys
import pandas as pd
import numpy as np

res_file=sys.argv[1]
out_file=sys.argv[2]

df=pd.read_csv(res_file, sep=',', header=0)

# Your two data lists (can be different lengths)
cols = list(df['target'])

x = np.arange(len(df))  # one position per variable (82 total)
width = 0.4

fig, ax = plt.subplots(figsize=(18, 6))
ax.bar(x - width/2, df['rf_r2'], width=width, label='rf_r2', color='tab:blue')
ax.bar(x + width/2, df['tabpfn_r2'], width=width, label='tabpfn_r2', color='tab:orange')

ax.axhline(0, color='gray', linewidth=1)  # reference line at R²=0
ax.set_ylabel('R² score')
ax.set_title('R² Scores by Model, per Variable')
ax.set_xticks([])  # no x-axis ticks or labels
ax.legend()

plt.tight_layout()
plt.savefig(out_file, dpi=150)
plt.show()
