import pandas as pd
import sys

df = pd.read_csv(sys.argv[1], sep=',', header=0)

# 1. Build the short phase label used in your column names (PRE, IN, POST)
df['phase_short'] = df['Test_Phase'].str.replace('_TEST', '', regex=False)

# 2. Build the "PRE:2" style period label
df['period'] = df['phase_short'] + ':' + df['BR_Day'].astype(str)

# 3. Assign a consistent replicate number (1/2) based on analyzer identity,
#    so the same analyzer always gets the same replicate slot across all periods
analyzer_order = sorted(df['analyzer'].unique())          # e.g. ['Tim C.', 'Tim M.']
analyzer_map = {name: i + 1 for i, name in enumerate(analyzer_order)}
df['rep'] = df['analyzer'].map(analyzer_map)

# 4. Melt the measurement columns into long format
value_cols = ['LV mass', 'LVDV', 'LVSV']
long = df.melt(
    id_vars=['Subject', 'Treatment', 'period', 'rep'],
    value_vars=value_cols,
    var_name='measure',
    value_name='value'
)

# 5. Clean measure names (spaces -> underscores) and build final column label
long['measure'] = long['measure'].str.replace(' ', '_', regex=False)
long['col_name'] = long['period'] + '_' + long['rep'].astype(str) + ':' + long['measure']

# 6. Pivot to one row per Subject/Treatment
flat = long.pivot_table(
    index=['Subject', 'Treatment'],
    columns='col_name',
    values='value',
    aggfunc='first'
).reset_index()

#flat.columns.name = None
#print(flat.head())
#print(list(flat.columns))

output_file=sys.argv[1].split('.csv')[0] + '_flat.csv'
flat.to_csv(output_file, sep=',', index=None)
