import pandas as pd
import sys

# ID,Subject,Treatment,analyzer,Test_Phase,BR_Day,LV mass,LVDV,LVSV,Comments

# ID,Subject,Group,Test_Phase,BR_Day,Session,Pre-Hgb (g/dL),Pre-Hct (%),Pre-COHgb Analysis 1 (%),Pre-COHgb Analysis 2 (%),Pre-COHgb Analysis 3 (%),Post-Hgb (g/dL),Post-Hct (%),Post-COHgb Analysis 1 (%),Post-COHgb Analysis 2 (%),Post-COHgb Analysis 3 (%),RBCvol,BVtot (L),PVtot (L)

df = pd.read_csv(sys.argv[1], sep=',', header=0)

# 1. Build the short phase label used in your column names (PRE, IN, POST)
df['phase_short'] = df['Test_Phase'].str.replace('_TEST', '', regex=False)

# 2. Build the "PRE:2" style period label
df['period'] = df['phase_short'] + ':' + df['BR_Day'].astype(str)

# 3. Assign a consistent replicate number (1/2) based on analyzer identity,
#    so the same analyzer always gets the same replicate slot across all periods
#analyzer_order = sorted(df['analyzer'].unique())          # e.g. ['Tim C.', 'Tim M.']
#analyzer_map = {name: i + 1 for i, name in enumerate(analyzer_order)}
#df['rep'] = df['analyzer'].map(analyzer_map)

# 4. Melt the measurement columns into long format
#value_cols = ['LV mass', 'LVDV', 'LVSV']
value_cols = ['Pre-Hgb (g/dL)','Pre-Hct (%)','Pre-COHgb Analysis 1 (%)','Pre-COHgb Analysis 2 (%)','Pre-COHgb Analysis 3 (%)','Post-Hgb (g/dL)','Post-Hct (%)','Post-COHgb Analysis 1 (%)','Post-COHgb Analysis 2 (%)','Post-COHgb Analysis 3 (%)','RBCvol','BVtot (L)','PVtot (L)']
long = df.melt(
    id_vars=['Subject', 'Group', 'period'],
    value_vars=value_cols,
    var_name='measure',
    value_name='value'
)

# 5. Clean measure names (spaces -> underscores) and build final column label
long['measure'] = long['measure'].str.replace(' ', '_', regex=False)
long['col_name'] = long['period'] +  ':' + long['measure']

# 6. Pivot to one row per Subject/Group
flat = long.pivot_table(
    index=['Subject', 'Group'],
    columns='col_name',
    values='value',
    aggfunc='first'
).reset_index()

#flat.columns.name = None
#print(flat.head())
#print(list(flat.columns))

output_file=sys.argv[1].split('.csv')[0] + '_flat.csv'
flat.to_csv(output_file, sep=',', index=None)
