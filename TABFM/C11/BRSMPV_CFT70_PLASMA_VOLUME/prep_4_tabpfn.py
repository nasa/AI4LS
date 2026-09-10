import pandas as pd

df=pd.read_csv('brsmpv_cft70_plasma_volume_flat.csv', sep=',', header=0)
df_nona=df.dropna()
groups=[]
for i in range(len(df_nona)):
	if df_nona.iloc[i]['Group'] == 'Ctrl':
		groups.append(1)
	elif df_nona.iloc[i]['Group'] == 'ExA':
		groups.append(2)
	elif df_nona.iloc[i]['Group'] == 'ExB':
		groups.append(3)
	elif df_nona.iloc[i]['Group'] == 'Fly':
		groups.append(4)
df_nona['Group'] = groups
df_nona.to_csv('brsmpv_cft70_plasma_volume_flat_tabpfn.csv', sep=',', index=None)
