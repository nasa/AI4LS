import pandas as pd

pre=pd.read_csv('MR080G_CFT70_PRE_ALL.csv', sep=',', header=0)
post1=pd.read_csv('MR080G_CFT70_POST1_ALL.csv', header=0, sep=',')
post2=pd.read_csv('MR080G_CFT70_POST2_ALL.csv', header=0, sep=',')

precols=['pre:' + col for col in list(pre.columns)]
post1cols=['post1:' + col for col in list(post1.columns)]
post2cols=['post2:' + col for col in list(post2.columns)]
pre.columns=precols
post1.columns=post1cols
post2.columns=post2cols
pre.rename(columns={'pre:SUBJECT':'SUBJECT'}, inplace=True)
post1.rename(columns={'post1:SUBJECT':'SUBJECT'}, inplace=True)
post2.rename(columns={'post2:SUBJECT':'SUBJECT'}, inplace=True)

from functools import reduce
dfs=[pre, post1, post2]
df_merged = reduce(lambda left, right: pd.merge(left, right, on="SUBJECT", how="inner"), dfs)

# Replace spaces with underscores
df_merged.columns = df_merged.columns.str.replace(' ', '_')

#feature_cols=['SUBJECT', 'pre:GROUP', 'pre:pBAR (mmHG)', 'pre:TEMP (0C)', 'pre:REL HUM (%)', 'pre:HRsupine', 'pre:SBPsupine', 'pre:DBPsupine', 'pre:HRseated', 'pre:SBPseated', 'pre:DBPseated', 'pre:TIMETOT', 'pre:WKLD @ VO2PK', 'pre:MAX WKLD',  'pre:HRPK', 'pre:VO2PK', 'pre:VCO2PK', 'pre:RERPK', 'pre:REL VO2PK', 'pre:VEPK', 'pre:VTPK', 'pre:BR FREQPK', 'pre:FEO2PK', 'pre:FECO2PK', 'pre:SBPPK', 'pre:DBPPK', 'pre:RPEPK', 'pre:COMMENTS', 'pre:VT1', 'pre:VT2', 'pre:VT3', 'pre:VT FINAL', 'pre:%VO2MAX @ VT', 'pre:HR @ VT', 'pre:%MHR @ VT', 'pre:MatLab VT']

feature_cols=['pre:GROUP', 'pre:HRsupine', 'pre:SBPsupine', 'pre:DBPsupine', 'pre:HRseated', 'pre:SBPseated', 'pre:DBPseated', 'pre:TIMETOT', 'pre:WKLD_@_VO2PK', 'pre:MAX_WKLD',  'pre:HRPK', 'pre:VO2PK', 'pre:VCO2PK', 'pre:RERPK', 'pre:REL_VO2PK', 'pre:VEPK', 'pre:VTPK', 'pre:BR_FREQPK', 'pre:FEO2PK', 'pre:FECO2PK', 'pre:SBPPK', 'pre:DBPPK', 'pre:RPEPK', 'pre:VT1', 'pre:VT2', 'pre:VT3', 'pre:VT_FINAL', 'pre:%VO2MAX_@_VT', 'pre:HR_@_VT', 'pre:%MHR_@_VT', 'pre:MatLab_VT']



#predict_cols=['post2:HRsupine', 'post2:SBPsupine', 'post2:DBPsupine', 'post2:HRseated', 'post2:SBPseated', 'post2:DBPseated', 'post2:TIMETOT', 'post2:WKLD @ VO2PK', 'post2:MAX WKLD', 'post2:TERMINATION', 'post2:HRPK', 'post2:VO2PK', 'post2:VCO2PK', 'post2:RERPK', 'post2:REL VO2PK', 'post2:VEPK', 'post2:VTPK', 'post2:BR FREQPK', 'post2:FEO2PK', 'post2:FECO2PK', 'post2:SBPPK', 'post2:DBPPK', 'post2:RPEPK', 'post2:COMMENTS', 'post2:VT1', 'post2:VT2', 'post2:VT3', 'post2:VT FINAL', 'post2:%VO2MAX @ VT', 'post2:HR @ VT', 'post2:%MHR @ VT', 'post2:MatLab VT']

predict_cols=['post2:HRsupine', 'post2:SBPsupine', 'post2:DBPsupine', 'post2:HRseated', 'post2:SBPseated', 'post2:DBPseated', 'post2:TIMETOT', 'post2:WKLD_@_VO2PK', 'post2:MAX_WKLD', 'post2:HRPK', 'post2:VO2PK', 'post2:VCO2PK', 'post2:RERPK', 'post2:REL_VO2PK', 'post2:VEPK', 'post2:VTPK', 'post2:BR_FREQPK', 'post2:FEO2PK', 'post2:FECO2PK', 'post2:SBPPK', 'post2:DBPPK', 'post2:RPEPK', 'post2:VT1', 'post2:VT2', 'post2:VT3', 'post2:VT_FINAL', 'post2:%VO2MAX_@_VT', 'post2:HR_@_VT', 'post2:%MHR_@_VT', 'post2:MatLab_VT']

for col in predict_cols:
   df_new=df_merged[feature_cols + [col]]
   df_new_clean=df_new.dropna(axis=1)

   for i in range(len(df_new_clean)):
      if df_new_clean.iloc[i]['pre:GROUP'] == 'CONTROL':
         df_new_clean.loc[i,'pre:GROUP'] = 1
      elif df_new_clean.iloc[i]['pre:GROUP'] == 'EXERCISE':
         df_new_clean.loc[i, 'pre:GROUP'] = 2
      elif df_new_clean.iloc[i]['pre:GROUP'] == 'FLY':
         df_new_clean.loc[i, 'pre:GROUP'] = 3
      else:
         df_new_clean.loc[i, 'pre:GROUP'] = 0

   df_new_clean.to_csv(col + '.csv', sep=',', index=None)


#df_merged.to_csv('mr080_merged.csv', sep=',', index=None)
