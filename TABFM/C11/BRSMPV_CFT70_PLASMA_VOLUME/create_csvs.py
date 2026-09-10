import pandas as pd
import sys 

input_file=sys.argv[1]

# Replace spaces with underscores
df=pd.read_csv(input_file, sep=',', header=0)

# Subject,Group,IN_TEST:Pre-Hgb (g/dL),IN_TEST:Pre-Hct (%),IN_TEST:Pre-COHgb Analysis 1 (%),IN_TEST:Pre-COHgb Analysis 2 (%),IN_TEST:Pre-COHgb Analysis 3 (%),IN_TEST:Post-Hgb (g/dL),IN_TEST:Post-Hct (%),IN_TEST:Post-COHgb Analysis 1 (%),IN_TEST:Post-COHgb Analysis 2 (%),IN_TEST:Post-COHgb Analysis 3 (%),IN_TEST:RBCvol,IN_TEST:BVtot (L),IN_TEST:PVtot (L),POST_TEST:Pre-Hgb (g/dL),POST_TEST:Pre-Hct (%),POST_TEST:Pre-COHgb Analysis 1 (%),POST_TEST:Pre-COHgb Analysis 2 (%),POST_TEST:Pre-COHgb Analysis 3 (%),POST_TEST:Post-Hgb (g/dL),POST_TEST:Post-Hct (%),POST_TEST:Post-COHgb Analysis 1 (%),POST_TEST:Post-COHgb Analysis 2 (%),POST_TEST:Post-COHgb Analysis 3 (%),POST_TEST:RBCvol,POST_TEST:BVtot (L),POST_TEST:PVtot (L),PRE_TEST:Pre-Hgb (g/dL),PRE_TEST:Pre-Hct (%),PRE_TEST:Pre-COHgb Analysis 1 (%),PRE_TEST:Pre-COHgb Analysis 2 (%),PRE_TEST:Pre-COHgb Analysis 3 (%),PRE_TEST:Post-Hgb (g/dL),PRE_TEST:Post-Hct (%),PRE_TEST:Post-COHgb Analysis 1 (%),PRE_TEST:Post-COHgb Analysis 2 (%),PRE_TEST:Post-COHgb Analysis 3 (%),PRE_TEST:RBCvol,PRE_TEST:BVtot (L),PRE_TEST:PVtot (L)

feature_cols =['Group','PRE_TEST:Pre-Hgb (g/dL)','PRE_TEST:Pre-Hct (%)','PRE_TEST:Pre-COHgb Analysis 1 (%)','PRE_TEST:Pre-COHgb Analysis 2 (%)','PRE_TEST:Pre-COHgb Analysis 3 (%)','PRE_TEST:Post-Hgb (g/dL)','PRE_TEST:Post-Hct (%)','PRE_TEST:Post-COHgb Analysis 1 (%)','PRE_TEST:Post-COHgb Analysis 2 (%)','PRE_TEST:Post-COHgb Analysis 3 (%)','PRE_TEST:RBCvol','PRE_TEST:BVtot (L)','PRE_TEST:PVtot (L)']

predict_cols=['POST_TEST:Pre-Hgb (g/dL)','POST_TEST:Pre-Hct (%)','POST_TEST:Pre-COHgb Analysis 1 (%)','POST_TEST:Pre-COHgb Analysis 2 (%)','POST_TEST:Pre-COHgb Analysis 3 (%)','POST_TEST:Post-Hgb (g/dL)','POST_TEST:Post-Hct (%)','POST_TEST:Post-COHgb Analysis 1 (%)','POST_TEST:Post-COHgb Analysis 2 (%)','POST_TEST:Post-COHgb Analysis 3 (%)','POST_TEST:RBCvol','POST_TEST:BVtot (L)','POST_TEST:PVtot (L)']




for col in predict_cols:
   df_new=df[feature_cols + [col]]
   df_new_clean = df_new.dropna(subset=[col])
   df_new_clean = df_new_clean.dropna(axis=1)
   if len(df_new_clean) >= 34:
       new_col=col.replace(" ", "")
       new_col=new_col.replace("(", "_")
       new_col=new_col.replace(")", "_")
       new_col=new_col.replace("/", "_")
       df_new_clean.rename(columns={col:new_col}, inplace=True)
       print('saving to ', 'EXPERIMENTS/DATA/' + new_col + '.csv')
       df_new_clean.to_csv('EXPERIMENTS/DATA/' + new_col + '.csv', sep=',', index=None)


