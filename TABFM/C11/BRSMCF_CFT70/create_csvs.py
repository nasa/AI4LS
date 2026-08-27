import pandas as pd
import sys 

input_file=sys.argv[1]

# Replace spaces with underscores
df=pd.read_csv(input_file, sep=',', header=0)

# Treatment,IN:21_2:LVDV,IN:21_2:LVSV,IN:21_2:LV_mass,IN:21_3:LVDV,IN:21_3:LVSV,IN:21_3:LV_mass,IN:30_2:LVDV,IN:30_2:LVSV,IN:30_2:LV_mass,IN:30_3:LVDV,IN:30_3:LVSV,IN:30_3:LV_mass,IN:70_2:LVDV,IN:70_2:LVSV,IN:70_2:LV_mass,IN:70_3:LVDV,IN:70_3:LVSV,IN:70_3:LV_mass,IN:7_1:LVDV,IN:7_1:LVSV,IN:7_1:LV_mass,IN:7_2:LVDV,IN:7_2:LVSV,IN:7_2:LV_mass,IN:7_3:LVDV,IN:7_3:LVSV,IN:7_3:LV_mass,POST:0 +4 hr_2:LVDV,POST:0 +4 hr_2:LVSV,POST:0 +4 hr_2:LV_mass,POST:0 +4 hr_3:LVDV,POST:0 +4 hr_3:LVSV,POST:0 +4 hr_3:LV_mass,POST:0 +4hr_2:LVDV,POST:0 +4hr_2:LVSV,POST:0 +4hr_2:LV_mass,POST:0 +4hr_3:LVDV,POST:0 +4hr_3:LVSV,POST:0 +4hr_3:LV_mass,POST:3_2:LVDV,POST:3_2:LVSV,POST:3_2:LV_mass,POST:3_3:LVDV,POST:3_3:LVSV,POST:3_3:LV_mass,PRE:2_1:LVDV,PRE:2_1:LVSV,PRE:2_1:LV_mass,PRE:2_2:LVDV,PRE:2_2:LVSV,PRE:2_2:LV_mass,PRE:2_3:LVDV,PRE:2_3:LVSV,PRE:2_3:LV_mass

feature_cols = ['Treatment','PRE:2_1:LVDV','PRE:2_1:LVSV','PRE:2_1:LV_mass','PRE:2_2:LVDV','PRE:2_2:LVSV','PRE:2_2:LV_mass','PRE:2_3:LVDV','PRE:2_3:LVSV','PRE:2_3:LV_mass']

predict_cols = ['POST:0_+4hr_2:LVDV','POST:0_+4hr_2:LVSV','POST:0_+4hr_2:LV_mass','POST:0_+4hr_3:LVDV','POST:0_+4hr_3:LVSV','POST:0_+4hr_3:LV_mass','POST:0_+4hr_2:LVDV','POST:0_+4hr_2:LVSV','POST:0_+4hr_2:LV_mass','POST:0_+4hr_3:LVDV','POST:0_+4hr_3:LVSV','POST:0_+4hr_3:LV_mass','POST:3_2:LVDV','POST:3_2:LVSV','POST:3_2:LV_mass','POST:3_3:LVDV','POST:3_3:LVSV','POST:3_3:LV_mass']




for col in predict_cols:
   df_new=df[feature_cols + [col]]
   df_new_clean = df_new.dropna(subset=[col])
   df_new_clean = df_new_clean.dropna(axis=1)
   if len(df_new_clean) >= 34:
       df_new_clean.to_csv(col + '.csv', sep=',', index=None)


#df_merged.to_csv('mr080_merged.csv', sep=',', index=None)
