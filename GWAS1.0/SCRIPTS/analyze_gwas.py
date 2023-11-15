import pandas as pd
import sys
import os
import json

in_dir=sys.argv[1]
suffix='glm.linear.adjusted'
prefix='out_filename.'

out_dir=in_dir + '/SIG_FDR_BH'
P_THRESHOLD=0.05
os.mkdir(out_dir)

sig_dict=dict()

for f in os.listdir(in_dir): 
    if suffix in f:
        df=pd.read_csv(in_dir + '/' + f, sep='\t', header=0)
        other_f=f.split('.adjusted')[0]
        other_df=pd.read_csv(in_dir + '/' + other_f, sep='\t', header=0)
        mylist=list()
        for i in range(len(df)):
            if df.iloc[i]['FDR_BH'] <= $P_THRESHOLD:
                id=df.iloc[i]['ID']
                chrom=other_df[other_df['ID']==id]['#CHROM'].values[0]
                pos=other_df[other_df['ID']==id]['POS'].values[0]
                ref=other_df[other_df['ID']==id]['REF'].values[0]
                alt=other_df[other_df['ID']==id]['ALT'].values[0]
                mylist.append(str([id, chrom, pos, ref, alt]))
        if len(mylist) != 0:
            # out_filename.Gamma_4h_Slope.glm.linear.adjusted
            f=f.split(prefix)[1].split('.glm')[0]
            sig_dict[f]=mylist

print(sig_dict)

with open(out_dir + '/results.json', 'w') as fp:
    json.dump(sig_dict, fp)

