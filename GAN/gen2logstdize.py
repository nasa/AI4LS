import pandas as pd
import numpy as np
from scipy.stats import zscore
import sys

fileName=sys.argv[1]

df=pd.read_csv(fileName, sep=',', header=0)

genes=list(df['gene'])

samples=list(df.columns)[1:]

x=df.drop(columns=['gene'])

x=np.log2(1+x)

x=np.float32(x)

x=x.T

x=zscore(x, axis=0)

x=x.T

x_pd=pd.DataFrame(x, columns=samples)

x_pd['gene'] = genes

x_pd = x_pd[['gene'] + samples]

x_pd.to_csv(fileName.split('.csv')[0] + '_log2stdize.csv', sep=',', index=None)
