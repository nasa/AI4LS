import pandas as pd
import numpy as np
import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--fake', help='fake expression file name', default=None, required=True)
    parser.add_argument('-r', '--real', help='real expression file name', default=None, required=True)
    parser.add_argument('-e', '--exp', help='exponent to raise the df', default=2, required=True)
    return parser.parse_args()

args = parse_args()
fakeDFFile = args.fake
realDFFile = args.real
exponent = int(args.exp)

fake_df = pd.read_csv(fakeDFFile, sep=',', header=0)
real_df = pd.read_csv(realDFFile, sep=',', header=0)

real_df = real_df.drop(columns=['gene'])
real_std = real_df.std()
real_mean = real_df.mean()

genes = list(fake_df['gene'])

fake_df = fake_df.drop(columns=['gene'])
theMin = min(fake_df.min())
print(theMin)
if theMin < 0:
    theMin = -1 * theMin
    fake_df = fake_df.add(theMin)

# 1. un-standardize
fake_unstdize = fake_df * real_std + real_mean
fake_unstdize.insert(0, 'gene', genes)
fake_unstdize.to_csv(fakeDFFile.split('.csv')[0] + '_unstdize.csv', sep=',', index=False)

# 2. unlog
fake_unlog = np.exp2(fake_df)
fake_df.insert(0, 'gene', genes)
fake_df.to_csv(fakeDFFile.split('.csv')[0] + '_unlog.csv', sep=',', index=False)

# both
fake_df.drop(columns=['gene'], inplace=True)
fake_ready = np.exp2(fake_df)
fake_ready = fake_ready * real_std + real_mean
fake_ready.insert(0, 'gene', genes)
fake_ready.to_csv(fakeDFFile.split('.csv')[0] + '_ready.csv', sep=',', index=False)
