import pandas as pd
import sys
import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--expr_file', help='expression file', default=None)
    parser.add_argument('-m', '--meta_file', help='metadata file', default=None)


    return parser.parse_args()

args = parse_args()
exprFile = args.expr_file
metaFile = args.meta_file
exprKey = 'gene'
metaKey = 'sample'

expr = pd.read_csv(exprFile, sep=',', header=0)
meta = pd.read_csv(metaFile, sep=',', header=0)

sample2index_dict = dict()

for i in range(meta.shape[0]):
    sample = meta.iloc[i][metaKey]
    j = expr.columns.get_loc(sample)
    #j = expr.index[expr['sample'] == sample].tolist()
    sample2index_dict[sample] = (i, j)

meta = meta.sample(frac=1)
meta.to_csv(metaFile.split('.csv')[0] + '_permuted.csv', sep=',', index=False)

sample_list = list(meta[metaKey])
#sample_list.sort()

expr = expr.reindex(columns=['gene'] + sample_list)
#expr = expr[[exprKey] + sample_list]
# df['Region'] = pd.Categorical(df['Region'], categories=desired_order, ordered=True)
#expr['sample'] = pd.Categorical(expr['sample'], categories=sample_list, ordered=True)
#expr = expr.sort_values('sample')
expr.to_csv(exprFile.split('.csv')[0] + '_permuted.csv', sep=',', index=False)



