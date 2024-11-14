import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.decomposition import PCA
import argparse
import json



def pcaPlot(pca, df, info_df, variable, title, gen_dir, use_meta_cols, use_palette=True):
    pcaDF = pd.DataFrame(data=pca.fit_transform(df), columns=['PC 1', 'PC 2'])
    pcaDF.index = info_df.index
    for meta_param in list(use_meta_cols['cat']):
        pcaDF = pd.concat([pcaDF, info_df[[meta_param]]], axis=1)

    sns.set(style="whitegrid", font_scale=0.5)
    fig, ax = plt.subplots(figsize=(5,5))
    uniq_values = set(list(pcaDF[variable]))
    #palette = ['green', 'orange', 'brown', 'blue', 'red', 'purple']
    palette = sns.color_palette("tab10")
    if use_palette:
        ax = sns.scatterplot(x=pcaDF['PC 1'], y=pcaDF['PC 2'], hue=pcaDF[variable], s=100, palette=palette[:len(uniq_values)])
    else:
        ax = sns.scatterplot(x=pcaDF['PC 1'], y=pcaDF['PC 2'], hue=pcaDF[variable], s=100)

    ax.set_xlabel('PC 1 ' + '(' + str(round(pca.explained_variance_ratio_[0]*100, 1)) + '% variance)', fontsize=16)
    ax.set_ylabel('PC 2 ' + '(' + str(round(pca.explained_variance_ratio_[1]*100, 1)) + '% variance)', fontsize=16)
    #ax.tick_params(axis='x',  labelsize=10)
    #ax.tick_params(axis='y',  labelsize=10)
    plt.xticks([])
    plt.yticks([])

    ax.set_title(title, fontsize=20)
    plt.legend(fontsize=15)
    #plt.show()
    if gen_dir is None:
        gen_dir = '.'
    plt.savefig(gen_dir + '/' + title, dpi=300)
    plt.close()



def myPlot(x, info_df, output_dir, use_meta_cols):

    pca = PCA(n_components=2)

    #x = standardize(x)

    for meta_param in list(use_meta_cols['cat']):
        if meta_param == 'ORO Positivity (%)':
            use_palette=False
        else:
            use_palette=True
        #pcaPlot(pca, x, info_df, meta_param, meta_param + '_Dataset_' + 'n=' + str(x.shape[0]), output_dir, use_meta_cols, use_palette)
        pcaPlot(pca, x, info_df, meta_param, meta_param, output_dir, use_meta_cols, use_palette)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-ie', '--input_expr', help='input expression data', default=None)
    parser.add_argument('-im', '--input_meta', help='input meta data', default=None)
    parser.add_argument('-umf', '--use_meta_file', help='file to specify meta data to use in analysis', default=None, required=True)
    parser.add_argument('-od', '--output_dir', help='output dir', default=None, required=True)

    return parser.parse_args()

def main():
    options = parse_args()
    with open(options.use_meta_file, 'r') as f:
        use_meta_cols = json.load(f)
    f.close()
    expr_df = pd.read_csv(options.input_expr, index_col=0)
    if 'env' in expr_df.columns:
        expr_df.drop(columns=['env'], inplace=True)
    if 'target' in expr_df.columns:
        expr_df.drop(columns=['target'])

    # pca expects rows to be samples and columns to be genes
    if len(expr_df.columns) < len(expr_df):
        x=expr_df.T
    else:
        x=expr_df

    #x = np.log10(1+ x)
    # standardize expression data
    #x = np.float32(x)
    #x = (x - x.mean()) / x.std()


    info_df = pd.read_csv(options.input_meta, header=0, sep=',')


    myPlot(x, info_df, options.output_dir, use_meta_cols)



if __name__ == "__main__":
    main()
