import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import random


class Correlation(object):
    def __init__(self, train_environments, val_environment, test_environment, args):
        self.intersected_features = None
        self.corr_threshold = args["correlation_threshold"]
        self.seed = args["seed"]
        self.verbose = args["verbose"]
        self.intersection_found = False
        self.targetkey = args['target'][0]
        self.all_columns = args["columns"]
        self.fname = args.get('fname', "")
        self.save_plot = args.get('save_plot', False)

        # Fix the seed
        random.seed(self.seed)
        np.random.seed(self.seed)

        correlation_set = []

        # Set up test set df        
        x_test, y_test = test_environment.get_all()
        df_test = pd.DataFrame(x_test.numpy(), columns=self.all_columns)
        df_test[self.targetkey] = y_test.numpy()
        self.df_test = df_test
        self.n_test = len(df_test)

        # Set env dfs
        for e, env in enumerate(train_environments):
            x, y = env.get_all()
            df = pd.DataFrame(x.numpy(), columns=self.all_columns)
            df[self.targetkey] = y.numpy()
            df = df.astype(float)
            correlatedcols = self.get_correlatedcol_set(df, self.corr_threshold, name='env_' + str(e))
            correlation_set.append(correlatedcols)

        # Set up validation set df
        x_val, y_val = val_environment.get_all()
        df_val = pd.DataFrame(x_val.numpy(), columns=self.all_columns)
        df_val[self.targetkey] = y_val.numpy()
        correlatedcols = self.get_correlatedcol_set(df_val, self.corr_threshold, name='val')
        correlation_set.append(correlatedcols)

        if len(correlation_set):
            self.intersected_features = list(set.intersection(*correlation_set))
            self.intersection_found = True
            if args["verbose"]:
                print("Intersection:", self.intersected_features)

    def get_correlatedcol_set(self, df, corr_threshold, name='correlation'):
        d = df.shape[1]
        corr = df.corr()
        npcorr = np.array(corr)

        if self.save_plot:
            f = plt.figure(figsize=(152, 120))
            plt.matshow(np.abs(corr.corr()), fignum=f.number)
            plt.xticks(range(corr.shape[1]), corr.columns, fontsize=10, rotation=45)
            plt.yticks(range(corr.shape[1]), corr.columns, fontsize=10)
            cb = plt.colorbar()
            cb.ax.tick_params(labelsize=10)
            plt.tight_layout()
            plt.savefig(self.fname + name + '_correlation.pdf')

        selectedcols = {}
        for i in range(d):
            for j in range(i, d):
                if i == j:
                    continue
                elif (npcorr[i, j] > corr_threshold) or (npcorr[i, j] < -corr_threshold):
                    column = corr.columns[j]
                    row = corr.columns[i]
                    if (row, column) not in selectedcols:
                        selectedcols[(row, column)] = True
                else:
                    continue
        selectedcols_set = []
        for key in selectedcols:
            selectedcols_set.append(key)
        selectedcols_set = set(selectedcols_set)

        return selectedcols_set



    def predictor_results(self):

        keptcols = []
        deletedcols = []
        for tupl in self.intersected_features:
            keptcols.append(tupl[0])
            deletedcols.append(tupl[1])

        invariant_correlations = pd.DataFrame(np.vstack((keptcols, deletedcols)).T, columns=['kept', 'deleted'])
        delcount = 0
        for idx, row_elem in enumerate(invariant_correlations['kept']):
            if row_elem in deletedcols:
                invariant_correlations = invariant_correlations.drop([invariant_correlations.index[idx - delcount]])
                delcount += 1

        retained_columns = []
        for elem in self.all_columns:
            if elem not in deletedcols:
                retained_columns.append(elem)

        results = {}
        results['retained_columns'] = retained_columns
        results['invariant_correlations'] = invariant_correlations

        return {
            "InvariantCorrelationsFound": self.intersection_found,
            "results": results
        }
