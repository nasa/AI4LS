import os
import numpy as np
import pandas as pd
import tensorflow as tf
import statsmodels.api as sm
import random
import time

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


class ITE(object):
    def __init__(self, train_environments, val_environment, test_environment, args):
        self.selected_features = []
        self.seed = args["seed"]
        self.verbose = args["verbose"]
        self.latent_representation_found = False
        self.Terminate = False
        self.latent_variables = None
        self.latent_dim = None
        self.model = None
        self.targetkey = args["target"][0]
        self.threshold = args["threshold"]
        self.pvalthreshold = args["pvalthreshold"]
        self.balance_epsilon = args["balance_epsilon"]

        # Fix the seed
        random.seed(self.seed)
        np.random.seed(self.seed)
        tf.set_random_seed(self.seed)

        # Set up test set df
        columns = args["columns"]
        self.columns = columns

        x_test, y_test = test_environment.get_all()
        df_test = pd.DataFrame(x_test.numpy(), columns=columns)
        df_test[self.targetkey] = y_test.numpy()
        self.df_test = df_test
        self.n_test = len(df_test)

        # Set up validation set df
        x_val, y_val = val_environment.get_all()
        df_val = pd.DataFrame(x_val.numpy(), columns=columns)
        df_val[self.targetkey] = y_val.numpy()
        self.df_val = df_val
        self.n_val = len(df_val)

        # Set up main df
        x_all = []
        y_all = []

        for e, env in enumerate(train_environments):
            x, y = env.get_all()
            x_all.append(x.numpy())
            y_all.append(y.numpy())

        x_all = np.vstack(x_all)
        y_all = np.vstack(y_all)

        df_train = pd.DataFrame(x_all, columns=columns)
        df_train[self.targetkey] = y_all
        self.df_train = df_train
        self.n_train = len(df_train)

        if self.verbose:
            print('testshape:', df_test.shape)
            print('valshape:', df_val.shape)
            print('trainshape:', df_train.shape)

        self.dfX = pd.concat([df_train, df_val], axis=0)
        self.dfy = self.dfX[self.targetkey]
        self.dfX = self.dfX.drop(columns=[self.targetkey], axis=1).astype(float)

        d = self.dfX.shape[1]
        if self.verbose:
            print('Number of features:', d)
        time.sleep(5)

        self.dfXmean = self.dfX.mean()
        self.dfXstd = self.dfX.std()

        ITE = np.zeros(shape=(d, 3))

        for i in range(d):
            ITE[i, 0] = int(i)

            if self.verbose:
                print('current feature number:', i)
            X = np.array(self.dfX)
            if self.threshold == 0:
                curr_threshold = min(X[:, i])
            else:
                print(
                    'Error: threshold not implemented - can only use 0 (i.e. min-value) threshold as we use normalised data')
            X[:, i] = np.array((X[:, i] > curr_threshold) * 1.0)

            ## __ Check if outcome and treatment variables are roughly balanced __ ##
            balanced = self.balance_test(X[:, i])
            if not balanced:
                ITE[i, 1] = np.nan
                ITE[i, 2] = np.nan
                continue
            ## __ _____________________________________________________________ __ ##

            X = sm.add_constant(X, has_constant='add')
            # print(min(np.std(X[:,1:], axis = 0)))
            y = np.array(self.dfy)

            def calculate_acc(y, yp):
                out = np.sum(y == yp) / len(y)
                return out

            logit_model = sm.Logit(y, X)  # , family=sm.families.Binomial())
            result = logit_model.fit_regularized(L1_wt=0.0, alpha=1, maxiter=5000, disp=0)
            treatment_coef = result.params[i + 1]
            treatment_pval = result.pvalues[i + 1]

            ITE[i, 1] = treatment_coef
            ITE[i, 2] = treatment_pval

        self.ITEdf = pd.DataFrame(ITE, columns=['index', 'effect', 'significance'])

    def balance_test(self, X):
        passed = True

        if len(self.dfy[X > 0]) == 0:
            return False
        if len(self.dfy[X == 0]) == 0:
            return False

        treatment_share = (sum(X) / len(X))
        treated_outcome_share = (sum(self.dfy[X > 0]) / len(self.dfy[X > 0]))
        untreated_outcome_share = (sum(self.dfy[X == 0]) / len(self.dfy[X == 0]))

        if (treatment_share < self.balance_epsilon) or (treatment_share > (1.0 - self.balance_epsilon)):
            passed = False
        if (treated_outcome_share < self.balance_epsilon) or (treated_outcome_share > (1.0 - self.balance_epsilon)):
            passed = False
        if (untreated_outcome_share < self.balance_epsilon) or (untreated_outcome_share > (1.0 - self.balance_epsilon)):
            passed = False

        return passed

    def results(self):
        outputdf = self.ITEdf.copy()
        outputdf = outputdf.dropna()
        dropped = 0
        for elem in outputdf['index']:
            print(self.ITEdf['significance'][elem])
            if self.ITEdf['significance'][elem] > self.pvalthreshold:
                outputdf = outputdf.drop([outputdf['index'][elem]])
                dropped += 1

        for i in range(outputdf.shape[0]):
            outputdf.iloc[i, 0] = int(outputdf.iloc[i, 0])

        print(outputdf)

        return {
            'output_df': outputdf,
            "to_bucket": {
                'method': 'ITE Discovery',
                'features': (np.array(self.columns)[np.array(outputdf['index'].values).astype(int)]).tolist(),
                'coefficients': np.array(outputdf['effect'].values).tolist(),
                'pvals': np.array(outputdf['significance'].values).tolist(),
                'test_acc': None,
                'test_acc_std': None
            }
        }
