import numpy as np
import pandas as pd


class ZeroVarianceChecker(object):
    def __init__(self, environment_datasets, val_dataset, test_dataset, args):
        self.in_each_env = args.get('in_each_env', False)
        self.predictor_columns = test_dataset.predictor_columns

        if self.in_each_env:
            envs = []
            for e in environment_datasets:
                e_x, e_y = e.get_all()
                e_x = pd.DataFrame(e_x.numpy().squeeze(), columns=self.predictor_columns)
                envs.append(e_x)
            v_x, v_y = val_dataset.get_all()
            v_x = pd.DataFrame(v_x.numpy().squeeze(), columns=self.predictor_columns)
            envs.append(v_x)
            t_x, t_y = test_dataset.get_all()
            t_x = pd.DataFrame(t_x.numpy().squeeze(), columns=self.predictor_columns)
            envs.append(t_x)
            zero_var_cols = self.check_vars_in_each_env(envs)
            self.zero_var_cols = zero_var_cols
        else:
            all_x = []
            for e in environment_datasets:
                e_x, e_y = e.get_all()
                all_x.append(e_x.numpy().squeeze())
            v_x, v_y = val_dataset.get_all()
            t_x, t_y = test_dataset.get_all()
            all_x.append(v_x.numpy().squeeze())
            all_x.append(t_x.numpy().squeeze())

            all_x = np.vstack(all_x)
            all_df = pd.DataFrame(all_x, columns=self.predictor_columns)

            zero_var_cols = self.check_vars_for_df(all_df)
            self.zero_var_cols = zero_var_cols

    def reduced_feature_list(self):
        red_list = [f for f in self.predictor_columns if f not in self.zero_var_cols]
        return red_list

    @staticmethod
    def check_vars_for_df(df):
        zero_var_cols = []
        for col in df.columns:
            if np.std(df[col].values) == 0:
                zero_var_cols.append(col)
        return zero_var_cols

    def check_vars_in_each_env(self, envs):
        zero_var_sets = []
        for df in envs:
            zero_var_cols = self.check_vars_for_df(df)
            if len(zero_var_cols) > 0:
                zero_var_sets += zero_var_cols

        zero_var_sets = np.unique(np.array(zero_var_sets).flatten()).tolist()
        return zero_var_sets


class ZeroVarianceCheckerTarget(object):
    def __init__(self, environment_datasets, in_any_env=True):
        self.in_any_env = in_any_env

        if self.in_any_env:
            envs = []
            for e in environment_datasets:
                e_x, e_y = e.get_all()
                e_y = pd.DataFrame(e_y.numpy().squeeze())
                envs.append(e_y)
            zero_variance = self.check_vars_in_any_env(envs)
            self.zero_var = zero_variance
        else:
            all_y = []
            for e in environment_datasets:
                e_x, e_y = e.get_all()
                all_y.append(e_y.numpy())
            all_y = np.vstack(all_y)
            all_df = pd.DataFrame(all_y)
            zero_var_cols = self.check_vars_for_df(all_df)
            if len(zero_var_cols) > 0:
                self.zero_var = True
            else:
                self.zero_var = False

    def check_vars_in_any_env(self, envs):
        for df in envs:
            zero_var_cols = self.check_vars_for_df(df)
            if len(zero_var_cols) > 0:
                return True
        return False

    @staticmethod
    def check_vars_for_df(df):
        zero_var_cols = []
        for col in df.columns:
            if np.std(df[col].values) == 0:
                zero_var_cols.append(col)
        return zero_var_cols
