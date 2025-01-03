from itertools import chain, combinations, compress
import numpy as np
import torch
from scipy.stats import f as fdist
from scipy.stats import ttest_ind

from models.TorchModelZoo import TorchLinearRegressionModule, TorchLogisticRegressionModule
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss
import pickle




def mean_var_test(x, y):
    pvalue_mean = ttest_ind(x, y, equal_var=False).pvalue
    pvalue_var1 = 1 - fdist.cdf(np.var(x, ddof=1) / np.var(y, ddof=1),
                                x.shape[0] - 1,
                                y.shape[0] - 1)

    pvalue_var2 = 2 * min(pvalue_var1, 1 - pvalue_var1)

    return 2 * min(pvalue_mean, pvalue_var2)

def computing_p_values(ordered_names, original_names, train_environments, first_n_elems = 100, epochs_regression=1000):
    
    # Train ICP

    x_all = []
    y_all = []
    e_all = []

    for e, env in enumerate(train_environments):
        x, y = env.get_all()
        x_all.append(x.numpy())
        y_all.append(y.numpy())
        e_all.append(np.full(x.shape[0], e))

    x_all = np.vstack(x_all)
    y_all = np.vstack(y_all)
    e_all = np.hstack(e_all)

    dim = x_all.shape[1]

    p_values = []
    counter = 0

    for j in range(1,first_n_elems):#len(ordered_names)):
        print(j/len(ordered_names))
        bool_names = []
        for original_name in original_names:
            if original_name in ordered_names[:j]:
                bool_names.append(True)
            else:
                bool_names.append(False)

        x_s = x_all[:, bool_names]
#         print(x_s.shape)
        #reg = LinearRegression(fit_intercept=False).fit(x_s, y_all)

        reg = TorchLinearRegressionModule(x_s.shape[1], y_all.shape[1], bias=False)
        reg.fit(x_s, y_all, epochs=epochs_regression) 
        p_values_ = []
        for e in range(len(train_environments)):
            e_in = np.where(e_all == e)[0]
            e_out = np.where(e_all != e)[0]

            res_in = (y_all[e_in] - reg.predict(x_s[e_in, :]).cpu().numpy()).ravel()
            res_out = (y_all[e_out] - reg.predict(x_s[e_out, :]).cpu().numpy()).ravel()

            p_values_.append(mean_var_test(res_in, res_out))

        p_values.append(min(p_values_) * len(train_environments))

    return p_values



def euristic_ICP(ordered_names, original_names, train_environments, first_n_elems=100, epochs_regression=1000):   
    # Train ICP
    x_all = []
    y_all = []
    e_all = []

    for e, env in enumerate(train_environments):
        x, y = env.get_all()
        x_all.append(x.numpy())
        y_all.append(y.numpy())
        e_all.append(np.full(x.shape[0], e))

    x_all = np.vstack(x_all)
    y_all = np.vstack(y_all)
    e_all = np.hstack(e_all)

    dim = x_all.shape[1]
    
    selected_names = []
    max_p_value = 0
    counter = 0

    for j in range(first_n_elems):#len(ordered_names)):
        
        selected_names.append(ordered_names[j])
        print(j/len(ordered_names))
        bool_names = []
        for original_name in original_names:
            if original_name in selected_names:
                bool_names.append(True)
            else:
                bool_names.append(False)

        x_s = x_all[:, bool_names]
#         print(x_s.shape)
        #reg = LinearRegression(fit_intercept=False).fit(x_s, y_all)

        reg = TorchLinearRegressionModule(x_s.shape[1], y_all.shape[1], bias=False)
        reg.fit(x_s, y_all, epochs=epochs_regression) 
        p_values_ = []
        for e in range(len(train_environments)):
            e_in = np.where(e_all == e)[0]
            e_out = np.where(e_all != e)[0]

            res_in = (y_all[e_in] - reg.predict(x_s[e_in, :]).cpu().numpy()).ravel()
            res_out = (y_all[e_out] - reg.predict(x_s[e_out, :]).cpu().numpy()).ravel()

            p_values_.append(mean_var_test(res_in, res_out))

        p_value = min(p_values_) * len(train_environments)
        
        if p_value > max_p_value:
            max_p_value=p_value
        else:
            selected_names.pop(-1)
        
    return selected_names
    