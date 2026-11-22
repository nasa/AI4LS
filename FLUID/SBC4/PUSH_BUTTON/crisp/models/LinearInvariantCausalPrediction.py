from itertools import chain, combinations
import numpy as np
import torch
from scipy.stats import f as fdist
from scipy.stats import ttest_ind
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import confusion_matrix

from utils.defining_sets import defining_sets


def pretty(vector):
    vlist = vector.view(-1).tolist()
    return "[" + ", ".join("{:+.4f}".format(vi) for vi in vlist) + "]"


"""
ICP class to use when dataset fits in memory
"""


class InvariantCausalPrediction(object):
    def __init__(self, train_environments, val_environment, test_environment, args):
        self.coefficients = None
        self.intersection_found = False
        self.defining_set_found = False
        self.alpha = args.get('alpha', 0.05)
        self.selected_features = None
        self.p_value = None
        self.model = None
        self.full_feature_set = train_environments[0].predictor_columns.copy()

        torch.manual_seed(args.get('seed', 0))
        np.random.seed(args.get('seed', 0))

        # Set up test set
        x_test, y_test = test_environment.get_all()
        self.x_test, self.y_test = x_test.numpy(), y_test.numpy()

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

        self.x_all = x_all
        self.y_all = y_all

        dim = x_all.shape[1]
        self.max_set_size = args.get('max_set_size', int(dim))

        accepted_subsets = []
        counter = 0
        subsets = self.powerset(range(dim), self.max_set_size)
        print('Testing ', len(list(subsets)), ' permutations with max set size: ', self.max_set_size)
        for subset in self.powerset(range(dim), self.max_set_size):
            if len(subset) == 0:
                continue

            x_s = x_all[:, subset]
            reg = LinearRegression(fit_intercept=False).fit(x_s, y_all)

            p_values = []
            for e in range(len(train_environments)):
                e_in = np.where(e_all == e)[0]
                e_out = np.where(e_all != e)[0]

                res_in = (y_all[e_in] - reg.predict(x_s[e_in, :])).ravel()
                res_out = (y_all[e_out] - reg.predict(x_s[e_out, :])).ravel()
                
                p_values.append(self.mean_var_test(res_in, res_out))
                
            p_value = min(p_values) * len(train_environments)
            
            # p_value is the probablity that there are more extreme cases -> the higher
            # the more likely that the residuals have all the same distribution in the
            # different environments
            
            if p_value > self.alpha:
                accepted_subsets.append(set(subset))
                if args["verbose"]:
                    print("Accepted subset:", subset, np.array(self.full_feature_set)[list(subset)])
            else:
                if counter % 10000 == 0:
                    print('Rejected subset', counter)
            counter += 1

        if len(accepted_subsets):
            intersection_features = list(set.intersection(*accepted_subsets))
            if args["verbose"]:
                print("Intersection:", intersection_features)
            self.coefficients = np.zeros(dim)

            if len(intersection_features):
                x_s = x_all[:, list(intersection_features)]
                reg = LinearRegression(fit_intercept=False).fit(x_s, y_all)
                self.coefficients[list(intersection_features)] = reg.coef_
                self.intersection_found = True
                self.selected_features = list(intersection_features)
            else:
                self.intersection_found = False
                accepted_subsets = [list(s) for s in accepted_subsets]
                print('No intersection, trying defining sets')
                def_sets = list(defining_sets(accepted_subsets))
                print('Found the defining sets!')
                def_sets = [[int(el) for el in s] for s in def_sets]
                if len(def_sets):
                    self.defining_set_found = True
                    if args["verbose"]:
                        print("Defining Sets:", def_sets)

                    def_err_values = []
                    def_p_values = []
                    for s in def_sets:

                        x_s = x_all[:, s]
                        reg = LinearRegression(fit_intercept=False).fit(x_s, y_all)

                        p_values = []
                        total_error_values = []
                        for e in range(len(train_environments)):
                            e_in = np.where(e_all == e)[0]
                            e_out = np.where(e_all != e)[0]

                            res_in = np.abs((y_all[e_in] - reg.predict(x_s[e_in, :])).ravel())
                            res_out = np.abs((y_all[e_out] - reg.predict(x_s[e_out, :])).ravel())

                            total_error_values.append(np.sum(np.sum(res_in) + np.sum(res_out)))

                            p_values.append(self.mean_var_test(res_in, res_out))
                        p_value = min(p_values) * len(train_environments)
                        def_p_values.append(p_value)
                        def_err_values.append(np.sum(total_error_values))

                    best_def_set = def_sets[np.where(def_err_values == min(def_err_values))[0][0]]
                    best_p_value = def_p_values[np.where(def_err_values == min(def_err_values))[0][0]]
                    best_def_set.sort()

                    self.coefficients = np.zeros(dim)

                    if len(best_def_set):
                        x_s = x_all[:, list(best_def_set)]
                        reg = LinearRegression(fit_intercept=False).fit(x_s, y_all)
                        self.coefficients[list(best_def_set)] = reg.coef_
                        self.selected_features = list(best_def_set)
                        self.p_value = best_p_value

            self.coefficients = torch.Tensor(self.coefficients)
        else:
            self.coefficients = torch.zeros(dim)

        if self.selected_features:
            self.model = LogisticRegression(penalty='l1', solver='liblinear').fit(x_all[:, self.selected_features],
                                                                                  y_all.astype('int'))
            self.test()
        else:
            print("No accepted sets found, please consider decreasing {alpha} or try non-linear ICP")

    def test(self):
        test_targets = self.y_test
        test_logits = self.model.predict(self.x_test[:, self.selected_features])
        test_probs = self.model.predict_proba(self.x_test[:, self.selected_features])

        conf_matrix = confusion_matrix(y_true=self.y_test, y_pred=test_logits)
        self.conf_matric = conf_matrix
        self.test_targets = torch.Tensor(test_targets).squeeze()
        self.test_logits = torch.Tensor(test_logits).squeeze()
        self.test_probs = torch.Tensor(test_probs)

    def coef_(self):
        if self.model:
            return self.model.coef_[0]
        else:
            return self.coefficients

    def results(self):

        if self.intersection_found or self.defining_set_found:
            test_nll = self.mean_nll(self.test_logits, self.test_targets)
            test_acc = self.mean_accuracy(self.test_logits, self.test_targets)
            test_acc_std = self.std_accuracy(self.test_logits, self.test_targets)

            return {

                "solution": self.intersection_found or self.defining_set_found,
                "intersection": self.intersection_found,
                "test_acc": test_acc.numpy().squeeze().tolist(),
                "test_nll": test_nll,
                "test_probs": self.test_probs,
                "test_labels": self.test_targets,
                "feature_coeffients": self.coef_(),
                "selected_features": np.array(self.full_feature_set)[self.selected_features],
                "selected_feature_indices": self.selected_features,
                "to_bucket": {
                    'method': 'Linear ICP',
                    'features': np.array(self.full_feature_set)[self.selected_features].tolist(),
                    'coefficients': np.array(self.coef_()).tolist(),
                    'pvals': np.array(self.p_value).tolist(),
                    'test_acc': test_acc.numpy().squeeze().tolist(),
                    'test_acc_std': test_acc_std.numpy().squeeze().tolist()
                }
            }
        else:
            return {
                "solution": False,
                "intersection": False,
                "test_acc": 0,
                "test_nll": 1e9,
                "test_probs": None,
                "test_labels": None,
                "feature_coeffients": self.coef_(),
                "selected_features": None,
                "selected_feature_indices": None,
                "to_bucket": {
                    'method': 'Linear ICP',
                    'features': None,
                    'coefficients': None,
                    'pvals': None,
                    'test_acc': None,
                    'test_acc_std': None
                }
            }

    def mean_nll(self, logits, y):
        return torch.nn.functional.binary_cross_entropy_with_logits(logits, y)

    def mean_accuracy(self, logits, y):
        preds = (logits > 0.).float()
        return ((preds - y).abs() < 1e-2).float().mean()

    def std_accuracy(self, logits, y):
        preds = (logits > 0.).float()
        return ((preds - y).abs() < 1e-2).float().std()

    def mean_var_test(self, x, y):
        # get the probability that there are more extreme cases of variables than x and y
        # considering that x and y come from the same mean
        pvalue_mean = ttest_ind(x, y, equal_var=False).pvalue
        
        # consider the F-test https://en.wikipedia.org/wiki/F-test
        # why not using simply the f_oneway function
        pvalue_var1 = 1 - fdist.cdf(np.var(x, ddof=1) / np.var(y, ddof=1),
                                    x.shape[0] - 1,
                                    y.shape[0] - 1)
        # To adress the fact that x could smaller than the media
        # then we need to watch the left tail.
        pvalue_var2 = 2 * min(pvalue_var1, 1 - pvalue_var1)
        return 2 * min(pvalue_mean, pvalue_var2)

    def powerset(self, s, max_set_size):
        return chain.from_iterable(combinations(s, r) for r in range(max_set_size + 1))