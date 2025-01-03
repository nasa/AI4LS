from itertools import chain, combinations

import numpy as np
import pandas as pd
import torch
from scipy.stats import levene, ranksums

from models.TorchModelZoo import MLP
from utils.defining_sets import defining_sets


def pretty(vector):
    vlist = vector.view(-1).tolist()
    return "[" + ", ".join("{:+.4f}".format(vi) for vi in vlist) + "]"


class NonLinearInvariantCausalPrediction(object):
    def __init__(self, train_environments, val_environment, test_environment,
                 args):  # define model nn.Module to fit to? default to MLP
        print("Non linear ICP with ", args["output_data_regime"], "target.")
        self.intersection_found = False
        self.defining_set_found = False
        self.alpha = args.get('alpha', 0.05)
        self.accepted_p_values = []
        self.accepted_subsets = []
        self.selected_features = []
        self.selected_p_value = None
        self.seed = args.get('seed', None)
        self.method = args.get('method', 'MLP')
        self.args = args
        self.cuda = torch.cuda.is_available() and args.get('cuda', False)
        self.max_iter = args.get('max_iter', 1000)
        self.full_feature_set = train_environments[0].predictor_columns.copy()
        

        self.output_dim = train_environments[0].get_output_dim()
        if self.args["output_data_regime"] == "multi-class":
            self.output_dim = len(np.unique(train_environments[0].targets))
        self.input_dim = train_environments[0].get_feature_dim()
        # Initialise Dataloaders (combined all environment for train; separated by environment for comparison across envs)
        self.batch_size = args.get('batch_size', 128)
        all_dataset = torch.utils.data.ConcatDataset(train_environments)
        self.all_loader = torch.utils.data.DataLoader(all_dataset, batch_size=self.batch_size, shuffle=True)
        train_loaders = []
        for ds in train_environments:
            dl = torch.torch.utils.data.DataLoader(ds, batch_size=self.batch_size)
            train_loaders.append(dl)
        self.train_loaders = train_loaders
        self.val_loader = torch.utils.data.DataLoader(val_environment, batch_size=self.batch_size, shuffle=False)
        self.test_loader = torch.utils.data.DataLoader(test_environment, batch_size=self.batch_size, shuffle=False)
        self.test_environment = test_environment
        
        # test every combination of features (up to max_set_size)
        self.max_set_size = (args['max_set_size'] if 'max_set_size' in args else int(self.input_dim))

        torch.manual_seed(args.get('seed', 0))
        np.random.seed(args.get('seed', 0))

        for subset in self.powerset(range(self.input_dim), self.max_set_size):
            if len(subset) == 0:
                continue

            # fit model on all_loader with this subset/feature mask
            self.feature_mask = subset
            # Initialise Model
            self.initialize_model()
            self.train(self.all_loader)

            # loop through each environment in train_environments, get true/preds and residuals within that env, map residuals to env id
            res_all = []
            e_all = []
            for e in range(len(self.train_loaders)):
                residuals = self.get_residuals(self.train_loaders[e])
                res_all.extend(residuals)
                e_all.extend([e] * len(residuals))

            # get p-value
            p_value = self.leveneAndWilcoxTest(res_all, e_all)

            if p_value > self.alpha:
                self.accepted_subsets.append(set(subset))
                self.accepted_p_values.append(p_value)
                if args["verbose"]:
                    print("Accepted subset:", subset)

        if len(self.accepted_subsets):
            intersection_features = list(set.intersection(*self.accepted_subsets))
            if args["verbose"]:
                print("Intersection:", intersection_features)

            if len(intersection_features):
                # fit model on all_loader with this subset/feature mask
                self.feature_mask = intersection_features
                ###fit nonlinear model here###
                self.initialize_model()
                self.train(self.all_loader)
                self.intersection_found = True
                self.selected_features = list(intersection_features)

                # loop through each environment in train_environments, get true/preds and residuals within that env, map residuals to env id
                res_all = []
                e_all = []
                for e in range(len(self.train_loaders)):
                    residuals = self.get_residuals(self.train_loaders[e])
                    res_all.extend(residuals)
                    e_all.extend([e] * len(residuals))

                # get p-value
                p_value = self.leveneAndWilcoxTest(res_all, e_all)
                self.selected_p_value = p_value

            else:
                self.intersection_found = False
                accepted_subsets = [list(s) for s in self.accepted_subsets]
                def_sets = list(defining_sets(accepted_subsets))
                def_sets = [[int(el) for el in s] for s in def_sets]
                if len(def_sets):
                    self.defining_set_found = True
                    if args["verbose"]:
                        print("Defining Sets:", def_sets)

                    def_p_values = []
                    for s in def_sets:
                        # fit model on all_loader with this subset/feature mask
                        self.feature_mask = s
                        ###fit nonlinear model here###
                        self.initialize_model()
                        self.train(self.all_loader)

                        # loop through each environment in train_environments, get true/preds and residuals within that env, map residuals to env id
                        res_all = []
                        e_all = []
                        for e in range(len(self.train_loaders)):
                            residuals = self.get_residuals(self.train_loaders[e])
                            res_all.extend(residuals)
                            e_all.extend([e] * len(residuals))

                        # get p-value
                        p_value = self.leveneAndWilcoxTest(res_all, e_all)
                        def_p_values.append(p_value)

                    best_def_set = def_sets[np.where(def_p_values == max(def_p_values))[0][0]]
                    best_def_set.sort()

                    if len(best_def_set):
                        self.feature_mask = list(best_def_set)
                        ###fit nonlinear model here###
                        self.initialize_model()
                        self.train(self.all_loader)

                        # loop through each environment in train_environments, get true/preds and residuals within that env, map residuals to env id
                        res_all = []
                        e_all = []
                        for e in range(len(self.train_loaders)):
                            residuals = self.get_residuals(self.train_loaders[e])
                            res_all.extend(residuals)
                            e_all.extend([e] * len(residuals))

                        # get p-value
                        p_value = self.leveneAndWilcoxTest(res_all, e_all)
                        self.selected_p_value = p_value

                        self.defining_set_found = True
                        self.selected_features = list(best_def_set)

            # test consensus model and return results
            self.test(loader=self.test_loader)
        else:
            print('no accepted sets found for nonlinear ICP')

    def leveneAndWilcoxTest(self, residuals, e_all):
        residuals = np.array(residuals)
        e_all = np.array(e_all)
        # get levene p-value
        res_groups = []
        wilcox_p = 1
        for e in range(len(set(e_all))):
            # for each environment, test invariance and record p-value
            e_in = np.where(e_all == e)  # in-group env indeces
            e_out = np.where(e_all != e)  # out-group env indeces
            res_in = residuals[e_in]
            res_out = residuals[e_out]
            res_groups.append(res_in)
            # wilcoxon rank sums test - to test that residuals drawn from same distribution
            stat, w_p = ranksums(res_in, res_out)
            wilcox_p = min(w_p, wilcox_p)
        # levene test - to test that residuals from all envs have equal variances
        W, levene_p = levene(*res_groups, center='mean')

        # bonf adj wilcoxon test if test multiple environments
        bonf_adj = (1 if len(set(e_all)) == 2 else len(set(e_all)))
        wilcox_p = wilcox_p * bonf_adj
        # accept minimum p-value of wilcoxon and levene tests; 2* is for bonferroni correction for the two tests
        p_value = 2 * min(wilcox_p, levene_p)

        return p_value

    def powerset(self, s, max_set_size):
        return chain.from_iterable(combinations(s, r) for r in range(max_set_size + 1))

    def initialize_model(self):
        if self.method == 'MLP':
            self.input_dim = len(self.feature_mask)
            self.model = MLP(self.args, self.input_dim, self.output_dim)
            if self.cuda:
                self.model.cuda()
        else:
            print('model not supported yet, use MLP')

    def train(self, loader=None):
        if loader is None:
            loader = self.train_loader

        epochs = self.args.get('epochs', 100)  # maybe reduce default to speed up?
        lr = self.args.get('lr', 0.001)
        if self.args["output_data_regime"] == "binary":
            criterion = torch.nn.BCEWithLogitsLoss()  # todo: maybe detect what loss to use for binary/multi-class cases
        elif self.args["output_data_regime"] == "real-valued":
            criterion = torch.nn.MSELoss()
        elif self.args["output_data_regime"] == "multi-class":
            criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

        # Start training loop
        i = 0
        for epoch in range(epochs):
            for i, (inputs, targets) in enumerate(loader):
                if self.cuda:
                    if self.feature_mask:
                        inputs, targets = inputs[:, self.feature_mask].cuda(), targets.cuda()
                    else:
                        inputs, targets = inputs.cuda(), targets.cuda()
                else:
                    if self.feature_mask:
                        inputs, targets = inputs[:, self.feature_mask], targets

                optimizer.zero_grad()
                outputs = self.model(inputs)
                if self.args["output_data_regime"] == "multi-class":
                    targets = targets.squeeze().long()
                    outputs = outputs.argmax(dim=1)
                    print(targets.shape, outputs.shape)
                    print(targets[:3], outputs[:3])
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                i += 1
                if i > self.max_iter:
                    break

    def get_residuals(self, loader):
        res = []
        i = 0
        with torch.no_grad():
            for inputs, targets in loader:
                inputs = inputs[:, self.feature_mask]
                if self.cuda:
                    inputs = inputs.to("cuda")
                pred = self.model(inputs)
                pred = pred.cpu().numpy()
                err = (pred.flatten() - targets.squeeze().numpy().flatten()).flatten()
                res.extend(err)
                i += 1
                if i > self.max_iter:
                    print('max iterations hit')
                    break
        return res

    def test(self, loader):

        test_targets = []
        test_logits = []
        test_probs = []

        sig = torch.nn.Sigmoid() 

        with torch.no_grad():
            for i, (inputs, targets) in enumerate(loader):
                if self.cuda:
                    if self.feature_mask:
                        inputs, targets = inputs[:, self.feature_mask].cuda(), targets.cuda()
                    else:
                        inputs, targets = inputs.cuda(), targets.cuda()
                else:
                    if self.feature_mask:
                        inputs, targets = inputs[:, self.feature_mask], targets

                outputs = self.model(inputs)

                if self.cuda:
                    test_targets.append(targets.squeeze().unsqueeze(0))
                    test_logits.append(outputs.cpu().squeeze().unsqueeze(0))
                    test_probs.append(sig(outputs).cpu().squeeze().unsqueeze(0))
                else:
                    test_targets.append(targets.squeeze().unsqueeze(0))
                    test_logits.append(outputs.squeeze().unsqueeze(0))
                    test_probs.append(sig(outputs).squeeze().unsqueeze(0))

        self.test_targets = torch.cat(test_targets, dim=1)
        self.test_logits = torch.cat(test_logits, dim=1)
        self.test_probs = torch.cat(test_probs, dim=1)

    def results(self):
        test_nll = self.mean_nll(self.test_logits, self.test_targets)
        test_acc = self.mean_accuracy(self.test_probs, self.test_targets)
        test_acc_std = self.std_accuracy(self.test_probs, self.test_targets)
        if len(self.selected_features):
            overall_sties, sties, npcorr = self.get_sensitivities()
            overall_sties = overall_sties.squeeze().tolist()
            npcorr = npcorr.tolist()
            sties = sties.tolist()

        return {
            "solution": self.intersection_found or self.defining_set_found,
            "intersection": self.intersection_found,
            "test_acc": test_acc.numpy().squeeze().tolist(),
            "test_acc_std": test_acc_std.numpy().squeeze().tolist(),
            "test_nll": test_nll,
            "test_probs": self.test_probs,
            "test_labels": self.test_targets,
            "feature_coeffients": None,
            "selected_p_value": self.selected_p_value,
            "selected_features": np.array(self.full_feature_set)[self.selected_features],
            "selected_feature_indices": self.selected_features,
            "to_bucket": {
                'method': "Non-Linear ICP",
                'features': list(np.array(self.full_feature_set)[self.selected_features]),
                'coefficients': overall_sties if len(self.selected_features) > 0 else None,
                'pvals': self.selected_p_value,
                'test_acc': test_acc.numpy().squeeze().tolist(),
                'test_acc_std': test_acc_std.numpy().squeeze().tolist(),
                'coefficient_correlation_matrix': npcorr if len(self.selected_features) > 0 else None,
                'test_data_sensitivities': sties if len(self.selected_features) > 0 else None
            }
        }

    def acc_preds(self, logits, y):
        if self.args["output_data_regime"] == "multi-class":
            return logits.argmax(dim=-1).float()
        elif self.args["output_data_regime"] == "real-valued":
            return logits.float()
        else:
            # binary classification case
            return  (logits > 0.).float()
    
    def mean_nll(self, logits, y):
        if self.args["output_data_regime"] == "multi-class":
            return CrossEntropyLoss()(logits.squeeze(), y.squeeze().long())
        else:
            return torch.nn.functional.binary_cross_entropy_with_logits(logits, y)

    def mean_accuracy(self, logits, y):
        preds = self.acc_preds(logits, y)
        if self.args["output_data_regime"] == "real-valued":
            return r2_score(y, preds)

    def std_accuracy(self, logits, y):
        preds = (logits > 0.).float()
        return ((preds - y).abs() < 1e-2).float().std()

    def get_sensitivities(self):
        dim = len(self.selected_features)

        # get min and max dim for each input feature across all_loader
        self.min_per_dim = np.ones(shape=(dim, 1)) * (100000.)
        self.max_per_dim = np.ones(shape=(dim, 1)) * (-100000.)
        for inputs, targets in self.all_loader:
            if self.cuda:
                if self.selected_features:
                    inputs, targets = inputs[:, self.selected_features].cuda(), targets.cuda()
                else:
                    inputs, targets = inputs.cuda(), targets.cuda()
            else:
                if self.selected_features:
                    inputs, targets = inputs[:, self.selected_features], targets

            for ii in range(len(self.selected_features)):
                dimmin = min(inputs.numpy()[:, ii])
                dimmax = max(inputs.numpy()[:, ii])
                if dimmin < self.min_per_dim[ii]:
                    self.min_per_dim[ii] = dimmin
                if dimmax > self.max_per_dim[ii]:
                    self.max_per_dim[ii] = dimmax

        overall_sties = np.zeros(shape=(dim,))
        sig = torch.nn.Sigmoid()
        with torch.no_grad():
            for i, (inputs, targets) in enumerate(self.test_loader):
                if self.cuda:
                    if self.selected_features:
                        inputs, targets = inputs[:, self.selected_features].cuda(), targets.cuda()
                    else:
                        inputs, targets = inputs.cuda(), targets.cuda()
                else:
                    if self.selected_features:
                        inputs, targets = inputs[:, self.selected_features], targets

                n = inputs.size(0)
                if self.cuda:
                    inputs = inputs.cuda()

                for ii in range(dim):
                    temp = inputs.clone()
                    temp[:, ii] = float(self.min_per_dim[ii])
                    outputs1 = self.model(temp)
                    temp[:, ii] = float(self.max_per_dim[ii])
                    outputs2 = self.model(temp)

                    overall_sties[ii] += (torch.sum(outputs2 - outputs1).numpy() / n) 

        # for items in test set, get feature correlation matrix of sensitivies
        x_test, y_test = self.test_environment.get_all()
        x_test = x_test[:, self.selected_features]
        sties = np.zeros(shape=x_test.shape)
        with torch.no_grad():
            for ii in range(dim):
                temp = x_test.clone()
                temp[:, ii] = float(self.min_per_dim[ii])
                outputs1 = self.model(temp)
                temp[:, ii] = float(self.max_per_dim[ii])
                outputs2 = self.model(temp)

                sties[:, ii] = (outputs2 - outputs1).numpy().ravel() 

        df_test = pd.DataFrame(sties, columns=np.array(self.full_feature_set)[self.selected_features])
        corr = df_test.corr()
        corr = corr.fillna(0)
        npcorr = np.array(corr)

        return overall_sties, sties, npcorr
