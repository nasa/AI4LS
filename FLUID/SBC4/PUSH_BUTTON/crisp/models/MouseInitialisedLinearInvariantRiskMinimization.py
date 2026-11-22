import numpy as np
import pandas as pd
import torch
from sklearn.metrics import r2_score 
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss

from .LinearInvariantRiskMinimization import LinearInvariantRiskMinimization

class MouseInitialiser(LinearInvariantRiskMinimization):
    def __init__(self, environment_datasets, val_dataset, test_dataset, args):
        super().__init__(environment_datasets, val_dataset, test_dataset, args)
    
        self.mouse_phi = self.phi.detach()


class MouseInitialisedLinearInvariantRiskMinimization(object):
    def __init__(self, environment_datasets, val_dataset, test_dataset, args, mouse_phi):
        self.args = args
        self.cuda = torch.cuda.is_available() and args.get('cuda', False)
        self.input_dim = environment_datasets[0].get_feature_dim()
        self.output_dim = len(np.unique(environment_datasets[0].targets))
        self.test_dataset = test_dataset
        self.args = args
        self.initialised_phi = mouse_phi

        self.use_icp_init = args.get('use_icp_initialization', False)
        if self.use_icp_init:
            self.icp_weights = args['ICP_weights']
            self.icp_weight_indices = args['ICP_weight_indices']

        self.feature_names = test_dataset.predictor_columns

        # Initialise Dataloaders (combine all environment datasets to as train)  
        self.batch_size = args.get('batch_size', 128)
        all_dataset = torch.utils.data.ConcatDataset(environment_datasets)
        self.all_loader = torch.utils.data.DataLoader(all_dataset, batch_size=self.batch_size, shuffle=True)
        train_loaders = []
        for ds in environment_datasets:
            dl = torch.torch.utils.data.DataLoader(ds, batch_size=self.batch_size)
            train_loaders.append(dl)
        self.train_loaders = train_loaders
        self.test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)

        # Start training procedure
        print("Start IRM training procedure with", self.args["output_data_regime"], "target")

        dim_x = self.input_dim + 1  
        best_phi = self.initialised_phi
        best_reg = 0
        best_err = 1e6

        # Penalty regularization parameter is an important hyperparameter. Grid search is performed to find the optimal hyperparameter.
        for reg in [0, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]:
            self.reg = reg
            self.train()

            error = 0
            for inputs, targets in self.all_loader:
                bias = torch.ones(inputs.size(0), 1)
                if self.cuda:
                    inputs = inputs.cuda()
                    targets = targets.cuda()
                    bias = bias.cuda()
                # Concatenating a vector of 1's to account for bias in linear classification
                inputs = torch.cat((bias, inputs), 1) 
                if self.args["output_data_regime"] == "real-valued":
                    error = (inputs @ self.solution() - targets).pow(2).mean().item()
                elif self.args["output_data_regime"] == "binary":
                    error = BCEWithLogitsLoss()(inputs @ self.solution(), targets).detach()
                elif self.args["output_data_regime"] == "multi-class":
                    targets = targets.squeeze().long()
                    error = CrossEntropyLoss()(inputs @ self.solution(), targets).detach()
                    
                error += error
            
            if args["verbose"]:
                print("IRM (reg={:.3f}) has {:.3f} validation error.".format(
                        reg, error))

            if error < best_err:
                if args["verbose"]:
                    print("\tnew best model:")
                    print("\terror:", error)
                    print("\treg:", reg)
                    print("\tphi.trace:", torch.trace(self.phi))
                best_err = error
                best_reg = reg
                best_phi = self.phi.clone()

        self.phi = best_phi
        self.reg = best_reg

        # Start testing procedure
        self.test(self.test_loader)

    def train(self):
        dim_x = self.input_dim + 1  
        dim_y = self.output_dim

        if self.cuda:
            self.phi = torch.nn.Parameter(self.initialised_phi.cuda())
            if self.use_icp_init:
                self.w = torch.ones(dim_x, 1).cuda()
                self.w[self.icp_weight_indices] = torch.Tensor(
                    np.sign(self.icp_weights) * (1 + self.icp_weights)).unsqueeze(-1)
            else:
                self.w = torch.ones(dim_x, 1).cuda()
                if self.args["output_data_regime"] == "multi-class":
                    self.w = (torch.randn(dim_x, dim_y).cuda())#.cuda()
        else:
            self.phi = torch.nn.Parameter(self.initialised_phi)
            if self.use_icp_init:
                self.w = torch.ones(dim_x, 1)
                self.w[self.icp_weight_indices] = torch.Tensor(
                    np.sign(self.icp_weights) * (1 + self.icp_weights)).unsqueeze(-1)
            else:
                self.w = torch.ones(dim_x, 1)
                if self.args["output_data_regime"] == "multi-class":
                    self.w = torch.randn(dim_x, dim_y)# / dim_y
        
        self.w.requires_grad = True

        opt = torch.optim.Adam([self.phi], lr=self.args["lr"])
        
        if self.args["output_data_regime"] == "real-valued":
            loss = torch.nn.MSELoss()
        elif self.args["output_data_regime"] == "multi-class":
            loss = torch.nn.CrossEntropyLoss()
        elif self.args["output_data_regime"] == "binary":
            loss = torch.nn.BCEWithLogitsLoss()
        else:
            raise NotImplementedError("IRM supports real-valued, binary, and multi-class target, not " + str(self.args["output_data_regime"]))

        for iteration in range(self.args["n_iterations"]):
            penalty = 0
            error = 0
            for env_loader in self.train_loaders:
                for inputs, targets in env_loader:
                    inputs = torch.cat((torch.ones(inputs.size(0), 1), inputs), 1)  ##
                    if self.cuda:
                        inputs = inputs.cuda()
                        targets = targets.cuda()
                        
                    if self.args["output_data_regime"] == "multi-class":
                        targets = targets.squeeze().long()

                    error_e = loss(inputs @ self.phi @ self.w, targets)
                    penalty += torch.autograd.grad(error_e, self.w,
                                                   create_graph=True)[0].pow(2).mean()
                    error += error_e

            if self.args["verbose"] and iteration % 200 == 0:
                print("iteration:", iteration, "error:", error.item())#, "r2:", r2_score(targets.detach().numpy(), pred.detach().numpy()))
                
            opt.zero_grad()
            (self.reg * error + (1 - self.reg) * penalty).backward()
            opt.step()

    def solution(self):
        return self.phi @ self.w

    def test(self, loader):

        test_targets = []
        test_logits = []
        test_probs = []

       # if self.args["output_data_regime"] == "real-valued":
        #    sig = torch.nn.Identity()
        #else:
            # ??
        sig = torch.nn.Sigmoid()

        with torch.no_grad():
            for i, (inputs, targets) in enumerate(self.test_loader):
                inputs = torch.cat((torch.ones(inputs.size(0), 1), inputs), 1)  ##
                if self.cuda:
                    inputs = inputs.cuda()

                outputs = inputs @ self.phi @ self.w

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

    #         print('Finished Testing')

    def results(self):
        test_nll = self.mean_nll(self.test_logits.squeeze(), self.test_targets.squeeze())
        test_acc = self.mean_accuracy(self.test_logits.squeeze(), self.test_targets.squeeze())
        test_acc_std = self.std_accuracy(self.test_logits.squeeze(), self.test_targets.squeeze())
        coefficients = self.solution().detach().cpu().numpy().squeeze()[1:].tolist()
        if self.args["output_data_regime"] == "multi-class":
            npcorr = None
        else:
            npcorr = self.get_corr_mat()


        return {
            "test_logits" : self.test_logits.squeeze().numpy().tolist(),
            "test_acc": test_acc, #test_acc.numpy().squeeze().tolist(),
            "test_nll": test_nll.item(),
            "test_probs": self.test_probs.squeeze().numpy().tolist(),
            "test_labels": self.test_targets.squeeze().numpy().tolist(),
            "feature_coeffients": self.solution().detach().cpu().numpy().squeeze()[1:].tolist(),
            'to_bucket': {
                "test_logits" : self.test_logits.squeeze().numpy().tolist(),
                'method': "Linear IRM",
                'features': self.feature_names,
                'coefficients': coefficients,
                'pvals': None,
                "test_logits" : self.test_logits.squeeze().numpy().tolist(),
                '''odhran - changed below back to list object from tensor'''
                'test_acc': test_acc.numpy().squeeze().tolist(),
                '''odhran - changed above back to list object from tensor'''
                'test_acc_std': test_acc_std.item(),# ,.numpy().squeeze().tolist(),
                'coefficient_correlation_matrix': npcorr
            }
        }

    def mean_nll(self, logits, y):
        if self.args["output_data_regime"] == "multi-class":
            return CrossEntropyLoss()(logits.squeeze(), y.squeeze().long())
        else:
            return torch.nn.functional.binary_cross_entropy_with_logits(logits, y)

    def acc_preds(self, logits, y):
        if self.args["output_data_regime"] == "multi-class":
            return logits.argmax(dim=-1).float()
        elif self.args["output_data_regime"] == "real-valued":
            return logits.float()
        else:
            # binary classification case
            return  (logits > 0.).float()

    def mean_accuracy(self, logits, y):
        preds = self.acc_preds(logits, y)
        if self.args["output_data_regime"] == "real-valued":
            return r2_score(y, preds)
        
        return ((preds - y).abs() < 1e-2).float().mean()

    def std_accuracy(self, logits, y):
        preds = self.acc_preds(logits, y)
        return ((preds - y).abs() < 1e-2).float().std()

    def pretty(self, vector):
        vlist = vector.view(-1).tolist()
        return "[" + ", ".join("{:+.4f}".format(vi) for vi in vlist) + "]"

    def get_corr_mat(self):
        x_test, y_test = self.test_dataset.get_all()
        inputs = torch.cat((torch.ones(x_test.size(0), 1), x_test), 1)  ##
        coefs = self.solution().detach().cpu().numpy().squeeze()
        with torch.no_grad():
            outputs = inputs * coefs
            outputs = outputs[:, 1:]
            sties_corr = outputs.numpy()

        df_test = pd.DataFrame(sties_corr, columns=np.array(self.feature_names))
        corr = df_test.corr()
        corr = corr.fillna(0)
        npcorr = np.array(corr).tolist()

        return npcorr

