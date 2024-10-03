import numpy as np
import pandas as pd
import torch
from sklearn.metrics import r2_score, mean_squared_error
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss
from sklearn.metrics import confusion_matrix


class LinearInvariantRiskMinimization(object):
    def __init__(self, environment_datasets, val_dataset, test_dataset, args):
        self.cuda = torch.cuda.is_available() and args.get('cuda', False)
        self.error_env_list = list()
        self.input_dim = environment_datasets[0].get_feature_dim()
        # self.output_dim = self.args["output_dim"]
        self.output_dim = environment_datasets[0].get_output_dim()
        self.test_dataset = test_dataset
        self.args = args
        self.args['output_data_regime'] = 'binary'
        self.logging_iteration = args.get('logging_iteration', 200)
        self.loss_per_iteration = []
        self.acc_per_iteration = []
        self.confusion_matrix_test = list()


        torch.manual_seed(args.get('seed', 0))
        np.random.seed(args.get('seed', 0))

        self.feature_names = test_dataset.predictor_columns

        # Initialise Dataloaders (combine all environment datasets to as train)
        self.batch_size = args.get('batch_size', 128)
        self.all_dataset = torch.utils.data.ConcatDataset(environment_datasets)
        self.all_loader = torch.utils.data.DataLoader(self.all_dataset, batch_size=self.batch_size, shuffle=True)
        train_loaders = []
        for ds in environment_datasets:
            dl = torch.torch.utils.data.DataLoader(ds, batch_size=self.batch_size)
            train_loaders.append(dl)
        self.train_loaders = train_loaders
        self.test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)
        self.val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)

        #self.reg = 0.95
        # JC
        self.reg = 1e-4 
        self.train()


        # Start testing procedure
        self.test(self.test_loader)

        # JC
        self.validate(self.val_loader)

    def compute_penalty(losses, dummy_w):
        # g1 is the even indices, g2 is the odd indices
        g1 = torch.autograd.grad(losses[0::2].mean(), dummy_w, create_graph=True)[0]
        g2 = torch.autograd.grad(losses[1::2].mean(), dummy_w, create_graph=True)[0]
        return (g1 * g2).sum()

    def train(self):
        dim_x = self.input_dim + 1
        dim_y = self.output_dim

        if self.cuda:
            self.phi = torch.nn.Linear(dim_x, dim_x, bias=False).cuda()
            self.w = torch.ones(dim_x, 1).cuda()
            if self.args["output_data_regime"] == "multi-class":
                self.w = (torch.ones(dim_x, dim_y).cuda())  # .cuda()
        else:
            self.phi = torch.nn.Linear(dim_x, dim_x, bias=False)
            self.w = torch.ones(dim_x, 1)
            if self.args["output_data_regime"] == "multi-class":
                self.w = torch.ones(dim_x, dim_y)  # / dim_y

        self.w.requires_grad = True

        opt = torch.optim.Adam([self.phi.weight], lr=self.args["lr"])

        if self.args["output_data_regime"] == "real-valued":
            loss = torch.nn.MSELoss()
        elif self.args["output_data_regime"] == "multi-class":
            loss = torch.nn.CrossEntropyLoss()
        elif self.args["output_data_regime"] == "binary":
            loss = torch.nn.BCEWithLogitsLoss()
        else:
            raise NotImplementedError(
                "IRM supports real-valued, binary, and multi-class target, not " + str(self.args["output_data_regime"]))

        for iteration in range(self.args["n_iterations"]):
            penalty = 0
            error = 0
            count = 0
            err_env = []
            for env_loader in self.train_loaders:
                error_e = 0
                count_e = 0
                for inputs, targets in env_loader:
                    inputs = torch.cat((torch.ones(inputs.size(0), 1), inputs), 1)  ##
                    if self.cuda:
                        inputs = inputs.cuda()
                        targets = targets.cuda()

                    if self.args["output_data_regime"] == "multi-class":
                        targets = targets.squeeze().long()

                    pred = self.phi(inputs) @ self.w
                    error_e += loss(pred, targets)
                    count += 1
                    count_e += 1

                err_env.append(error_e.item() / count_e)

                penalty += torch.autograd.grad(outputs=error_e, inputs=self.w, create_graph=True)[0].pow(2).mean()
                error += error_e

            if iteration % self.logging_iteration == 0:
                if self.args["verbose"]:
                    print('logging accuracy and loss')
                self.test(self.test_loader)
                self.acc_per_iteration.append(
                    self.mean_accuracy(self.test_logits.squeeze(), self.test_targets.squeeze()))
                self.loss_per_iteration.append(error / len(env_loader))

                if self.args["verbose"]:
                    print('logging accuracy and loss', error.item() / count, penalty.item(), self.acc_per_iteration[-1])

            if self.args["verbose"] and iteration % 100 == 0:
                print("iteration:", iteration, "training error:", error.item() / count)

            opt.zero_grad()
            (self.reg * error / count + (1 - self.reg) * penalty / count).backward()
            opt.step()
            self.error_env_list.append(err_env)

    def solution(self):
        coef_unnormalized = sum([w.data.abs().sum(axis=1) for w in self.phi.parameters()])
        if self.cuda:
            W = torch.eye(self.input_dim + 1).cuda()
        else:
            W = torch.eye(self.input_dim + 1)

        for w in self.phi.parameters():
            W = W @ w.T

        # JC
        coef = W @ self.w
        # coef_unnormalized = w.data.abs().sum(axis=1) for w in self.phi.parameters()])
        coef = coef / coef.sum()
        return coef

    def get_sensitivities(self):
        sties = np.zeros(shape=(self.input_dim,))

        if self.args["output_data_regime"] == "real-valued":
            loss = torch.nn.MSELoss()
        elif self.args["output_data_regime"] == "multi-class":
            loss = torch.nn.CrossEntropyLoss()
        elif self.args["output_data_regime"] == "binary":
            loss = torch.nn.BCEWithLogitsLoss()
        else:
            raise NotImplementedError(
                "IRM supports real-valued, binary, and multi-class target, not " + str(self.args["output_data_regime"]))

        with torch.no_grad():
            for i, (inputs, targets) in enumerate(self.test_loader):
                n = inputs.size(0)
                inputs = torch.cat((torch.ones(inputs.size(0), 1), inputs), 1)  ##
                if self.cuda:
                    inputs = inputs.cuda()
                    targets = targets.cuda()
                # JC
                output = self.phi(inputs) @ self.w
                l0 = loss(output, targets)

                for ii in range(self.input_dim):
                    if False:
                        sties[ii] += 0
                    else:
                        temp = inputs.clone()
                        temp[:, ii] = float(self.min_per_dim[ii])
                        outputs1 = self.phi(temp) @ self.w
                        l1 = loss(outputs1, targets)
                        temp[:, ii] = float(self.max_per_dim[ii])
                        outputs2 = self.phi(temp) @ self.w
                        l2 = loss(outputs2, targets)

                        sties[ii] += torch.sum(l1 - l0 + l2 - l0).cpu().numpy()
        return sties

    def test(self, loader):
        print('calling test')
        test_targets = []
        test_logits = []
        test_probs = []

        # if self.args["output_data_regime"] == "real-valued":
        #    sig = torch.nn.Identity()
        sig = torch.nn.Sigmoid()

        with torch.no_grad():
            for i, (inputs, targets) in enumerate(self.test_loader):
                inputs = torch.cat((torch.ones(inputs.size(0), 1), inputs), 1)  ##
                if self.cuda:
                    inputs = inputs.cuda()

                outputs = self.phi(inputs) @ self.w

                if self.cuda:
                    test_targets.append(targets.squeeze().unsqueeze(0))
                    test_logits.append(outputs.cpu().squeeze().unsqueeze(0))
                    test_probs.append(sig(outputs).cpu().squeeze().unsqueeze(0))
                else:
                    test_targets.append(targets.squeeze().unsqueeze(0))
                    # JC
                    test_logits.append(outputs.squeeze().unsqueeze(0))
                    test_probs.append(sig(outputs).squeeze().unsqueeze(0))

        self.test_targets = torch.cat(test_targets, dim=1)
        self.test_logits = torch.cat(test_logits, dim=1)
        self.test_probs = torch.cat(test_probs, dim=1)

        # JC
        preds = (self.test_logits > 0.).float().tolist()[0]
        y = self.test_targets.tolist()[0]
        conf_matrix = confusion_matrix(y_true=y, y_pred=preds)
        self.confusion_matrix_test.append(conf_matrix)

    #         print('Finished Testing')

    def validate(self, loader):

        validate_targets = []
        validate_logits = []
        validate_probs = []

        # if self.args["output_data_regime"] == "real-valued":
        #    sig = torch.nn.Identity()
        sig = torch.nn.Sigmoid()

        with torch.no_grad():
            for i, (inputs, targets) in enumerate(self.val_loader):
                inputs = torch.cat((torch.ones(inputs.size(0), 1), inputs), 1)  ##
                if self.cuda:
                    inputs = inputs.cuda()

                outputs = self.phi(inputs) @ self.w

                if self.cuda:
                    validate_targets.append(targets.squeeze().unsqueeze(0))
                    validate_logits.append(outputs.cpu().squeeze().unsqueeze(0))
                    validate_probs.append(sig(outputs).cpu().squeeze().unsqueeze(0))
                else:
                    validate_targets.append(targets.squeeze().unsqueeze(0))
                    # JC
                    validate_logits.append(outputs.squeeze().unsqueeze(0))
                    validate_probs.append(sig(outputs).squeeze().unsqueeze(0))

        self.validate_targets = torch.cat(validate_targets, dim=1)
        self.validate_logits = torch.cat(validate_logits, dim=1)
        self.validate_probs = torch.cat(validate_probs, dim=1)

    #         print('Finished validating')

    def get_input_gradients(self):
        sig = torch.nn.Sigmoid()

        n_samples = len(self.all_dataset)

        input_gradients = torch.zeros((n_samples, self.input_dim))
        # print("input_gradients:", input_gradients.shape)

        if self.args["output_data_regime"] == "real-valued":
            criterion = torch.nn.MSELoss()
        elif self.args["output_data_regime"] == "multi-class":
            criterion = torch.nn.CrossEntropyLoss()
        elif self.args["output_data_regime"] == "binary":
            criterion = torch.nn.BCEWithLogitsLoss()

        for i, (inputs, targets) in enumerate(self.all_loader):
            inputs = torch.cat((torch.ones(inputs.size(0), 1), inputs), 1)  ##
            inputs.requires_grad = True

            if self.cuda:
                inputs = inputs.cuda()
                targets = targets.cuda()

            if self.args["output_data_regime"] == "multi-class":
                targets = targets.squeeze().long()

            pred = self.phi(inputs) @ self.w

            # JC fixed bug - don't take sigmoid here
            loss = criterion(pred, targets)
            # loss = criterion(sig(pred), targets)
            loss.backward()

            if self.cuda:
                grad = inputs.grad.squeeze().detach().cuda()
            else:
                grad = inputs.grad.squeeze().detach().cpu()
            # print("grad:", grad.shape)

            if np.ndim(grad) == 2:
                input_gradients[i * self.batch_size:i * self.batch_size + self.batch_size, :] = grad[:,
                                                                                                1:]  # don't record the gradient of the intercept
            else:
                break
                # return input_gradients.mean(dim=0)
        return input_gradients.norm(2, dim=0)
        # return torch.abs(input_gradients).sum(dim=0)

    def results(self):
        test_nll = self.mean_nll(self.test_logits.squeeze(), self.test_targets.squeeze())
        # test_acc = self.mean_accuracy(self.test_logits.squeeze(), self.test_targets.squeeze())
        tn = 0
        fp = 0
        fn = 0
        tp = 0
        for cm in self.confusion_matrix_test:
            tn += cm.ravel()[0]
            fp += cm.ravel()[1]
            fn += cm.ravel()[2]
            tp += cm.ravel()[3]
        test_acc = (tp + tn) / (tp + tn + fp + fn)
        test_acc_std = self.std_accuracy(self.test_logits.squeeze(), self.test_targets.squeeze())
        coefficients = self.solution().detach().cpu().numpy().squeeze()[1:].tolist()
        if self.cuda:
            feature_gradients = None
        else:
            feature_gradients = self.get_input_gradients().cpu().numpy().tolist()

        if self.args["output_data_regime"] == "multi-class":
            npcorr = None
        else:
            npcorr = self.get_corr_mat()

        print('accuracy: ', test_acc)
        return {
            #"test_logits": self.test_logits.squeeze().numpy().tolist(),
            #"test_acc": test_acc.numpy().squeeze().tolist(),
            "test_acc": test_acc,
            #"test_nll": test_nll.item(),
            "test_probs": self.test_probs.squeeze().numpy().tolist(),
            "test_labels": self.test_targets.squeeze().numpy().tolist(),
            "feature_coeffients": self.solution().detach().cpu().numpy().squeeze()[1:].tolist(),
            "loss_over_time": [x.tolist() for x in self.loss_per_iteration],
            'acc_over_time': [x.tolist() for x in self.acc_per_iteration],
            'to_bucket': {
                'method': "Linear IRM",
                'features': self.feature_names,
                'coefficients': coefficients,
                # 'feature_gradients' : feature_gradients,
                'pvals': None,
                #"test_logits": self.test_logits.squeeze().numpy().tolist(),
                #'test_acc': test_acc.numpy().squeeze().tolist(),
                'test_acc': test_acc,
                'test_acc_std': test_acc_std.item(),  # ,.numpy().squeeze().tolist(),
                "confusion_matrix_test": str(self.confusion_matrix_test)
                #'coefficient_correlation_matrix': None,
                # 'sensitivities': self.get_sensitivities().tolist()
            }
        }

    def validation_results(self):
        validate_nll = self.mean_nll(self.validate_logits.squeeze(), self.validate_targets.squeeze())
        validate_acc = self.mean_accuracy(self.validate_logits.squeeze(), self.validate_targets.squeeze())
        validate_acc_std = self.std_accuracy(self.validate_logits.squeeze(), self.validate_targets.squeeze())
        coefficients = self.solution().detach().cpu().numpy().squeeze()[1:].tolist()
        if self.cuda:
            feature_gradients = None
        else:
            feature_gradients = self.get_input_gradients().cpu().numpy().tolist()

        if self.args["output_data_regime"] == "multi-class":
            npcorr = None
        else:
            npcorr = self.get_corr_mat()

        print('validation accuracy: ', validate_acc.numpy().squeeze().tolist())
        return {
            "validate_logits": self.validate_logits.squeeze().numpy().tolist(),
            "validate_acc": validate_acc.numpy().squeeze().tolist(),
            "validate_nll": validate_nll.item(),
            "validate_probs": self.validate_probs.squeeze().numpy().tolist(),
            "validate_labels": self.validate_targets.squeeze().numpy().tolist(),
            "feature_coeffients": self.solution().detach().cpu().numpy().squeeze()[1:].tolist(),
            "val_loss_over_time": [x.tolist() for x in self.loss_per_iteration],
            'val_acc_over_time': [x.tolist() for x in self.acc_per_iteration],
            'validate_to_bucket': {
                'method': "Linear IRM",
                'features': self.feature_names,
                'coefficients': coefficients,
                # 'feature_gradients' : feature_gradients,
                'pvals': None,
                "validate_logits": self.validate_logits.squeeze().numpy().tolist(),
                'validate_acc': validate_acc.numpy().squeeze().tolist(),
                'validate_acc_std': validate_acc_std.item(),  # ,.numpy().squeeze().tolist(),
                'coefficient_correlation_matrix': None,
                # 'sensitivities': self.get_sensitivities().tolist()
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
            return (logits > 0.).float()

    def mean_accuracy(self, logits, y):
        preds = self.acc_preds(logits, y)
        if self.args["output_data_regime"] == "real-valued":
            return mean_squared_error(y, preds)

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
