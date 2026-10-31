import numpy as np
import torch

from models.TorchModelZoo import MLP, MLP2
from sklearn.metrics import confusion_matrix


class NonLinearInvariantRiskMinimization(object):
    def __init__(self, environment_datasets, val_dataset, test_dataset, args):
        self.args = args
        self.cuda = torch.cuda.is_available() and args.get('cuda', False)

        torch.manual_seed(args.get('seed', 0))
        np.random.seed(args.get('seed', 0))

        self.feature_names = test_dataset.predictor_columns
        # Initialise Model
        method = args.get('NN_method') 
        self.method = method
        self.confusion_matrix_test = list()
        if method == 'NN':
            print('NLIRM using MLP')
            self.input_dim = environment_datasets[0].get_feature_dim()
            self.output_dim = environment_datasets[0].get_output_dim()
            self.model = MLP(self.args, self.input_dim, self.output_dim)
            # JC
            self.model.train()
            if self.cuda:
                self.model.cuda()
        if method == 'DNN':
            print('NLIRM using deep MLP2')
            self.input_dim = environment_datasets[0].get_feature_dim()
            self.output_dim = environment_datasets[0].get_output_dim()
            self.model = MLP2(self.args, self.input_dim, self.output_dim)
            self.model.train() # self.model.eval() to turn dropout off
            if self.cuda:
                self.model.cuda()


        # Initialise Dataloaders (combine all environment datasets to as train)
        self.batch_size = args.get('batch_size', 256)
        all_dataset = torch.utils.data.ConcatDataset(environment_datasets)
        self.all_loader = torch.utils.data.DataLoader(all_dataset, batch_size=self.batch_size, shuffle=True)
        train_loaders = []
        for ds in environment_datasets:
            dl = torch.torch.utils.data.DataLoader(ds, batch_size=self.batch_size)
            train_loaders.append(dl)
        self.train_loaders = train_loaders
        self.val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=self.batch_size)
        self.test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)

        ### ______________ Find min and max values per feature to conduct sensitivity analysis _____________ ###

        self.min_per_dim = np.ones(shape=(self.input_dim ,1) ) *(100000.)
        self.max_per_dim = np.ones(shape=(self.input_dim ,1) ) *(-100000.)

        for inputs, targets in self.all_loader:
            for ii in range(self.input_dim):
                dimmin = min(inputs.numpy()[: ,ii])
                dimmax = max(inputs.numpy()[: ,ii])
                if dimmin < self.min_per_dim[ii]:
                    self.min_per_dim[ii] = dimmin
                if dimmax > self.max_per_dim[ii]:
                    self.max_per_dim[ii] = dimmax

        ### ####################################################################### ###


        self.train()
        self.test()
        self.validate()

    def get_sensitivities(self):
        sties = np.zeros(shape=(self.input_dim,))
        sig = torch.nn.Sigmoid()
        with torch.no_grad():
            for i, (inputs, targets) in enumerate(self.val_loader):
                n = inputs.size(0)
                if self.cuda:
                    inputs = inputs.cuda()

                for ii in range(self.input_dim):
                    if False:  
                        sties[ii] += 0
                    else:
                        temp = inputs.clone()
                        temp[: ,ii] = float(self.min_per_dim[ii])
                        outputs1 = self.model(temp)
                        temp[: ,ii] = float(self.max_per_dim[ii])
                        outputs2 = self.model(temp)

                        sties[ii] += torch.sum \
                            (outputs2 -outputs1).numpy( ) /n  

        return sties

    def train(self):

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args['lr'])

        for step in range(self.args['n_iterations']):
            env_nlls = []
            env_accs = []
            env_pens = []
            for env_loader in self.train_loaders:
                env_logits = []
                env_targets = []
                for inputs, targets in env_loader:
                    if self.cuda:
                        inputs = inputs.cuda()
                        env_logits.append(self.model(inputs).cpu().unsqueeze(-1))
                    else:
                        e_log = self.model(inputs).unsqueeze(-1).unsqueeze(0)
                        env_logits.append(e_log)
                    env_targets.append(targets.unsqueeze(-1).unsqueeze(0))

                env_logits = torch.cat(env_logits, dim=1)
                env_targets = torch.cat(env_targets, dim=1)
                env_nlls.append(self.mean_nll(env_logits, env_targets))
                env_accs.append(self.mean_accuracy(env_logits, env_targets))
                env_pens.append(self.penalty(env_logits, env_targets))

            train_nll = torch.stack(env_nlls).mean()
            train_acc = torch.stack(env_accs).mean()
            train_penalty = torch.stack(env_pens).mean()

            weight_norm = torch.tensor(0.)
            for w in self.model.parameters():
                # TODO JC why square this?
                weight_norm += w.norm().pow(2)
            # JC (take sqrt for L2 norm?)
            #weight_norm = math.sqrt(weight_norm)
            loss = train_nll.clone()
            loss += self.args['l2_regularizer_weight'] * weight_norm
            
            # Penalty weight is an important hyper parameter, which balances the invariance penalty vs. the empirical risk
            # Defaul penalty weight is 1 for the first 100 iterations, and thereon it is increased to 1e4 (based on Arjovsky et al.'s implementation)
            # Other suggested variations in the literature include a sequence of monotonically increasing penalty weights 

            penalty_weight = (self.args['penalty_weight']
                              if step >= self.args['penalty_anneal_iters'] else 1.0)
            # JC
            #penalty_weight = 1.1 * step
            loss += penalty_weight * train_penalty
            if penalty_weight > 1.0:
                loss /= penalty_weight

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if step % 100 == 0 and self.args['verbose']:
                self.pretty_print(np.int32(step),
                                  train_nll.detach().cpu().numpy(),
                                  train_acc.detach().cpu().numpy(),
                                  train_penalty.detach().cpu().numpy()
                                  )

    def test(self):

        test_targets = []
        test_logits = []
        test_probs = []

        sig = torch.nn.Sigmoid()

        with torch.no_grad():
            for i, (inputs, targets) in enumerate(self.test_loader):
                if self.cuda:
                    inputs = inputs.cuda()

                outputs = self.model(inputs)

                if self.cuda:
                    test_targets.append(targets.squeeze().unsqueeze(0))
                    test_logits.append(outputs.cpu().squeeze().unsqueeze(0))
                    test_probs.append(sig(outputs).cpu().squeeze().unsqueeze(0))
                    print('using cuda')
                else:
                    test_targets.append(targets.squeeze().unsqueeze(0))
                    test_logits.append(outputs.squeeze().unsqueeze(0))
                    test_probs.append(sig(outputs).squeeze().unsqueeze(0))
                    print('not using cuda')

        self.test_targets = torch.cat(test_targets, dim=1)
        self.test_logits = torch.cat(test_logits, dim=1)
        self.test_probs = torch.cat(test_probs, dim=1)

        # JC
        preds = (self.test_logits > 0.).float().tolist()[0]
        y = self.test_targets.tolist()[0]
        conf_matrix = confusion_matrix(y_true=y, y_pred=preds)
        self.confusion_matrix_test.append(conf_matrix)
    #         print('Finished Testing')

    def validate(self):

        validate_targets = []
        validate_logits = []
        validate_probs = []

        sig = torch.nn.Sigmoid()

        with torch.no_grad():
            for i, (inputs, targets) in enumerate(self.val_loader):
                if self.cuda:
                    inputs = inputs.cuda()

                outputs = self.model(inputs)

                if self.cuda:
                    validate_targets.append(targets.squeeze().unsqueeze(0))
                    validate_logits.append(outputs.cpu().squeeze().unsqueeze(0))
                    validate_probs.append(sig(outputs).cpu().squeeze().unsqueeze(0))
                    print('using cuda')
                else:
                    validate_targets.append(targets.squeeze().unsqueeze(0))
                    validate_logits.append(outputs.squeeze().unsqueeze(0))
                    validate_probs.append(sig(outputs).squeeze().unsqueeze(0))
                    print('not using cuda')

        self.validate_targets = torch.cat(validate_targets, dim=1)
        self.validate_logits = torch.cat(validate_logits, dim=1)
        self.validate_probs = torch.cat(validate_probs, dim=1)
    #         print('Finished Testing')


    def results(self):
        test_nll = self.mean_nll(self.test_logits, self.test_targets)
        #test_acc = self.mean_accuracy(self.test_logits, self.test_targets)
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
        test_acc_std = self.std_accuracy(self.test_logits, self.test_targets)
        validate_nll = self.mean_nll(self.validate_logits, self.validate_targets)
        validate_acc = self.mean_accuracy(self.validate_logits, self.validate_targets)
        validate_acc_std = self.std_accuracy(self.validate_logits, self.validate_targets)

        return {
            #"test_acc": test_acc.numpy().squeeze().tolist(),
            "test_acc": test_acc,
            #"test_nll": test_nll,
            "test_probs": self.test_probs,
            "test_labels": self.test_targets,
            #"validate_acc": validate_acc.numpy().squeeze().tolist(),
            #"validate_nll": validate_nll,
            #"validate_probs": self.validate_probs,
            #"validate_labels": self.validate_targets,
            "feature_coeffients": self.model.linear.weight.data
                [0].detach().numpy().squeeze().tolist() if self.method == "Linear" else None,
            "to_bucket": {
                'method': "Non-Linear IRM",
                'features': self.feature_names,
                'coefficients': self.model.linear.weight.data
                    [0].detach().numpy().squeeze().tolist() if self.method == "Linear" else self.get_sensitivities().squeeze().tolist(),
                'pvals': None,
                #'test_acc': test_acc.numpy().squeeze().tolist(),
                'test_acc': test_acc,
                'test_acc_std': test_acc_std.numpy().squeeze().tolist(),
                "confusion_matrix_test": str(self.confusion_matrix_test)
                #'validate_acc': validate_acc.numpy().squeeze().tolist(),
                #'validate_acc_std': validate_acc_std.numpy().squeeze().tolist()
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

    def penalty(self, logits, y):
        scale = torch.tensor(1.).requires_grad_()
        loss = self.mean_nll(logits * scale, y)
        grad = torch.autograd.grad(loss, [scale], create_graph=True)[0]
        return torch.sum(grad**2)

    def pretty_print(self, *values):
        col_width = 13
        def format_val(v):
            if not isinstance(v, str):
                v = np.array2string(v, precision=5, floatmode='fixed')
            return v.ljust(col_width)
        str_values = [format_val(v) for v in values]
        print("   ".join(str_values))

