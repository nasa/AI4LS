import torch
import numpy as np

from models.TorchModelZoo import TorchLinearRegressionModule, MLP


class EmpericalRiskMinimization(object):
    def __init__(self, environment_datasets, val_dataset, test_dataset, args):
        self.args = args
        self.cuda = torch.cuda.is_available() and args.get('cuda', False)

        torch.manual_seed(args.get('seed', 0))
        np.random.seed(args.get('seed', 0))

        self.feature_mask = args.get('feature_mask', None)
        self.features = test_dataset.predictor_columns

        # Initialise Model
        self.method = args.get('method', 'Linear')
        if self.method == 'Linear':
            if self.feature_mask:
                self.input_dim = len(np.array(environment_datasets[0].predictor_columns)[self.feature_mask])
            else:
                self.input_dim = environment_datasets[0].get_feature_dim()
            self.output_dim = environment_datasets[0].get_output_dim()
            self.model = TorchLinearRegressionModule(self.input_dim, self.output_dim)
            if self.cuda:
                self.model.cuda()
        if self.method == 'NN':
            mlp_args = {
                "hidden_dim": args.get("hidden_dim", 256)
            }
            if self.feature_mask:
                self.input_dim = len(np.array(environment_datasets[0].predictor_columns)[self.feature_mask])
            else:
                self.input_dim = environment_datasets[0].get_input_dim()
            self.output_dim = environment_datasets[0].get_output_dim()
            self.model = MLP(mlp_args, self.input_dim, self.output_dim)
            if self.cuda:
                self.model.cuda()

        # Initialise Dataloaders (combine all environment datasets as train)  
        self.batch_size = args.get('batch_size', 256)
        all_dataset = torch.utils.data.ConcatDataset(environment_datasets)
        self.train_loader = torch.utils.data.DataLoader(all_dataset, batch_size=self.batch_size, shuffle=True)
        self.test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)

        # Start training procedure
        self.train()
        # Start testing procedure
        self.test()

    def train(self, loader=None):

        if loader is None:
            loader = self.train_loader

        epochs = self.args.get('epochs', 100)
        lr = self.args.get('lr', 0.001)
        criterion = torch.nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

        # Start training loop
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
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()

    def test(self, loader=None):

        if loader is None:
            loader = self.test_loader

        test_targets = []
        test_logits = []
        test_probs = []

        sig = torch.nn.Sigmoid()

        with torch.no_grad():
            for i, (inputs, targets) in enumerate(loader):
                if self.cuda:
                    if self.feature_mask:
                        inputs = inputs[:, self.feature_mask].cuda()
                    else:
                        inputs = inputs.cuda()
                else:
                    if self.feature_mask:
                        inputs = inputs[:, self.feature_mask]

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
        test_acc = self.mean_accuracy(self.test_logits, self.test_targets)
        test_acc_std = self.std_accuracy(self.test_logits, self.test_targets)

        return {
            "test_acc": test_acc.numpy().squeeze().tolist(),
            "test_nll": test_nll.numpy().squeeze().tolist(),
            "test_probs": self.test_probs.numpy().squeeze(),
            "test_labels": self.test_targets.numpy().squeeze(),
            "feature_coeffients": self.model.linear.weight.data.numpy().squeeze() if self.method == 'Linear' else None,
            "to_bucket": {
                "method": "Non-Causal ERM",
                "features": np.array(self.features).tolist(),
                "coefficients": self.model.linear.weight.data.numpy().squeeze().tolist() if self.method == 'Linear' else None,
                "pvals": None,
                "test_acc": test_acc.numpy().squeeze().tolist(),
                "test_acc_std": test_acc_std.numpy().squeeze().tolist()
            }
        }

    @staticmethod
    def mean_nll(logits, y):
        return torch.nn.functional.binary_cross_entropy_with_logits(logits, y)

    @staticmethod
    def mean_accuracy(logits, y):
        preds = (logits > 0.).float()
        return ((preds - y).abs() < 1e-2).float().mean()

    @staticmethod
    def std_accuracy(logits, y):
        preds = (logits > 0.).float()
        return ((preds - y).abs() < 1e-2).float().std()
