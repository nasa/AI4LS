import torch
import numpy as np
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.metrics import accuracy_score


class RandomForest(object):
    """ Defines the Random Forest classifier.
    """
    
    def __init__(self, environment_datasets, val_dataset, test_dataset, args):
        """ Method used to initialize the Random Forest classifier.
        
        param environment_datasets: Pytorch dataset, environment datasets
        param val_dataset: Pytorch dataset, validation dataset
        param test_dataset: Pytorch dataset, test dataset
        param args: dict, model parameters
        """
            
        self.args = args
        self.running_params_model = args['running_params']  # get parameters used when running the model
        
        self.feature_mask = args.get('feature_mask', None)  # get feature mask
        
        self.features = test_dataset.predictor_columns

        self.model = RFC(**args['model_params'])  # initialize Random Forest with the given parameters
        
        if self.feature_mask:
            self.input_dim = len(np.array(environment_datasets[0].predictor_columns)[self.feature_mask])
        else:
            self.input_dim = environment_datasets[0].get_feature_dim()
        
        # Initialise Dataloaders (combine all environment datasets as train)  
        self.batch_size = self.input_dim
        all_dataset = torch.utils.data.ConcatDataset(environment_datasets)
        self.train_loader = torch.utils.data.DataLoader(all_dataset, batch_size=self.batch_size, shuffle=True)
        self.test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=5, shuffle=False)
        self.val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=5, shuffle=False)
        
        # Start training procedure
        self.train()
        
        # Start testing procedure
        self.test()

        # JC
        self.validate()
        
    def train(self, loader=None):
        """ Method used to train the model.
        
        param loader: Pytorch loader used to load data to evaluate the model. If None, it uses the loader created when initializing the estimator object.
        """
        
        if loader is None:
            loader = self.train_loader

        for inputs_loader, targets_loader in loader:
            inputs, targets = inputs_loader, targets_loader
            
        if self.feature_mask:
            inputs = inputs[:, self.feature_mask]
        
        targets = np.ravel(targets)

        # fit model to the training data
        self.model.fit(inputs, targets, self.running_params_model['sample_weight'])    
        
        # get feature coefficients
        self.feature_coefficients = self.model.feature_importances_
    
    def test(self, loader=None):
        """ Method used to evaluate the model.
        
        param loader: Pytorch loader used to load data to evaluate the model. If None, it uses the loader created when initializing the estimator object.
        """
        
        test_targets, test_class, test_probs = [], [], []
        
        if loader is None:
            loader = self.test_loader
        
        for inputs_loader, targets_loader in loader:
            inputs, targets = inputs_loader, targets_loader
            
            if self.feature_mask:
                inputs = inputs[:, self.feature_mask]
        
            test_targets.append(targets)
            test_class.append(self.model.predict(inputs))
            test_probs.append(self.model.predict_proba(inputs))

        self.test_targets = np.concatenate(test_targets, axis=0)
        self.test_class = np.concatenate(test_class, axis=0)
        self.test_probs = np.concatenate(test_probs, axis=0)

    def validate(self, loader=None):
        """ Method used to evaluate the model.

        param loader: Pytorch loader used to load data to evaluate the model. If None, it uses the loader created when initializing the estimator object.
        """

        validate_targets, validate_class, validate_probs = [], [], []

        if loader is None:
            loader = self.val_loader

        for inputs_loader, targets_loader in loader:
            inputs, targets = inputs_loader, targets_loader

            if self.feature_mask:
                inputs = inputs[:, self.feature_mask]

            validate_targets.append(targets)
            validate_class.append(self.model.predict(inputs))
            validate_probs.append(self.model.predict_proba(inputs))

        self.validate_targets = np.concatenate(validate_targets, axis=0)
        self.validate_class = np.concatenate(validate_class, axis=0)
        self.validate_probs = np.concatenate(validate_probs, axis=0)

    def results(self):
        """ Compute performance metrics and return the results.
        
        Returns:
            dict, holds results such as performance metrics and feature coefficients
        """

        test_acc = accuracy_score(self.test_targets, self.test_class, normalize=True, sample_weight=None)
        
        return {
            "test_acc": test_acc, 
            "test_probs": self.test_probs[:, 1].tolist(),  
            "test_labels": self.test_targets.tolist(),
            "feature_coeffients": self.feature_coefficients.tolist(),
            "to_bucket": {
                "method": "RF",
                "features": np.array(self.features).tolist(),
                "coefficients": self.feature_coefficients.tolist(),
                "pvals": None,
                "test_acc": test_acc
            }
        }

    def validation_results(self):
        """ Compute performance metrics and return the results.

        Returns:
            dict, holds results such as performance metrics and feature coefficients
        """

        validation_acc = accuracy_score(self.validate_targets, self.validate_class, normalize=True, sample_weight=None)

        return {
            "validate_acc": validation_acc,
            "validate_probs": self.validate_probs[:, 1].tolist(),
            "validate_labels": self.validate_targets.tolist(),
            "feature_coeffients": self.feature_coefficients.tolist(),
            "to_bucket_val": {
                "method": "RF",
                "features": np.array(self.features).tolist(),
                "coefficients": self.feature_coefficients.tolist(),
                "pvals": None,
                "validate_acc": validation_acc
            }
        }
