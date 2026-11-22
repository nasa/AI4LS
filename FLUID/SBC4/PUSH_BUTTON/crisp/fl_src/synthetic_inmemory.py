# Copyright (C) 2020-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""You may copy this file as the starting point of your own model."""

# from .mnist_utils import load_mnist_shard
from .utils import load_data_shard
from openfl.federated import PyTorchDataLoader


class PyTorchSyntheticInMemory(PyTorchDataLoader):
    """PyTorch data loader for Synthetic datasets."""

    def __init__(self, data_path, batch_size, seed, index = False, **kwargs):
        """Instantiate the data object.

        Args:
            data_path: The file path to the data
            batch_size: The batch size of the data loader
            seed: int
            **kwargs: Additional arguments, passed to super
             init and load_mnist_shard
        """

        super().__init__(batch_size, seed, **kwargs)

        index_column = None
        if index: index_column = 0
    
        print()
        print("synthetic in memory kwargs:")
        print(kwargs)
        print()

        X_train, y_train, X_valid, y_valid = load_data_shard(
            data_path, kwargs.pop("collaborator_count"), index_column, **kwargs)

        self.X_train = X_train

        # overwrite the batch size of the data loader, if number of samples is too small
        if self.X_train.shape[0] < batch_size:
            self.batch_size = self.X_train.shape[0]

        self.y_train = y_train
        self.training_samples = X_train.shape[0]

        self.train_loader = self.get_train_loader()

        self.X_valid = X_valid
        self.y_valid = y_valid
        self.val_loader = self.get_valid_loader()
