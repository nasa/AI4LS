# Copyright (C) 2020-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os
import numpy as np
import pandas as pd
from logging import getLogger
import torch
from torchvision import datasets, transforms
import torch.nn.functional as F
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss, SoftMarginLoss

logger = getLogger(__name__)

def irm_loss(output, target, reg, w, output_data_regime):
    """Invariant Risk Minimization loss function

    Args:
        output: The mode prediction
        target: The target (ground truth label)
        reg: regularizer (lambda in the paper)
        w: static linear classifier of ones. 
        output_data_regime: string

    Returns:
        irm_loss

    """

    loss = empirical_loss(output, target,  reg, w, output_data_regime)

    penalty = torch.autograd.grad(loss.float(), w.float(), create_graph=True)[0].pow(2).mean()
    irm_loss = (reg * loss + (1 - reg) * penalty)

    return irm_loss


def empirical_loss(output, target,  reg=0, w=None, output_data_regime="real-valued"):
    """Empirical Risk Minimization loss function

    Args:
        output: The mode prediction
        target: The target (ground truth label)
        reg: dummy
        w: dummy 
        output_data_regime: string

    Returns:
        loss: empirical risk

    """

    if output_data_regime == "real-valued":
        loss = MSELoss()(output, target)
    elif output_data_regime == "binary":
        #loss = BCEWithLogitsLoss()(output, target)
        # JC
        loss = SoftMarginLoss()(output, target)
    elif output_data_regime == "multi-class":
        loss = CrossEntropyLoss()(output.float(), target.float().detach())

    return loss

def load_data_shard(shard_path, collaborator_count, index_column=None,
                     categorical=False, channels_last=True, **kwargs):
    """
    Load the CRISP Synthetic dataset.

    Args:
        shard_path (str): The path  to use from the dataset
        collaborator_count (int): The number of collaborators in the
                                  federation
        categorical (bool): True = convert the labels to one-hot encoded
                            vectors (Default = True)
        channels_last (bool): True = The input images have the channels
                              last (Default = True)
        **kwargs: Additional parameters to pass to the function

    Returns:
        list: The input shape
        int: The number of classes
        numpy.ndarray: The training data
        numpy.ndarray: The training labels
        numpy.ndarray: The validation data
        numpy.ndarray: The validation labels
    """

    df = pd.read_csv(os.path.join(shard_path, 'train/data.csv'), index_col = index_column)
    X_train = df.drop("Target", axis=1).to_numpy()
    y_train = np.expand_dims(df["Target"].to_numpy(), axis=1)
    #print(X_train, "\n", y_train)

    df = pd.read_csv(os.path.join(shard_path, 'test/data.csv'), index_col = index_column)
    X_valid = df.drop("Target", axis=1).to_numpy()
    y_valid = np.expand_dims(df["Target"].to_numpy(), axis=1)
    #print(X_valid, "\n", y_valid)

    # (X_train, y_train), (X_valid, y_valid) = _load_raw_datashards(
    #     shard_path, collaborator_count, transform=transforms.ToTensor())

    logger.info(f'DATA > X_train Shape : {X_train.shape}')
    logger.info(f'DATA > y_train Shape : {y_train.shape}')
    logger.info(f'DATA > Train Samples : {X_train.shape[0]}')
    logger.info(f'DATA > Valid Samples : {X_valid.shape[0]}')

    # if categorical:
    #     # convert class vectors to binary class matrices
    #     y_train = one_hot(y_train, num_classes)
    #     y_valid = one_hot(y_valid, num_classes)

    return X_train, y_train, X_valid, y_valid
