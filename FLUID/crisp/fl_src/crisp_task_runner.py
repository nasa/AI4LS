
# Copyright (C) 2020-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""You may copy this file as the starting point of your own model."""
import numpy as np
import torch
from sklearn.metrics import r2_score, mean_squared_error
from torch.nn import CrossEntropyLoss
import tqdm
from typing import Iterator, Tuple

from src.irm_module import IRMModule, NLIRMModule
from src.erm_module import ERMModule, NLERMModule

from openfl.federated.task import TaskRunner
from openfl.utilities import TensorKey, split_tensor_dict_for_holdouts, Metric


class CRISPTaskRunner(TaskRunner):
    """PyTorch Model class for Federated Multiple Learning Models."""

    def __init__(self, num_classes, device='cpu', **kwargs):
        """Initialize.

        Args:
            data: The data loader class
            device (string): Compute device (default="cpu") The hardware device to use for training (Default = "cpu")
            **kwargs: Additional arguments to pass to the function

        """
        super().__init__(device=device, **kwargs)
        TaskRunner.__init__(self, **kwargs)
        if device:
            self.device = device
        else:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.output_data_regime = kwargs["output_data_regime"]

        # This is a map of all the required tensors for each of the public
        # functions in PyTorchTaskRunner
        self.required_tensorkeys_for_function = {}

        # overwrite attribute to account for one optimizer param (in every
        # child model that does not overwrite get and set tensordict) that is
        # not a numpy array
        self.tensor_dict_split_fn_kwargs.update({
            'holdout_tensor_names': ['__opt_state_needed']
        })

        self.dim_x = self.data_loader.X_train.shape[1] #+ 1  # for the bias
        self.num_classes = num_classes

        print()
        print("======batch_size=======:", self.data_loader.batch_size)
        print()

        # Define an explicit nn.Module:
        self.models = {
                #"irm0"  : IRMModule(logger=self.logger, device=self.device, reg=1.0, prefix="irm0_"),
                #"irm1"  : IRMModule(logger=self.logger, device=self.device, reg=0.9, prefix="irm1_"),
                #"irm2"  : IRMModule(logger=self.logger, device=self.device, reg=0.95, prefix="irm2_"),
                "nlerm" : NLERMModule(logger=self.logger, device=self.device, reg=1.0, prefix="nlerm_"),
                #"nlirm" : NLIRMModule(logger=self.logger, device=self.device, reg=1.0, prefix="nlirm_"),
                #"nlirm2" : NLIRMModule(logger=self.logger, device=self.device, reg=0.95, prefix="nlirm2_"),
                #"erm0" : ERMModule(logger=self.logger, device=self.device, prefix="erm0_")
                # "nlerm0" : NLERMModule(logger=self.logger, device=self.device, prefix="nlerm0_")
                }
#        self.models = {"irm" + str(n) : IRMModule(logger=self.logger, devices=self.device, prefix="irm" + str(n) + "_")
#                for n in range(5)}
#        self.irm_model = IRMModule(logger=self.logger, device=self.device)
        self.num_models = len(self.models.keys())
        self.validate_iter = 0

        # cheating:
        for model_name, model in self.models.items():
            model.tensor_dict_split_fn_kwargs = self.tensor_dict_split_fn_kwargs
#           self.models.get("irm0").required_tensorkeys_for_function = {} # self.required_tensorkeys_for_function
#           self.models.get("irm0").required_tensorkeys_for_function = self.irm_model.required_tensorkeys_for_function

            # init each model:
            model.init_network(input_size=self.dim_x, num_classes=self.num_classes, **kwargs)
            model.required_tensorkeys_for_function = {}
            model.initialize_tensorkeys_for_functions()
            model.training_round_completed = False

            self.required_tensorkeys_for_function.update(model.required_tensorkeys_for_function)

    def train_epoch(self, model, model_name, batch_generator: Iterator[Tuple[np.ndarray, np.ndarray]]) -> Metric:
        """Train single epoch.

        Override this function in order to use custom training.

        Args:
            batch_generator: Train dataset batch generator. Yields (samples, targets) tuples of
            size = `self.data_loader.batch_size`.
        Returns:
            Metric: An object containing name and np.ndarray value.
        """
        losses = []
        loss = 0

        batches_in_an_epoch = int(self.data_loader.training_samples / self.data_loader.batch_size)

        for i, (data, target) in enumerate(batch_generator):

            data = torch.tensor(data).to(self.device).float()
            data.requires_grad = True
            target = torch.tensor(target).to(self.device).float()
            target.requires_grad = True

            # Concatenating a vector of 1's to account for bias in linear classification
            #bias = torch.ones(target.shape[0], 1)
            #inputs = torch.cat((bias, data), 1)
            #inputs = data
            output = model(data)

            loss += model.loss_fn(output=output, target=target.reshape(target.shape[0],1), w=model.w, reg=model.reg, output_data_regime=model.output_data_regime)
            losses.append(loss.detach().to(self.device).numpy())

            if i % batches_in_an_epoch == 0 :
                model.optimizer.zero_grad()
                loss.backward()
                model.optimizer.step()
                loss = 0

        loss = np.mean(losses)
        name = model.loss_fn.__name__ + "_" + model_name
        # print(">> loss:", model_name, loss)
        return Metric(name=name, value=np.array(loss))


    def validate(self, col_name, round_num, input_tensor_dict, use_tqdm=False, **kwargs):
        """Validate.

        Run validation of the model on the local data.

        Args:
            col_name:            Name of the collaborator
            round_num:           What round is it
            input_tensor_dict:   Required input tensors (for model)
            use_tqdm (bool):     Use tqdm to print a progress bar (Default=True)

        Returns:
            global_output_dict:  Tensors to send back to the aggregator
            local_output_dict:   Tensors to maintain in the local TensorDB

        """
        output_tensor_dict = {}
        for model_name, model in self.models.items():
            if not model.optimize:
                continue
            model.rebuild_model(round_num, input_tensor_dict, validation=True)
            model.eval()
            val_score = 0
            total_samples = 0

            loader = self.data_loader.get_valid_loader()
            if use_tqdm:
                loader = tqdm.tqdm(loader, desc='validate')

            test_targets = []
            test_logits = []
            test_probs = []
            sig = torch.nn.Sigmoid()

            with torch.no_grad():
                for data, targets in loader:
                    n_samples = targets.shape[0]
                    total_samples += n_samples
                    data = torch.tensor(data).to(self.device)

                    if self.output_data_regime == "real-valued":
                        targets = torch.tensor(targets).to(self.device).float()
                    else:
                        targets = torch.tensor(targets).to(self.device, dtype=torch.int64)

                    # Concatenate a vector of 1's to account for bias in linear classification
                    #bias = torch.ones(n_samples, 1)
                    #inputs = torch.cat((bias, data), 1)
                    #outputs = model(inputs)
                    outputs = model(data)

                    test_targets.append(targets.squeeze().unsqueeze(0))
                    test_logits.append(outputs.squeeze().unsqueeze(0))
                    test_probs.append(sig(outputs).squeeze().unsqueeze(0))

            self.test_targets = torch.cat(test_targets, dim=1)
            self.test_logits = torch.cat(test_logits, dim=1)
            self.test_probs = torch.cat(test_probs, dim=1)

            # if self.output_data_regime == "binary":
            #     pred = output > 0.5 # .argmax(dim=1)
            #     test_probs.append(sig(outputs).squeeze().unsqueeze(0))
            #     val_score += pred.eq(target).sum().cpu().numpy()
            # elif :
            #print(">>", self.test_logits)
            #print(">>", self.test_targets)

            val_score = self.mean_accuracy(self.test_logits.squeeze(), self.test_targets.squeeze())

            origin = col_name
            suffix = 'validate'
            if kwargs['apply'] == 'local':
                suffix += '_local'
            else:
                suffix += '_agg'

            self.validate_iter+=1
            if self.output_data_regime == "real-valued":
                print("round:%s %s %s: MSE: %0.2f" % (round_num, suffix, model_name, val_score))
                metric_name = "MSE"
            else:
                print("round:%s %s %s: Classification rate: %0.2f" % (round_num, suffix, model_name, val_score))
                metric_name = "CCR"
            if self.validate_iter% (2*self.num_models) == 0:

                print('------------------')

            tags = ('metric', suffix)

        # TODO figure out a better way to pass
        #  in metric for this pytorch validate function
            tensor_dict = {
                TensorKey(metric_name + "_" + model_name, origin, round_num, True, tags):
                np.array(val_score) # / total_samples)
            }

            output_tensor_dict.update(tensor_dict)

        # Empty list represents metrics that should only be stored locally
        return output_tensor_dict, {}

    def train_batches(self, col_name, round_num, input_tensor_dict,
                        use_tqdm=False, **kwargs):
            """Train batches.

            Train the model on the requested number of batches.

            Args:
                col_name:            Name of the collaborator
                round_num:           What round is it
                input_tensor_dict:   Required input tensors (for model)
                num_epochs:         The number of epochs to train on before
                                    returning
                use_tqdm (bool):     Use tqdm to print a progress bar (Default=True)

            Returns:
                global_output_dict:  Tensors to send back to the aggregator
                local_output_dict:   Tensors to maintain in the local TensorDB
            """

            output_model_dict = {}
            global_tensor_dict = {}
            local_tensor_dict = {}


            print("num epochs = ", kwargs['num_epochs'])

            batches_in_an_epoch = int(self.data_loader.training_samples / self.data_loader.batch_size)
            print("batches_in_an_epoch ", batches_in_an_epoch)

            num_batches = int(kwargs['num_epochs'] * batches_in_an_epoch)

            for model_name, model in self.models.items():
                if not model.optimize:
                    continue
                loader = self.data_loader.get_train_loader(num_batches)
                model.rebuild_model(round_num, input_tensor_dict)

                # set to "training" mode
                model.train()
                model.to(self.device)

                # Add regression model training loop here
                tags = ('trained',)
                origin = col_name

                if "nlirm" in model_name and round_num >750:
                    #print(model_name, ": change reg from 1 -> 0.05")
                    model.reg = 0.001

                metric = self.train_epoch(model, model_name, loader)

                # Output metric tensors (scalar)
                output_metric_dict = {
                    TensorKey(
                            metric.name, origin, round_num, True, ('metric',)
                        ): metric.value
                    }
                #else:
                #    output_metric_dict = {}

            # output model tensors (Doesn't include TensorKey)
                model_dict = model.get_tensor_dict(with_opt_vars=True)
                global_model_dict, local_model_dict = split_tensor_dict_for_holdouts(
                    self.logger, model_dict,
                    **self.tensor_dict_split_fn_kwargs
                )

            # Create global tensorkeys
                global_tensorkey_model_dict = {
                    TensorKey(tensor_name, origin, round_num, False, tags):
                        nparray for tensor_name, nparray in global_model_dict.items()
                }
            # Create tensorkeys that should stay local
                local_tensorkey_model_dict= {
                    TensorKey(tensor_name, origin, round_num, False, tags):
                        nparray for tensor_name, nparray in local_model_dict.items()
                }
            # The train/validate aggregated function of the next round will look
            # for the updated model parameters.
            # This ensures they will be resolved locally
                next_local_tensorkey_model_dict = {
                    TensorKey(tensor_name, origin, round_num + 1, False, ('model',)): nparray
                    for tensor_name, nparray in local_model_dict.items()}

                global_tensor_dict_per_model = {
                    **output_metric_dict,
                    **global_tensorkey_model_dict
                }
                local_tensor_dict_per_model = {
                    **local_tensorkey_model_dict,
                    **next_local_tensorkey_model_dict
                }

            # Update the required tensors if they need to be pulled from the
            # aggregator
            # TODO this logic can break if different collaborators have different
            # roles between rounds.
            # For example, if a collaborator only performs validation in the first
            # round but training in the second, it has no way of knowing the
            # optimizer state tensor names to request from the aggregator because
            # these are only created after training occurs. A work around could
            # involve doing a single epoch of training on random data to get the
            # optimizer names, and then throwing away the model.
                if model.opt_treatment == 'CONTINUE_GLOBAL':
                    model.initialize_tensorkeys_for_functions(with_opt_vars=True)

            # This will signal that the optimizer values are now present,
            # and can be loaded when the model is rebuilt
                model.train_round_completed = True

                global_tensor_dict.update(global_tensor_dict_per_model)
                local_tensor_dict.update(local_tensor_dict_per_model)

            # Return global_tensor_dict, local_tensor_dict
            return global_tensor_dict, local_tensor_dict

    def mean_nll(self, logits, y):
        if self.output_data_regime == "multi-class":
            return CrossEntropyLoss()(logits.squeeze(), y.squeeze().long())
        else:
            return torch.nn.functional.binary_cross_entropy_with_logits(logits, y)

    def acc_preds(self, logits, y):
        if self.output_data_regime == "multi-class":
            return logits.argmax(dim=-1).float()
        elif self.output_data_regime == "real-valued":
            return logits.float()
        else:
            # binary classification case
            return  (logits > 0.).float()

    def mean_accuracy(self, logits, y):
        preds = self.acc_preds(logits, y)
        if self.output_data_regime == "real-valued":
            return -mean_squared_error(y, preds)

        return ((preds - y).abs() < 1e-2).float().mean()

    def std_accuracy(self, logits, y):
        preds = self.acc_preds(logits, y)
        return ((preds - y).abs() < 1e-2).float().std()

    def compute_variances(self, col_name, round_num, input_tensor_dict, num_batches=None, **kwargs):
        loader = self.data_loader.get_train_loader(num_batches)

        data = []
        for x,y in loader:
            data.append(x)
        if len(data) == 1:
            data = data[0]
        else:
            data = torch.tensor(data)

        var = data.var(axis=0)
        n_samples = data.shape[0]

        print("=========data.shape=======:", data.shape)
        print("=========var.shape=======:", var.shape)
        print("=========n_samples========:", n_samples)

        origin = col_name
        tags = ('variances', 'n_samples: ' + str(n_samples))

        output_metric_dict = {
                TensorKey("feature_variances", origin, round_num, False, tags): var,
        }

        global_tensor_dict = {
                    **output_metric_dict,
                }
        local_tensor_dict = {
                **output_metric_dict,
                }

        print("=========global_tensor_dict.keys():=====", global_tensor_dict.keys())
        print("=========local_tensor_dict.keys():=====", local_tensor_dict.keys())

        return global_tensor_dict, local_tensor_dict


    def get_tensor_dict(self, model=None, with_opt_vars = False):
        """Return the tensor dictionary.

        Args:
            with_opt_vars (bool): Return the tensor dictionary including the
                                  optimizer tensors (Default=False)

        Returns:
            dict: Tensor dictionary {**dict, **optimizer_dict}

        """
        tensor_dict = {}
        for model_name, model in self.models.items():
            tensor_dict.update(model.get_tensor_dict(with_opt_vars))

        return tensor_dict
#        return self.irm_model.get_tensor_dict(with_opt_vars) # + This needs to accumulate all models tensor dicts.

    def get_required_tensorkeys_for_function(self, func_name, model=None, **kwargs):
        """
        Get the required tensors for specified function that could be called \
        as part of a task. By default, this is just all of the layers and \
        optimizer of the model.

        Args:
            func_name

        Returns:
            list : [TensorKey]
        """
        output = []
        for model_name, model in self.models.items():
            output.extend(model.get_required_tensorkeys_for_function(func_name, **kwargs))

        return output
#        return self.irm_model.get_required_tensorkeys_for_function(func_name, **kwargs) # + This needs to accumulate all tensorkeys dict

    def set_tensor_dict(self, tensor_dict, with_opt_vars=False, device='cpu'):
        for model_name, model in self.models.items():
            model.set_tensor_dict(tensor_dict, with_opt_vars) # subset tensor_dict for model specific data?


    def save_native(self, filepath, model_state_dict_key='model_state_dict',
                    optimizer_state_dict_key='optimizer_state_dict', **kwargs):

        for model_name, model in self.models.items():
            pickle_dict = {
                model_state_dict_key: model.state_dict(),
                optimizer_state_dict_key: model.optimizer.state_dict()
            }
            filepath = filepath.split(".pt")[0]
            torch.save(pickle_dict, filepath + model_name + ".pt")





def _derive_opt_state_dict(opt_state_dict):
    """Separate optimizer tensors from the tensor dictionary.

    Flattens the optimizer state dict so as to have key, value pairs with
    values as numpy arrays.
    The keys have sufficient info to restore opt_state_dict using
    expand_derived_opt_state_dict.

    Args:
        opt_state_dict: The optimizer state dictionary

    """
    derived_opt_state_dict = {}

    # Determine if state is needed for this optimizer.
    if len(opt_state_dict['state']) == 0:
        derived_opt_state_dict['__opt_state_needed'] = 'false'
        return derived_opt_state_dict

    derived_opt_state_dict['__opt_state_needed'] = 'true'

    # Using one example state key, we collect keys for the corresponding
    # dictionary value.
    example_state_key = opt_state_dict['param_groups'][0]['params'][0]
    example_state_subkeys = set(
        opt_state_dict['state'][example_state_key].keys()
    )

    # We assume that the state collected for all params in all param groups is
    # the same.
    # We also assume that whether or not the associated values to these state
    # subkeys is a tensor depends only on the subkey.
    # Using assert statements to break the routine if these assumptions are
    # incorrect.
    for state_key in opt_state_dict['state'].keys():
        assert example_state_subkeys == set(opt_state_dict['state'][state_key].keys())
        for state_subkey in example_state_subkeys:
            assert (isinstance(
                opt_state_dict['state'][example_state_key][state_subkey],
                torch.Tensor)
                == isinstance(
                    opt_state_dict['state'][state_key][state_subkey],
                    torch.Tensor))

    state_subkeys = list(opt_state_dict['state'][example_state_key].keys())

    # Tags will record whether the value associated to the subkey is a
    # tensor or not.
    state_subkey_tags = []
    for state_subkey in state_subkeys:
        if isinstance(
                opt_state_dict['state'][example_state_key][state_subkey],
                torch.Tensor
        ):
            state_subkey_tags.append('istensor')
        else:
            state_subkey_tags.append('')
    state_subkeys_and_tags = list(zip(state_subkeys, state_subkey_tags))

    # Forming the flattened dict, using a concatenation of group index,
    # subindex, tag, and subkey inserted into the flattened dict key -
    # needed for reconstruction.
    nb_params_per_group = []
    for group_idx, group in enumerate(opt_state_dict['param_groups']):
        for idx, param_id in enumerate(group['params']):
            for subkey, tag in state_subkeys_and_tags:
                if tag == 'istensor':
                    new_v = opt_state_dict['state'][param_id][
                        subkey].cpu().numpy()
                else:
                    new_v = np.array(
                        [opt_state_dict['state'][param_id][subkey]]
                    )
                derived_opt_state_dict[f'__opt_state_{group_idx}_{idx}_{tag}_{subkey}'] = new_v
        nb_params_per_group.append(idx + 1)
    # group lengths are also helpful for reconstructing
    # original opt_state_dict structure
    derived_opt_state_dict['__opt_group_lengths'] = np.array(
        nb_params_per_group
    )

    return derived_opt_state_dict


def expand_derived_opt_state_dict(derived_opt_state_dict, device):
    """Expand the optimizer state dictionary.

    Takes a derived opt_state_dict and creates an opt_state_dict suitable as
    input for load_state_dict for restoring optimizer state.

    Reconstructing state_subkeys_and_tags using the example key
    prefix, "__opt_state_0_0_", certain to be present.

    Args:
        derived_opt_state_dict: Optimizer state dictionary

    Returns:
        dict: Optimizer state dictionary
    """
    state_subkeys_and_tags = []
    for key in derived_opt_state_dict:
        if key.startswith('__opt_state_0_0_'):
            stripped_key = key[16:]
            if stripped_key.startswith('istensor_'):
                this_tag = 'istensor'
                subkey = stripped_key[9:]
            else:
                this_tag = ''
                subkey = stripped_key[1:]
            state_subkeys_and_tags.append((subkey, this_tag))

    opt_state_dict = {'param_groups': [], 'state': {}}
    nb_params_per_group = list(
        derived_opt_state_dict.pop('__opt_group_lengths').astype(np.int)
    )

    # Construct the expanded dict.
    for group_idx, nb_params in enumerate(nb_params_per_group):
        these_group_ids = [f'{group_idx}_{idx}' for idx in range(nb_params)]
        opt_state_dict['param_groups'].append({'params': these_group_ids})
        for this_id in these_group_ids:
            opt_state_dict['state'][this_id] = {}
            for subkey, tag in state_subkeys_and_tags:
                flat_key = f'__opt_state_{this_id}_{tag}_{subkey}'
                if tag == 'istensor':
                    new_v = torch.from_numpy(derived_opt_state_dict.pop(flat_key))
                else:
                    # Here (for currrently supported optimizers) the subkey
                    # should be 'step' and the length of array should be one.
                    assert subkey == 'step'
                    assert len(derived_opt_state_dict[flat_key]) == 1
                    new_v = int(derived_opt_state_dict.pop(flat_key))
                opt_state_dict['state'][this_id][subkey] = new_v

    # sanity check that we did not miss any optimizer state
    assert len(derived_opt_state_dict) == 0

    return opt_state_dict