# Author: Paul Duckworth
# Date: 15th July 2021

import numpy as np
import torch
import torch.optim as optim

try:
    from src.model_module import ModelModule
    from src.utils import irm_loss
except ModuleNotFoundError:
    from fl_src.model_module import ModelModule
    from fl_src.utils import irm_loss

class IRMModule(ModelModule):
    """PyTorch Model class for IRM model."""

    def __init__(self, logger, device='cpu',  prefix='irm_0_', reg=0, **kwargs):
        """Initialize.

        Args:
            data: The data loader class
            device (string): Compute device (default="cpu") The hardware device to use for training (Default = "cpu")
            prefix (string): Prefix name for the model layers
            **kwargs: Additional arguments to pass to the function

        """

        super().__init__(logger, device,  prefix, **kwargs)
        self.phi_name = prefix + "phi"
        self.reg = reg

    def init_network(self,
                     input_size,
                     num_classes,
                     l1=0.5,
                     print_model=True,
                     **kwargs):
        """Create the IRM Linear network (model).

        Args:
            print_model (bool): Print the model topology (Default=True)
            input_size (int):
            output size (int):
            l1 (float):
            **kwargs: Additional arguments to pass to the function

        """

        self.inputSize = input_size
        self.opt_treatment = 'RESET'
        self.output_data_regime = kwargs["output_data_regime"]

        # set seed
        torch.manual_seed(kwargs["seed"])

        # phi
        setattr(self, self.phi_name, torch.nn.Linear(self.inputSize, self.inputSize, bias=False))

        # w
        self.w = torch.ones(input_size, 1).float()
        if self.output_data_regime == "multi-class":
            self.w = torch.rand(input_size, num_classes).float()
        self.w.requires_grad = True

        self.loss_fn = irm_loss
        self._init_optimizer()

        if print_model:
            print(self)
        self.to(self.device)

    def _init_optimizer(self):
        """Initialize the optimizer."""
        #self.optimizer = optim.Adam([self.phi.weight], lr=1e-4)
        self.optimizer = optim.Adam([self.phi().weight], lr=1e-3)

    def coef_(self):
        #return (self.phi().weight @ self.w).detach().numpy()
        return self.solution()
#        return self.phi().detach.numpy()


class NLIRMModule(ModelModule):
    """PyTorch Model class for Simple Non-Linear IRM model."""

    def __init__(self, logger, device='cpu', prefix='nlirm_0_', reg=0, **kwargs):
        """Initialize.

        Args:
            data: The data loader class
            device (string): Compute device (default="cpu") The hardware device to use for training (Default = "cpu")
            prefix (string): Prefix name for the model layers
            **kwargs: Additional arguments to pass to the function

        """

        super().__init__(logger, device,  prefix, **kwargs)
        self.phi_name = prefix + "phi"
        self.reg = reg

    def init_network(self,
                     input_size,
                     num_classes,
                     l1=0.5,
                     print_model=True,
                     **kwargs):
        """Create the IRM Linear network (model).

        Args:
            print_model (bool): Print the model topology (Default=True)
            input_size (int):
            output size (int):
            l1 (float):
            **kwargs: Additional arguments to pass to the function

        """

        self.inputSize = input_size
        self.opt_treatment = 'RESET'
        self.output_data_regime = kwargs["output_data_regime"]

        # set seed
        torch.manual_seed(kwargs["seed"])

        # copied from non-federated SNLIRM model
        hidden_dim = 30
        #setattr(self, self.phi_name, torch.nn.Sequential(
        #                torch.nn.Linear(self.inputSize, hidden_dim, bias=True),
        #                torch.nn.Tanh(),
        #                torch.nn.Linear(hidden_dim, self.inputSize, bias=True),
        #                torch.nn.Tanh(),
        #                torch.nn.Linear(self.inputSize, 1, bias=True))
        #)

        # phi
        setattr(self, self.phi_name, torch.nn.Sequential(
            torch.nn.Linear(self.inputSize, hidden_dim, bias=True),
            torch.nn.Tanh(),
            torch.nn.Linear(hidden_dim, self.inputSize, bias=True)
            ))
            #torch.nn.Tanh())
        # w
        self.w = torch.ones(self.inputSize, 1).float()
        if self.output_data_regime == "multi-class":
            self.w = torch.rand(input_size, num_classes).float()
        self.w.requires_grad = True

        # nonlineartiy
        #self.nonlinearity = torch.nn.Sigmoid()
        #self.nonlinearity = torch.nn.Tanh()

        self.loss_fn = irm_loss
        self._init_optimizer()

        if print_model:
            print(self)
        self.to(self.device)

    def _init_optimizer(self):
        """Initialize the optimizer."""
        #self.optimizer = optim.Adam([self.phi.weight], lr=1e-4)
        self.optimizer = optim.Adam(self.phi().parameters(), lr=1e-3)

    def coef_(self):
#        return self.phi.weight.detach().numpy()
#        return self.nonlinearity(self.phi()).detach.numpy()
#        coef_unnormalized = sum([w.data.abs().sum(axis=1) for w in self.phi.parameters()])
#        coef = coef_unnormalized / coef_unnormalized.sum()

#        return coef
        return self.solution()


    def set_optimizer_treatment(self, opt_treatment):
        """Change the treatment of current instance optimizer."""
        self.opt_treatment = opt_treatment
