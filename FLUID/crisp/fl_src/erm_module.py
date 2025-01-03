# Author: Paul Duckworth
# Date: 30th July 2021

import numpy as np
import torch
import torch.optim as optim
# from models.TorchModelZoo import MLP

try:
    from src.model_module import ModelModule 
    from src.utils import empirical_loss
except ModuleNotFoundError: 
    from fl_src.model_module import ModelModule 
    from fl_src.utils import empirical_loss

class ERMModule(ModelModule):
    """PyTorch Model class for ERM model."""

    def __init__(self, logger, device='cpu',  prefix='erm_0_', **kwargs):
        """Initialize.

        Args:
            data: The data loader class
            device (string): Compute device (default="cpu") The hardware device to use for training (Default = "cpu")
            prefix (string): Prefix name for the model layers
            **kwargs: Additional arguments to pass to the function

        """

        super().__init__(logger, device,  prefix, **kwargs)
        self.phi_name = prefix + "phi"


    def init_network(self,
                     input_size,
                     num_classes,
                     l1=0.5,
                     print_model=True,
                     **kwargs):
        """Create the ERM MLP network (model).

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

        # setattr(self, self.phi_name, 
        #     torch.nn.Linear(self.inputSize, 1, bias=False))
            # setattr(self, self.phi_name, torch.nn.Linear(self.inputSize, num_classes, bias=False))

        # MLP model (copied from the ModelZoo)
        flags = { "hidden_dim" : int(self.inputSize / 2)}

        #print("FLAGS: ", flags)
        if self.output_data_regime != "multi-class":
            output_dim = 1
        else:
            output_dim = num_classes

        lin1 = torch.nn.Linear(self.inputSize, flags['hidden_dim'], bias=False)
        lin2 = torch.nn.Linear(flags['hidden_dim'], flags['hidden_dim'], bias=False)
        lin3 = torch.nn.Linear(flags['hidden_dim'], output_dim, bias=False)

        for lin in [lin1, lin2, lin3]:
            torch.nn.init.xavier_uniform_(lin.weight)
            #torch.nn.init.zeros_(lin.bias)
        # self._main = torch.nn.Sequential(lin1, torch.nn.ReLU(True), lin2, torch.nn.ReLU(True), lin3)

        setattr(self, self.phi_name, 
                torch.nn.Sequential(
                    lin1, 
                    torch.nn.ReLU(True), 
                    lin2, 
                    torch.nn.ReLU(True), 
                    lin3
                ) )

        # dummy : to allow the TaskRunner to loop over federated model losses
        self.w = torch.ones(input_size, 1).float() # dummy var (i.e. not used)
        self.reg = 0  # dummy var (i.e. not used)

        # model loss:
        self.loss_fn = empirical_loss
        self._init_optimizer()

        if print_model:
            print(self)
        self.to(self.device)

    def _init_optimizer(self):
        """Initialize the optimizer."""
        #self.optimizer = optim.Adam([self.phi.weight], lr=1e-4)
        #self.optimizer = optim.Adam(self.phi().parameters(), lr=1e-3)
        # JC
        self.optimizer = optim.Adam(self.phi().parameters(), lr=1e-2)

    def solution(self):
        W = torch.eye(self.inputSize)
        for w in self.phi().parameters():
            W = W@w.T
        coef = W / W.sum()
        return coef

    def coef_(self):
    #   return (self.phi().weight @ self.w).detach().numpy()
        return self.solution()
#        return self.phi().detach.numpy()


class NLERMModule(ModelModule):
    """PyTorch Model class for Simple Non-Linear ERM model."""

    def __init__(self, logger, device='cpu',  prefix='erm_0_', **kwargs):
        """Initialize.

        Args:
            data: The data loader class
            device (string): Compute device (default="cpu") The hardware device to use for training (Default = "cpu")
            prefix (string): Prefix name for the model layers
            **kwargs: Additional arguments to pass to the function

        """

        super().__init__(logger, device,  prefix, **kwargs)
        self.phi_name = prefix + "phi"

    def init_network(self,
                     input_size,
                     num_classes,
                     l1=0.5,
                     print_model=True,
                     **kwargs):
        """Create the ERM Non-Linear network (model).

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
            #)

        # w: just used as a dummy to pass into loss function
        self.w = torch.ones(input_size, 1).float()
        #self.w = torch.ones(1,1).float()

        self.nonlinearity = torch.nn.Tanh()
        self.reg = 0  # dummy (i.e. not used)

        self.loss_fn = empirical_loss
        self._init_optimizer()

        if print_model:
            print(self)
        self.to(self.device)

    def _init_optimizer(self):
        """Initialize the optimizer."""
        #self.optimizer = optim.Adam([self.phi.weight], lr=1e-4)
        self.optimizer = optim.Adam(self.phi().parameters(), lr=1e-3)

#    def forward(self, x):
#        return self.nonlinearity( self.phi()(x.float())) @ self.w

    def solution(self):
        W = torch.eye(self.inputSize)
        for w in self.phi().parameters():
            W = W@w.T
        coef = W / W.sum()

        return coef

    def coef_(self):
#        return (self.phi().weight @ self.w).detach().numpy()
        return self.solution()
#        return self.phi().detach.numpy()
