from typing import Any

import torch


class TorchLinearRegressionModule(torch.nn.Module):
    def __init__(self, inputSize, outputSize):
        super(TorchLinearRegressionModule, self).__init__()
        self.linear = torch.nn.Linear(inputSize, outputSize, bias=True)

    def forward(self, x):
        out = self.linear(x)
        return out


class MLP(torch.nn.Module):
    def __init__(self, flags, input_dim, output_dim):
        super(MLP, self).__init__()
        lin1 = torch.nn.Linear(input_dim, flags['hidden_dim'], bias=True)
        lin2 = torch.nn.Linear(flags['hidden_dim'], flags['hidden_dim'], bias=True)
        lin3 = torch.nn.Linear(flags['hidden_dim'], output_dim, bias=True)

        for lin in [lin1, lin2, lin3]:
            torch.nn.init.xavier_uniform_(lin.weight)
            torch.nn.init.zeros_(lin.bias)
        self._main = torch.nn.Sequential(lin1, torch.nn.ReLU(True), lin2, torch.nn.ReLU(True), lin3)
        self.input_dim = input_dim
        self.output_dim = output_dim

    def forward(self, input):
        out = input.view(input.shape[0], self.input_dim)
        out = self._main(out)
        return out


class MLP2(torch.nn.Module):
    def __init__(self, flags, input_dim, output_dim):
        super(MLP2, self).__init__()
        hid_dim1 = flags['hidden_dim']
        hid_dim2 = int(hid_dim1 / 2)
        hid_dim3 = int(hid_dim2 / 2)
        dropout_p_low = 0.4
        dropout_p_high = 0.4

        lin1 = torch.nn.Linear(input_dim, hid_dim1, bias=True)
        d1 = torch.nn.Dropout(dropout_p_low)
        lin2 = torch.nn.Linear(hid_dim1, hid_dim1, bias=True)
        d2 = torch.nn.Dropout(dropout_p_low)
        lin3 = torch.nn.Linear(hid_dim1, hid_dim2, bias=True)
        d3 = torch.nn.Dropout(dropout_p_low)
        lin4 = torch.nn.Linear(hid_dim2, hid_dim2, bias=True)
        d4 = torch.nn.Dropout(dropout_p_low)
        lin5 = torch.nn.Linear(hid_dim2, hid_dim3, bias=True)
        d5 = torch.nn.Dropout(dropout_p_low)
        lin6 = torch.nn.Linear(hid_dim3, hid_dim3, bias=True)
        d6 = torch.nn.Dropout(dropout_p_high)
        lin7 = torch.nn.Linear(hid_dim3, output_dim, bias=True)

        for lin in [lin1, lin2, lin3, lin4, lin5, lin6, lin7]:
            torch.nn.init.xavier_uniform_(lin.weight)
            torch.nn.init.zeros_(lin.bias)
        self._main = torch.nn.Sequential(lin1, torch.nn.ReLU(True), d1,
                                         lin2, torch.nn.ReLU(True), d2,
                                         lin3, torch.nn.ReLU(True), d3,
                                         lin4, torch.nn.ReLU(True), d4,
                                         lin5, torch.nn.ReLU(True), d5,
                                         lin6, torch.nn.ReLU(True), d6,
                                         lin7
                                         )
        self.input_dim = input_dim
        self.output_dim = output_dim

    def forward(self, input):
        out = input.view(input.shape[0], self.input_dim)
        out = self._main(out)
        return out
