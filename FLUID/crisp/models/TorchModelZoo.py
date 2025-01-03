from typing import Any

import torch

'''
should apply l1 or l2 penalty to the linear reg for high d low n?
'''
class TorchLinearRegressionModule(torch.nn.Module):
    def __init__(self, inputSize, outputSize, bias=True, device="cpu"):
        super(TorchLinearRegressionModule, self).__init__()
        self.device = device
        self.linear = torch.nn.Linear(inputSize, outputSize, bias=bias).to(self.device)
        self.criterion = torch.nn.MSELoss()
        self.optimizer = torch.optim.SGD(self.parameters(), lr=0.001)

    def forward(self, x):
        out = self.linear(x)
        return out
    
    def coef_(self):
        return self.linear.weight.detach().cpu().numpy()
    
    def fit(self, X_train, y_train, epochs=500):
#         add tolerance to this for tensors already in correct formmat
        X_train = torch.from_numpy(X_train).float().to(self.device)
        y_train = torch.from_numpy(y_train).float().to(self.device)
        for epoch in range(epochs):
            #forward feed
            y_pred = self.forward(X_train.requires_grad_())

            #calculate the loss
            loss = self.criterion(y_pred, y_train)

            #backward propagation: calculate gradients
            loss.backward()

            #update the weights
            self.optimizer.step()

            #clear out the gradients from the last step loss.backward()
            self.optimizer.zero_grad()
            
    def predict(self, x):
        x = torch.from_numpy(x).float().to(self.device)
        with torch.no_grad():
            return self.forward(x)
        
    def predict_proba(self, x):
        return None
    
class TorchLogisticRegressionModule(torch.nn.Module):
    def __init__(self, inputSize, outputSize, bias=True, l1=0.5, device="cpu"):
        super(TorchLogisticRegressionModule, self).__init__()
        self.device = device
        self.outputSize = outputSize
        
        if self.outputSize <= 2:
            self.outputSize = 1
        self.linear = torch.nn.Linear(inputSize, self.outputSize, bias=bias).to(self.device)

        if self.outputSize <= 2:
            self.criterion = torch.nn.BCEWithLogitsLoss()
        else:
            self.criterion = torch.nn.functional.cross_entropy
        
        self.optimizer = torch.optim.SGD(self.parameters(), lr=0.001, momentum=0.8)
        self.lambda1 = l1

    def forward(self, x):
        out = self.linear(x)
        return out
    
    def coef_(self):
        return self.linear.weight.detach().cpu().numpy()
    
    def fit(self, X_train, y_train, epochs=500):
#         add tolerance to this for tensors already in correct formmat
        X_train = torch.from_numpy(X_train).float().to(self.device)
        y_train = torch.from_numpy(y_train).float().to(self.device)
        
        if self.outputSize >2:
            y_train = y_train.long()
        else:
            y_train = y_train.unsqueeze(dim=1)

        loss_trajectory = []
        for epoch in range(epochs):
            #forward feed
            y_pred = self.forward(X_train.requires_grad_())
    
            #calculate the loss
            l1_reg = self.lambda1 * torch.norm(torch.from_numpy(self.coef_()), 1)
            
            criteria = self.criterion(y_pred, y_train)
            
            loss_trajectory.append(criteria.item())
            
            loss = l1_reg + criteria
            #backward propagation: calculate gradients
            loss.backward()

            #update the weights
            self.optimizer.step()

            #clear out the gradients from the last step loss.backward()
            self.optimizer.zero_grad()
            
        return loss_trajectory
            
    def predict(self, x):
        x = torch.from_numpy(x).float().to(self.device)
        with torch.no_grad():
            y_pred = self.forward(x)
            
            if self.outputSize > 2:
                return y_pred.argmax(1)
            else:
                return (y_pred > 0.5).long()

    def predict_proba(self, x):
        x = torch.from_numpy(x).to(self.device).float()
        with torch.no_grad():
            y_pred = self.forward(x)
            y_pred = torch.sigmoid(y_pred)
            return torch.cat((y_pred, 1-y_pred), dim=1)
        
        
        
        
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
