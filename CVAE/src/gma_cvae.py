import torch
import numpy as np
import torch.nn.functional as F
from torch import nn, optim
import src.utils as utils
import scanpy as sc
from scipy import sparse
from src.customized_linear import CustomizedLinear

class GMA_CVAE(torch.nn.Module):
    def __init__(self, pathway_mask, n_labels, **kwargs):
        super(GMA_CVAE, self).__init__()

        self.pathway_mask = pathway_mask
        self.n_pathways = self.pathway_mask.shape[1] - n_labels
        self.n_genes = self.pathway_mask.shape[0]
        self.n_labels = n_labels
        self.beta = kwargs.get('beta', 0.05)
        self.dropout = kwargs.get('dropout', 0.03)
        self.learn_rate = kwargs.get('lr', 0.001)
        self.weight_decay = kwargs.get('wd', 0.001)
        self.init_w = kwargs.get('init_w', False)
        self.model = kwargs.get('model', 'trained_models/trained_gma_cvae.pt')
        self.verbose = kwargs.get('verbose', True)
        self.device = kwargs.get('device', torch.device('cpu'))

        self.encoder = nn.Sequential(
            nn.Linear(self.n_genes + self.n_labels, 800),
            nn.BatchNorm1d(800),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(800, 400),
            nn.BatchNorm1d(400),
            nn.ReLU(),
            nn.Dropout(self.dropout))
        
        self.mean = nn.Sequential(
            nn.Linear(400, self.n_pathways),
            nn.Dropout(self.dropout))
        
        self.logvar = nn.Sequential(
            nn.Linear(400, self.n_pathways),
            nn.Dropout(self.dropout))
        
        self.decoder = CustomizedLinear(self.pathway_mask.T)
        
        if self.init_w:
            self.encoder.apply(self.init_weight)
            self.decoder(self.init_weight)
            # should init weights for mean/logvar?
            self.mean.apply(self.init_weight)
            self.logvar(self.init_weight)
        

    def init_weight(m):
        if type(m) == nn.Linear:
            torch.nn.init.xavier_uniform_(m.weight)


    def get_sample(self, mean, logvar):
        """ Sample latent space with reparameterization method """
        std = logvar.mul(0.5).exp()
        eps = torch.FloatTensor(std.size()).normal_().to(self.device)
        z = eps.mul(std).add(mean)
        return z


    def encode(self, x, y):
        if y.dim() == 1:
            y = y.unsqueeze(1)
        out = self.encoder(torch.cat((x, y), dim = 1))
        mean, logvar = self.mean(out), self.logvar(out)
        z = self.get_sample(mean, logvar)
        return z, mean, logvar
    
    
    def decode(self, z, y):
        if y.dim() == 1:
            y = y.unsqueeze(1)
        x_reconstructed = self.decoder(torch.cat((z, y), dim = 1)) 
        return x_reconstructed
    

    def forward(self, x, y):
        """ Full forward pass through network """
        z, mean, logvar = self.encode(x, y)
        X_reconstructed = self.decode(z, y)
        return X_reconstructed, mean, logvar, z
    

    def criterion(self, X_reconstructed, X_true, mean, logvar):
        """ VAE Loss, KLD * beta parameter + MSE """
        kld = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp())
        mse = F.mse_loss(X_reconstructed, X_true, reduction = 'sum')
        return torch.mean(mse + self.beta * kld)
    

    def train_net(self, train_loader, lr = 0.001, weight_decay = 0.01, n_epochs = 100, patience = 10, 
                  test_loader = False, save_model = True):
        epoch_hist = {'train_loss': [], 'valid_loss': []}
        optimizer = optim.Adam(self.parameters(), lr = lr, weight_decay = weight_decay)
        epoch = 0
        count_to_patience = 0
        global_min_loss = np.inf
        self.to(self.device)  # Move model to the device

        print("TRAINING IN PROGRESS :D", flush=True)
        while epoch < n_epochs and count_to_patience < patience:
            self.train()
            loss_val = 0
            print("epoch", epoch, flush=True)
            for x_train, y_train in train_loader:
                x_train, y_train = x_train.to(self.device), y_train.to(self.device)  # Move data to the device
                optimizer.zero_grad()
                X_reconstructed, mean, logvar, z = self.forward(x_train, y_train)
                loss = self.criterion(X_reconstructed, x_train, mean, logvar)
                loss_val += loss.item()
                loss.backward()
                optimizer.step()

            epoch_loss = loss_val / len(train_loader)
            epoch_hist['train_loss'].append(epoch_loss)
            if test_loader:
                self.eval()
                test_dict = self.test_model(test_loader)
                test_loss = test_dict['loss']
                epoch_hist['valid_loss'].append(test_loss)
                count_to_patience, global_min_loss = utils.early_stopping(test_loss, 
                                                                     count_to_patience, global_min_loss)
                if self.verbose:
                    print(f'------------Epoch {epoch + 1}------------', flush=True)
                    print(f'loss: {epoch_loss:.3f} | test loss: {test_loss:.3f}', flush=True)
            epoch += 1

        
        print("Training done.")
        if save_model:
            print('Saving model to ', self.model)
            torch.save(self.state_dict(), self.model)


    def test_model(self, test_loader):
        """ Evaluate the model on the test dataset """
        loss_val = 0
        self.eval()
        with torch.no_grad():
            for x_test, y_test in test_loader:
                x_test, y_test = x_test.to(self.device), y_test.to(self.device)
                x_reconstructed, mean, logvar, z = self.forward(x_test, y_test)
                loss = self.criterion(x_reconstructed, x_test, mean, logvar)
                loss_val += loss.item()
        test_loss = loss_val / (len(test_loader) * test_loader.batch_size)
        return {'loss': test_loss}
    
    def generate(self, c, num_samples):
        self.eval()
        with torch.no_grad():
            z = torch.randn(num_samples, self.n_pathways)
            c = c.repeat(num_samples, 1)
            samples = self.decode(z, c)
        return samples
