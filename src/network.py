import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import sys

sys.dont_write_bytecode = True

# Define the network training and prediction
class Net:
    def __init__(self, net, params, device):
        self.reg = net
        self.params = params
        self.device = device
        
    def train(self, data):
        #self.reg = self.net().to(self.device)
        self.reg.train()
        optimizer = torch.optim.Adam(self.reg.parameters(), **self.params['optim'])

        loader = DataLoader(data, shuffle=True, **self.params['train'])
        for epoch in tqdm(range(1, self.params['n_epoch']+1), ncols=100):
            losses = []
            for input, label, idx in loader:
                input, label = input.float().to(self.device), label.float().to(self.device)
                optimizer.zero_grad()
                out, _ = self.reg(input)
                loss = F.mse_loss(out.reshape(-1,1), label.reshape(-1,1))
                losses.append(loss.item())
                loss.backward()
                optimizer.step()
            # if epoch % 10 == 0:
            #     print(f'Epoch {epoch} Loss: {np.mean(losses)}')
            

    def predict(self, data):
        self.reg.eval()
        loader = DataLoader(data, shuffle=False, **self.params['test'])
        losses = []
        preds, targets = [], []

        with torch.no_grad():
            for input, target, idx in loader:
                input, target = input.float().to(self.device), target.float().to(self.device)
                out, _ = self.reg(input)
                loss = F.mse_loss(out.reshape(-1,1), target.reshape(-1,1))
                losses.append(loss.item())
                preds.append(out.detach().numpy())
                targets.append(target.detach().numpy())

        return preds, targets, np.mean(losses)
    
    def get_embeddings(self, data):
        self.reg.eval()
        embeddings = torch.zeros([len(data), self.reg.get_embedding_dim()])
        loader = DataLoader(data, shuffle=False, **self.params['test'])
        with torch.no_grad():
            for x, y, idxs in loader:
                x, y = x.to(self.device), y.to(self.device)
                _, e1 = self.clf(x)
                embeddings[idxs] = e1.cpu()
        return embeddings

# Define the network
class Regressor(nn.Module):
    def __init__(self, input_dim=1, n_hidden_size=10):
        super(Regressor, self).__init__()
        self.fc1 = nn.Linear(input_dim, n_hidden_size)
        self.fc2 = nn.Linear(n_hidden_size, n_hidden_size)
        self.fc3 = nn.Linear(n_hidden_size, int(n_hidden_size/2))
        self.fc4 = nn.Linear(int(n_hidden_size/2), 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        out = self.fc4(x)
        return out, x
    
    def get_embedding_dim(self):
        return 30
