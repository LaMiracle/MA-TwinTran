import torch
import torch.nn as nn
from torchdiffeq import odeint

class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(Encoder, self).__init__()
        self.linear = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        # x = x.contiguous().view(x.size(0), -1)  # Flatten the input
        return self.relu(self.linear(x))

class ODEFunc(nn.Module):
    def __init__(self, hidden_dim):
        super(ODEFunc, self).__init__()
        self.linear = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, t, x):
        return self.linear(x)

class Decoder(nn.Module):
    def __init__(self, hidden_dim, output_dim):
        super(Decoder, self).__init__()
        self.linear = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        return self.linear(x)

class NP_ODE_Model(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, device):
        super(NP_ODE_Model, self).__init__()
        self.encoder = Encoder(input_dim, hidden_dim)
        self.ode_func = ODEFunc(hidden_dim)
        self.decoder = Decoder(hidden_dim, output_dim)
        self.device = device

    def forward(self, x, t):
        x = x.to(self.device)
        t = t.to(self.device)
        encoded = self.encoder(x)
        ode_sol = odeint(self.ode_func, encoded, t, method='dopri5') # ode方程由128*128NN代替，对初始的encoder积分到t，并求解最终时刻t的encoder网络，再通过decoder转换回系统状态预测
        return self.decoder(ode_sol[-1])