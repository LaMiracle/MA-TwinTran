import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        is_odd_d_model = False
        if d_model % 2 != 0:
            is_odd_d_model = True
            d_model += 1
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0) # 1 * max_len * d_model
        if is_odd_d_model:
            d_model -= 1
            pe = pe[:, :, :-1]
        # print(pe.size())
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class RotatePositionalEncoding(nn.Module):
    def __init__(self, seq_len, d_model):
        super(RotatePositionalEncoding, self).__init__()
        position = torch.arange(0, seq_len).unsqueeze(1).float() # seq_len * 1
        theta = torch.pow(10000, -2 * (torch.arange(1, int(d_model/2)+1) - 1) / d_model).repeat(2).unsqueeze(0) # 1 * d_model
        pe_theta = torch.mm(position, theta)
        rotate_cos = torch.cos(pe_theta).unsqueeze(0)
        rotate_sin = torch.sin(pe_theta).unsqueeze(0)
        self.register_buffer('rotate_cos', rotate_cos)
        self.register_buffer('rotate_sin', rotate_sin)

    def forward(self, x):
        # 读入x：batch_size * decay_len * feature_size
        dif_x = x.clone()
        dif_x[:, :, 1::2] *= -1
        R = torch.multiply(x, self.rotate_cos) + torch.multiply(dif_x, self.rotate_sin)

        return R
