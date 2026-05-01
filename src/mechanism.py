import torch
import torch.nn as nn
import numpy as np
import math
from copy import deepcopy

device = 'cpu'
if torch.cuda.is_available():
    torch.set_default_tensor_type(torch.cuda.FloatTensor)
    device = 'cuda'

class MechanismModel(nn.Module):
    '''
    ['ALT', 'IVV', 'TAS', 'GS', 'AOA1', 'AOA2', 'PTCH', 'WS', 'WD', 'SAT', 'TAT', 'PI', 'PT', 'GAMMA']
    注意：nasa数据集中ALTR的单位是英尺每分钟，TAS的单位是节，地速GS的单位是节；风速的单位是节
    考虑到后续风切变强度计算的基本单位是英尺/秒，需要进行单位转化
    1节 = 101.27英尺每分钟 = 1.6878英尺每秒
    '''

    def __init__(self, feature_size, pred_len):
        super().__init__()
        self.fc = nn.Linear(feature_size, feature_size)
        self.ReLU = nn.ReLU()
        self.kernel_size = 3
        self.k = feature_size
        self.seq_len = pred_len
        self.conv1 = nn.Conv1d(self.k, 16, kernel_size=self.kernel_size, padding=1)
        self.conv2 = nn.Conv1d(16, 32, kernel_size=self.kernel_size, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=2)
        self.Linear = nn.Linear(32 * int(pred_len/5), self.k * self.seq_len)

    def mechaismRead(self, formula_str):
        return 0

    def mechanism(self, Y, scaler):  # batch_size * pred_len * feature_size
        Y_mechanism = deepcopy(Y)
        Y_mechanism = Y_mechanism.cpu().numpy()
        for iBatch in range(Y_mechanism.shape[0]):
            Y_mechanism[iBatch, :, :] = scaler.inverse_transform(Y_mechanism[iBatch, :, :])
        Y_mechanism[:, :, 7] = np.abs(
            Y_mechanism[:, :, 3] - Y_mechanism[:, :, 2] * np.cos(Y_mechanism[:, :, -1] / 180 * math.pi))
        for iBatch in range(Y_mechanism.shape[0]):
            Y_mechanism[iBatch, :, :] = scaler.transform(Y_mechanism[iBatch, :, :])
        return torch.from_numpy(Y_mechanism).to(device)

    def cnn(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = x.view(x.size(0), x.size(1) * x.size(2))
        x = self.Linear(x)
        x = x.view(x.size(0), self.k, self.seq_len)
        return x

    def forward(self, x):
        x = self.cnn(x.transpose(1, 2)).transpose(1, 2) + x
        return x