import torch.nn as nn

# CNN Feature Extraction Block
class CNNnetwork(nn.Module):
    def __init__(self, k, seq_len, kernel_size=3):
        super().__init__()
        self.conv1 = nn.Conv1d(k, 16, kernel_size=kernel_size, padding=1)
        self.conv2 = nn.Conv1d(16, 32, kernel_size=kernel_size, padding=1)
        self.conv3 = nn.Conv1d(32, 64, kernel_size=kernel_size, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool1 = nn.MaxPool1d(kernel_size=2)
        self.maxpool2 = nn.MaxPool1d(kernel_size=2)
        self.Linear = nn.Linear(32*11, k*seq_len) # 16->32
        self.k = k
        self.seq_len = seq_len
    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.maxpool1(x)
        x = x.view(x.size(0), x.size(1) * x.size(2))
        x = self.Linear(x)
        x = x.view(x.size(0), self.k, self.seq_len)
        return x