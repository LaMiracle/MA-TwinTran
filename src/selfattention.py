import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np

# Self Attention Class
class SelfAttention(nn.Module):
    def __init__(self, k, headers, decay_len, pred_len, kernel_size=1, mask_next=True, mask_diag=False, sparse = True):
        super().__init__()

        self.k, self.headers, self.kernel_size = k, headers, kernel_size # input_size, header_size, kernel_size
        self.win_len = decay_len
        self.mask_next = mask_next
        self.mask_diag = mask_diag
        self.sparse = sparse

        h = headers # 注意力头数

        # Query, Key and Value Transformations
        padding = (kernel_size - 1)
        self.padding_opertor = nn.ConstantPad1d((padding, 0), 0) # 该方法将在目标张量的各个维度的首尾各添加padding和0个0值

        # 同源产生Q/K/V矩阵，输入为k维的时序向量，通过多头拆分成k*h个通道，形成kernel_size*k大小的卷积核
        self.toqueries = nn.Conv1d(k, k * h, kernel_size, padding=0, bias=True)
        self.tokeys = nn.Conv1d(k, k * h, kernel_size, padding=0, bias=True)
        self.tovalues = nn.Conv1d(k, k * h, kernel_size=1, padding=0, bias=False)  # No convolution operated
        # kernel_size=1就是原始transformer，>1就是卷积transformer，卷积核能够收集到更多上下文趋势信息

        # Heads unifier
        self.unifyheads = nn.Linear(k * h, k)

        # predictor unifier
        self.unifypreds = nn.Linear(k * 2, k)

    def log_sparse_mask(self, win_len, sub_len=1):
        """
        revised based on https://github.com/AIStream-Peelout/flow-forecast/blob/master/flood_forecast/transformer_xl/transformer_bottleneck.py
        Remark:
        1 . Currently, dense matrices with sparse multiplication are not supported by Pytorch. Efficient implementation
            should deal with CUDA kernel, which we haven't implemented yet.

        2 . Our default setting here use Local attention and Restart attention.

        3 . For index-th row, if its past is smaller than the number of cells the last
            cell can attend, we can allow current cell to attend all past cells to fully
            utilize parallel computing in dense matrices with sparse multiplication."""
        mask = torch.zeros((win_len, win_len), dtype=torch.float)
        for i in range(win_len):
            log_l = math.ceil(np.log2(sub_len))
            mask[i] = torch.zeros((win_len), dtype=torch.float)

            if ((win_len // sub_len) * 2 * (log_l) > i):
                mask[i][:(i + 1)] = 1
            else:
                while (i >= 0):
                    if ((i - log_l + 1) < 0):
                        mask[i][:i] = 1
                        break
                    mask[i][i - log_l + 1:(i + 1)] = 1  # Local attention
                    for i in range(0, log_l):
                        new_index = i - log_l + 1 - 2**i
                        if ((i - new_index) <= sub_len and new_index >= 0):
                            mask[i][new_index] = 1
                    i -= sub_len
        return mask.view(1, mask.size(0), mask.size(1))

    def forward(self, x):
        # Extraction dimensions
        b, t, k = x.size()  # batch_size, number_of_timesteps, input_size

        # Checking Embedding dimension,
        assert self.k == k, 'Number of time series ' + str(k) + ' didn t much the number of k ' + str(
            self.k) + ' in the initiaalization of the attention layer.'
        h = self.headers

        #  Transpose to see the different time series as different channels
        x = x.transpose(1, 2) # 将最后两个维度转置，获得[batch_size, input_size, number_of_timesteps]形状的张量
        x_padded = self.padding_opertor(x) # 由于padding为0，并没有在张量外围补充0值行列

        # Query, Key and Value Transformations，并将卷积的多头通道拆分
        queries = self.toqueries(x_padded).view(b, k, h, t)
        keys = self.tokeys(x_padded).view(b, k, h, t)
        values = self.tovalues(x).view(b, k, h, t)

        # Transposition to return the canonical format
        queries = queries.transpose(1, 2)  # batch, header, input_size, time step (b, h, k, t)
        queries = queries.transpose(2, 3)  # batch, header, time step, input_size (b, h, t, k)

        values = values.transpose(1, 2)  # batch, header, input_size, time step (b, h, k, t)
        values = values.transpose(2, 3)  # batch, header, time step, input_size (b, h, t, k)

        keys = keys.transpose(1, 2)  # batch, header, input_size, time step (b, h, k, t)
        keys = keys.transpose(2, 3)  # batch, header, time step, input_size (b, h, t, k)

        '''开始计算自注意力矩阵'''

        # Weights
        queries = queries / (k ** (.25))
        keys = keys / (k ** (.25))

        # 将转置后的Q/K/V矩阵进行深拷贝并将不同batch的多头通道合并，即将被分到子空间的原属于不同batch的词向量整合
        queries = queries.transpose(1, 2).contiguous().view(b * h, t, k)
        keys = keys.transpose(1, 2).contiguous().view(b * h, t, k)
        values = values.transpose(1, 2).contiguous().view(b * h, t, k)

        # 计算QK^T， weights是t*t的矩阵
        temporal_weights = torch.bmm(queries, keys.transpose(1, 2))
        # logSparse mask
        if self.sparse:
            temporal_mask = self.log_sparse_mask(win_len=self.win_len)
            temporal_weights = temporal_weights * temporal_mask + -1e9 * (1 - temporal_mask)
        # Mask the upper & diag of the attention matrix，使用掩码机制
        if self.mask_next:
            if self.mask_diag:
                indices = torch.triu_indices(t, t, offset=0) # 获取一个t行t列的矩阵的包括对角线和上三角部分元素的[row_idx, cow_idx]
                temporal_weights[:, indices[0], indices[1]] = float('-inf') # 屏蔽对角线和上三角部分的注意力权重，即第一轮什么都不知道
            else:
                indices = torch.triu_indices(t, t, offset=1) # 获取一个t行t列的矩阵的上三角部分元素的[row_idx, cow_idx]
                temporal_weights[:, indices[0], indices[1]] = float('-inf') # 屏蔽上三角部分的注意力权重，即第一轮是知道一个特征的
        # Softmax
        temporal_weights = F.softmax(temporal_weights/np.sqrt(k), dim=2)
        # Output
        temporal_output = torch.bmm(temporal_weights, values) # 计算softmax(QK^T)V
        temporal_output = temporal_output.view(b, h, t, k)
        temporal_output = temporal_output.transpose(1, 2).contiguous().view(b, t, k * h)
        # unify
        unified_temporal_output = self.unifyheads(temporal_output)  # shape (b,t,k)

        # 计算QK^T， weights是k*k的矩阵
        feature_weights = torch.bmm(queries.transpose(1, 2), keys)
        # logSparse mask
        if self.sparse:
            feature_mask = self.log_sparse_mask(win_len=self.k)
            feature_weights = feature_weights * feature_mask + -1e9 * (1 - feature_mask)
        # Softmax
        feature_weights = F.softmax(feature_weights/np.sqrt(self.win_len), dim=2)
        # Output
        feature_output = torch.bmm(feature_weights, values.transpose(1, 2)) # k * t
        feature_output = feature_output.transpose(1, 2) # t * k
        feature_output = feature_output.view(b, h, t, k)
        feature_output = feature_output.transpose(1, 2).contiguous().view(b, t, k * h)
        # unify
        unified_feature_output = self.unifyheads(feature_output)  # shape (b,t,k)

        # predictor unify
        output = torch.cat([unified_temporal_output, unified_feature_output], dim=2)
        # print(output.size())
        unified_output = self.unifypreds(output)

        return unified_output
        # return unified_temporal_output