import torch
import torch.nn as nn
from cnn import CNNnetwork
from convTrans import ConvTransformerBLock

# Forcasting Conv Transformer :
class ForecastConvTransformer(nn.Module):
    '''
    CNN特征提取层
    位置编码层
    堆叠的卷积tran层
    实现时序压缩的FC层
    '''

    def __init__(self, k, headers, depth, decay_len, pred_len, kernel_size=9, mask_next=True, mask_diag=False,
                 sparse=True, dropout_proba=0.2, num_tokens=None):
        super().__init__()
        self.depth = depth
        # Embedding
        self.tokens_in_count = False
        if num_tokens:
            self.tokens_in_count = True
            self.token_embedding = nn.Embedding(num_tokens, k)  # （369, 1）= (nb_ts, k)

        # Embedding the position，进行位置编码，没有padding填充
        self.position_embedding = nn.Embedding(decay_len, k)  # (500, 1) = (windows_size, k)

        # Number of input channels
        self.k = k  # 没有协变量的情况下，k=1
        self.seq_length = decay_len  # 历史序列长度

        # Feature extraction block
        self.cnn = CNNnetwork(k, self.seq_length)

        tblocks = []
        # 多层ConvTrans层堆叠
        self.transformer = ConvTransformerBLock(k, headers, decay_len, pred_len, kernel_size, mask_next, mask_diag, sparse, dropout_proba)
        for t in range(depth):
            tblocks.append(self.transformer)
        self.TransformerBlocks = nn.Sequential(*tblocks)

        self.fc = nn.Linear(decay_len, pred_len)

        self.fc2 = nn.Linear(2 * k, k)

    def forward(self, x, tokens=None, is_test=False):
        b, t, k = x.size()

        # checking that the given batch had same number of time series as the BLock had
        assert k == self.k, 'The k :' + str(
            self.k) + ' number of timeseries given in the initialization is different than what given in the x :' + str(
            k)
        assert t == self.seq_length, 'The lenght of the timeseries given t ' + str(
            t) + ' miss much with the lenght sequence given in the Tranformers initialisation self.seq_length: ' + str(
            self.seq_length)

        # Position embedding
        pos = torch.arange(t)  # 准备历史长度的向量
        self.pos_emb = self.position_embedding(pos).expand(b, t, k)  # 将历史长度向量的位置编码结果扩展（复制）为[b, t, k]的张量

        # Checking token embedding
        assert self.tokens_in_count == (not (tokens is None)), 'self.tokens_in_count = ' + str(
            self.tokens_in_count) + ' should be equal to (not (tokens is None)) = ' + str((not (tokens is None)))
        if not (tokens is None):
            ## checking that the number of tockens corresponde to the number of batch elements
            assert tokens.size(0) == b
            self.tok_emb = self.token_embedding(tokens)
            self.tok_emb = self.tok_emb.expand(t, b, k).transpose(0, 1)

        # Adding Pos Embedding and token Embedding to the variable
        if not (tokens is None):
            x = self.pos_emb + self.tok_emb + x
        else:
            x = self.pos_emb + x  # 向输入中添加位置编码

        # # Feature extracting
        # x_cnn = x.contiguous().transpose(1, 2)  # (b, k, t)
        # x_cnn = self.cnn(x_cnn)
        # # print(x.size())
        # x_cnn = x_cnn.transpose(1, 2)  # (b, t, k)
        #
        # # x = x_cnn #与transformer联合时注释掉

        # Transformer :
        x_tran = x.contiguous()
        # x_tran = self.TransformerBlocks(x, is_test=is_test)
        for _ in range(self.depth):
            x_tran = self.transformer(x, is_test=is_test)
        x = x_tran

        # # 特征联合和特征维数变换
        # x = torch.cat((x_cnn, x_tran), dim=2)
        # x = self.fc2(x)

        # 时间长度变换
        x = x.transpose(1, 2)
        x = self.fc(x)
        x = x.transpose(1, 2)

        return x