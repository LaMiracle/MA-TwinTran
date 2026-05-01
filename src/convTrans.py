import torch.nn as nn
from selfattention import SelfAttention

# Conv Transformer Block
class ConvTransformerBLock(nn.Module):
    '''
    自注意力层
    dropout层
    归一化层
    *4 FL
    Relu层
    /4 FL
    归一化层
    '''

    def __init__(self, k, headers, decay_len, pred_len, kernel_size=9, mask_next=False, mask_diag=False, sparse=True,
                 dropout_proba=0.2):
        super().__init__()

        # Self attention
        self.attention = SelfAttention(k, headers, decay_len, pred_len, kernel_size, mask_next, mask_diag, sparse)

        # First & Second Norm
        self.norm1 = nn.LayerNorm(k)
        self.norm2 = nn.LayerNorm(k)

        # Feed Forward Network
        self.feedforward = nn.Sequential(
            nn.Linear(k, 4 * k),
            nn.ReLU(),
            nn.Linear(4 * k, k)
        )

        # Dropout funtcion  & Relu:
        self.dropout = nn.Dropout(p=dropout_proba)
        self.activation = nn.ReLU()

    def TransformerLayer(self, x, is_test=False):
        # Self attention + Residual
        x = self.attention(x) + x

        # Dropout attention
        if not is_test:
            x = self.dropout(x)

        # First Normalization
        x = self.norm1(x)

        # Feed Froward network + residual
        x = self.feedforward(x) + x

        # Second Normalization
        x = self.norm2(x)

        return x

    def forward(self, x, is_test=False):
        transformer_layer_out = self.TransformerLayer(x, is_test=is_test)

        x = transformer_layer_out + x  # 加入残差结构

        return x  # batch_size * decay_len * feature_size