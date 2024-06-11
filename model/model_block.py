import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np

# ResNet_basic_block0 L->L
class ResNet_basic_block0(nn.Module): # L->L
    def __init__(self, in_channels, out_channels, kernel_size, kernel_size2):
        super().__init__()
        # k-2p=s=1 (k,p,s)=(k=3,p=1,s=1)=(k=5,p=2,s=1)=(k=7,p=3,s=1) L->L
        # 1_3
        kernel_size_3 = kernel_size-2
        self.conv1_3 = nn.Conv1d(in_channels=in_channels,
                               out_channels=out_channels,
                               kernel_size=kernel_size_3,
                               stride=1,
                               padding=int((kernel_size_3-1)/2)) # L->L
        # self.bn1_3 = nn.BatchNorm1d(num_features=out_channels)
        self.relu1_3 = nn.ReLU()
        # 1_5
        kernel_size_5 = kernel_size
        self.conv1_5 = nn.Conv1d(in_channels=in_channels,
                               out_channels=out_channels,
                               kernel_size=kernel_size_5,
                               stride=1,
                               padding=int((kernel_size_5 - 1) / 2))  # L->L
        # self.bn1_5 = nn.BatchNorm1d(num_features=out_channels)
        self.relu1_5 = nn.ReLU()
        # 1_7
        kernel_size_7 = kernel_size+2
        self.conv1_7 = nn.Conv1d(in_channels=in_channels,
                               out_channels=out_channels,
                               kernel_size=kernel_size_7,
                               stride=1,
                               padding=int((kernel_size_7 - 1) / 2))  # L->L
        # self.bn1_7 = nn.BatchNorm1d(num_features=out_channels)
        self.relu1_7 = nn.ReLU()

        self.conv2 = nn.Conv1d(in_channels=out_channels,
                               out_channels=out_channels,
                               kernel_size=kernel_size2,
                               stride=1,
                               padding=int((kernel_size2-1)/2)) # L->L
        # self.bn2 = nn.BatchNorm1d(num_features=out_channels)
        self.relu2 = nn.ReLU()

        kernel_size3 = kernel_size
        self.conv3 = nn.Conv1d(in_channels=in_channels,
                               out_channels=out_channels,
                               kernel_size=kernel_size3,
                               stride=1,
                               padding=int((kernel_size3 - 1) / 2))  # L->L

        kernel_size4 = 1
        self.conv4 = nn.Conv1d(in_channels=in_channels,
                               out_channels=out_channels,
                               kernel_size=kernel_size4,
                               stride=1,
                               padding=int((kernel_size4 - 1) / 2))  # L->L

    def forward(self, x):
        # (N, C, L)
        # 1_3
        out_3 = self.conv1_3(x)
        # out_3 = self.bn1_3(out_3)
        out_3 = self.relu1_3(out_3)
        # 1_5
        out_5 = self.conv1_5(x)
        # out_5 = self.bn1_5(out_5)
        out_5 = self.relu1_5(out_5)
        # 1_7
        out_7 = self.conv1_7(x)
        # out_7 = self.bn1_7(out_7)
        out_7 = self.relu1_7(out_7)
        # add
        out = out_3 + out_5 + out_7

        out = self.conv2(out)
        # out = self.bn2(out)

        xn = self.conv3(x)
        # xn = self.bn2(xn)
        out = out + xn  # 残差
        x1 = self.conv4(x)
        # x1 = self.bn2(x1)
        out = out + x1 # 残差
        r = self.relu2(out)
        return r

# ResNet_basic_block1 L->1/2L
class ResNet_basic_block1(nn.Module): # L->L/2
    def __init__(self, in_channels, out_channels, kernel_size, kernel_size2):
        super().__init__()
        # k-2p=s=2 (k,p,s)=(k=4,p=1,s=2)=(k=6,p=2,s=2)=(k=8,p=3,s=2) L->L/2
        # 1_4
        kernel_size_4 = kernel_size-2
        self.conv1_4 = nn.Conv1d(in_channels=in_channels,
                                 out_channels=out_channels,
                                 kernel_size=kernel_size_4,
                                 stride=2,
                                 padding=int((kernel_size_4 - 2) / 2))  # L->L
        # self.bn1_4 = nn.BatchNorm1d(num_features=out_channels)
        self.relu1_4 = nn.ReLU()
        # 1_6
        kernel_size_6 = kernel_size
        self.conv1_6 = nn.Conv1d(in_channels=in_channels,
                                 out_channels=out_channels,
                                 kernel_size=kernel_size_6,
                                 stride=2,
                                 padding=int((kernel_size_6 - 2) / 2))  # L->L
        # self.bn1_6 = nn.BatchNorm1d(num_features=out_channels)
        self.relu1_6 = nn.ReLU()
        # 1_8
        kernel_size_8 = kernel_size+2
        self.conv1_8 = nn.Conv1d(in_channels=in_channels,
                                 out_channels=out_channels,
                                 kernel_size=kernel_size_8,
                                 stride=2,
                                 padding=int((kernel_size_8 - 2) / 2))  # L->L
        # self.bn1_8 = nn.BatchNorm1d(num_features=out_channels)
        self.relu1_8 = nn.ReLU()

        self.conv2 = nn.Conv1d(in_channels=out_channels,
                               out_channels=out_channels,
                               kernel_size=kernel_size2,
                               stride=1,
                               padding=int((kernel_size2-1)/2)) # L->L

        # self.bn2 = nn.BatchNorm1d(num_features=out_channels)
        self.relu2 = nn.ReLU()

        kernel_size3 = kernel_size
        self.conv3 = nn.Conv1d(in_channels=in_channels,
                               out_channels=out_channels,
                               kernel_size=kernel_size3,
                               stride=2,
                               padding=int((kernel_size3-2)/2)) # L->L/2

        kernel_size4 = 2
        self.conv4 = nn.Conv1d(in_channels=in_channels,
                               out_channels=out_channels,
                               kernel_size=kernel_size4,
                               stride=2,
                               padding=int((kernel_size4 - 2) / 2))  # L->L

    def forward(self, x):
        # (N, C, L)
        # 1_4
        out_4 = self.conv1_4(x)
        # out_4 = self.bn1_4(out_4)
        out_4 = self.relu1_4(out_4)
        # 1_6
        out_6 = self.conv1_6(x)
        # out_6 = self.bn1_6(out_6)
        out_6 = self.relu1_6(out_6)
        # 1_8
        out_8 = self.conv1_8(x)
        # out_8 = self.bn1_8(out_8)
        out_8 = self.relu1_8(out_8)
        # add
        out = out_4 + out_6 + out_8

        out = self.conv2(out)
        # out = self.bn2(out)

        xn = self.conv3(x)
        # xn = self.bn2(xn)
        out = out + xn  # 残差
        x1 = self.conv4(x)
        # x1 = self.bn2(x1)
        out = out + x1 # 残差
        r = self.relu2(out)
        return r


# ResNet_basic_block0 L->L
class ResNet_basic_block0_simple(nn.Module): # L->L
    def __init__(self, in_channels, out_channels, kernel_size, kernel_size2):
        super().__init__()
        # k-2p=s=1 (k,p,s)=(k=3,p=1,s=1)=(k=5,p=2,s=1)=(k=7,p=3,s=1) L->L
        # 1_5
        kernel_size_5 = kernel_size
        self.conv1_5 = nn.Conv1d(in_channels=in_channels,
                               out_channels=out_channels,
                               kernel_size=kernel_size_5,
                               stride=1,
                               padding=int((kernel_size_5 - 1) / 2))  # L->L
        # self.bn1_5 = nn.BatchNorm1d(num_features=out_channels)
        self.relu1_5 = nn.ReLU()

        self.conv2 = nn.Conv1d(in_channels=out_channels,
                               out_channels=out_channels,
                               kernel_size=kernel_size2,
                               stride=1,
                               padding=int((kernel_size2-1)/2)) # L->L
        # self.bn2 = nn.BatchNorm1d(num_features=out_channels)
        self.relu2 = nn.ReLU()

    def forward(self, x):
        # (N, C, L)
        # 1_5
        out_5 = self.conv1_5(x)
        # out_5 = self.bn1_5(out_5)
        out_5 = self.relu1_5(out_5)
        # add
        out = out_5

        out = self.conv2(out)
        # out = self.bn2(out)
        r = self.relu2(out)
        return r

# ResNet_basic_block1 L->1/2L
class ResNet_basic_block1_simple(nn.Module): # L->L/2
    def __init__(self, in_channels, out_channels, kernel_size, kernel_size2):
        super().__init__()
        # k-2p=s=2 (k,p,s)=(k=4,p=1,s=2)=(k=6,p=2,s=2)=(k=8,p=3,s=2) L->L/2
        # 1_6
        kernel_size_6 = kernel_size
        self.conv1_6 = nn.Conv1d(in_channels=in_channels,
                                 out_channels=out_channels,
                                 kernel_size=kernel_size_6,
                                 stride=2,
                                 padding=int((kernel_size_6 - 2) / 2))  # L->L
        # self.bn1_6 = nn.BatchNorm1d(num_features=out_channels)
        self.relu1_6 = nn.ReLU()

        self.conv2 = nn.Conv1d(in_channels=out_channels,
                               out_channels=out_channels,
                               kernel_size=kernel_size2,
                               stride=1,
                               padding=int((kernel_size2-1)/2)) # L->L

        # self.bn2 = nn.BatchNorm1d(num_features=out_channels)
        self.relu2 = nn.ReLU()

    def forward(self, x):
        # (N, C, L)
        # 1_6
        out_6 = self.conv1_6(x)
        # out_6 = self.bn1_6(out_6)
        out_6 = self.relu1_6(out_6)
        # add
        out = out_6

        out = self.conv2(out)
        # out = self.bn2(out)
        r = self.relu2(out)
        return r


# Attention
class ScaledDotProductAttention(nn.Module):
    """ Scaled Dot-Product Attention """

    def __init__(self, scale):
        super().__init__()

        self.scale = scale
        self.softmax = nn.Softmax(dim=2)

    def forward(self, q, k, v, mask=None):

        u = torch.bmm(q, k.transpose(1, 2)) # 1.Matmul
        u = u / self.scale # 2.Scale

        if mask is not None:
            u = u.masked_fill(mask, -np.inf) # 3.Mask

        attn = self.softmax(u) # 4.Softmax
        output = torch.bmm(attn, v) # 5.Output

        return attn, output


# MultiHead-Attention
class MultiHeadAttention(nn.Module):
    """ Multi-Head Attention """

    def __init__(self, n_head, d_k_, d_v_, d_k, d_v, d_o):
        super().__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.fc_q = nn.Linear(d_k_, n_head * d_k)
        self.fc_k = nn.Linear(d_k_, n_head * d_k)
        self.fc_v = nn.Linear(d_v_, n_head * d_v)

        self.attention = ScaledDotProductAttention(scale=np.power(d_k, 0.5))

        self.fc_o = nn.Linear(n_head * d_v, d_o)

    def forward(self, q, k, v, mask=None):

        n_head, d_q, d_k, d_v = self.n_head, self.d_k, self.d_k, self.d_v

        batch, n_q, d_q_ = q.size()
        batch, n_k, d_k_ = k.size()
        batch, n_v, d_v_ = v.size()

        q = self.fc_q(q) # 1.单头变多头 (N, L, C*n_head)
        k = self.fc_k(k) # (N, L, C*n_head)
        v = self.fc_v(v) # (N, L, C*n_head)
        q = q.view(batch, n_q, n_head, d_q).permute(2, 0, 1, 3).contiguous().view(-1, n_q, d_q) # (N*n_head, L, C)
        k = k.view(batch, n_k, n_head, d_k).permute(2, 0, 1, 3).contiguous().view(-1, n_k, d_k) # (N*n_head, L, C)
        v = v.view(batch, n_v, n_head, d_v).permute(2, 0, 1, 3).contiguous().view(-1, n_v, d_v) # (N*n_head, L, C)

        if mask is not None:
            mask = mask.repeat(n_head, 1, 1)
        attn, output = self.attention(q, k, v, mask=mask) # 2.当成单头注意力求输出

        output = output.view(n_head, batch, n_q, d_v).permute(1, 2, 0, 3).contiguous().view(batch, n_q, -1) # 3.Concat
        output = self.fc_o(output) # 4.仿射变换得到最终输出

        return attn, output


class MultiHead_SelfAttention(nn.Module):
    """ Self-Attention """

    def __init__(self, n_head, d_k, d_v, d_x, d_o):
        super().__init__()

        self.wq = nn.Parameter(torch.Tensor(d_x, d_k))
        self.wk = nn.Parameter(torch.Tensor(d_x, d_k))
        self.wv = nn.Parameter(torch.Tensor(d_x, d_v))

        self.mha = MultiHeadAttention(n_head=n_head, d_k_=d_k, d_v_=d_v, d_k=d_k, d_v=d_v, d_o=d_o)

        self.init_parameters()

    def init_parameters(self):
        for param in self.parameters():
            stdv = 1. / np.power(param.size(-1), 0.5)
            param.data.uniform_(-stdv, stdv)

    def forward(self, x, y, mask=None):
        # x(batch=256, n=100, d=300)
        # y(batch=256, n=1000, d=300)
        q = torch.matmul(x, self.wq) # (N=256, L_x=100, d_k=128)
        k = torch.matmul(y, self.wk) # (N=256, L_y=1000, d_k=128)
        v = torch.matmul(y, self.wv) # (N=256, L_y=1000, d_v=64)

        attn, output = self.mha(q, k, v, mask=mask)
        r = output

        return r


