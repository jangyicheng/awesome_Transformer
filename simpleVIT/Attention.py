import copy

import numpy as np
from torch import nn
from torch.nn import functional as F
from einops import rearrange
import torch


def clones(module, n):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(n)])

class multiHeadAttention(nn.Module):
    def __init__(self, model_dim, head_dim, dropout=0., qkv_bias=False):
        super().__init__()
        assert model_dim % head_dim == 0
        self.dim = model_dim
        self.head_dim = head_dim
        self.qkv = nn.Linear(model_dim, model_dim * 3, bias=qkv_bias)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):  # [B,N,C]
        b, n, c = x.shape
        dk = self.head_dim ** 0.5
        qkv = self.qkv(x).split([c, c, c], dim=-1)
        qkv = self.qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.head_dim), qkv)
        # attn = torch.einsum('b h q n,b h k n->b h q k', q, k)  # [batchsize,head_dim,len_q,len_k]，要求q_dim=k_dim
        attn = torch.matmul(q, k.transpose(-1, -2)) / dk
        if mask is not None:
            attn = attn.masked_fill(mask == 0, -np.inf)

        attn = self.dropout(F.softmax(attn, dim=-1))
        # output = torch.einsum('b h q n,b h n v->b h q v', attn, v)  #  [batchsize,head_dim,len_q,v_dim],要求len_k=len_v
        output = torch.matmul(attn, v)
        output = output.transpose(1, 2).reshape(b, n, c)
        # output = rearrange(output, 'b h n d -> b n (h d) ', h=self.head_dim)
        return output, attn


class MultiHeadAttention(nn.Module):
    def __init__(self, model_dim, num_heads, dropout=0., qkv_bias=False):
        super().__init__()
        assert model_dim % num_heads == 0
        self.dim = model_dim
        self.num_heads = num_heads
        self.dk=model_dim // num_heads
        self.qkv = nn.Linear(model_dim, model_dim * 3, bias=qkv_bias)
        self.linears = clones(nn.Linear(model_dim, model_dim), 3)
        self.attn = None
        self.project = nn.Linear(model_dim, model_dim)
        self.dropout = nn.Dropout(dropout)
    def forward(self, q, k, v, mask=None, dropout=0.):  # [B,N,C]
        b, n, c = q.shape
        q, k, v = \
            [l(x).view(b, -1, self.num_heads, self.dk).transpose(1, 2)
             for l, x in zip(self.linears, (q, k, v))]
        output, self.attn = self.attention(q, k, v, mask, dropout)
        output = output.transpose(1, 2).reshape(b, n, c)#[B,H,N_q,C]->[B,N_q,H,C]->[B,N_q,H,C]
        output = self.project(output)
        return output, self.attn



    def attention(self,query, key, value, mask=None, dropout=None):
        "Compute 'Scaled Dot Product Attention'"
        dk = query.size(-1)
        scores = torch.matmul(query, key.transpose(-2, -1)) \
                 / dk ** 0.5# [batchsize,head_dim,len_q,len_k]，要求q_dim=k_dim
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -np.inf)
        attn = F.softmax(scores, dim=-1)
        if dropout is not None:
            attn = self.dropout(attn)
        return torch.matmul(attn, value), attn #  [batchsize,head_dim,len_q,v_dim],要求len_k=len_v
