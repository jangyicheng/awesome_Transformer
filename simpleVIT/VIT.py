from Attention import MultiHeadAttention
from einops.layers.torch import Rearrange
from torch import nn
import torch
import copy
import torch.nn.functional as F
from einops import repeat


def clones(module, n):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(n)])


def pair(t):
    return t if isinstance(t, tuple) else (t, t)


class DropPath(nn.Module):
    def __init__(self, dropout):
        super().__init__()
        self.dropout = dropout

    def forward(self, x):
        n_dim = len(x.shape)
        keep_prob = 1.0 - self.dropout
        shape = (x.shape[0],) + (1,) * (n_dim - 1)
        mask = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        mask.floor_()
        output = x.div(keep_prob) * mask
        return output


class MLPBlock(nn.Module):
    def __init__(self, embed_dim, mlp_ratio=4, dropout=0.1):
        super().__init__()
        self.Linear1 = nn.Linear(embed_dim, int(embed_dim * mlp_ratio))
        self.Linear2 = nn.Linear(int(embed_dim * mlp_ratio), embed_dim)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        x = self.dropout(F.gelu(self.Linear1(x)))
        x = self.Linear2(x)
        return x


class EncoderLayer(nn.Module):
    def __init__(self, model_dim, num_heads, dropout=0.1):
        super().__init__()
        self.Attention = MultiHeadAttention(model_dim, num_heads)
        self.Layernorm = nn.LayerNorm(model_dim)
        self.model_dim = model_dim
        self.droppath = DropPath(dropout)
        self.MLP_block = MLPBlock(model_dim)

    def forward(self, x, mask=None):
        shortcut = x
        x = self.Layernorm(x)
        x, attn = self.Attention(x, x, x, mask)
        x = self.droppath(x)
        x = shortcut + x

        shortcut = x
        x = self.Layernorm(x)
        x = self.MLP_block(x)
        x = self.droppath(x)
        output = shortcut + x
        return output


class Encoder(nn.Module):
    def __init__(self, layer_num, model_dim, num_heads, encoder_layer=EncoderLayer):
        super().__init__()
        self.layer_num = layer_num
        self.layers = clones(encoder_layer(model_dim, num_heads), layer_num)

    def forward(self, x, mask=None):
        for i in range(self.layer_num):
            x = self.layers[i](x, mask)
        return x


class VIT(nn.Module):
    def __init__(
            self,
            imgsize,
            patchsize,
            embed_dim,
            num_classes,
            layer_num=6,
            num_heads=8,
            dropout=0.1,
    ):
        super().__init__()
        assert imgsize % patchsize == 0
        image_height, image_width = pair(imgsize)
        patch_height, patch_width = pair(patchsize)
        assert (
                image_height % patch_height == 0 and image_width % patch_width == 0
        ), "Image dimensions must be divisible by the patch size."

        # self.patch_size = patchsize
        # self.embed_dim = embed_dim
        self.patch_num = int(image_height // patch_height) * int(
            image_width // patch_width
        )
        self.rearrange = Rearrange(
            "b c (h p1) (w p2) -> b (h w) (p1 p2 c)", p1=patch_height, p2=patch_width
        )
        self.patch_embedding = nn.Linear(patch_height * patch_width * 3, embed_dim)
        self.encoder = Encoder(layer_num, embed_dim, num_heads)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        ##事实上正常的 embed_dim=patch_size**2*channel_num，不过也可以指定作为一个恒定的维度，并使用线性层强制压缩到这个维度
        self.pos_embedding = nn.Parameter(torch.zeros(1, self.patch_num + 1, embed_dim))
        self.Conv2d = nn.Conv2d(
            3,
            embed_dim,
            kernel_size=(patch_height, patch_width),
            stride=(patch_height, patch_width),
        )
        self.layernorm = nn.LayerNorm(embed_dim)
        self.mlp_head = nn.Linear(embed_dim, num_classes)

    def forward(self, img):
        assert img.ndim == 4, "输入张量必须是四维的 (batch_size, channels, height, width)"
        # 下面假设img 形状是 [batch_size, channel, height, width]
        img_patch = self.rearrange(img)  #shape [batch_size,patch_num, *patch_size]

        # img_patch = self.Conv2d(img)#shape [batch_size, embed_dim, patch_size, patch_size]
        # img_patch = rearrange(img_patch,'b,c,h,w->b,(h w),c')#shape [batch_size,patch_num, embed_dim]

        patch_embedding = self.patch_embedding(img_patch)#shape [batch_size,patch_num,embed_dim]
        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b = img.shape[0])#shape [batch_size,1,embed_dim]
        patch_embedding = torch.cat(
            [cls_tokens, patch_embedding], dim=1
        )  # shape [batch_size,patch_num+1, embed_dim]

        img_embedding = (
                patch_embedding + self.pos_embedding
        )  # shape [batch_size,patch_num+1, embed_dim]
        encode = self.encoder(img_embedding)# shape [batch_size,patch_num+1, embed_dim]

        encode = self.layernorm(encode)
        cls = encode[:, 0]
        return self.mlp_head(cls)
