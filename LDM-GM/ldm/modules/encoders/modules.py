import torch
import torch.nn as nn
from functools import partial
import clip
from einops import rearrange, repeat
import kornia

from ldm.modules.x_transformer import Encoder, TransformerWrapper  # TODO: can we directly rely on lucidrains code and simply add this as a reuirement? --> test


class E2P_projection(nn.Module):
    def __init__(self, in_dim=4, embed_dim=48, res_dim=512, resolution=10):
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels=in_dim, out_channels=embed_dim, kernel_size=4, stride=2, padding=2) if resolution == 10 else nn.Conv3d(in_channels=in_dim, out_channels=embed_dim, kernel_size=4, stride=1, padding=2)
        self.conv2 = nn.Conv3d(in_channels=embed_dim, out_channels=embed_dim * 4, kernel_size=4, stride=2, padding=2)
        self.conv3 = nn.Conv3d(in_channels=embed_dim * 4, out_channels=embed_dim * 24, kernel_size=4, stride=1, padding=0)

        self.LReLu = nn.LeakyReLU()
        self.fc1 = nn.Linear(embed_dim * 24, res_dim * 2)
        self.fc2 = nn.Linear(res_dim * 2, res_dim)
        self.dropout = nn.Dropout(0.3)

    def forward(self, z):

        b = z.shape[0]
        emb = self.conv1(z)
        # print(z.shape, emb.shape)
        emb = self.conv2(emb)
        emb = self.dropout(emb)
        # print(emb.shape)
        emb = self.conv3(emb)
        # print(emb.shape)
        emb = torch.reshape(emb, [b, -1])
        emb = self.fc1(emb)
        emb = self.dropout(emb)
        emb = self.LReLu(emb)
        emb = self.fc2(emb)
        # emb = emb / emb.norm(dim=1, keepdim=True)

        # print(emb.shape)

        return emb


class AbstractEncoder(nn.Module):
    def __init__(self):
        super().__init__()

    def encode(self, *args, **kwargs):
        raise NotImplementedError


class ClassEmbedder(nn.Module):
    def __init__(self, embed_dim, n_classes=1000, key='class'):
        super().__init__()
        self.key = key
        self.embedding = nn.Embedding(n_classes, embed_dim)

    def forward(self, batch, key=None):
        if key is None:
            key = self.key
        # this is for use in crossattn
        c = batch[key][:, None]
        c = self.embedding(c)
        # print(c.shape)
        return c


class CEmbedder(nn.Module):
    def __init__(self, embed_dim, in_dim=36, key='C'):
        super().__init__()
        self.key = key
        self.fc1 = nn.Linear(in_dim, embed_dim // 2)
        self.fc2 = nn.Linear(embed_dim // 2, embed_dim)
        self.LReLu = nn.LeakyReLU()

    def forward(self, x):
        # this is for use in crossattn
        # print(key, batch.keys())
        # print(x.shape)
        # print(self.fc2.weight)
        # print(self.fc2.weight.grad)
        # print(x.shape)
        b = x.shape[0]
        emb = torch.reshape(x, [b, 1, -1])
        emb = self.fc1(emb)
        emb = self.LReLu(emb)
        emb = self.fc2(emb)
        # emb = torch.cat([emb, emb], dim=1)
        # print(emb.shape)


        return emb


import math


class PositionalEncoding(nn.Module):

    def __init__(self, dim, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()

        if dim % 2 != 0:
            raise ValueError("Cannot use sin/cos positional encoding with "
                             "odd dim (got dim={:d})".format(dim))

        """
        构建位置编码pe
        pe公式为：
        PE(pos,2i/2i+1) = sin/cos(pos/10000^{2i/d_{model}})
        """
        pe = torch.zeros(max_len, dim)  # max_len 是解码器生成句子的最长的长度，假设是 10
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2, dtype=torch.float) * -(math.log(10000.0) / dim))
        print(position.float() * div_term)

        pe[:, 0::2] = torch.sin(position.float() * div_term)
        pe[:, 1::2] = torch.cos(position.float() * div_term)
        # print(torch.sin(position.float() * div_term))
        print(pe[:, 0::2])
        print(pe[:, 1::2])
        pe = pe.unsqueeze(1)
        # print(pe.shape)
        # print(pe)
        self.register_buffer('pe', pe)
        self.drop_out = nn.Dropout(p=dropout)
        self.dim = dim

    def forward(self, emb, step=None):

        emb = emb * math.sqrt(self.dim)

        if step is None:
            emb = emb + self.pe[:emb.size(0)]
        else:
            emb = emb + self.pe[step]
        emb = self.drop_out(emb)
        return emb


def Pembedding(x, dim=512, range_p=[-4.0, 4.0], sigma=1.0):
    # print("---------------", sigma, range_p)
    pos = torch.range(start=range_p[0], end=range_p[1], step=(range_p[1] - range_p[0]) / (dim - 1), dtype=x.dtype).to(x.device)
    pos = torch.reshape(pos, [1,]*len(x.shape) + [dim,])
    pos = pos.repeat(list(x.shape) + [1, ])
    emb = (pos - torch.unsqueeze(x, dim=-1)) ** 2.0
    emb = -0.5 * emb / (sigma ** 2)
    emb = torch.exp(emb)
    emb = emb / torch.sum(emb, dim=-1, keepdim=True)
    return emb


def sim_C(C_mat, bs):
    sample_C = torch.reshape(C_mat, [bs, 6, 6])
    C11 = torch.unsqueeze(sample_C[:, 0, 0] + sample_C[:, 1, 1] + sample_C[:, 2, 2], dim=-1) / 3.0
    C12 = torch.unsqueeze(sample_C[:, 0, 1] + sample_C[:, 1, 2] + sample_C[:, 0, 2] + sample_C[:, 1, 0] + sample_C[:, 2, 1] + sample_C[:, 2, 0], dim=-1) / 6.0
    C44 = torch.unsqueeze(sample_C[:, 3, 3] + sample_C[:, 4, 4] + sample_C[:, 5, 5], dim=-1) / 3.0
    return C11, C12, C44


class CEmbedder_vf(nn.Module):
    def __init__(self, embed_dim, res_dim, vf=False, key='C', sigma_c=1.0, sigma_vf=1.0, range_c=[-4.0, 4.0], range_vf=[0.0, 0.4], cond_zero=False):
        super().__init__()
        self.cond_zero = cond_zero
        self.key = key
        self.fc1_c11 = nn.Linear(embed_dim, res_dim)
        self.fc2_c11 = nn.Linear(res_dim, res_dim * 2)
        self.fc3_c11 = nn.Linear(res_dim * 2, res_dim)

        self.fc1_c12 = nn.Linear(embed_dim, res_dim)
        self.fc2_c12 = nn.Linear(res_dim, res_dim * 2)
        self.fc3_c12 = nn.Linear(res_dim * 2, res_dim)

        self.fc1_c44 = nn.Linear(embed_dim, res_dim)
        self.fc2_c44 = nn.Linear(res_dim, res_dim * 2)
        self.fc3_c44 = nn.Linear(res_dim * 2, res_dim)

        self.LReLu = nn.LeakyReLU()

        self.fc3_out = nn.Linear(res_dim * 3, res_dim * 2)
        self.fc4_out = nn.Linear(res_dim * 2, res_dim)
        self.sigma_c = sigma_c
        self.range_c = range_c
        self.vf = vf

        if self.vf:
            self.fc1_vf = nn.Linear(embed_dim, res_dim)
            self.fc2_vf = nn.Linear(res_dim, res_dim * 2)
            self.fc3_vf = nn.Linear(res_dim * 2, res_dim)

            self.fc3_out = nn.Linear(res_dim * 4, res_dim * 2)

            self.sigma_vf = sigma_vf
            self.range_vf = range_vf

    def forward(self, x, return_sim=False):
        # this is for use in crossattn
        # print(key, batch.keys())
        # print(x.shape)
        # print(self.fc2.weight)
        # print(self.fc2.weight.grad)
        # print(x.shape)
        # x = x[..., :-1]

        b = x.shape[0]
        if self.vf:
            assert x.shape[-1] == 37
            vf = x[..., -1]
            vf_s = torch.reshape(vf, [b, 1])
            vf = Pembedding(vf_s[..., 0], range_p=self.range_vf, sigma=self.sigma_vf)
            vf = torch.tensor(vf, dtype=self.fc1_vf.weight.dtype)
            # print(vf.dtype, self.fc1_vf.weight.dtype)

            vf = self.fc1_vf(vf)
            vf = self.fc2_vf(vf)
            vf = self.fc3_vf(vf)

            x = x[..., :36]
        C11, C12, C44 = sim_C(x, b)
        # sample_C = torch.reshape(x, [b, 6, 6])
        # C11 = torch.unsqueeze(sample_C[:, 0, 0] + sample_C[:, 1, 1] + sample_C[:, 2, 2], dim=-1) / 3.0
        # C12 = torch.unsqueeze(sample_C[:, 0, 1] + sample_C[:, 1, 2] + sample_C[:, 0, 2] + sample_C[:, 1, 0] + sample_C[:, 2, 1] + sample_C[:, 2, 0], dim=-1) / 6.0
        # C44 = torch.unsqueeze(sample_C[:, 3, 3] + sample_C[:, 4, 4] + sample_C[:, 5, 5], dim=-1) / 3.0
        C_sim = torch.cat([C11, C12, C44, vf_s], dim=-1) if self.vf else torch.cat([C11, C12, C44], dim=-1)
        if self.cond_zero:
            emb = torch.ones([b, 512], dtype=x.dtype).to(x.device) * 0.0000001
            # print("cond zero True", emb.shape)
            if return_sim:
                return emb, C_sim
            else:
                return torch.unsqueeze(emb, dim=1)
        # print("[INFO]")
        # print(C11, C12, C44)
        # print(C11.shape, vf.shape)
        # print(C_sim)
        # print(C_sim.shape)
        C_embedding = Pembedding(torch.cat([C11, C12, C44], dim=-1), range_p=self.range_c, sigma=self.sigma_c)
        C_embedding = torch.tensor(C_embedding, dtype=self.fc1_c11.weight.dtype)

        # print(torch.max(C_embedding[:, 0, :]).detach().cpu().numpy(), torch.min(C_embedding[:, 0, :]).detach().cpu().numpy())
        # print(torch.max(C_embedding[:, 1, :]).detach().cpu().numpy(),
        #       torch.min(C_embedding[:, 1, :]).detach().cpu().numpy())
        # print(torch.max(C_embedding[:, 2, :]).detach().cpu().numpy(),
        #       torch.min(C_embedding[:, 2, :]).detach().cpu().numpy())

        # print(torch.sum(C_embedding, dim=-1))
        # print()

        C11 = self.fc1_c11(C_embedding[:, 0, :])
        C11 = self.fc2_c11(C11)
        C11 = self.fc3_c11(C11)

        C12 = self.fc1_c12(C_embedding[:, 1, :])
        C12 = self.fc2_c12(C12)
        C12 = self.fc3_c12(C12)

        C44 = self.fc1_c44(C_embedding[:, 2, :])
        C44 = self.fc2_c44(C44)
        C44 = self.fc3_c44(C44)

        emb = torch.cat([C11, C12, C44], dim=-1) if not self.vf else torch.cat([C11, C12, C44, vf], dim=-1)

        emb = self.fc3_out(emb)
        emb = self.LReLu(emb)
        emb = self.fc4_out(emb)
        # emb = self.LReLu(emb)
        # print("[INFO] END!")

        # print("cond zero False", emb.shape)
        if return_sim:
            return emb, C_sim
        # print(emb.shape)
        return torch.unsqueeze(emb, dim=1)


class CEmbedder_vf_tiny(nn.Module):
    def __init__(self, embed_dim, res_dim, mid_dim, vf=False, key='C', sigma_c=1.0, sigma_vf=1.0, range_c=[-4.0, 4.0], range_vf=[0.0, 0.4], cond_zero=False):
        super().__init__()
        self.cond_zero = cond_zero
        self.key = key
        self.embed_dim = embed_dim
        self.fc1_c11 = nn.Linear(embed_dim, mid_dim)
        self.fc2_c11 = nn.Linear(mid_dim, mid_dim)
        self.fc3_c11 = nn.Linear(mid_dim, mid_dim)

        self.fc1_c12 = nn.Linear(embed_dim, mid_dim)
        self.fc2_c12 = nn.Linear(mid_dim, mid_dim)
        self.fc3_c12 = nn.Linear(mid_dim, mid_dim)

        self.fc1_c44 = nn.Linear(embed_dim, mid_dim)
        self.fc2_c44 = nn.Linear(mid_dim, mid_dim)
        self.fc3_c44 = nn.Linear(mid_dim, mid_dim)

        self.LReLu = nn.LeakyReLU()

        self.fc3_out = nn.Linear(mid_dim * 3, mid_dim * 2)
        self.fc4_out = nn.Linear(mid_dim * 2, res_dim)
        self.sigma_c = sigma_c
        self.range_c = range_c
        self.vf = vf

        if self.vf:
            self.fc1_vf = nn.Linear(embed_dim, mid_dim)
            self.fc2_vf = nn.Linear(mid_dim, mid_dim)
            self.fc3_vf = nn.Linear(mid_dim, mid_dim)

            self.fc3_out = nn.Linear(mid_dim * 4, mid_dim * 2)

            self.sigma_vf = sigma_vf
            self.range_vf = range_vf

    def forward(self, x, return_sim=False):
        # this is for use in crossattn
        # print(key, batch.keys())
        # print(x.shape)
        # print(self.fc2.weight)
        # print(self.fc2.weight.grad)
        # print(x.shape)
        # x = x[..., :-1]

        b = x.shape[0]
        if self.vf:
            assert x.shape[-1] == 37
            vf = x[..., -1]
            vf_s = torch.reshape(vf, [b, 1])
            vf = Pembedding(vf_s[..., 0], dim=self.embed_dim, range_p=self.range_vf, sigma=self.sigma_vf)
            vf = torch.tensor(vf, dtype=self.fc1_vf.weight.dtype)
            # print(vf.dtype, self.fc1_vf.weight.dtype)

            vf = self.fc1_vf(vf)
            vf = self.fc2_vf(vf)
            vf = self.fc3_vf(vf)

            x = x[..., :36]
        C11, C12, C44 = sim_C(x, b)
        # sample_C = torch.reshape(x, [b, 6, 6])
        # C11 = torch.unsqueeze(sample_C[:, 0, 0] + sample_C[:, 1, 1] + sample_C[:, 2, 2], dim=-1) / 3.0
        # C12 = torch.unsqueeze(sample_C[:, 0, 1] + sample_C[:, 1, 2] + sample_C[:, 0, 2] + sample_C[:, 1, 0] + sample_C[:, 2, 1] + sample_C[:, 2, 0], dim=-1) / 6.0
        # C44 = torch.unsqueeze(sample_C[:, 3, 3] + sample_C[:, 4, 4] + sample_C[:, 5, 5], dim=-1) / 3.0
        C_sim = torch.cat([C11, C12, C44, vf_s], dim=-1) if self.vf else torch.cat([C11, C12, C44], dim=-1)
        if self.cond_zero:
            emb = torch.ones([b, 512], dtype=x.dtype).to(x.device) * 0.0000001
            # print("cond zero True", emb.shape)
            if return_sim:
                return emb, C_sim
            else:
                return torch.unsqueeze(emb, dim=1)
        # print("[INFO]")
        # print(C11, C12, C44)
        # print(C11.shape, vf.shape)
        # print(C_sim)
        # print(C_sim.shape)
        C_embedding = Pembedding(torch.cat([C11, C12, C44], dim=-1), dim=self.embed_dim, range_p=self.range_c, sigma=self.sigma_c)
        C_embedding = torch.tensor(C_embedding, dtype=self.fc1_c11.weight.dtype)

        # print(torch.max(C_embedding[:, 0, :]).detach().cpu().numpy(), torch.min(C_embedding[:, 0, :]).detach().cpu().numpy())
        # print(torch.max(C_embedding[:, 1, :]).detach().cpu().numpy(),
        #       torch.min(C_embedding[:, 1, :]).detach().cpu().numpy())
        # print(torch.max(C_embedding[:, 2, :]).detach().cpu().numpy(),
        #       torch.min(C_embedding[:, 2, :]).detach().cpu().numpy())

        # print(torch.sum(C_embedding, dim=-1))
        # print()

        C11 = self.fc1_c11(C_embedding[:, 0, :])
        C11 = self.fc2_c11(C11)
        C11 = self.fc3_c11(C11)

        C12 = self.fc1_c12(C_embedding[:, 1, :])
        C12 = self.fc2_c12(C12)
        C12 = self.fc3_c12(C12)

        C44 = self.fc1_c44(C_embedding[:, 2, :])
        C44 = self.fc2_c44(C44)
        C44 = self.fc3_c44(C44)

        emb = torch.cat([C11, C12, C44], dim=-1) if not self.vf else torch.cat([C11, C12, C44, vf], dim=-1)

        emb = self.fc3_out(emb)
        emb = self.LReLu(emb)
        emb = self.fc4_out(emb)
        # emb = self.LReLu(emb)
        # print("[INFO] END!")

        # print("cond zero False", emb.shape)
        if return_sim:
            return emb, C_sim
        # print(emb.shape)
        return torch.unsqueeze(emb, dim=1)


class Decoder_C_vf(nn.Module):
    def __init__(self, in_dim=512,  vf=False):
        super().__init__()
        self.fc1_c11 = nn.Linear(in_dim, in_dim * 2)
        self.fc2_c11 = nn.Linear(in_dim * 2, in_dim)
        self.fc3_c11 = nn.Linear(in_dim, 1)

        self.fc1_c12 = nn.Linear(in_dim, in_dim * 2)
        self.fc2_c12 = nn.Linear(in_dim * 2, in_dim)
        self.fc3_c12 = nn.Linear(in_dim, 1)

        self.fc1_c44 = nn.Linear(in_dim, in_dim * 2)
        self.fc2_c44 = nn.Linear(in_dim * 2, in_dim)
        self.fc3_c44 = nn.Linear(in_dim, 1)

        self.LReLu = nn.LeakyReLU()

        self.vf = vf

        if self.vf:
            self.fc1_vf = nn.Linear(in_dim, in_dim * 2)
            self.fc2_vf = nn.Linear(in_dim * 2, in_dim)
            self.fc3_vf = nn.Linear(in_dim, 1)

    def forward(self, x):

        # b = x.shape[0]
        # print()
        # print("before:", torch.max(x).detach().cpu().numpy(), torch.min(x).detach().cpu().numpy())
        c11 = self.fc1_c11(x)
        c11 = self.LReLu(c11)
        c11 = self.fc2_c11(c11)
        # print("after", torch.max(c11).detach().cpu().numpy(), torch.min(c11).detach().cpu().numpy())
        c11 = self.LReLu(c11)
        c11 = self.fc3_c11(c11)
        # print("after 1", torch.max(c11).detach().cpu().numpy(), torch.min(c11).detach().cpu().numpy())
        # print()

        c12 = self.fc1_c12(x)
        c12 = self.LReLu(c12)
        c12 = self.fc2_c12(c12)
        c12 = self.LReLu(c12)
        c12 = self.fc3_c12(c12)

        c44 = self.fc1_c44(x)
        c44 = self.LReLu(c44)
        c44 = self.fc2_c44(c44)
        c44 = self.LReLu(c44)
        c44 = self.fc3_c44(c44)

        if self.vf:
            vf = self.fc1_vf(x)
            vf = self.LReLu(vf)
            vf = self.fc2_vf(vf)
            vf = self.LReLu(vf)
            vf = self.fc3_vf(vf)
            return torch.cat([c11, c12, c44, vf], dim=-1)

        return torch.cat([c11, c12, c44], dim=-1)



class CEmbedder_ml(nn.Module):
    def __init__(self, embed_dim, in_dim=12, key='C', log=False):
        super().__init__()
        self.key = key
        self.fc1 = nn.Linear(in_dim, embed_dim // 2)
        self.fc2 = nn.Linear(embed_dim // 2, embed_dim)
        self.fc3 = nn.Linear(embed_dim, embed_dim * 2)
        self.fc4 = nn.Linear(embed_dim * 2, embed_dim * 4)
        self.fc5 = nn.Linear(embed_dim * 4, embed_dim)
        self.LReLu = nn.LeakyReLU(negative_slope=0.1)
        self.sigmod = nn.Sigmoid()
        self.log = log
        self.mask = torch.tensor([[1.0, 1.0, 1.0, 0.0, 0.0, 0.0],
                                  [1.0, 1.0, 1.0, 0.0, 0.0, 0.0],
                                  [1.0, 1.0, 1.0, 0.0, 0.0, 0.0],
                                  [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
                                  [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                                  [0.0, 0.0, 0.0, 0.0, 0.0, 1.0]])
        self.ind = (self.mask > 0.5)
        if self.log:
            c_11_b = 5.0
            c_12_b = 20.0
            c_44_b = 20.0
            bias = torch.tensor([[c_11_b, c_12_b, c_12_b, 0.0, 0.0, 0.0],
                                 [c_12_b, c_11_b, c_12_b, 0.0, 0.0, 0.0],
                                 [c_12_b, c_12_b, c_11_b, 0.0, 0.0, 0.0],
                                 [0.0, 0.0, 0.0, c_44_b, 0.0, 0.0],
                                 [0.0, 0.0, 0.0, 0.0, c_44_b, 0.0],
                                 [0.0, 0.0, 0.0, 0.0, 0.0, c_44_b]])
            self.bias = bias[self.ind]
            c_11_w = 1.25
            c_12_w = 1.6
            c_44_w = 1.6
            weight = torch.tensor([[c_11_w, c_12_w, c_12_w, 0.0, 0.0, 0.0],
                                   [c_12_w, c_11_w, c_12_w, 0.0, 0.0, 0.0],
                                   [c_12_w, c_12_w, c_11_w, 0.0, 0.0, 0.0],
                                   [0.0, 0.0, 0.0, c_44_w, 0.0, 0.0],
                                   [0.0, 0.0, 0.0, 0.0, c_44_w, 0.0],
                                   [0.0, 0.0, 0.0, 0.0, 0.0, c_44_w]])
            self.weights = weight[self.ind]
            self.LN = nn.LayerNorm(embed_dim)

    def forward(self, x):
        # this is for use in crossattn
        # print(key, batch.keys())
        # print(x.shape)
        # print(self.fc2.weight)
        # print(self.fc2.weight.grad)
        x = x[:, :, self.ind]
        b = x.shape[0]
        if self.log:
            x_log = 0.4 * torch.log(x)
            # print(torch.max(x_log), torch.min(x_log))
            x = x*self.weights.to(x.device) + x_log + self.bias.to(x.device)
        # print(x)
        emb = torch.reshape(x, [b, 1, -1])
        emb = self.fc1(emb)
        emb = self.LReLu(emb)
        emb = self.fc2(emb)
        emb = self.LReLu(emb)
        emb = self.fc3(emb)
        emb = self.LReLu(emb)
        emb = self.fc4(emb)
        emb = self.sigmod(emb)
        emb = self.fc5(emb)
        emb = self.LN(emb) if self.log else emb

        # print(emb.shape)


        return emb


class CEmbedder_L(nn.Module):
    def __init__(self, embed_dim, in_dim=36, key='C'):
        super().__init__()
        self.key = key
        self.fc1_l = nn.ModuleList([nn.Linear(1, embed_dim // 2) for _ in range(in_dim)])
        self.fc2_l = nn.ModuleList([nn.Linear(embed_dim // 2, embed_dim) for _ in range(in_dim)])
        self.LReLu = nn.LeakyReLU()

    def forward(self, x):
        # this is for use in crossattn
        # print(key, batch.keys())
        # print(x.shape)
        # print(self.fc2.weight)
        # print(self.fc2.weight.grad)
        b = x.shape[0]
        emb = torch.reshape(x, [b, 1, -1])
        res = []
        for i in range(len(self.fc1_l)):
            tmp = self.fc1_l[i](emb[..., i:i+1])
            tmp = self.LReLu(tmp)
            tmp = self.fc2_l[i](tmp)
            res.append(tmp)

        res = torch.cat(res, dim=-2)
        # print(res.shape)

        return res


class AttentionLayer(nn.Module):
    def __init__(self, input_dim):
        super(AttentionLayer, self).__init__()
        self.q_linear = nn.Linear(input_dim, input_dim // 8)
        self.k_linear = nn.Linear(input_dim, input_dim // 8)
        self.v_linear = nn.Linear(input_dim, input_dim // 8)
        self.out_linear = nn.Linear(input_dim // 8, input_dim)

    def forward(self, x):
        q = self.q_linear(x)
        k = self.k_linear(x)
        v = self.v_linear(x)
        # print("b:", q.shape, k.shape, v.shape)
        attn_weights = torch.softmax(torch.matmul(q.transpose(-2, -1), k) / (v.shape[-1] ** 0.5), dim=-1)
        # print(attn_weights.shape)
        weighted_v = torch.matmul(attn_weights, v.transpose(-2, -1)).transpose(-2, -1)
        # print(weighted_v.shape)
        # print(self.out_linear(weighted_v).shape)
        return self.out_linear(weighted_v)


class CEmbedder_attn(nn.Module):
    def __init__(self, embed_dim, in_dim=36, key='C'):
        super().__init__()
        self.key = key
        self.fc1 = nn.Linear(in_dim, embed_dim // 2)
        self.fc2 = nn.Linear(embed_dim // 2, embed_dim)
        self.attn = AttentionLayer(embed_dim // 2)
        self.LReLu = nn.LeakyReLU()

    def forward(self, x):
        b = x.shape[0]
        emb = self.fc1(torch.reshape(x, [b, 1, -1]))
        emb = self.LReLu(emb)
        emb = self.attn(emb)
        emb = self.fc2(emb)
        # print(c.shape)
        return emb


class TransformerEmbedder(AbstractEncoder):
    """Some transformer encoder layers"""
    def __init__(self, n_embed, n_layer, vocab_size, max_seq_len=77, device="cuda"):
        super().__init__()
        self.device = device
        self.transformer = TransformerWrapper(num_tokens=vocab_size, max_seq_len=max_seq_len,
                                              attn_layers=Encoder(dim=n_embed, depth=n_layer))

    def forward(self, tokens):
        tokens = tokens.to(self.device)  # meh
        z = self.transformer(tokens, return_embeddings=True)
        return z

    def encode(self, x):
        return self(x)


class BERTTokenizer(AbstractEncoder):
    """ Uses a pretrained BERT tokenizer by huggingface. Vocab size: 30522 (?)"""
    def __init__(self, device="cuda", vq_interface=True, max_length=77):
        super().__init__()
        from transformers import BertTokenizerFast  # TODO: add to reuquirements
        self.tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
        self.device = device
        self.vq_interface = vq_interface
        self.max_length = max_length

    def forward(self, text):
        batch_encoding = self.tokenizer(text, truncation=True, max_length=self.max_length, return_length=True,
                                        return_overflowing_tokens=False, padding="max_length", return_tensors="pt")
        tokens = batch_encoding["input_ids"].to(self.device)
        return tokens

    @torch.no_grad()
    def encode(self, text):
        tokens = self(text)
        if not self.vq_interface:
            return tokens
        return None, None, [None, None, tokens]

    def decode(self, text):
        return text


class BERTEmbedder(AbstractEncoder):
    """Uses the BERT tokenizr model and add some transformer encoder layers"""
    def __init__(self, n_embed, n_layer, vocab_size=30522, max_seq_len=77,
                 device="cuda",use_tokenizer=True, embedding_dropout=0.0):
        super().__init__()
        self.use_tknz_fn = use_tokenizer
        if self.use_tknz_fn:
            self.tknz_fn = BERTTokenizer(vq_interface=False, max_length=max_seq_len)
        self.device = device
        self.transformer = TransformerWrapper(num_tokens=vocab_size, max_seq_len=max_seq_len,
                                              attn_layers=Encoder(dim=n_embed, depth=n_layer),
                                              emb_dropout=embedding_dropout)

    def forward(self, text):
        if self.use_tknz_fn:
            tokens = self.tknz_fn(text)#.to(self.device)
        else:
            tokens = text
        z = self.transformer(tokens, return_embeddings=True)
        return z

    def encode(self, text):
        # output of length 77
        return self(text)


class SpatialRescaler(nn.Module):
    def __init__(self,
                 n_stages=1,
                 method='bilinear',
                 multiplier=0.5,
                 in_channels=3,
                 out_channels=None,
                 bias=False):
        super().__init__()
        self.n_stages = n_stages
        assert self.n_stages >= 0
        assert method in ['nearest','linear','bilinear','trilinear','bicubic','area']
        self.multiplier = multiplier
        self.interpolator = partial(torch.nn.functional.interpolate, mode=method)
        self.remap_output = out_channels is not None
        if self.remap_output:
            print(f'Spatial Rescaler mapping from {in_channels} to {out_channels} channels after resizing.')
            self.channel_mapper = nn.Conv2d(in_channels,out_channels,1,bias=bias)

    def forward(self,x):
        for stage in range(self.n_stages):
            x = self.interpolator(x, scale_factor=self.multiplier)


        if self.remap_output:
            x = self.channel_mapper(x)
        return x

    def encode(self, x):
        return self(x)


class FrozenCLIPTextEmbedder(nn.Module):
    """
    Uses the CLIP transformer encoder for text.
    """
    def __init__(self, version='ViT-L/14', device="cuda", max_length=77, n_repeat=1, normalize=True):
        super().__init__()
        self.model, _ = clip.load(version, jit=False, device="cpu")
        self.device = device
        self.max_length = max_length
        self.n_repeat = n_repeat
        self.normalize = normalize

    def freeze(self):
        self.model = self.model.eval()
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, text):
        tokens = clip.tokenize(text).to(self.device)
        z = self.model.encode_text(tokens)
        if self.normalize:
            z = z / torch.linalg.norm(z, dim=1, keepdim=True)
        return z

    def encode(self, text):
        z = self(text)
        if z.ndim==2:
            z = z[:, None, :]
        z = repeat(z, 'b 1 d -> b k d', k=self.n_repeat)
        return z


class FrozenClipImageEmbedder(nn.Module):
    """
        Uses the CLIP image encoder.
        """
    def __init__(
            self,
            model,
            jit=False,
            device='cuda' if torch.cuda.is_available() else 'cpu',
            antialias=False,
        ):
        super().__init__()
        self.model, _ = clip.load(name=model, device=device, jit=jit)

        self.antialias = antialias

        self.register_buffer('mean', torch.Tensor([0.48145466, 0.4578275, 0.40821073]), persistent=False)
        self.register_buffer('std', torch.Tensor([0.26862954, 0.26130258, 0.27577711]), persistent=False)

    def preprocess(self, x):
        # normalize to [0,1]
        x = kornia.geometry.resize(x, (224, 224),
                                   interpolation='bicubic',align_corners=True,
                                   antialias=self.antialias)
        x = (x + 1.) / 2.
        # renormalize according to clip
        x = kornia.enhance.normalize(x, self.mean, self.std)
        return x

    def forward(self, x):
        # x is assumed to be in range [-1,1]
        return self.model.encode_image(self.preprocess(x))

