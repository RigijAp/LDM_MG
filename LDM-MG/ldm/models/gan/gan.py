import copy

import numpy as np
import torch
import torch.nn as nn
from ldm.modules.encoders.modules import CEmbedder_vf


class Generator(nn.Module):
    def __init__(self, opt):
        super(Generator, self).__init__()
        self.opt = opt

        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(opt.latent_dim + opt.n_classes, 128, normalize=False),
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            nn.Linear(1024, int(np.prod(opt.img_shape))),
            nn.Tanh()
        )

    def forward(self, z, c_embedding):
        # Concatenate label embedding and image to produce input
        gen_input = torch.cat((c_embedding, z), -1)
        img = self.model(gen_input)
        img = img.view(img.shape[0], * self.opt.img_shape)
        return img


class Discriminator(nn.Module):
    def __init__(self, opt):
        super(Discriminator, self).__init__()

        # Copied from cgan.py
        self.conv_model = nn.Sequential(
            nn.Conv3d(4, 128, 3, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(128, 256, 3, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(256, 512, 3, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(512, 1024, 3, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(1024, 2048, 2, 1),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.model = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 512),
            # nn.Dropout(0.4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 1),
        )

        # self.model = nn.Sequential(
        #     nn.Linear(int(np.prod(opt.img_shape)), 512),
        #     nn.LeakyReLU(0.2, inplace=True),
        #     nn.Linear(512, 512),
        #     # nn.Dropout(0.4),
        #     nn.LeakyReLU(0.2, inplace=True),
        #     nn.Linear(512, 1),
        # )

    def forward(self, img):
        # d_in = img.view(img.size(0), -1)
        # validity = self.model(d_in)
        for layer in self.conv_model:
            img = layer(img)
            # print(img.shape)
            # print(torch.max(img), torch.min(img))
        # img = self.conv_model(img)
        # print(img.shape)
        # print(torch.max(img), torch.min(img))
        b = img.shape[0]
        img = torch.reshape(img, [b, -1])
        validity = self.model(img)
        return validity


class Discriminator_p(nn.Module):
    def __init__(self, VAE):
        super(Discriminator_p, self).__init__()

        # Copied from cgan.py
        self.projection = copy.deepcopy(VAE.projection)
        self.pred_C = copy.deepcopy(VAE.decoder_p)

    def forward(self, img):
        z_emb = self.projection(img)
        z_emb = z_emb / z_emb.norm(dim=-1, keepdim=True)
        validity = self.pred_C(z_emb)
        return validity


class Discriminator_cls(nn.Module):
    def __init__(self, opt):
        super(Discriminator_cls, self).__init__()

        self.label_embedding = nn.Embedding(opt.n_classes, opt.n_classes)

        # Copied from cgan.py
        self.model = nn.Sequential(
            nn.Linear(opt.n_classes + int(np.prod(opt.img_shape)), 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 512),
            nn.Dropout(0.4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 1),
        )

    def forward(self, img, labels):
        # Concatenate label embedding and image to produce input
        d_in = torch.cat((img.view(img.size(0), -1), self.label_embedding(labels)), -1)
        validity = self.model(d_in)
        return validity