import torch
import pytorch_lightning as pl
import torch.nn.functional as F
from contextlib import contextmanager

from taming.modules.vqvae.quantize import VectorQuantizer2 as VectorQuantizer

from ldm.modules.distributions.distributions import DiagonalGaussianDistribution
from ldm.modules.encoders.modules import CEmbedder_vf, Decoder_C_vf, E2P_projection

from ldm.util import instantiate_from_config


def product_of_experts(q_distr_set):
    mu_q_set, sigma_q_set = q_distr_set
    tmp1 = 1.0
    for i in range(len(mu_q_set)):
        tmp1 = tmp1 + (1.0 / (sigma_q_set[i] ** 2))
    poe_var = torch.sqrt(1.0 / tmp1)
    tmp2 = 0.0
    for i in range(len(mu_q_set)):
        tmp2 = tmp2 + torch.div(mu_q_set[i], sigma_q_set[i]**2)
    poe_u = torch.div(tmp2, tmp1)
    return poe_u, poe_var


class VQModel(pl.LightningModule):
    def __init__(self,
                 ddconfig,
                 lossconfig,
                 n_embed,
                 embed_dim,
                 ckpt_path=None,
                 ignore_keys=[],
                 image_key="image",
                 colorize_nlabels=None,
                 monitor=None,
                 batch_resize_range=None,
                 scheduler_config=None,
                 lr_g_factor=1.0,
                 remap=None,
                 sane_index_shape=False, # tell vector quantizer to return indices as bhw
                 use_ema=False
                 ):
        super().__init__()
        self.embed_dim = embed_dim
        self.n_embed = n_embed
        self.image_key = image_key
        try:
            if ddconfig["3d"]:
                from ldm.modules.diffusionmodules.model_3d import Encoder, Decoder
            else:
                from ldm.modules.diffusionmodules.model import Encoder, Decoder
        except:
            from ldm.modules.diffusionmodules.model import Encoder, Decoder
        self.encoder = Encoder(**ddconfig)
        self.decoder = Decoder(**ddconfig)
        self.loss = instantiate_from_config(lossconfig)
        self.quantize = VectorQuantizer(n_embed, embed_dim, beta=0.25,
                                        remap=remap,
                                        sane_index_shape=sane_index_shape)
        self.quant_conv = torch.nn.Conv2d(ddconfig["z_channels"], embed_dim, 1)
        self.post_quant_conv = torch.nn.Conv2d(embed_dim, ddconfig["z_channels"], 1)
        if colorize_nlabels is not None:
            assert type(colorize_nlabels)==int
            self.register_buffer("colorize", torch.randn(3, colorize_nlabels, 1, 1))
        if monitor is not None:
            self.monitor = monitor
        self.batch_resize_range = batch_resize_range
        if self.batch_resize_range is not None:
            print(f"{self.__class__.__name__}: Using per-batch resizing in range {batch_resize_range}.")

        self.use_ema = use_ema
        if self.use_ema:
            self.model_ema = LitEma(self)
            print(f"Keeping EMAs of {len(list(self.model_ema.buffers()))}.")

        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)
        self.scheduler_config = scheduler_config
        self.lr_g_factor = lr_g_factor

    @contextmanager
    def ema_scope(self, context=None):
        if self.use_ema:
            self.model_ema.store(self.parameters())
            self.model_ema.copy_to(self)
            if context is not None:
                print(f"{context}: Switched to EMA weights")
        try:
            yield None
        finally:
            if self.use_ema:
                self.model_ema.restore(self.parameters())
                if context is not None:
                    print(f"{context}: Restored training weights")

    def init_from_ckpt(self, path, ignore_keys=list()):
        sd = torch.load(path, map_location="cpu")["state_dict"]
        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print("Deleting key {} from state_dict.".format(k))
                    del sd[k]
        missing, unexpected = self.load_state_dict(sd, strict=False)
        print(f"Restored from {path} with {len(missing)} missing and {len(unexpected)} unexpected keys")
        if len(missing) > 0:
            print(f"Missing Keys: {missing}")
            print(f"Unexpected Keys: {unexpected}")

    def on_train_batch_end(self, *args, **kwargs):
        if self.use_ema:
            self.model_ema(self)

    def encode(self, x):
        h = self.encoder(x)
        h = self.quant_conv(h)
        quant, emb_loss, info = self.quantize(h)
        return quant, emb_loss, info

    def encode_to_prequant(self, x):
        h = self.encoder(x)
        h = self.quant_conv(h)
        return h

    def decode(self, quant):
        quant = self.post_quant_conv(quant)
        dec = self.decoder(quant)
        return dec

    def decode_code(self, code_b):
        quant_b = self.quantize.embed_code(code_b)
        dec = self.decode(quant_b)
        return dec

    def forward(self, input, return_pred_indices=False):
        quant, diff, (_,_,ind) = self.encode(input)
        dec = self.decode(quant)
        if return_pred_indices:
            return dec, diff, ind
        return dec, diff

    def get_input(self, batch, k):
        x = batch[k]
        if len(x.shape) == 3:
            x = x[..., None]
        x = x.permute(0, 3, 1, 2).to(memory_format=torch.contiguous_format).float()
        if self.batch_resize_range is not None:
            lower_size = self.batch_resize_range[0]
            upper_size = self.batch_resize_range[1]
            if self.global_step <= 4:
                # do the first few batches with max size to avoid later oom
                new_resize = upper_size
            else:
                new_resize = np.random.choice(np.arange(lower_size, upper_size+16, 16))
            if new_resize != x.shape[2]:
                x = F.interpolate(x, size=new_resize, mode="bicubic")
            x = x.detach()
        return x

    def training_step(self, batch, batch_idx, optimizer_idx):
        # https://github.com/pytorch/pytorch/issues/37142
        # try not to fool the heuristics
        x = self.get_input(batch, self.image_key)
        xrec, qloss, ind = self(x, return_pred_indices=True)

        if optimizer_idx == 0:
            # autoencode
            aeloss, log_dict_ae = self.loss(qloss, x, xrec, optimizer_idx, self.global_step,
                                            last_layer=self.get_last_layer(), split="train",
                                            predicted_indices=ind)

            self.log_dict(log_dict_ae, prog_bar=False, logger=True, on_step=True, on_epoch=True)
            return aeloss

        if optimizer_idx == 1:
            # discriminator
            discloss, log_dict_disc = self.loss(qloss, x, xrec, optimizer_idx, self.global_step,
                                            last_layer=self.get_last_layer(), split="train")
            self.log_dict(log_dict_disc, prog_bar=False, logger=True, on_step=True, on_epoch=True)
            return discloss

    def validation_step(self, batch, batch_idx):
        log_dict = self._validation_step(batch, batch_idx)
        with self.ema_scope():
            log_dict_ema = self._validation_step(batch, batch_idx, suffix="_ema")
        return log_dict

    def _validation_step(self, batch, batch_idx, suffix=""):
        x = self.get_input(batch, self.image_key)
        xrec, qloss, ind = self(x, return_pred_indices=True)
        aeloss, log_dict_ae = self.loss(qloss, x, xrec, 0,
                                        self.global_step,
                                        last_layer=self.get_last_layer(),
                                        split="val"+suffix,
                                        predicted_indices=ind
                                        )

        discloss, log_dict_disc = self.loss(qloss, x, xrec, 1,
                                            self.global_step,
                                            last_layer=self.get_last_layer(),
                                            split="val"+suffix,
                                            predicted_indices=ind
                                            )
        rec_loss = log_dict_ae[f"val{suffix}/rec_loss"]
        self.log(f"val{suffix}/rec_loss", rec_loss,
                   prog_bar=True, logger=True, on_step=False, on_epoch=True, sync_dist=True)
        self.log(f"val{suffix}/aeloss", aeloss,
                   prog_bar=True, logger=True, on_step=False, on_epoch=True, sync_dist=True)
        if version.parse(pl.__version__) >= version.parse('1.4.0'):
            del log_dict_ae[f"val{suffix}/rec_loss"]
        self.log_dict(log_dict_ae)
        self.log_dict(log_dict_disc)
        return self.log_dict

    def configure_optimizers(self):
        lr_d = self.learning_rate
        lr_g = self.lr_g_factor*self.learning_rate
        print("lr_d", lr_d)
        print("lr_g", lr_g)
        opt_ae = torch.optim.Adam(list(self.encoder.parameters())+
                                  list(self.decoder.parameters())+
                                  list(self.quantize.parameters())+
                                  list(self.quant_conv.parameters())+
                                  list(self.post_quant_conv.parameters()),
                                  lr=lr_g, betas=(0.5, 0.9))
        opt_disc = torch.optim.Adam(self.loss.discriminator.parameters(),
                                    lr=lr_d, betas=(0.5, 0.9))

        if self.scheduler_config is not None:
            scheduler = instantiate_from_config(self.scheduler_config)

            print("Setting up LambdaLR scheduler...")
            scheduler = [
                {
                    'scheduler': LambdaLR(opt_ae, lr_lambda=scheduler.schedule),
                    'interval': 'step',
                    'frequency': 1
                },
                {
                    'scheduler': LambdaLR(opt_disc, lr_lambda=scheduler.schedule),
                    'interval': 'step',
                    'frequency': 1
                },
            ]
            return [opt_ae, opt_disc], scheduler
        return [opt_ae, opt_disc], []

    def get_last_layer(self):
        return self.decoder.conv_out.weight

    def log_images(self, batch, only_inputs=False, plot_ema=False, **kwargs):
        log = dict()
        x = self.get_input(batch, self.image_key)
        x = x.to(self.device)
        if only_inputs:
            log["inputs"] = x
            return log
        xrec, _ = self(x)
        if x.shape[1] > 3:
            # colorize with random projection
            assert xrec.shape[1] > 3
            x = self.to_rgb(x)
            xrec = self.to_rgb(xrec)
        log["inputs"] = x
        log["reconstructions"] = xrec
        if plot_ema:
            with self.ema_scope():
                xrec_ema, _ = self(x)
                if x.shape[1] > 3: xrec_ema = self.to_rgb(xrec_ema)
                log["reconstructions_ema"] = xrec_ema
        return log

    def to_rgb(self, x):
        assert self.image_key == "segmentation"
        if not hasattr(self, "colorize"):
            self.register_buffer("colorize", torch.randn(3, x.shape[1], 1, 1).to(x))
        x = F.conv2d(x, weight=self.colorize)
        x = 2.*(x-x.min())/(x.max()-x.min()) - 1.
        return x


class VQModelInterface(VQModel):
    def __init__(self, embed_dim, *args, **kwargs):
        super().__init__(embed_dim=embed_dim, *args, **kwargs)
        self.embed_dim = embed_dim

    def encode(self, x):
        h = self.encoder(x)
        h = self.quant_conv(h)
        return h

    def decode(self, h, force_not_quantize=False):
        # also go through quantization layer
        if not force_not_quantize:
            quant, emb_loss, info = self.quantize(h)
        else:
            quant = h
        quant = self.post_quant_conv(quant)
        dec = self.decoder(quant)
        return dec


class AutoencoderKL_ori(pl.LightningModule):
    def __init__(self,
                 ddconfig,
                 lossconfig,
                 embed_dim,
                 ckpt_path=None,
                 ignore_keys=[],
                 image_key="image",
                 colorize_nlabels=None,
                 monitor=None,
                 projection=False,
                 gan_inter=False,
                 ):
        super().__init__()
        self.image_key = image_key

        assert ddconfig["double_z"]
        try:
            if ddconfig["3d"]:
                from ldm.modules.diffusionmodules.model_3d import Encoder, Decoder, z2p
                self.quant_conv = torch.nn.Conv3d(2 * ddconfig["z_channels"], 2 * embed_dim, 1)
                self.post_quant_conv = torch.nn.Conv3d(embed_dim, ddconfig["z_channels"], 1)
            else:
                from ldm.modules.diffusionmodules.model import Encoder, Decoder
                self.quant_conv = torch.nn.Conv2d(2 * ddconfig["z_channels"], 2 * embed_dim, 1)
                self.post_quant_conv = torch.nn.Conv2d(embed_dim, ddconfig["z_channels"], 1)
        except:
            from ldm.modules.diffusionmodules.model import Encoder, Decoder
            self.quant_conv = torch.nn.Conv2d(2 * ddconfig["z_channels"], 2 * embed_dim, 1)
            self.post_quant_conv = torch.nn.Conv2d(embed_dim, ddconfig["z_channels"], 1)
            print("INFO except!")
        self.encoder = Encoder(**ddconfig)
        self.decoder = Decoder(**ddconfig)
        self.projection = z2p(4000) if projection else None
        self.c_k = "C"  if projection else None
        self.gan_inter = gan_inter
        self.loss = instantiate_from_config(lossconfig)
        self.embed_dim = embed_dim
        if colorize_nlabels is not None:
            assert type(colorize_nlabels)==int
            self.register_buffer("colorize", torch.randn(3, colorize_nlabels, 1, 1))
        if monitor is not None:
            self.monitor = monitor
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)

    def init_from_ckpt(self, path, ignore_keys=list()):
        sd = torch.load(path, map_location="cpu")["state_dict"]
        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print("Deleting key {} from state_dict.".format(k))
                    del sd[k]
        self.load_state_dict(sd, strict=False)
        print(f"Restored from {path}")

    def encode(self, x):
        h = self.encoder(x)
        moments = self.quant_conv(h)
        posterior = DiagonalGaussianDistribution(moments)
        # print(h.shape, moments.shape, posterior.mean.shape, posterior.var.shape)
        return posterior

    def decode(self, z):
        z = self.post_quant_conv(z)
        dec = self.decoder(z)
        return dec

    def forward(self, input, sample_posterior=True, gan_inter=False):
        posterior = self.encode(input)
        if gan_inter:
            rand_id1 = torch.tensor([1, 0, 3, 2], dtype=torch.int64).cuda()  if input.size(0)==4 else torch.randperm(input.size(0))
            rand_id2 = torch.tensor([3, 2, 0, 1], dtype=torch.int64).cuda()  if input.size(0)==4 else torch.randperm(input.size(0))
            posterior_inter = posterior.mean, posterior.std
            posterior_inter1 = posterior.mean[rand_id1], posterior.std[rand_id1]
            posterior_inter2 = posterior.mean[rand_id2], posterior.std[rand_id2]
            if torch.rand(1)[0] > 0.5:
                mix = torch.cat([torch.unsqueeze(p[0], dim=0) for p in [posterior_inter, posterior_inter1, posterior_inter2]], dim=0), torch.cat([torch.unsqueeze(p[1], dim=0) for p in [posterior_inter, posterior_inter1, posterior_inter2]], dim=0)
            else:
                mix = torch.cat(
                    [torch.unsqueeze(p[0], dim=0) for p in [posterior_inter, posterior_inter2]],
                    dim=0), torch.cat(
                    [torch.unsqueeze(p[1], dim=0) for p in [posterior_inter, posterior_inter2]], dim=0)

            mean, std = product_of_experts(mix)
            z = mean + std * torch.randn(mean.shape).cuda()
            # print(mean.shape, posterior.mean.shape)
            # print(type(rand_id1), rand_id1.shape, rand_id1.dtype)
            # print(mean[0,0,5:9, 5:9, 5])
            # print(posterior.mean[0,0,5:9, 5:9, 5])
            # print(posterior_inter[0][0,0,5:9, 5:9, 5])
            # print(posterior_inter1[0][0, 0, 5:9, 5:9, 5])
            # print(posterior_inter2[0][0, 0, 5:9, 5:9, 5])
            # print(posterior.mean.shape, posterior.std.shape, mean.shape, std.shape)
        elif sample_posterior:
            z = posterior.sample()
        else:
            z = posterior.mode()

        # print(input.shape, posterior.mean.shape, posterior.var.shape, z.shape)
        print(torch.max(z[0]).detach().cpu().numpy(), torch.min(z[0]).detach().cpu().numpy())
        dec = self.decode(z)
        # print(input.shape, posterior.mean.shape, posterior.var.shape, z.shape, dec.shape)
        # print(torch.max(dec), torch.min(dec))
        return dec, posterior

    def get_input(self, batch, k):
        x = batch[k]
        if len(x.shape) == 3:
            x = x[..., None]
        if len(x.shape) == 4:
            x = x.permute(0, 3, 1, 2).to(memory_format=torch.contiguous_format).float()
        elif len(x.shape) == 5:
            x = x.permute(0, 4, 1, 2, 3).to(memory_format=torch.contiguous_format).float()
        #     b, c, h, w, l = x.shape
        # print(x.shape)
        # x = torch.ones([b, 1, 80, 80, 80], dtype=x.dtype).cuda()
        # print(x.shape)
        return x

    def training_step(self, batch, batch_idx, optimizer_idx):
        inputs= self.get_input(batch, self.image_key)

        gan_inter = self.gan_inter and torch.rand(1)[0] > 0.5

        reconstructions, posterior = self(inputs, gan_inter=gan_inter)

        if self.projection and (self.c_k is not None):
            z = posterior.sample()
            # z = posterior.mode()
            b = z.shape[0]
            # print(z.shape)
            proj2p = self.projection(torch.reshape(z, [b, -1]))
            cond = torch.reshape(batch[self.c_k], [b, -1])
            # print(proj2p.shape, cond.shape)
            # print(torch.reshape(proj2p[0], [6, 6]))
            # print(torch.reshape(cond[0], [6, 6]))
            proj2p = [proj2p, cond]
        else:
            proj2p = None

        if optimizer_idx == 0:
            # train encoder+decoder+logvar
            aeloss, log_dict_ae = self.loss(inputs, reconstructions, posterior, optimizer_idx, self.global_step,
                                            last_layer=self.get_last_layer(), split="train", proj2p=proj2p, gan_pure=gan_inter)
            self.log("aeloss", aeloss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
            self.log_dict(log_dict_ae, prog_bar=False, logger=True, on_step=True, on_epoch=False)
            return aeloss

        if optimizer_idx == 1:
            # train the discriminator
            discloss, log_dict_disc = self.loss(inputs, reconstructions, posterior, optimizer_idx, self.global_step,
                                                last_layer=self.get_last_layer(), split="train", proj2p=proj2p)

            self.log("discloss", discloss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
            self.log_dict(log_dict_disc, prog_bar=False, logger=True, on_step=True, on_epoch=False)
            return discloss

    def validation_step(self, batch, batch_idx):
        inputs = self.get_input(batch, self.image_key)
        gan_inter = self.gan_inter and torch.rand(1)[0] > 0.5

        reconstructions, posterior = self(inputs, gan_inter=gan_inter)

        if self.projection and (self.c_k is not None):
            z = posterior.sample()
            # z = posterior.mode()
            b = z.shape[0]
            # print(z.shape)
            proj2p = self.projection(torch.reshape(z, [b, -1]))
            cond = torch.reshape(batch[self.c_k], [b, -1])
            proj2p = [proj2p, cond]
        else:
            proj2p = None

        aeloss, log_dict_ae = self.loss(inputs, reconstructions, posterior, 0, self.global_step,
                                        last_layer=self.get_last_layer(), split="val", proj2p=proj2p, gan_pure=gan_inter)

        discloss, log_dict_disc = self.loss(inputs, reconstructions, posterior, 1, self.global_step,
                                            last_layer=self.get_last_layer(), split="val", proj2p=proj2p)

        self.log("val/rec_loss", log_dict_ae["val/rec_loss"])
        self.log_dict(log_dict_ae)
        self.log_dict(log_dict_disc)
        return self.log_dict

    def configure_optimizers(self):
        lr = self.learning_rate
        opt_ae = torch.optim.Adam(
                                  list(self.encoder.parameters())+
                                  list(self.decoder.parameters())+
                                  list(self.quant_conv.parameters())+
                                  list(self.post_quant_conv.parameters())+
                                  list(self.projection.parameters())+
                                  [],
                                  lr=lr, betas=(0.5, 0.9))
        opt_disc = torch.optim.Adam(self.loss.discriminator.parameters(),
                                    lr=lr, betas=(0.5, 0.9))
        return [opt_ae, opt_disc], []

    def get_last_layer(self):
        return self.decoder.conv_out.weight

    @torch.no_grad()
    def log_images(self, batch, only_inputs=False, **kwargs):
        log = dict()
        x = self.get_input(batch, self.image_key)
        x = x.to(self.device)
        # print(x.shape)
        if not only_inputs:
            xrec, posterior = self(x)
            x_inter_sample, _ = self(x, gan_inter=True)
            if x.shape[1] > 3:
                # colorize with random projection
                assert xrec.shape[1] > 3
                x = self.to_rgb(x)
                xrec = self.to_rgb(xrec)
            log["samples"] = self.decode(torch.randn_like(posterior.sample()))
            log["inter_samples"] = x_inter_sample
            log["reconstructions"] = xrec
        log["inputs"] = x
        return log

    def to_rgb(self, x):
        assert self.image_key == "segmentation"
        if not hasattr(self, "colorize"):
            self.register_buffer("colorize", torch.randn(3, x.shape[1], 1, 1).to(x))
        x = F.conv2d(x, weight=self.colorize)
        x = 2.*(x-x.min())/(x.max()-x.min()) - 1.
        return x


class AutoencoderKL(pl.LightningModule):
    def __init__(self,
                 ddconfig,
                 lossconfig,
                 embed_dim,
                 projection_s_config=None,
                 encoder_p_config=None,
                 decoder_p_config=None,
                 ckpt_path=None,
                 ignore_keys=[],
                 image_key="image",
                 colorize_nlabels=None,
                 monitor=None,
                 projection=False,
                 gan_inter=False,
                 z_norm=False,
                 ):
        super().__init__()
        self.image_key = image_key
        self.z_norm = z_norm

        assert ddconfig["double_z"]
        try:
            if ddconfig["3d"]:
                from ldm.modules.diffusionmodules.model_3d import Encoder, Decoder, z2p
                self.quant_conv = torch.nn.Conv3d(2 * ddconfig["z_channels"], 2 * embed_dim, 1)
                self.post_quant_conv = torch.nn.Conv3d(embed_dim, ddconfig["z_channels"], 1)
            else:
                from ldm.modules.diffusionmodules.model import Encoder, Decoder
                self.quant_conv = torch.nn.Conv2d(2 * ddconfig["z_channels"], 2 * embed_dim, 1)
                self.post_quant_conv = torch.nn.Conv2d(embed_dim, ddconfig["z_channels"], 1)
        except:
            from ldm.modules.diffusionmodules.model import Encoder, Decoder
            self.quant_conv = torch.nn.Conv2d(2 * ddconfig["z_channels"], 2 * embed_dim, 1)
            self.post_quant_conv = torch.nn.Conv2d(embed_dim, ddconfig["z_channels"], 1)
            print("INFO except!")
        self.encoder = Encoder(**ddconfig)
        self.decoder = Decoder(**ddconfig)
        # self.projection = E2P_projection(**projection_s_config) if projection else None
        self.encoder_p = CEmbedder_vf(**encoder_p_config) if encoder_p_config is not None else None
        self.decoder_p = Decoder_C_vf(**decoder_p_config) if decoder_p_config is not None else None
        if self.encoder_p is not None and (self.decoder_p is not None) and projection:
            self.projection = E2P_projection(**projection_s_config)
        elif projection:
            self.projection = z2p(4000)
        else:
            self.projection =None

        self.c_k = "C"  if projection else None
        self.gan_inter = gan_inter
        self.loss = instantiate_from_config(lossconfig)
        self.embed_dim = embed_dim
        if colorize_nlabels is not None:
            assert type(colorize_nlabels)==int
            self.register_buffer("colorize", torch.randn(3, colorize_nlabels, 1, 1))
        if monitor is not None:
            self.monitor = monitor
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)

    def init_from_ckpt(self, path, ignore_keys=list()):
        sd = torch.load(path, map_location="cpu")["state_dict"]
        keys = list(sd.keys())
        # print(keys)
        # print(ignore_keys)
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print("Deleting key {} from state_dict.".format(k))
                    del sd[k]
        self.load_state_dict(sd, strict=False)
        print(f"Restored from {path}")

    def encode(self, x):
        h = self.encoder(x)
        moments = self.quant_conv(h)
        posterior = DiagonalGaussianDistribution(moments)
        # print(h.shape, moments.shape, posterior.mean.shape, posterior.var.shape)
        return posterior

    def decode(self, z):
        z = self.post_quant_conv(z)
        dec = self.decoder(z)
        return dec

    def forward(self, input, sample_posterior=True, gan_inter=False, inter_idx=None):
        posterior = self.encode(input)
        if gan_inter and (input.size(0) >= 4):
            rand_id1 = torch.tensor([1, 0, 3, 2], dtype=torch.int64).cuda()  if input.size(0)==4 else torch.randperm(input.size(0))
            rand_id2 = torch.tensor([3, 2, 0, 1], dtype=torch.int64).cuda()  if input.size(0)==4 else torch.randperm(input.size(0))
            posterior_inter = posterior.mean, posterior.std
            posterior_inter1 = posterior.mean[rand_id1], posterior.std[rand_id1]
            posterior_inter2 = posterior.mean[rand_id2], posterior.std[rand_id2]
            if torch.rand(1)[0] > 0.5:
                mix = torch.cat([torch.unsqueeze(p[0], dim=0) for p in [posterior_inter, posterior_inter1, posterior_inter2]], dim=0), torch.cat([torch.unsqueeze(p[1], dim=0) for p in [posterior_inter, posterior_inter1, posterior_inter2]], dim=0)
            else:
                mix = torch.cat([torch.unsqueeze(p[0], dim=0) for p in [posterior_inter, posterior_inter2]], dim=0), torch.cat([torch.unsqueeze(p[1], dim=0) for p in [posterior_inter, posterior_inter2]], dim=0)

            mean, std = product_of_experts(mix)
            z = mean + std * torch.randn(mean.shape).cuda()
        elif gan_inter and (input.size(0) == 3):
            rand_id1 = torch.tensor([1, 2, 0], dtype=torch.int64).cuda()
            posterior_inter = posterior.mean, posterior.std
            posterior_inter1 = posterior.mean[rand_id1], posterior.std[rand_id1]
            mix = torch.cat([torch.unsqueeze(p[0], dim=0) for p in [posterior_inter, posterior_inter1]], dim=0), torch.cat([torch.unsqueeze(p[1], dim=0) for p in [posterior_inter, posterior_inter1]], dim=0)
            # print(mix[0].shape, mix[1].shape)
            mean, std = product_of_experts(mix)
            # print(mean.shape, std.shape)
            mean1, std1 = product_of_experts((torch.unsqueeze(posterior.mean, dim=1), torch.unsqueeze(posterior.std, dim=1)))
            # print(mean1.shape, std1.shape)
            pr = torch.rand(1)[0]
            if inter_idx is None:
                mean, std = torch.cat([mean, mean1], dim=0), torch.cat([std, std1], dim=0)
                # print("[INFO ]all mixed z embeddings!")
            elif pr > 0.6:
                mean, std = torch.cat([posterior.mean, mean1], dim=0), torch.cat([posterior.std, std1], dim=0)
                # print("[INFO ] mixed z embeddings! [0, 1, 2]")
            elif pr > 0.4:
                mean, std = torch.cat([posterior.mean, mean[0:1]], dim=0), torch.cat([posterior.std, std[0:1]], dim=0)
                # print("[INFO ] mixed z embeddings! [0, 1]")
            elif pr > 0.2:
                mean, std = torch.cat([posterior.mean, mean[1:2]], dim=0), torch.cat([posterior.std, std[1:2]], dim=0)
                # print("[INFO ] mixed z embeddings! [1, 2]")
            else:
                mean, std = torch.cat([posterior.mean, mean[2:3]], dim=0), torch.cat([posterior.std, std[2:3]], dim=0)
                # print("[INFO ] mixed z embeddings! [0, 2]")
            z = mean + std * torch.randn(mean.shape).cuda()
            # print(z.shape)
        elif sample_posterior:
            # print("[INFO] right right!")
            z = posterior.sample()
        else:
            z = posterior.mode()

        # print(input.shape, posterior.mean.shape, posterior.var.shape, z.shape)
        print("z range", torch.max(z[0]).detach().cpu().numpy(), torch.min(z[0]).detach().cpu().numpy())
        dec = self.decode(z)
        # print(input.shape, posterior.mean.shape, posterior.var.shape, z.shape, dec.shape)
        # print(torch.max(dec), torch.min(dec))
        return dec, posterior

    def get_input(self, batch, k):
        x = batch[k]
        if len(x.shape) == 3:
            x = x[..., None]
        if len(x.shape) == 4:
            x = x.permute(0, 3, 1, 2).to(memory_format=torch.contiguous_format).float()
        elif len(x.shape) == 5:
            x = x.permute(0, 4, 1, 2, 3).to(memory_format=torch.contiguous_format).float()
        #     b, c, h, w, l = x.shape
        # print(x.shape)
        # x = torch.ones([b, 1, 80, 80, 80], dtype=x.dtype).cuda()
        # print(x.shape)
        return x

    def training_step(self, batch, batch_idx, optimizer_idx):

        inputs= self.get_input(batch, self.image_key)

        # print(inputs.shape, batch[self.c_k].shape)
        # print(inputs[0, 0, 0:5, 0:5, 5])
        # print(inputs[1, 0, 0:5, 0:5, 5])
        # print(inputs[2, 0, 0:5, 0:5, 5])
        # print(inputs[3, 0, 0:5, 0:5, 5])

        # gan_inter = self.gan_inter and torch.rand(1)[0] > 0.5

        reconstructions, posterior = self(inputs, gan_inter=self.gan_inter, inter_idx=1)

        if self.projection and (self.c_k is not None) and (self.encoder_p is None):
            z = posterior.sample()
            # z = posterior.mode()
            b = z.shape[0]
            # print(z.shape)
            proj2p = self.projection(torch.reshape(z, [b, -1]))
            cond = torch.reshape(batch[self.c_k], [b, -1])
            proj2p = [proj2p, cond]
            # print(proj2p[0].shape, cond.shape)
            # print(proj2p[0][0])
            # print(cond[0])
        elif self.projection and (self.c_k is not None) and (self.encoder_p is not None):
            z = posterior.sample()
            # z = posterior.mode()
            # print(z.shape)
            b = z.shape[0]
            pred_z_cond = self.projection(z)
            pred_z_cond = pred_z_cond / pred_z_cond.norm(dim=-1, keepdim=True)  if self.z_norm else pred_z_cond
            pred_cond = self.decoder_p(pred_z_cond)

            cond = torch.reshape(batch[self.c_k], [b, -1])
            z_cond, cond_sim = self.encoder_p(cond, return_sim=True)

            z_cond = z_cond / z_cond.norm(dim=-1, keepdim=True) if self.z_norm else z_cond
            recon_cond = self.decoder_p(z_cond)

            # print(pred_z_cond.shape, pred_cond.shape, cond.shape, z_cond.shape, recon_cond.shape)

            # print(torch.max(pred_z_cond).detach().cpu().numpy(), torch.min(pred_z_cond).detach().cpu().numpy(), torch.max(z_cond).detach().cpu().numpy(), torch.min(z_cond).detach().cpu().numpy())
            # print(torch.max(pred_cond).detach().cpu().numpy(), torch.min(pred_cond).detach().cpu().numpy(), torch.max(cond_sim).detach().cpu().numpy(), torch.min(cond_sim).detach().cpu().numpy())
            # print("[INFO] END")

            proj2p = [pred_z_cond, z_cond, pred_cond, recon_cond, cond_sim]
        else:
            proj2p = None

        if optimizer_idx == 0:
            # train encoder+decoder+logvar
            aeloss, log_dict_ae = self.loss(inputs, reconstructions, posterior, optimizer_idx, self.global_step,
                                            last_layer=self.get_last_layer(), split="train", proj2p=proj2p, gan_pure=False)
            self.log("aeloss", aeloss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
            self.log_dict(log_dict_ae, prog_bar=False, logger=True, on_step=True, on_epoch=False)
            return aeloss

        if optimizer_idx == 1:
            # train the discriminator
            discloss, log_dict_disc = self.loss(inputs, reconstructions, posterior, optimizer_idx, self.global_step,
                                                last_layer=self.get_last_layer(), split="train", proj2p=proj2p)

            self.log("discloss", discloss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
            self.log_dict(log_dict_disc, prog_bar=False, logger=True, on_step=True, on_epoch=False)
            return discloss

    def validation_step(self, batch, batch_idx):
        inputs = self.get_input(batch, self.image_key)
        # gan_inter = self.gan_inter and torch.rand(1)[0] > 0.5

        reconstructions, posterior = self(inputs, gan_inter=self.gan_inter, inter_idx=1)

        if self.projection and (self.c_k is not None) and (self.encoder_p is None):
            z = posterior.sample()
            # z = posterior.mode()
            b = z.shape[0]
            # print(z.shape)
            proj2p = self.projection(torch.reshape(z, [b, -1]))
            cond = torch.reshape(batch[self.c_k], [b, -1])
            proj2p = [proj2p, cond]
        elif self.projection and (self.c_k is not None) and (self.encoder_p is not None):
            z = posterior.sample()
            # z = posterior.mode()
            # print(z.shape)
            b = z.shape[0]
            pred_z_cond = self.projection(z)
            pred_z_cond = pred_z_cond / pred_z_cond.norm(dim=-1, keepdim=True) if self.z_norm else pred_z_cond
            pred_cond = self.decoder_p(pred_z_cond)

            cond = torch.reshape(batch[self.c_k], [b, -1])
            z_cond, cond_sim = self.encoder_p(cond, return_sim=True)
            z_cond = z_cond / z_cond.norm(dim=-1, keepdim=True) if self.z_norm else z_cond
            recon_cond = self.decoder_p(z_cond)

            # print(pred_z_cond.shape, pred_cond.shape, cond.shape, z_cond.shape, recon_cond.shape)
            # print(torch.max(pred_z_cond).detach().cpu().numpy(), torch.min(pred_z_cond).detach().cpu().numpy(), torch.max(z_cond).detach().cpu().numpy(), torch.min(z_cond).detach().cpu().numpy())
            # print(torch.max(pred_cond).detach().cpu().numpy(), torch.min(pred_cond).detach().cpu().numpy(), torch.max(cond_sim).detach().cpu().numpy(), torch.min(cond_sim).detach().cpu().numpy())

            proj2p = [pred_z_cond, z_cond, pred_cond, recon_cond, cond_sim]
        else:
            proj2p = None

        aeloss, log_dict_ae = self.loss(inputs, reconstructions, posterior, 0, self.global_step,
                                        last_layer=self.get_last_layer(), split="val", proj2p=proj2p, gan_pure=False)

        discloss, log_dict_disc = self.loss(inputs, reconstructions, posterior, 1, self.global_step,
                                            last_layer=self.get_last_layer(), split="val", proj2p=proj2p)

        self.log("val/rec_loss", log_dict_ae["val/rec_loss"])
        self.log_dict(log_dict_ae)
        self.log_dict(log_dict_disc)
        return self.log_dict

    def configure_optimizers(self):
        lr = self.learning_rate

        # opt_ae = torch.optim.Adam(
        #     [
        #         {'params': list(self.encoder.parameters()), 'lr': lr},
        #         {'params': list(self.decoder.parameters()), 'lr': lr},
        #         {'params': list(self.quant_conv.parameters()), 'lr': lr},
        #         {'params': list(self.post_quant_conv.parameters()), 'lr': lr},
        #         {'params': list(self.projection.parameters()), 'lr': lr},
        #         # {'params': list(self.encoder_p.parameters()), 'lr': lr},
        #         {'params': list(self.decoder_p.parameters()), 'lr': lr},
        #     ], betas=(0.5, 0.9))

        opt_ae = torch.optim.Adam(
            # list(self.encoder.parameters()) +
            list(self.decoder.parameters()) +
            # list(self.quant_conv.parameters()) +
            list(self.post_quant_conv.parameters()) +
            # list(self.projection.parameters()) +
            # list(self.encoder_p.parameters()) +
            # list(self.decoder_p.parameters()) +
            [], lr=lr, betas=(0.5, 0.9))
        opt_disc = torch.optim.Adam(self.loss.discriminator.parameters(),
                                    lr=lr, betas=(0.5, 0.9))
        return [opt_ae, opt_disc], []

    def get_last_layer(self):
        return self.decoder.conv_out.weight

    @torch.no_grad()
    def log_images(self, batch, only_inputs=False, **kwargs):
        log = dict()
        x = self.get_input(batch, self.image_key)
        x = x.to(self.device)
        # print(x.shape)
        if not only_inputs:
            xrec, posterior = self(x)
            x_inter_sample, _ = self(x, gan_inter=True)
            if x.shape[1] > 3:
                # colorize with random projection
                assert xrec.shape[1] > 3
                x = self.to_rgb(x)
                xrec = self.to_rgb(xrec)
            log["samples"] = self.decode(torch.randn_like(posterior.sample()))
            log["inter_samples"] = x_inter_sample
            log["reconstructions"] = xrec
        log["inputs"] = x
        return log

    def to_rgb(self, x):
        assert self.image_key == "segmentation"
        if not hasattr(self, "colorize"):
            self.register_buffer("colorize", torch.randn(3, x.shape[1], 1, 1).to(x))
        x = F.conv2d(x, weight=self.colorize)
        x = 2.*(x-x.min())/(x.max()-x.min()) - 1.
        return x


class IdentityFirstStage(torch.nn.Module):
    def __init__(self, *args, vq_interface=False, **kwargs):
        self.vq_interface = vq_interface  # TODO: Should be true by default but check to not break older stuff
        super().__init__()

    def encode(self, x, *args, **kwargs):
        return x

    def decode(self, x, *args, **kwargs):
        return x

    def quantize(self, x, *args, **kwargs):
        if self.vq_interface:
            return x, None, [None, None, None]
        return x

    def forward(self, x, *args, **kwargs):
        return x
