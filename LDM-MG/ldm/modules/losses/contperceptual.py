import torch
import torch.nn as nn
from taming.modules.losses.vqperceptual import *  # TODO: taming dependency yes/no?
from ldm.modules.diffusionmodules.discriminator import weights_init


class LPIPSWithDiscriminator_ori(nn.Module):
    def __init__(self, disc_start, logvar_init=0.0, kl_weight=1.0, pixelloss_weight=1.0,
                 disc_num_layers=3, disc_in_channels=3, disc_factor=1.0, disc_weight=1.0,
                 perceptual_weight=1.0, use_actnorm=False, disc_conditional=False,
                 disc_loss="hinge", is_3d=False):

        super().__init__()
        assert disc_loss in ["hinge", "vanilla"]
        self.kl_weight = kl_weight
        self.pixel_weight = pixelloss_weight
        self.perceptual_loss = LPIPS().eval()
        self.perceptual_weight = perceptual_weight
        # output log variance
        self.logvar = nn.Parameter(torch.ones(size=()) * logvar_init)
        if is_3d:
            from ldm.modules.diffusionmodules.discriminator import NLayerDiscriminator_3d as NLayerDiscriminator
        else:
            from ldm.modules.diffusionmodules.discriminator import NLayerDiscriminator
        self.discriminator = NLayerDiscriminator(input_nc=disc_in_channels,
                                                 n_layers=disc_num_layers,
                                                 use_actnorm=use_actnorm
                                                 ).apply(weights_init)

        self.discriminator_iter_start = disc_start
        self.disc_loss = hinge_d_loss if disc_loss == "hinge" else vanilla_d_loss
        self.disc_factor = disc_factor
        self.discriminator_weight = disc_weight
        self.disc_conditional = disc_conditional

    def calculate_adaptive_weight(self, nll_loss, g_loss, last_layer=None):
        if last_layer is not None:
            nll_grads = torch.autograd.grad(nll_loss, last_layer, retain_graph=True)[0]
            g_grads = torch.autograd.grad(g_loss, last_layer, retain_graph=True)[0]
        else:
            nll_grads = torch.autograd.grad(nll_loss, self.last_layer[0], retain_graph=True)[0]
            g_grads = torch.autograd.grad(g_loss, self.last_layer[0], retain_graph=True)[0]

        d_weight = torch.norm(nll_grads) / (torch.norm(g_grads) + 1e-4)
        d_weight = torch.clamp(d_weight, 0.0, 1e4).detach()
        d_weight = d_weight * self.discriminator_weight
        return d_weight

    def forward(self, inputs, reconstructions, posteriors, optimizer_idx,
                global_step, last_layer=None, cond=None, split="train",
                weights=0.00005, proj2p=None, gan_pure=False):
        # print(inputs.shape, reconstructions.shape, posteriors.mean.shape, optimizer_idx)

        # print(proj2p[0].shape, proj2p[1].shape)
        mask = torch.tensor([
            [2.0, 1.0, 1.0, 0, 0, 0],
            [1.0, 2.0, 1.0, 0, 0, 0],
            [1.0, 1.0, 2.0, 0, 0, 0],
            [0, 0, 0, 2.0, 0, 0],
            [0, 0, 0, 0, 2.0, 0],
            [0, 0, 0, 0, 0, 2.0]
        ], dtype=torch.float32).to(proj2p[0].device)
        sum_mask = torch.sum(mask)
        # print(sum_mask)
        mask = torch.reshape(mask, [1, -1]) / sum_mask * (6 ** 2)
        # print(mask.shape, proj2p[0].shape, proj2p[1].shape)

        if proj2p is not None:
            b = proj2p[1].shape[0]
            label = proj2p[1].reshape([b, -1])
            # print(label.shape) #, proj2p.shape, mask.shape
            label_i = 1.0 / label
            if torch.isnan(label_i).any() or torch.isinf(label_i).any():
                print("[WARNING] there are nan in label_C_inverse!")
                change_label_i = torch.where(
                    torch.isinf(label_i),
                    torch.full_like(label_i, 0.1),
                    label_i)
                change_label_i1 = torch.where(
                    torch.isinf(change_label_i),
                    torch.full_like(change_label_i, 0.1),
                    change_label_i)
                label_i = change_label_i1
            cls_loss_l1_r = (torch.abs((proj2p[0] - label) * label_i) * mask)
            if torch.isnan(cls_loss_l1_r).any() or torch.isinf(cls_loss_l1_r).any():
                print("[WARNING] there are nan in cls loss l1_r!")
                cls_loss_l1_r = (torch.abs((proj2p[0] - label) / 1e-1) * mask).mean()
            else:
                cls_loss_l1_r = cls_loss_l1_r.mean()
            cls_loss_l1 = ((torch.abs(proj2p[0] - label)) * mask).mean()
            cls_loss_l2 = (((proj2p[0] - label) ** 2) * mask).mean()


        # print(torch.diag(proj2p[0][0].reshape([6, 6]) / 20.0).detach())
        # print(torch.diag(proj2p[1][0].reshape([6, 6])))
        print("std range", torch.max(posteriors.std[0]).detach().cpu().numpy(), torch.min(posteriors.std[0]).detach().cpu().numpy())
        # print(torch.max(proj2p[0] / 20.0))


        rec_loss = torch.abs(inputs.contiguous() - reconstructions.contiguous())
        if self.perceptual_weight > 0:
            # print(inputs.shape, reconstructions.shape)
            p_loss = self.perceptual_loss(inputs.contiguous(), reconstructions.contiguous())
            rec_loss = rec_loss + self.perceptual_weight * p_loss

        nll_loss = rec_loss / torch.exp(self.logvar) + self.logvar
        weighted_nll_loss = nll_loss
        if weights is not None:
            weighted_nll_loss = weights*nll_loss
        weighted_nll_loss = torch.sum(weighted_nll_loss) / weighted_nll_loss.shape[0]
        nll_loss = torch.sum(nll_loss) / nll_loss.shape[0]
        kl_loss = posteriors.kl()
        kl_loss = torch.sum(kl_loss) / kl_loss.shape[0]
        # print(optimizer_idx)

        # now the GAN part
        if optimizer_idx == 0:
            # generator update
            if cond is None:
                assert not self.disc_conditional
                logits_fake = self.discriminator(reconstructions.contiguous())
            else:
                assert self.disc_conditional
                logits_fake = self.discriminator(torch.cat((reconstructions.contiguous(), cond), dim=1))
            g_loss = -torch.mean(logits_fake)

            if self.disc_factor > 0.0:
                try:
                    d_weight = self.calculate_adaptive_weight(nll_loss, g_loss, last_layer=last_layer)
                except RuntimeError:
                    assert not self.training
                    d_weight = torch.tensor(0.0)
            else:
                d_weight = torch.tensor(0.0)

            disc_factor = adopt_weight(self.disc_factor, global_step, threshold=self.discriminator_iter_start)

            if not gan_pure:
                loss = weighted_nll_loss + self.kl_weight * kl_loss + d_weight * disc_factor * g_loss * 0.0001
            else:
                loss = d_weight * disc_factor * g_loss
            print(gan_pure, "kl", self.kl_weight, kl_loss.detach().cpu().numpy())
            print("rec", weighted_nll_loss.detach().cpu().numpy(), self.logvar)
            print("gan", d_weight.detach().cpu().numpy(), disc_factor, g_loss.detach().cpu().numpy())
            print("total_loss", loss.detach().cpu().numpy())

            log = {"{}/total_loss".format(split): loss.clone().detach().mean(), "{}/logvar".format(split): self.logvar.detach(),
                   "{}/kl_loss".format(split): kl_loss.detach().mean(), "{}/nll_loss".format(split): nll_loss.detach().mean(),
                   "{}/rec_loss".format(split): rec_loss.detach().mean(),
                   "{}/d_weight".format(split): d_weight.detach(),
                   "{}/disc_factor".format(split): torch.tensor(disc_factor),
                   "{}/g_loss".format(split): g_loss.detach().mean()
                   }

            if proj2p is not None:
                # loss = self.kl_weight * kl_loss
                loss = loss + cls_loss_l2 * 20 + cls_loss_l1_r * 0.5 #+ cls_loss_l1 * 10 #+ cls_loss_l1_r # loss + cls_loss * 300000
                print("cls loss l2", cls_loss_l2.detach().cpu().numpy())
                print("cls loss l1", cls_loss_l1.detach().cpu().numpy())
                print("cls loss l1_r", cls_loss_l1_r.detach().cpu().numpy())
                log["{}/cls_loss_l2".format(split)] = cls_loss_l2.detach().mean()
                log["{}/cls_loss_l1".format(split)] = cls_loss_l1.detach().mean()
                log["{}/cls_loss_l1_r".format(split)] = cls_loss_l1_r.detach().mean()
                # print(torch.diag(proj2p[0][0].reshape([6, 6]) / 20.0).detach())
                # print(torch.diag(proj2p[1][0].reshape([6, 6])).detach())
            return loss, log

        if optimizer_idx == 1:
            # second pass for discriminator update
            if cond is None:
                logits_real = self.discriminator(inputs.contiguous().detach())
                logits_fake = self.discriminator(reconstructions.contiguous().detach())
            else:
                logits_real = self.discriminator(torch.cat((inputs.contiguous().detach(), cond), dim=1))
                logits_fake = self.discriminator(torch.cat((reconstructions.contiguous().detach(), cond), dim=1))

            disc_factor = adopt_weight(self.disc_factor, global_step, threshold=self.discriminator_iter_start)
            g_loss = self.disc_loss(logits_real, logits_fake)
            d_loss = disc_factor * g_loss

            print("gan_d", disc_factor, g_loss.detach().cpu().numpy())

            log = {"{}/disc_loss".format(split): d_loss.clone().detach().mean(),
                   "{}/logits_real".format(split): logits_real.detach().mean(),
                   "{}/logits_fake".format(split): logits_fake.detach().mean(),
                   # "{}/cls_loss".format(split): cls_loss.detach().mean(),
                   }
            return d_loss, log



class LPIPSWithDiscriminator(nn.Module):
    def __init__(self, disc_start, logvar_init=0.0, kl_weight=1.0, pixelloss_weight=1.0,
                 disc_num_layers=3, disc_in_channels=3, disc_factor=1.0, disc_weight=1.0,
                 perceptual_weight=1.0, use_actnorm=False, disc_conditional=False,
                 disc_loss="hinge", is_3d=False):

        super().__init__()
        assert disc_loss in ["hinge", "vanilla"]
        self.kl_weight = kl_weight
        self.pixel_weight = pixelloss_weight
        # self.perceptual_loss = LPIPS().eval()
        self.perceptual_weight = perceptual_weight
        # output log variance
        self.logvar = nn.Parameter(torch.ones(size=()) * logvar_init)
        if is_3d:
            from ldm.modules.diffusionmodules.discriminator import NLayerDiscriminator_3d as NLayerDiscriminator
        else:
            from ldm.modules.diffusionmodules.discriminator import NLayerDiscriminator
        self.discriminator = NLayerDiscriminator(input_nc=disc_in_channels,
                                                 n_layers=disc_num_layers,
                                                 use_actnorm=use_actnorm
                                                 ).apply(weights_init)

        self.discriminator_iter_start = disc_start
        self.disc_loss = hinge_d_loss if disc_loss == "hinge" else vanilla_d_loss
        self.disc_factor = disc_factor
        self.discriminator_weight = disc_weight
        self.disc_conditional = disc_conditional

    def calculate_adaptive_weight(self, nll_loss, g_loss, last_layer=None):
        if last_layer is not None:
            nll_grads = torch.autograd.grad(nll_loss, last_layer, retain_graph=True)[0]
            g_grads = torch.autograd.grad(g_loss, last_layer, retain_graph=True)[0]
        else:
            nll_grads = torch.autograd.grad(nll_loss, self.last_layer[0], retain_graph=True)[0]
            g_grads = torch.autograd.grad(g_loss, self.last_layer[0], retain_graph=True)[0]

        d_weight = torch.norm(nll_grads) / (torch.norm(g_grads) + 1e-4)
        d_weight = torch.clamp(d_weight, 0.0, 1e4).detach()
        d_weight = d_weight * self.discriminator_weight
        return d_weight

    def forward(self, inputs, reconstructions, posteriors, optimizer_idx,
                global_step, last_layer=None, cond=None, split="train",
                weights=0.0005, proj2p=None, gan_pure=False):
        # print(inputs.shape, reconstructions.shape, posteriors.mean.shape, optimizer_idx)

        # reconstructions = reconstructions[:inputs.size(0)]

        # print(proj2p[0].shape, proj2p[1].shape)
        mask = torch.tensor([
            [2.0, 1.0, 1.0, 0, 0, 0],
            [1.0, 2.0, 1.0, 0, 0, 0],
            [1.0, 1.0, 2.0, 0, 0, 0],
            [0, 0, 0, 2.0, 0, 0],
            [0, 0, 0, 0, 2.0, 0],
            [0, 0, 0, 0, 0, 2.0]
        ], dtype=torch.float32).to(proj2p[0].device)
        sum_mask = torch.sum(mask)
        # print(sum_mask)
        mask = torch.reshape(mask, [1, -1]) / sum_mask * (6 ** 2)
        # print(mask.shape, proj2p[0].shape, proj2p[1].shape)


        if proj2p is not None and len(proj2p) == 2:
            b = proj2p[1].shape[0]
            label = proj2p[1].reshape([b, -1])
            # print(label.shape) #, proj2p.shape, mask.shape
            # print("label range", torch.max(label*mask).detach().cpu().numpy(), torch.min(label*mask).detach().cpu().numpy(), label.shape)
            # print("pred range", torch.max(proj2p[0]*mask).detach().cpu().numpy(), torch.min(proj2p[0]*mask).detach().cpu().numpy(), proj2p[0].shape)
            label_i = 1.0 / label
            if torch.isnan(label_i).any() or torch.isinf(label_i).any():
                print("[WARNING] there are nan in label_C_inverse!")
                change_label_i = torch.where(
                    torch.isinf(label_i),
                    torch.full_like(label_i, 0.1),
                    label_i)
                change_label_i1 = torch.where(
                    torch.isinf(change_label_i),
                    torch.full_like(change_label_i, 0.1),
                    change_label_i)
                label_i = change_label_i1
            cls_loss_l1_r = (torch.abs((proj2p[0] - label) * label_i) * mask)
            if torch.isnan(cls_loss_l1_r).any() or torch.isinf(cls_loss_l1_r).any():
                print("[WARNING] there are nan in cls loss l1_r!")
                cls_loss_l1_r = (torch.abs((proj2p[0] - label) / 1e-1) * mask).mean()
            else:
                cls_loss_l1_r = cls_loss_l1_r.mean()
            cls_loss_l2 = (((proj2p[0] - label) ** 2) * mask).mean()
            cls_loss_l1 = ((torch.abs(proj2p[0] - label)) * mask).mean()

        elif proj2p is not None and len(proj2p) == 5:
            # for tensor in proj2p:
            #     print(tensor.shape)
            pred_z_cond, z_cond, pred_cond, recon_cond, label_cond = proj2p
            # pred_z_cond = pred_z_cond / pred_z_cond.norm(dim=-1, keepdim=True)
            # z_cond = z_cond / z_cond.norm(dim=-1, keepdim=True)
            z_cond_loss_l1 = torch.abs(pred_z_cond - z_cond).mean()
            z_cond_loss_l2 = ((pred_z_cond - z_cond) ** 2).mean()

            label_cond_i = 1.0 / label_cond
            if torch.isnan(label_cond_i).any() or torch.isinf(label_cond_i).any():
                print("[WARNING] there are nan or inf in label_C_inverse!")
                change_label_i = torch.where(
                    torch.isinf(label_cond_i),
                    torch.full_like(label_cond_i, 0.001),
                    label_cond_i)
                change_label_i1 = torch.where(
                    torch.isinf(change_label_i),
                    torch.full_like(change_label_i, 0.001),
                    change_label_i)
                label_cond_i = change_label_i1

            rec_cond_loss_l1 = torch.abs(recon_cond - label_cond).mean()
            rec_cond_loss_l2 = ((recon_cond - label_cond) ** 2).mean()
            rec_cond_loss_l1_r = torch.abs((recon_cond - label_cond) * label_cond_i).mean()
            pred_cond_loss_l1 = torch.abs(pred_cond - label_cond).mean()
            pred_cond_loss_l2 = ((pred_cond - label_cond) ** 2).mean()
            pred_cond_loss_l1_r = torch.abs((pred_cond - label_cond) * label_cond_i).mean()


        # print(torch.diag(proj2p[0][0].reshape([6, 6]) / 20.0).detach())
        # print(torch.diag(proj2p[1][0].reshape([6, 6])))
        print("std range", torch.max(posteriors.std[0]).detach().cpu().numpy(), torch.min(posteriors.std[0]).detach().cpu().numpy())
        # print(torch.max(proj2p[0] / 20.0))


        rec_loss = torch.abs(inputs.contiguous() - reconstructions[:inputs.size(0)].contiguous())
        if self.perceptual_weight > 0:
            # print(inputs.shape, reconstructions.shape)
            p_loss = self.perceptual_loss(inputs.contiguous(), reconstructions[:inputs.size(0)].contiguous())
            rec_loss = rec_loss + self.perceptual_weight * p_loss

        nll_loss = rec_loss / torch.exp(self.logvar) + self.logvar
        weighted_nll_loss = nll_loss
        if weights is not None:
            weighted_nll_loss = weights*nll_loss
        weighted_nll_loss = torch.sum(weighted_nll_loss) / weighted_nll_loss.shape[0]
        nll_loss = torch.sum(nll_loss) / nll_loss.shape[0]
        kl_loss = posteriors.kl()
        kl_loss = torch.sum(kl_loss) / kl_loss.shape[0]
        # print(optimizer_idx)

        # now the GAN part
        if optimizer_idx == 0:
            # generator update
            if cond is None:
                assert not self.disc_conditional
                logits_fake = self.discriminator(reconstructions.contiguous())
            else:
                assert self.disc_conditional
                logits_fake = self.discriminator(torch.cat((reconstructions.contiguous(), cond), dim=1))
            g_loss = -torch.mean(logits_fake)

            if self.disc_factor > 0.0:
                try:
                    d_weight = self.calculate_adaptive_weight(nll_loss, g_loss, last_layer=last_layer)
                except RuntimeError:
                    assert not self.training
                    d_weight = torch.tensor(0.0)
            else:
                d_weight = torch.tensor(0.0)

            disc_factor = adopt_weight(self.disc_factor, global_step, threshold=self.discriminator_iter_start)

            if not gan_pure:
                loss = weighted_nll_loss + self.kl_weight * kl_loss + d_weight * disc_factor * g_loss * 0.01
            else:
                loss = d_weight * disc_factor * g_loss
            print(gan_pure, "kl", self.kl_weight, kl_loss.detach().cpu().numpy())
            print("rec", rec_loss.detach().mean().cpu().numpy(), weighted_nll_loss.detach().cpu().numpy(), self.logvar.detach().cpu().numpy())
            print("gan", d_weight.detach().cpu().numpy(), disc_factor, g_loss.detach().cpu().numpy())

            log = {"{}/total_loss".format(split): loss.clone().detach().mean(), "{}/logvar".format(split): self.logvar.detach(),
                   "{}/kl_loss".format(split): kl_loss.detach().mean(), "{}/nll_loss".format(split): nll_loss.detach().mean(),
                   "{}/rec_loss".format(split): rec_loss.detach().mean(),
                   "{}/d_weight".format(split): d_weight.detach(),
                   "{}/disc_factor".format(split): torch.tensor(disc_factor),
                   "{}/g_loss".format(split): g_loss.detach().mean()
                   }

            if proj2p is not None and len(proj2p) == 2:
                # loss = self.kl_weight * kl_loss
                # loss = 0
                loss = loss
                # loss = loss + cls_loss_l2 * 20 + cls_loss_l1_r * 2 + cls_loss_l1 * 10  #+ cls_loss_l1_r # loss + cls_loss * 300000
                print("cls loss l2", cls_loss_l2.detach().cpu().numpy())
                print("cls loss l1", cls_loss_l1.detach().cpu().numpy())
                print("cls loss l1_r", cls_loss_l1_r.detach().cpu().numpy())
                log["{}/cls_loss_l2".format(split)] = cls_loss_l2.detach().mean()
                log["{}/cls_loss_l1".format(split)] = cls_loss_l1.detach().mean()
                log["{}/cls_loss_l1_r".format(split)] = cls_loss_l1_r.detach().mean()
                # print(torch.diag(proj2p[0][0].reshape([6, 6]) / 20.0).detach())
                # print(torch.diag(proj2p[1][0].reshape([6, 6])).detach())
            elif proj2p is not None and len(proj2p) == 5:
                rec_cond_loss = rec_cond_loss_l1 * 0.0 + rec_cond_loss_l2 * 300.0 + rec_cond_loss_l1_r * 0.0
                z_cond_loss = z_cond_loss_l1 * 0.5 + z_cond_loss_l2 * 500
                pred_cond_loss = pred_cond_loss_l1 * 0.0 + pred_cond_loss_l2 * 400.0 + pred_cond_loss_l1_r * 0.0
                # loss = 0
                # loss = self.kl_weight * kl_loss
                loss = loss + rec_cond_loss * 0.0 + z_cond_loss * 0.0 + pred_cond_loss * 0.0
                print("rec cond loss: l1 {}; l2 {}; l1_r {}".format(str(rec_cond_loss_l1.detach().cpu().numpy()), str(rec_cond_loss_l2.detach().cpu().numpy()), str(rec_cond_loss_l1_r.detach().cpu().numpy())))
                print("pred cond loss: l1 {}; l2 {}; l1_r {}".format(str(pred_cond_loss_l1.detach().cpu().numpy()), str(pred_cond_loss_l2.detach().cpu().numpy()), str(pred_cond_loss_l1_r.detach().cpu().numpy())))
                print("z cond loss: l1 {}; l2 {}".format(str(z_cond_loss_l1.detach().cpu().numpy()), str(z_cond_loss_l2.detach().cpu().numpy())))
                log["{}/rec_cond_loss_l1".format(split)] = rec_cond_loss_l1.detach().mean()
                log["{}/rec_cond_loss_l2".format(split)] = rec_cond_loss_l2.detach().mean()
                log["{}/rec_cond_loss_l1_r".format(split)] = rec_cond_loss_l1_r.detach().mean()
                log["{}/z_cond_loss_l1".format(split)] = z_cond_loss_l1.detach().mean()
                log["{}/z_cond_loss_l2".format(split)] = z_cond_loss_l2.detach().mean()
                log["{}/pred_cond_loss_l1".format(split)] = pred_cond_loss_l1.detach().mean()
                log["{}/pred_cond_loss_l2".format(split)] = pred_cond_loss_l2.detach().mean()
                log["{}/pred_cond_loss_l1_r".format(split)] = pred_cond_loss_l1_r.detach().mean()
            print("total_loss", loss.detach().cpu().numpy())

            return loss, log

        if optimizer_idx == 1:
            # second pass for discriminator update
            if cond is None:
                logits_real = self.discriminator(inputs.contiguous().detach())
                logits_fake = self.discriminator(reconstructions.contiguous().detach())
            else:
                logits_real = self.discriminator(torch.cat((inputs.contiguous().detach(), cond), dim=1))
                logits_fake = self.discriminator(torch.cat((reconstructions.contiguous().detach(), cond), dim=1))

            disc_factor = adopt_weight(self.disc_factor, global_step, threshold=self.discriminator_iter_start)
            g_loss = self.disc_loss(logits_real, logits_fake)
            d_loss = disc_factor * g_loss

            print("gan_d", disc_factor, g_loss.detach().cpu().numpy())

            log = {"{}/disc_loss".format(split): d_loss.clone().detach().mean(),
                   "{}/logits_real".format(split): logits_real.detach().mean(),
                   "{}/logits_fake".format(split): logits_fake.detach().mean(),
                   # "{}/cls_loss".format(split): cls_loss.detach().mean(),
                   }
            return d_loss, log

