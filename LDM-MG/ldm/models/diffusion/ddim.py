"""SAMPLING ONLY."""

import torch
import numpy as np
from tqdm import tqdm
from functools import partial

from ldm.modules.diffusionmodules.util import make_ddim_sampling_parameters, make_ddim_timesteps, noise_like
import math


class DDIMSampler(object):
    def __init__(self, model, schedule="linear", **kwargs):
        super().__init__()
        self.model = model
        self.ddpm_num_timesteps = model.num_timesteps
        self.schedule = schedule

    def register_buffer(self, name, attr):
        if type(attr) == torch.Tensor:
            if attr.device != torch.device("cuda"):
                attr = attr.to(torch.device("cuda"))
        setattr(self, name, attr)

    def make_schedule(self, ddim_num_steps, ddim_discretize="uniform", ddim_eta=0., verbose=True):
        self.ddim_timesteps = make_ddim_timesteps(ddim_discr_method=ddim_discretize, num_ddim_timesteps=ddim_num_steps,
                                                  num_ddpm_timesteps=self.ddpm_num_timesteps,verbose=verbose)
        alphas_cumprod = self.model.alphas_cumprod
        assert alphas_cumprod.shape[0] == self.ddpm_num_timesteps, 'alphas have to be defined for each timestep'
        to_torch = lambda x: x.clone().detach().to(torch.float32).to(self.model.device)

        self.register_buffer('betas', to_torch(self.model.betas))
        self.register_buffer('alphas_cumprod', to_torch(alphas_cumprod))
        self.register_buffer('alphas_cumprod_prev', to_torch(self.model.alphas_cumprod_prev))

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_cumprod', to_torch(np.sqrt(alphas_cumprod.cpu())))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', to_torch(np.sqrt(1. - alphas_cumprod.cpu())))
        self.register_buffer('log_one_minus_alphas_cumprod', to_torch(np.log(1. - alphas_cumprod.cpu())))
        self.register_buffer('sqrt_recip_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod.cpu())))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod.cpu() - 1)))

        # ddim sampling parameters
        ddim_sigmas, ddim_alphas, ddim_alphas_prev = make_ddim_sampling_parameters(alphacums=alphas_cumprod.cpu(),
                                                                                   ddim_timesteps=self.ddim_timesteps,
                                                                                   eta=ddim_eta,verbose=verbose)
        self.register_buffer('ddim_sigmas', ddim_sigmas)
        self.register_buffer('ddim_alphas', ddim_alphas)
        self.register_buffer('ddim_alphas_prev', ddim_alphas_prev)
        self.register_buffer('ddim_sqrt_one_minus_alphas', np.sqrt(1. - ddim_alphas))
        sigmas_for_original_sampling_steps = ddim_eta * torch.sqrt(
            (1 - self.alphas_cumprod_prev) / (1 - self.alphas_cumprod) * (
                        1 - self.alphas_cumprod / self.alphas_cumprod_prev))
        self.register_buffer('ddim_sigmas_for_original_num_steps', sigmas_for_original_sampling_steps)

    @torch.no_grad()
    def sample(self,
               S,
               batch_size,
               shape,
               conditioning=None,
               callback=None,
               normals_sequence=None,
               img_callback=None,
               quantize_x0=False,
               eta=0.,
               mask=None,
               x0=None,
               temperature=1.,
               noise_dropout=0.,
               score_corrector=None,
               corrector_kwargs=None,
               verbose=True,
               x_T=None,
               log_every_t=100,
               unconditional_guidance_scale=1.,
               unconditional_conditioning=None,
               ddim_discretize="uniform",
               # this has to come in the same format as the conditioning, # e.g. as encoded tokens, ...
               **kwargs
               ):
        # print(conditioning.shape)
        if conditioning is not None:
            if isinstance(conditioning, dict):
                cbs = conditioning[list(conditioning.keys())[0]].shape[0]
                if cbs != batch_size:
                    print(f"Warning: Got {cbs} conditionings but batch-size is {batch_size}")
            else:
                if conditioning.shape[0] != batch_size:
                    print(f"Warning: Got {conditioning.shape[0]} conditionings but batch-size is {batch_size}")

        self.make_schedule(ddim_num_steps=S, ddim_discretize=ddim_discretize, ddim_eta=eta, verbose=verbose)
        # sampling
        if len(shape) == 3:
            C, H, W = shape
            size = (batch_size, C, H, W)
        else:
            C, H, W, L = shape
            size = (batch_size, C, H, W, L)
        print(f'Data shape for DDIM sampling is {size}, eta {eta}')

        samples, intermediates = self.ddim_sampling(conditioning, size,
                                                    callback=callback,
                                                    img_callback=img_callback,
                                                    quantize_denoised=quantize_x0,
                                                    mask=mask, x0=x0,
                                                    ddim_use_original_steps=False,
                                                    noise_dropout=noise_dropout,
                                                    temperature=temperature,
                                                    score_corrector=score_corrector,
                                                    corrector_kwargs=corrector_kwargs,
                                                    x_T=x_T,
                                                    log_every_t=log_every_t,
                                                    unconditional_guidance_scale=unconditional_guidance_scale,
                                                    unconditional_conditioning=unconditional_conditioning,
                                                    )
        return samples, intermediates

    @torch.no_grad()
    def ddim_sampling(self, cond, shape,
                      x_T=None, ddim_use_original_steps=False,
                      callback=None, timesteps=None, quantize_denoised=False,
                      mask=None, x0=None, img_callback=None, log_every_t=100,
                      temperature=1., noise_dropout=0., score_corrector=None, corrector_kwargs=None,
                      unconditional_guidance_scale=1., unconditional_conditioning=None,):
        device = self.model.betas.device
        b = shape[0]
        if x_T is None:
            img = torch.randn(shape, device=device)
        else:
            img = x_T

        if timesteps is None:
            timesteps = self.ddpm_num_timesteps if ddim_use_original_steps else self.ddim_timesteps
        elif timesteps is not None and not ddim_use_original_steps:
            subset_end = int(min(timesteps / self.ddim_timesteps.shape[0], 1) * self.ddim_timesteps.shape[0]) - 1
            timesteps = self.ddim_timesteps[:subset_end]

        intermediates = {'x_inter': [img], 'pred_x0': [img]}
        time_range = reversed(range(0,timesteps)) if ddim_use_original_steps else np.flip(timesteps)
        total_steps = timesteps if ddim_use_original_steps else timesteps.shape[0]
        print(f"Running DDIM Sampling with {total_steps} timesteps")

        iterator = tqdm(time_range, desc='DDIM Sampler', total=total_steps)

        # print()

        for i, step in enumerate(iterator):
            index = total_steps - i - 1
            ts = torch.full((b,), step, device=device, dtype=torch.long)

            # print(ts, mask)
            # print(i, step)
            # print(time_range[index])

            if mask is not None:
                assert x0 is not None
                img_orig = self.model.q_sample(x0, ts)  # TODO: deterministic forward pass?
                img = img_orig * mask + (1. - mask) * img

            outs = self.p_sample_ddim(img, cond, ts, index=index, use_original_steps=ddim_use_original_steps,
                                      quantize_denoised=quantize_denoised, temperature=temperature,
                                      noise_dropout=noise_dropout, score_corrector=score_corrector,
                                      corrector_kwargs=corrector_kwargs,
                                      unconditional_guidance_scale=unconditional_guidance_scale,
                                      unconditional_conditioning=unconditional_conditioning)
            img, pred_x0 = outs
            if callback: callback(i)
            if img_callback: img_callback(pred_x0, i)

            if index % log_every_t == 0 or index == total_steps - 1:
                intermediates['x_inter'].append(img)
                intermediates['pred_x0'].append(pred_x0)

        return img, intermediates

    @torch.no_grad()
    def p_sample_ddim(self, x, c, t, index, repeat_noise=False, use_original_steps=False, quantize_denoised=False,
                      temperature=1., noise_dropout=0., score_corrector=None, corrector_kwargs=None,
                      unconditional_guidance_scale=1., unconditional_conditioning=None):
        b, *_, device = *x.shape, x.device

        # print(c.shape, t.shape)
        # print(t)

        if unconditional_conditioning is None or unconditional_guidance_scale == 1.:
            # print("condition", t.shape, c.shape, t)
            # print(torch.max(c[0]), torch.min(c[0]))
            e_t = self.model.apply_model(x, t, c)
        else:
            x_in = torch.cat([x] * 2)
            t_in = torch.cat([t] * 2)
            c_in = torch.cat([unconditional_conditioning, c])
            e_t_uncond, e_t = self.model.apply_model(x_in, t_in, c_in).chunk(2)
            e_t = e_t_uncond + unconditional_guidance_scale * (e_t - e_t_uncond)

        # print(self.model.parameterization,score_corrector)

        if score_corrector is not None:
            assert self.model.parameterization == "eps"
            e_t = score_corrector.modify_score(self.model, e_t, x, t, c, **corrector_kwargs)

        # print(use_original_steps, )
        # print(self.ddim_alphas.shape)
        # print(self.ddim_alphas_prev.shape)
        # print(self.ddim_sqrt_one_minus_alphas.shape)
        # print(self.ddim_sigmas.shape)
        # print(self.ddim_alphas)
        # print(self.ddim_sigmas)


        alphas = self.model.alphas_cumprod if use_original_steps else self.ddim_alphas
        alphas_prev = self.model.alphas_cumprod_prev if use_original_steps else self.ddim_alphas_prev
        sqrt_one_minus_alphas = self.model.sqrt_one_minus_alphas_cumprod if use_original_steps else self.ddim_sqrt_one_minus_alphas
        sigmas = self.model.ddim_sigmas_for_original_num_steps if use_original_steps else self.ddim_sigmas
        # select parameters corresponding to the currently considered timestep
        # print(len(alphas), len(alphas_prev), len(sqrt_one_minus_alphas), len(sigmas))
        # print(sigmas)
        # print(alphas)
        # print(sqrt_one_minus_alphas)

        if len(x.shape) == 5:
            a_t = torch.full((b, 1, 1, 1, 1), alphas[index], device=device)
            a_prev = torch.full((b, 1, 1, 1, 1), alphas_prev[index], device=device)
            sigma_t = torch.full((b, 1, 1, 1, 1), sigmas[index], device=device)
            sqrt_one_minus_at = torch.full((b, 1, 1, 1, 1), sqrt_one_minus_alphas[index], device=device)
        else:
            a_t = torch.full((b, 1, 1, 1), alphas[index], device=device)
            a_prev = torch.full((b, 1, 1, 1), alphas_prev[index], device=device)
            sigma_t = torch.full((b, 1, 1, 1), sigmas[index], device=device)
            sqrt_one_minus_at = torch.full((b, 1, 1, 1), sqrt_one_minus_alphas[index], device=device)
        sigma_t = sigma_t * 0.0


        # current prediction for x_0
        # print(sqrt_one_minus_at, a_t.sqrt())
        pred_x0 = e_t  #(x - sqrt_one_minus_at * e_t) / a_t.sqrt()
        # print(x.shape, pred_x0.shape, a_t.shape, sqrt_one_minus_at.shape)
        # print(a_t)
        # print(a_t.shape, sqrt_one_minus_at.shape)
        # print(a_t + sqrt_one_minus_at ** 2.0, sigma_t)
        e_t = (x - pred_x0 * a_t.sqrt()) / sqrt_one_minus_at
        if quantize_denoised:
            pred_x0, _, *_ = self.model.first_stage_model.quantize(pred_x0)
        # direction pointing to x_t
        dir_xt = (1. - a_prev - sigma_t**2).sqrt() * e_t
        noise = sigma_t * noise_like(x.shape, device, repeat_noise) * temperature
        if noise_dropout > 0.:
            noise = torch.nn.functional.dropout(noise, p=noise_dropout)
        x_prev = a_prev.sqrt() * pred_x0 + dir_xt + noise
        return x_prev, pred_x0


    @torch.no_grad()
    def sample_interp(self,
                       S,
                       batch_size,
                       shape,
                       img1=None,
                       img2=None,
                       conditioning=None,
                       callback=None,
                       normals_sequence=None,
                       img_callback=None,
                       quantize_x0=False,
                       eta=0.,
                       mask=None,
                       x0=None,
                       temperature=1.,
                       noise_dropout=0.,
                       score_corrector=None,
                       corrector_kwargs=None,
                       verbose=True,
                       x_T=None,
                       log_every_t=100,
                       unconditional_guidance_scale=1.,
                       unconditional_conditioning=None,
                       ddim_discretize="uniform",
                       # this has to come in the same format as the conditioning, # e.g. as encoded tokens, ...
                       **kwargs
                       ):
                if conditioning is not None:
                    if isinstance(conditioning, dict):
                        cbs = conditioning[list(conditioning.keys())[0]].shape[0]
                        if cbs != batch_size:
                            print(f"Warning: Got {cbs} conditionings but batch-size is {batch_size}")
                    else:
                        if conditioning.shape[0] != batch_size:
                            print(f"Warning: Got {conditioning.shape[0]} conditionings but batch-size is {batch_size}")

                self.make_schedule(ddim_num_steps=S, ddim_discretize=ddim_discretize, ddim_eta=eta, verbose=verbose)
                # sampling
                if len(shape) == 3:
                    C, H, W = shape
                    size = (1, C, H, W)
                else:
                    C, H, W, L = shape
                    size = (1, C, H, W, L)
                print(f'Data shape for inverse DDIM sampling is {size}, eta {eta}')
                # print(img1.shape, img2.shape)

                noise1, _ = self.ddim_interp(conditioning, size,
                                                          callback=callback,
                                                          img_callback=img_callback,
                                                          quantize_denoised=quantize_x0,
                                                          mask=mask, x0=x0,
                                                          ddim_use_original_steps=False,
                                                          noise_dropout=noise_dropout,
                                                          temperature=temperature,
                                                          score_corrector=score_corrector,
                                                          corrector_kwargs=corrector_kwargs,
                                                          x_T=img1,
                                                          log_every_t=log_every_t,
                                                          unconditional_guidance_scale=unconditional_guidance_scale,
                                                          unconditional_conditioning=unconditional_conditioning,
                                                          )
                noise2, _ = self.ddim_interp(conditioning, size,
                                             callback=callback,
                                             img_callback=img_callback,
                                             quantize_denoised=quantize_x0,
                                             mask=mask, x0=x0,
                                             ddim_use_original_steps=False,
                                             noise_dropout=noise_dropout,
                                             temperature=temperature,
                                             score_corrector=score_corrector,
                                             corrector_kwargs=corrector_kwargs,
                                             x_T=img2,
                                             log_every_t=log_every_t,
                                             unconditional_guidance_scale=unconditional_guidance_scale,
                                             unconditional_conditioning=unconditional_conditioning,
                                             )

                # print(noise1.shape, noise2.shape)
                noise_interp = []
                for i in range(batch_size):
                    theta = i / (batch_size - 1) * math.pi / 4 + math.pi / 8 #4 + math.pi / 8 #  2  #3 + math.pi / 12
                    noise_interp.append(noise1 * math.cos(theta) + noise2 * math.sin(theta))
                noise_interp = torch.cat(noise_interp, dim=0)
                # print(noise_interp.shape)

                # sampling
                if len(shape) == 3:
                    C, H, W = shape
                    size = (batch_size, C, H, W)
                else:
                    C, H, W, L = shape
                    size = (batch_size, C, H, W, L)
                print(f'Data shape for DDIM sampling is {size}, eta {eta}')
                conditioning = conditioning.repeat(batch_size, 1, 1)
                # print(size, conditioning.shape, noise_interp.shape)

                samples, intermediates = self.ddim_sampling(conditioning, size,
                                                            callback=callback,
                                                            img_callback=img_callback,
                                                            quantize_denoised=quantize_x0,
                                                            mask=mask, x0=x0,
                                                            ddim_use_original_steps=False,
                                                            noise_dropout=noise_dropout,
                                                            temperature=temperature,
                                                            score_corrector=score_corrector,
                                                            corrector_kwargs=corrector_kwargs,
                                                            x_T=noise_interp,
                                                            log_every_t=log_every_t,
                                                            unconditional_guidance_scale=unconditional_guidance_scale,
                                                            unconditional_conditioning=unconditional_conditioning,
                                                            )

                return samples, intermediates


    @torch.no_grad()
    def ddim_interp(self, cond, shape,
                      x_T=None, ddim_use_original_steps=False,
                      callback=None, timesteps=None, quantize_denoised=False,
                      mask=None, x0=None, img_callback=None, log_every_t=100,
                      temperature=1., noise_dropout=0., score_corrector=None, corrector_kwargs=None,
                      unconditional_guidance_scale=1., unconditional_conditioning=None,):
        device = self.model.betas.device
        b = shape[0]
        if x_T is None:
            img = torch.randn(shape, device=device)
        else:
            img = x_T
        # print(img.shape)

        if timesteps is None:
            timesteps = self.ddpm_num_timesteps if ddim_use_original_steps else self.ddim_timesteps
        elif timesteps is not None and not ddim_use_original_steps:
            subset_end = int(min(timesteps / self.ddim_timesteps.shape[0], 1) * self.ddim_timesteps.shape[0]) - 1
            timesteps = self.ddim_timesteps[:subset_end]

        intermediates = {'x_inter': [img], 'pred_x0': [img]}
        time_range = reversed(range(0,timesteps)) if ddim_use_original_steps else np.flip(timesteps)
        total_steps = timesteps if ddim_use_original_steps else timesteps.shape[0]
        print(f"Running DDIM Sampling with {total_steps} timesteps")

        iterator = tqdm(time_range, desc='DDIM Sampler', total=total_steps)

        # print(time_range)

        for i, step in enumerate(iterator):
            index = i
            ts = torch.full((b,), time_range[total_steps - i - 1], device=device, dtype=torch.long)

            # print(ts, mask)

            if mask is not None:
                assert x0 is not None
                img_orig = self.model.q_sample(x0, ts)  # TODO: deterministic forward pass?
                img = img_orig * mask + (1. - mask) * img

            outs = self.p_sample_ddim_reverse(img, cond, ts, index=index, use_original_steps=ddim_use_original_steps,
                                              quantize_denoised=quantize_denoised, temperature=temperature,
                                              noise_dropout=noise_dropout, score_corrector=score_corrector,
                                              corrector_kwargs=corrector_kwargs,
                                              unconditional_guidance_scale=unconditional_guidance_scale,
                                              unconditional_conditioning=unconditional_conditioning)
            img, pred_x0 = outs
            if callback: callback(i)
            if img_callback: img_callback(pred_x0, i)

            if index % log_every_t == 0 or index == total_steps - 1:
                intermediates['x_inter'].append(img)
                intermediates['pred_x0'].append(pred_x0)

        return img, intermediates


    @torch.no_grad()
    def p_sample_ddim_reverse(self, x, c, t, index, repeat_noise=False, use_original_steps=False, quantize_denoised=False,
                      temperature=1., noise_dropout=0., score_corrector=None, corrector_kwargs=None,
                      unconditional_guidance_scale=1., unconditional_conditioning=None):
        b, *_, device = *x.shape, x.device

        if unconditional_conditioning is None or unconditional_guidance_scale == 1.:
            # print("condition", t.shape, c.shape, t)
            # print(torch.max(c[0]), torch.min(c[0]))
            e_t = self.model.apply_model(x, t, c)
        else:
            x_in = torch.cat([x] * 2)
            t_in = torch.cat([t] * 2)
            c_in = torch.cat([unconditional_conditioning, c])
            e_t_uncond, e_t = self.model.apply_model(x_in, t_in, c_in).chunk(2)
            e_t = e_t_uncond + unconditional_guidance_scale * (e_t - e_t_uncond)

        # print(self.model.parameterization,score_corrector)

        if score_corrector is not None:
            assert self.model.parameterization == "eps"
            e_t = score_corrector.modify_score(self.model, e_t, x, t, c, **corrector_kwargs)

        # print(use_original_steps, )
        # print(self.ddim_alphas.shape)
        # print(self.ddim_alphas_prev.shape)
        # print(self.ddim_sqrt_one_minus_alphas.shape)
        # print(self.ddim_sigmas.shape)
        # print(self.ddim_alphas)
        # print(self.ddim_sigmas)


        alphas = self.model.alphas_cumprod if use_original_steps else self.ddim_alphas
        alphas_prev = self.model.alphas_cumprod_prev if use_original_steps else self.ddim_alphas_prev
        # sqrt_one_minus_alphas = self.model.sqrt_one_minus_alphas_cumprod if use_original_steps else self.ddim_sqrt_one_minus_alphas
        sigmas = self.model.ddim_sigmas_for_original_num_steps if use_original_steps else self.ddim_sigmas
        # select parameters corresponding to the currently considered timestep
        # print(len(alphas), len(alphas_prev), len(sqrt_one_minus_alphas), len(sigmas))
        # print(sigmas)
        # alphas = alphas[::-1]
        # alphas_prev = alphas_prev[::-1]
        # sqrt_one_minus_alphas = sqrt_one_minus_alphas[::-1]


        if len(x.shape) == 5:
            a_prev = torch.full((b, 1, 1, 1, 1), alphas[index], device=device)
            a_t = torch.full((b, 1, 1, 1, 1), alphas_prev[index], device=device)
            sigma_t = torch.full((b, 1, 1, 1, 1), sigmas[index], device=device)
            # sqrt_one_minus_at_prev = torch.full((b, 1, 1, 1, 1), sqrt_one_minus_alphas[index], device=device)
            sqrt_one_minus_at = torch.full((b, 1, 1, 1, 1), 1.0 - alphas_prev[index], device=device)
            sqrt_one_minus_at = sqrt_one_minus_at.sqrt()
        else:
            a_prev = torch.full((b, 1, 1, 1), alphas[index], device=device)
            a_t = torch.full((b, 1, 1, 1), alphas_prev[index], device=device)
            sigma_t = torch.full((b, 1, 1, 1), sigmas[index], device=device)
            # sqrt_one_minus_at_prev = torch.full((b, 1, 1, 1), sqrt_one_minus_alphas[index], device=device)
            sqrt_one_minus_at = torch.full((b, 1, 1, 1), 1.0 - alphas_prev[index], device=device)
            sqrt_one_minus_at = sqrt_one_minus_at.sqrt()
        sigma_t = sigma_t * 0.0


        # current prediction for x_0
        # print(sqrt_one_minus_at, a_t.sqrt())
        pred_x0 = e_t  #(x - sqrt_one_minus_at * e_t) / a_t.sqrt()
        # print(x.shape, pred_x0.shape, a_t.shape, sqrt_one_minus_at.shape)
        # print(a_t)
        # print(a_t.shape, sqrt_one_minus_at.shape)
        # print(a_t + sqrt_one_minus_at ** 2.0, sigma_t)
        e_t = (x - pred_x0 * a_t.sqrt()) / sqrt_one_minus_at
        if quantize_denoised:
            pred_x0, _, *_ = self.model.first_stage_model.quantize(pred_x0)
        # direction pointing to x_t
        dir_xt = (1. - a_prev - sigma_t**2).sqrt() * e_t
        noise = sigma_t * noise_like(x.shape, device, repeat_noise) * temperature
        if noise_dropout > 0.:
            noise = torch.nn.functional.dropout(noise, p=noise_dropout)
        x_prev = a_prev.sqrt() * pred_x0 + dir_xt + noise
        return x_prev, pred_x0