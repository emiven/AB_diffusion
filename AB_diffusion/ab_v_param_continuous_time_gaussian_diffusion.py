import math
import torch
from torch import sqrt
from torch import nn, einsum
import torch.nn.functional as F
from torch.special import expm1
from torch.cuda.amp import autocast

from tqdm import tqdm
from einops import rearrange, repeat, reduce
from einops.layers.torch import Rearrange

# helpers

def exists(val):
    return val is not None

def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d


# diffusion helpers

def right_pad_dims_to(x, t):
    padding_dims = x.ndim - t.ndim
    if padding_dims <= 0:
        return t
    return t.view(*t.shape, *((1,) * padding_dims))

# continuous schedules
# log(snr) that approximates the original linear schedule

def log(t, eps = 1e-20):
    return torch.log(t.clamp(min = eps))

def alpha_cosine_log_snr(t, s = 0.008):
    return -log((torch.cos((t + s) / (1 + s) * math.pi * 0.5) ** -2) - 1, eps = 1e-5)

class ABVParamContinuousTimeGaussianDiffusion(nn.Module):
    """
    a new type of parameterization in v-space proposed in https://arxiv.org/abs/2202.00512 that
    (1) allows for improved distillation over noise prediction objective and
    (2) noted in imagen-video to improve upsampling unets by removing the color shifting artifacts
    """

    def __init__(
        self,
        model,
        *,
        image_size,

        num_timesteps = 500,
        clip_sample_denoised = True,
    ):
        super().__init__()
        assert model.random_or_learned_sinusoidal_cond

        self.objective = 'pred_v'
        self.beta_schedule = 'cosine'
        self.offset_noise_strength = 0
        self.image_size = image_size
        self.model = model

        # image dimensions

        self.channels = self.model.out_dim

        # continuous noise schedule related stuff

        self.log_snr = alpha_cosine_log_snr

        # sampling

        self.num_timesteps = num_timesteps
        self.clip_sample_denoised = clip_sample_denoised        

    @property
    def device(self):
        return next(self.model.parameters()).device

    def p_mean_variance(self, x, conditioning,time, time_next):
        # reviewer found an error in the equation in the paper (missing sigma)
        # following - https://openreview.net/forum?id=2LdBqxc1Yv&noteId=rIQgH0zKsRt

        log_snr = self.log_snr(time)
        log_snr_next = self.log_snr(time_next)
        c = -expm1(log_snr - log_snr_next)

        squared_alpha, squared_alpha_next = log_snr.sigmoid(), log_snr_next.sigmoid()
        squared_sigma, squared_sigma_next = (-log_snr).sigmoid(), (-log_snr_next).sigmoid()

        alpha, sigma, alpha_next = map(sqrt, (squared_alpha, squared_sigma, squared_alpha_next))

        batch_log_snr = repeat(log_snr, ' -> b', b = x.shape[0])

        pred_v = self.model(x, batch_log_snr,conditioning)

        # shown in Appendix D in the paper
        x_start = alpha * x - sigma * pred_v

        if self.clip_sample_denoised:
            x_start.clamp_(-1., 1.)

        model_mean = alpha_next * (x * (1 - c) / alpha + c * x_start)

        posterior_variance = squared_sigma_next * c

        return model_mean, posterior_variance

    # sampling related functions

    @torch.no_grad()
    def p_sample(self, x, time, time_next,conditioning):
        batch, *_, device = *x.shape, x.device

        model_mean, model_variance = self.p_mean_variance(x = x, time = time,
                                                           time_next = time_next,
                                                           conditioning = conditioning
                                                           )

        if time_next == 0:
            return model_mean

        noise = torch.randn_like(x)
        return model_mean + sqrt(model_variance) * noise

    @torch.no_grad()
    def p_sample_loop(self,conditioning, shape):
        batch = shape[0]

        img = torch.randn(shape, device = self.device)
        steps = torch.linspace(1., 0., self.num_timesteps + 1, device = self.device)

        for i in tqdm(range(self.num_timesteps), desc = 'sampling loop time step', total = self.num_timesteps):
            times = steps[i]
            times_next = steps[i + 1]
            img = self.p_sample(img, times, times_next,conditioning)

        img.clamp_(-1., 1.)
        return img

    @torch.no_grad()
    def sample(self,conditioning):
        batch_size,height,width = conditioning.shape[0], conditioning.shape[-2],conditioning.shape[-1]

        return self.p_sample_loop(conditioning,(batch_size, self.channels, height, width))

    # training related functions - noise prediction

    @autocast(enabled = False)
    def q_sample(self, x_start, times, noise = None):
        noise = default(noise, lambda: torch.randn_like(x_start))

        log_snr = self.log_snr(times)

        log_snr_padded = right_pad_dims_to(x_start, log_snr)
        alpha, sigma = sqrt(log_snr_padded.sigmoid()), sqrt((-log_snr_padded).sigmoid())
        x_noised =  x_start * alpha + noise * sigma

        return x_noised, log_snr, alpha, sigma

    def random_times(self, batch_size):
        return torch.zeros((batch_size,), device = self.device).float().uniform_(0, 1)

    def p_losses(self, x_start, times,conditioning, noise = None):
        noise = default(noise, lambda: torch.randn_like(x_start))

        x, log_snr, alpha, sigma = self.q_sample(x_start = x_start, times = times, noise = noise)

        # described in section 4 as the prediction objective, with derivation in Appendix D
        v = alpha * noise - sigma * x_start

        model_out = self.model(x, log_snr, conditioning)

        return F.mse_loss(model_out, v)

    def forward(self, img,conditioning, *args, **kwargs):
        b, c, h, w, device, img_size, = *img.shape, img.device, conditioning.shape[-1]
        assert h == img_size and w == img_size, f'height and width of image must be {img_size}'

        times = self.random_times(b)
        return self.p_losses(img, times,conditioning, *args, **kwargs)
